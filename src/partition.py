"""
ZeRO-Q Partition Module.

Core partitioning logic for quantized tensors. Handles:
- Computing aligned partition sizes
- Quantizing and partitioning tensors
- Gathering and dequantizing partitions
"""

from dataclasses import dataclass
from typing import Tuple, Optional, TYPE_CHECKING
import torch
import torch.nn.functional as F
import torch.distributed as dist

# Conditional import for bitsandbytes
try:
    import bitsandbytes as bnb
    from bitsandbytes.functional import quantize_4bit, dequantize_4bit
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False
    quantize_4bit = None
    dequantize_4bit = None

if TYPE_CHECKING:
    from bitsandbytes.functional import QuantState


@dataclass
class PartitionInfo:
    """
    Information about a quantized partition.
    
    Stores all metadata needed to reconstruct the original tensor
    from partitioned quantized shards.
    
    Attributes:
        rank: This process's rank
        world_size: Total number of processes
        original_shape: Shape of the original tensor
        original_numel: Number of elements in original tensor
        original_dtype: Original tensor dtype
        packed_partition_size: Size of packed data per rank
        absmax_partition_size: Size of absmax per rank
        blocksize: Quantization block size
        elements_per_rank: Elements per rank (after alignment)
        padded_numel: Total elements after padding
    """
    rank: int
    world_size: int
    
    # Original tensor info
    original_shape: torch.Size
    original_numel: int
    original_dtype: torch.dtype
    
    # Partition sizes
    packed_partition_size: int
    absmax_partition_size: int
    
    # Alignment info
    blocksize: int
    elements_per_rank: int
    padded_numel: int
    
    @property
    def needs_padding(self) -> bool:
        """Check if tensor needed padding for alignment."""
        return self.padded_numel > self.original_numel
    
    @property
    def padding_elements(self) -> int:
        """Number of padding elements added."""
        return self.padded_numel - self.original_numel


def compute_aligned_partition_sizes(
    numel: int,
    world_size: int,
    blocksize: int = 64
) -> Tuple[int, int, int, int]:
    """
    Compute partition sizes aligned to quantization blocks.
    
    For proper quantization, partition boundaries must align with
    blocksize boundaries. This ensures each partition contains
    complete quantization groups (weights + their scale values).
    
    Args:
        numel: Total number of elements in the tensor
        world_size: Number of partitions (GPUs)
        blocksize: Quantization block size (default 64)
    
    Returns:
        elements_per_rank: Elements per GPU (aligned to blocksize)
        packed_per_rank: Packed bytes per GPU (numel // 2 for 4-bit)
        absmax_per_rank: Number of absmax values per GPU
        padded_numel: Total elements after padding
    
    Example:
        >>> elements, packed, absmax, total = compute_aligned_partition_sizes(
        ...     numel=4096, world_size=4, blocksize=64
        ... )
        >>> print(f"Each GPU gets {elements} elements, {packed} packed bytes")
        Each GPU gets 1024 elements, 512 packed bytes
    """
    # Calculate total blocks needed
    num_blocks = (numel + blocksize - 1) // blocksize
    
    # Ensure blocks divide evenly across ranks
    # Round up to nearest multiple of world_size
    if num_blocks % world_size != 0:
        aligned_blocks = ((num_blocks + world_size - 1) // world_size) * world_size
    else:
        aligned_blocks = num_blocks
    
    # Calculate per-rank sizes
    blocks_per_rank = aligned_blocks // world_size
    elements_per_rank = blocks_per_rank * blocksize
    padded_numel = aligned_blocks * blocksize
    
    # For 4-bit quantization: 2 elements packed into 1 byte
    packed_per_rank = elements_per_rank // 2
    
    # One absmax value per block
    absmax_per_rank = blocks_per_rank
    
    return elements_per_rank, packed_per_rank, absmax_per_rank, padded_numel


def partition_quantized_tensor(
    weight: torch.Tensor,
    rank: int,
    world_size: int,
    blocksize: int = 64,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.float32,
) -> Tuple[torch.Tensor, torch.Tensor, "QuantState", PartitionInfo]:
    """
    Quantize a weight tensor and partition it across ranks.
    
    This is the core ZeRO-Q operation: take a full weight tensor,
    quantize it to 4-bit, and return only this rank's partition.
    
    The tensor is:
    1. Flattened to 1D
    2. Padded to align with blocksize × world_size
    3. Quantized to 4-bit NF4/FP4
    4. Partitioned - each rank gets 1/world_size of packed data and absmax
    
    Args:
        weight: Full weight tensor to partition
        rank: This process's rank (0 to world_size-1)
        world_size: Total number of processes
        blocksize: Quantization block size (64, 128, 256, etc.)
        quant_type: Quantization type ('nf4' or 'fp4')
        compute_dtype: Dtype for dequantized computation
    
    Returns:
        local_packed: This rank's packed weights (uint8, N/2 bytes)
        local_absmax: This rank's absmax values (float16/32)
        quant_state: Quantization state (codebook, metadata)
        partition_info: Metadata for reconstruction
    
    Raises:
        ImportError: If bitsandbytes is not installed
        ValueError: If rank >= world_size
    
    Example:
        >>> weight = torch.randn(4096, 4096)
        >>> packed, absmax, state, info = partition_quantized_tensor(
        ...     weight, rank=0, world_size=4
        ... )
        >>> print(f"Local partition: {packed.shape}, {absmax.shape}")
    """
    if not BNB_AVAILABLE:
        raise ImportError(
            "bitsandbytes is required for ZeRO-Q quantization. "
            "Install with: pip install bitsandbytes>=0.43.1"
        )
    
    if rank >= world_size:
        raise ValueError(f"rank ({rank}) must be < world_size ({world_size})")
    
    # Store original info
    original_shape = weight.shape
    original_numel = weight.numel()
    original_dtype = weight.dtype
    
    # Flatten for quantization
    weight_flat = weight.detach().contiguous().view(-1)
    
    # Compute aligned partition sizes
    elements_per_rank, packed_per_rank, absmax_per_rank, padded_numel = \
        compute_aligned_partition_sizes(original_numel, world_size, blocksize)
    
    # Pad if necessary to ensure alignment
    if padded_numel > original_numel:
        padding = padded_numel - original_numel
        weight_flat = torch.nn.functional.pad(weight_flat, (0, padding), value=0.0)
    
    # Convert to compute dtype for quantization
    weight_flat = weight_flat.to(compute_dtype)
    
    # Quantize the full tensor
    # NOTE: In production, we'd want to quantize directly to partitions
    # to avoid materializing the full tensor. This is for initial implementation.
    packed, quant_state = quantize_4bit(
        weight_flat,
        blocksize=blocksize,
        quant_type=quant_type,
    )
    
    # Extract this rank's partition
    packed_start = rank * packed_per_rank
    packed_end = (rank + 1) * packed_per_rank
    local_packed = packed[packed_start:packed_end].clone()
    
    absmax_start = rank * absmax_per_rank
    absmax_end = (rank + 1) * absmax_per_rank
    local_absmax = quant_state.absmax[absmax_start:absmax_end].clone()
    
    # Create partition info
    partition_info = PartitionInfo(
        rank=rank,
        world_size=world_size,
        original_shape=original_shape,
        original_numel=original_numel,
        original_dtype=original_dtype,
        packed_partition_size=packed_per_rank,
        absmax_partition_size=absmax_per_rank,
        blocksize=blocksize,
        elements_per_rank=elements_per_rank,
        padded_numel=padded_numel,
    )
    
    return local_packed, local_absmax, quant_state, partition_info


def gather_and_dequantize(
    local_packed: torch.Tensor,
    local_absmax: torch.Tensor,
    quant_state: "QuantState",
    partition_info: PartitionInfo,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
) -> torch.Tensor:
    """
    Gather quantized partitions from all ranks and dequantize.
    
    This is called before forward/backward passes to reconstruct
    the full weight tensor for computation.
    
    Process:
    1. All-gather packed data from all ranks
    2. All-gather absmax from all ranks
    3. Concatenate to form full quantized tensor
    4. Dequantize using bitsandbytes
    5. Remove padding and reshape to original shape
    
    Args:
        local_packed: This rank's packed weights
        local_absmax: This rank's absmax values
        quant_state: Quantization state with codebook
        partition_info: Partition metadata
        group: Process group (None = default world group)
        async_op: If True, returns handle instead of waiting
    
    Returns:
        Full dequantized weight tensor with original shape
    
    Example:
        >>> # After partition_quantized_tensor...
        >>> full_weight = gather_and_dequantize(
        ...     local_packed, local_absmax, quant_state, partition_info
        ... )
        >>> assert full_weight.shape == original_shape
    """
    if not BNB_AVAILABLE:
        raise ImportError("bitsandbytes is required for ZeRO-Q")
    
    world_size = partition_info.world_size
    
    # Handle non-distributed case (single GPU or testing)
    if not dist.is_initialized() or world_size == 1:
        # Already have full tensor (no gather needed)
        from bitsandbytes.functional import QuantState as BnBQuantState
        
        full_state = BnBQuantState(
            absmax=local_absmax,
            shape=torch.Size([partition_info.elements_per_rank]),
            dtype=quant_state.dtype,
            blocksize=quant_state.blocksize,
            code=quant_state.code,
            quant_type=quant_state.quant_type,
        )
        
        full_weight = dequantize_4bit(local_packed, full_state)
        full_weight = full_weight[:partition_info.original_numel]
        return full_weight.view(partition_info.original_shape)
    
    # Allocate gather buffers
    packed_buffers = [
        torch.empty_like(local_packed) for _ in range(world_size)
    ]
    absmax_buffers = [
        torch.empty_like(local_absmax) for _ in range(world_size)
    ]
    
    # All-gather packed data
    packed_handle = dist.all_gather(
        packed_buffers,
        local_packed,
        group=group,
        async_op=True,  # Always async internally
    )
    
    # All-gather absmax
    absmax_handle = dist.all_gather(
        absmax_buffers,
        local_absmax,
        group=group,
        async_op=True,
    )
    
    # Wait for communication
    packed_handle.wait()
    absmax_handle.wait()
    
    # Concatenate gathered tensors
    full_packed = torch.cat(packed_buffers, dim=0)
    full_absmax = torch.cat(absmax_buffers, dim=0)
    
    # Rebuild QuantState with gathered absmax
    from bitsandbytes.functional import QuantState as BnBQuantState
    
    full_state = BnBQuantState(
        absmax=full_absmax,
        shape=torch.Size([partition_info.padded_numel]),
        dtype=quant_state.dtype,
        blocksize=quant_state.blocksize,
        code=quant_state.code,
        quant_type=quant_state.quant_type,
    )
    
    # Dequantize
    full_weight = dequantize_4bit(full_packed, full_state)
    
    # Remove padding and reshape to original
    full_weight = full_weight[:partition_info.original_numel]
    full_weight = full_weight.view(partition_info.original_shape)
    
    return full_weight


def repartition_tensor(
    full_weight: torch.Tensor,
    partition_info: PartitionInfo,
    quant_type: str = "nf4",
) -> Tuple[torch.Tensor, torch.Tensor, "QuantState"]:
    """
    Re-quantize and re-partition a tensor after modification.
    
    Called after optimizer step if weights were updated.
    For QLoRA with frozen base, this is rarely needed since
    base weights don't change during training.
    
    Args:
        full_weight: Full weight tensor to re-partition
        partition_info: Original partition info
        quant_type: Quantization type
    
    Returns:
        local_packed: Re-partitioned packed weights
        local_absmax: Re-partitioned absmax values
        quant_state: New quantization state
    """
    if not BNB_AVAILABLE:
        raise ImportError("bitsandbytes is required for ZeRO-Q")
    
    rank = partition_info.rank
    world_size = partition_info.world_size
    blocksize = partition_info.blocksize
    
    # Flatten and pad
    weight_flat = full_weight.contiguous().view(-1)
    if weight_flat.numel() < partition_info.padded_numel:
        padding = partition_info.padded_numel - weight_flat.numel()
        weight_flat = torch.nn.functional.pad(weight_flat, (0, padding), value=0.0)
    
    # Re-quantize
    packed, quant_state = quantize_4bit(
        weight_flat,
        blocksize=blocksize,
        quant_type=quant_type,
    )
    
    # Extract local partition
    packed_start = rank * partition_info.packed_partition_size
    packed_end = (rank + 1) * partition_info.packed_partition_size
    local_packed = packed[packed_start:packed_end].clone()
    
    absmax_start = rank * partition_info.absmax_partition_size
    absmax_end = (rank + 1) * partition_info.absmax_partition_size
    local_absmax = quant_state.absmax[absmax_start:absmax_end].clone()
    
    return local_packed, local_absmax, quant_state


# Utility functions

def estimate_memory_savings(
    numel: int,
    world_size: int,
    original_dtype: torch.dtype = torch.float16,
) -> dict:
    """
    Estimate memory savings from ZeRO-Q.
    
    Args:
        numel: Number of elements in tensor
        world_size: Number of GPUs
        original_dtype: Original tensor dtype
    
    Returns:
        Dictionary with memory statistics
    """
    # Original size
    dtype_bytes = {
        torch.float32: 4,
        torch.float16: 2,
        torch.bfloat16: 2,
    }
    original_bytes = numel * dtype_bytes.get(original_dtype, 4)
    
    # ZeRO-3 (partitioned fp16)
    zero3_per_gpu = (numel * 2) // world_size
    
    # ZeRO-Q (partitioned 4-bit + absmax)
    packed_per_gpu = (numel // 2) // world_size  # 4-bit = 0.5 bytes/element
    absmax_per_gpu = (numel // 64) // world_size * 2  # float16 absmax
    zeroq_per_gpu = packed_per_gpu + absmax_per_gpu
    
    return {
        "original_total_bytes": original_bytes,
        "zero3_per_gpu_bytes": zero3_per_gpu,
        "zeroq_per_gpu_bytes": zeroq_per_gpu,
        "memory_reduction": zero3_per_gpu / zeroq_per_gpu if zeroq_per_gpu > 0 else 0,
        "communication_reduction": 2.0 / 0.53,  # FP16 vs 4-bit+absmax
    }


def partition_fp32(
    tensor: torch.Tensor,
    world_size: int,
    rank: int,
) -> Tuple[torch.Tensor, torch.Size]:
    """
    Shard an fp32 tensor evenly across ranks. No quantization.

    Each rank receives a contiguous slice of the flattened tensor.
    The last rank's slice is zero-padded to match the chunk size of
    other ranks so that ``all_gather`` works with uniform tensor sizes.

    Args:
        tensor: Full weight tensor (any dtype — will be cast to float32).
        world_size: Number of GPUs / ranks.
        rank: This process's rank (0-indexed).

    Returns:
        local_shard: This rank's fp32 shard (shape ``[chunk_size]``).
        original_shape: Shape of the original tensor (for reconstruction).
    """
    flat = tensor.detach().contiguous().view(-1).to(torch.float32)
    chunk_size = (flat.numel() + world_size - 1) // world_size
    start = rank * chunk_size
    end = min(start + chunk_size, flat.numel())
    shard = flat[start:end].clone()
    if shard.numel() < chunk_size:
        shard = F.pad(shard, (0, chunk_size - shard.numel()))
    return shard, tensor.shape


def gather_fp32(
    local_shard: torch.Tensor,
    original_shape: torch.Size,
    world_size: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    All-gather fp32 shards and reconstruct the full tensor.

    Args:
        local_shard: This rank's fp32 shard (from ``partition_fp32``).
        original_shape: Original tensor shape (for reshape + trim).
        world_size: Number of GPUs / ranks.
        group: ``torch.distributed`` process group (``None`` = default).

    Returns:
        Full fp32 tensor with ``original_shape``.
    """
    original_numel = 1
    for s in original_shape:
        original_numel *= s

    if not dist.is_initialized() or world_size == 1:
        return local_shard[:original_numel].view(original_shape)

    gathered = [torch.empty_like(local_shard) for _ in range(world_size)]
    dist.all_gather(gathered, local_shard, group=group)
    full = torch.cat(gathered)[:original_numel]
    return full.view(original_shape)
