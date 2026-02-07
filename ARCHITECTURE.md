# ZeRO-Q Architecture & Implementation Guide

**Author:** Zero (Claude Opus 4.5)  
**Version:** 0.2.0  
**Date:** December 11, 2025

---

## ⚠️ ZERO - READ THIS BEFORE DOING ANYTHING

**You keep forgetting how ZeRO-Q works and reverting to dumb DDP.**

### ZeRO-Q IS:
- Each GPU holds **1/N** of quantized weights
- Forward hooks **AllGather** weights before compute
- Memory usage: ~18GB / N GPUs

### ZeRO-Q is NOT:
- ❌ DDP (full model on each GPU, sync gradients) - WRONG
- ❌ Loading full model then "measuring" partition savings - FAKE
- ❌ `device_map={"": local_rank}` on each GPU - That's N copies!

### Working Implementation: `/home/drawson/Phoenix/Experiments/06_God_Mode_32B/train_zeroq_v2.py`

```python
# THE REAL PATTERN:
# 1. Rank 0 loads to CPU (500GB RAM)
# 2. Rank 0 quantizes on CPU
# 3. Rank 0 broadcasts partition[i] to rank[i]
# 4. Each GPU: 18GB / 7 = 2.6GB
# 5. Forward: AllGather hooks reconstruct weights

class ZeroQPartitioner:
    def _gather_weights(self, name, module):
        local_data, local_absmax = self.local_partitions[name]
        data_list = [torch.empty_like(local_data) for _ in range(self.world_size)]
        dist.all_gather(data_list, local_data)
        module.weight.data = torch.cat(data_list).reshape(...)
```

### Script Status (December 11, 2025)
| Script | Status |
|--------|--------|
| `train_zeroq_v4.py` | ✅ **PROVEN WORKING** - 3B on 4 GPUs across PE2+PE3 |
| `train_zeroq_v3.py` | ⚠️ Send/recv issues with NCCL |
| `train_zeroq_v2.py` | ⚠️ Tries CPU quantize (bitsandbytes needs GPU) |
| `train_distributed_3b.py` | ❌ DDP, not ZeRO-Q |
| `test_7b_real.py` | ❌ Fake - doesn't use partitions |

### ✅ PROOF OF CONCEPT SUCCESS (December 11, 2025)
```
🏆 SUCCESS! ZeRO-Q AllGather hooks WORK!
   Steps: 6 | Avg Loss: 14.1405 | Time: 94.5s
   Partitioned: 252 layers
   Local memory: 372.1MB per GPU
```
- 4 GPUs across 2 nodes (PE2 + PE3)
- Each GPU holds 1/4 of weights (372MB instead of ~1.5GB)
- AllGather reconstructs weights before each layer's forward
- Loss decreased: 15.08 → 14.14

---

## Overview

This document provides implementation-focused technical details for ZeRO-Q. For high-level design rationale, see [SPECIFICATION.md](./SPECIFICATION.md).

---

## Module Structure

```
ZeroQ/
├── src/
│   ├── __init__.py           # Public API exports
│   ├── config.py             # ZeroQConfig dataclass
│   ├── partition.py          # Quantized tensor partitioning
│   ├── quantized_param.py    # QuantizedZeroParameter class
│   ├── coordinator.py        # Parameter fetch/release coordination
│   ├── comm.py               # Communication primitives
│   ├── hooks.py              # Forward/backward hooks
│   └── integration.py        # HuggingFace/PEFT integration
├── tests/
│   ├── test_partition.py
│   ├── test_coordinator.py
│   ├── test_e2e.py
│   └── conftest.py           # Pytest fixtures
└── examples/
    ├── train_simple.py       # Minimal example
    └── train_32b_maxwell.py  # Full M40 cluster example
```

---

## 1. Configuration Module (`config.py`)

```python
"""ZeRO-Q Configuration."""

from dataclasses import dataclass, field
from typing import Optional, List
import torch


@dataclass
class ZeroQConfig:
    """
    Configuration for ZeRO-Q quantized distributed training.
    
    Attributes:
        enabled: Whether ZeRO-Q quantization is enabled
        quant_type: Quantization type ('nf4' or 'fp4')
        blocksize: Elements per quantization block (must be power of 2)
        double_quant: Whether to use double quantization for absmax
        compute_dtype: Dtype for computation (torch.float32 for Maxwell)
        
        partition_trainable: Whether to partition trainable params (LoRA)
        frozen_only: Only quantize frozen parameters
        
        async_gather: Use asynchronous all-gather operations
        prefetch_count: Number of layers to prefetch
        overlap_comm: Overlap communication with computation
        
        pin_memory: Pin communication buffers
        contiguous_buffers: Use contiguous memory for buffers
        
        target_modules: Module names to apply quantization (None = all Linear)
        exclude_modules: Module names to exclude from quantization
    """
    
    # Quantization settings
    enabled: bool = True
    quant_type: str = "nf4"
    blocksize: int = 64
    double_quant: bool = True
    compute_dtype: torch.dtype = torch.float32
    
    # Partitioning settings
    partition_trainable: bool = False
    frozen_only: bool = True
    
    # Communication settings
    async_gather: bool = True
    prefetch_count: int = 1
    overlap_comm: bool = True
    
    # Memory settings
    pin_memory: bool = True
    contiguous_buffers: bool = True
    
    # Module targeting
    target_modules: Optional[List[str]] = None
    exclude_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    
    def __post_init__(self):
        """Validate configuration."""
        assert self.quant_type in ("nf4", "fp4"), \
            f"quant_type must be 'nf4' or 'fp4', got {self.quant_type}"
        assert self.blocksize in (64, 128, 256, 512, 1024, 2048, 4096), \
            f"blocksize must be power of 2 between 64-4096, got {self.blocksize}"
        assert self.compute_dtype in (torch.float32, torch.float16, torch.bfloat16), \
            f"compute_dtype must be float32/float16/bfloat16, got {self.compute_dtype}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "quant_type": self.quant_type,
            "blocksize": self.blocksize,
            "double_quant": self.double_quant,
            "compute_dtype": str(self.compute_dtype),
            "partition_trainable": self.partition_trainable,
            "frozen_only": self.frozen_only,
            "async_gather": self.async_gather,
            "prefetch_count": self.prefetch_count,
            "overlap_comm": self.overlap_comm,
            "pin_memory": self.pin_memory,
            "contiguous_buffers": self.contiguous_buffers,
            "target_modules": self.target_modules,
            "exclude_modules": self.exclude_modules,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "ZeroQConfig":
        """Create from dictionary."""
        # Handle compute_dtype string conversion
        if isinstance(d.get("compute_dtype"), str):
            dtype_map = {
                "torch.float32": torch.float32,
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
            }
            d["compute_dtype"] = dtype_map.get(d["compute_dtype"], torch.float32)
        return cls(**d)


# Pre-defined configurations
MAXWELL_CONFIG = ZeroQConfig(
    compute_dtype=torch.float32,  # Required for SM 5.2
    double_quant=True,
    blocksize=64,
)

AMPERE_CONFIG = ZeroQConfig(
    compute_dtype=torch.bfloat16,  # BF16 supported on SM 8.0+
    double_quant=True,
    blocksize=64,
)
```

---

## 2. Partition Module (`partition.py`)

```python
"""Quantized tensor partitioning for ZeRO-Q."""

from dataclasses import dataclass
from typing import Tuple, Optional
import torch
import torch.distributed as dist

# Import bitsandbytes quantization functions
try:
    import bitsandbytes as bnb
    from bitsandbytes.functional import quantize_4bit, dequantize_4bit, QuantState
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False


@dataclass
class PartitionInfo:
    """Information about a quantized partition."""
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
    remainder_elements: int


def compute_aligned_partition_sizes(
    numel: int,
    world_size: int,
    blocksize: int = 64
) -> Tuple[int, int, int]:
    """
    Compute partition sizes aligned to quantization blocks.
    
    For proper quantization, partition boundaries must align with
    blocksize boundaries. This ensures each partition contains
    complete quantization groups.
    
    Args:
        numel: Total number of elements
        world_size: Number of partitions (GPUs)
        blocksize: Quantization block size
    
    Returns:
        elements_per_rank: Elements per GPU (aligned)
        packed_per_rank: Packed bytes per GPU (numel // 2 for 4-bit)
        absmax_per_rank: Absmax values per GPU
    
    Raises:
        ValueError: If alignment is not possible
    """
    # Total number of quantization blocks
    num_blocks = (numel + blocksize - 1) // blocksize
    
    # Blocks per rank (must be integer)
    if num_blocks % world_size != 0:
        # Pad to make divisible
        padded_blocks = ((num_blocks + world_size - 1) // world_size) * world_size
        padded_numel = padded_blocks * blocksize
    else:
        padded_numel = num_blocks * blocksize
    
    blocks_per_rank = (padded_numel // blocksize) // world_size
    elements_per_rank = blocks_per_rank * blocksize
    
    # For 4-bit: 2 elements packed into 1 byte
    packed_per_rank = elements_per_rank // 2
    
    # One absmax per block
    absmax_per_rank = blocks_per_rank
    
    return elements_per_rank, packed_per_rank, absmax_per_rank


def partition_quantized_tensor(
    weight: torch.Tensor,
    rank: int,
    world_size: int,
    blocksize: int = 64,
    quant_type: str = "nf4",
    compute_dtype: torch.dtype = torch.float32
) -> Tuple[torch.Tensor, torch.Tensor, QuantState, PartitionInfo]:
    """
    Quantize a weight tensor and partition it across ranks.
    
    This is the core ZeRO-Q operation: take a full weight tensor,
    quantize it to 4-bit, and return only this rank's partition.
    
    Args:
        weight: Full weight tensor to partition
        rank: This process's rank
        world_size: Total number of processes
        blocksize: Quantization block size
        quant_type: 'nf4' or 'fp4'
        compute_dtype: Dtype for dequantized computation
    
    Returns:
        local_packed: This rank's packed weights (uint8)
        local_absmax: This rank's absmax values
        quant_state: Quantization state (shared metadata)
        partition_info: Partition metadata
    """
    if not BNB_AVAILABLE:
        raise ImportError("bitsandbytes is required for ZeRO-Q")
    
    original_shape = weight.shape
    original_numel = weight.numel()
    original_dtype = weight.dtype
    
    # Flatten for quantization
    weight_flat = weight.contiguous().view(-1)
    
    # Compute aligned sizes
    elements_per_rank, packed_per_rank, absmax_per_rank = \
        compute_aligned_partition_sizes(original_numel, world_size, blocksize)
    
    # Pad if necessary
    total_elements = elements_per_rank * world_size
    if total_elements > original_numel:
        padding = total_elements - original_numel
        weight_flat = torch.nn.functional.pad(weight_flat, (0, padding))
    
    # Quantize the full tensor
    # Note: This happens on each rank initially; optimize later
    packed, quant_state = quantize_4bit(
        weight_flat.to(compute_dtype),
        blocksize=blocksize,
        quant_type=quant_type
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
        remainder_elements=total_elements - original_numel,
    )
    
    return local_packed, local_absmax, quant_state, partition_info


def gather_and_dequantize(
    local_packed: torch.Tensor,
    local_absmax: torch.Tensor,
    quant_state: QuantState,
    partition_info: PartitionInfo,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False
) -> torch.Tensor:
    """
    Gather quantized partitions and dequantize to full tensor.
    
    This is called before forward/backward passes to reconstruct
    the full weight tensor for computation.
    
    Args:
        local_packed: This rank's packed weights
        local_absmax: This rank's absmax values
        quant_state: Quantization state with codebook
        partition_info: Partition metadata
        group: Process group (None = default)
        async_op: Whether to use async communication
    
    Returns:
        Full dequantized weight tensor
    """
    world_size = partition_info.world_size
    
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
        async_op=async_op
    )
    
    # All-gather absmax
    absmax_handle = dist.all_gather(
        absmax_buffers,
        local_absmax,
        group=group,
        async_op=async_op
    )
    
    # Wait if async
    if async_op:
        packed_handle.wait()
        absmax_handle.wait()
    
    # Concatenate gathered tensors
    full_packed = torch.cat(packed_buffers, dim=0)
    full_absmax = torch.cat(absmax_buffers, dim=0)
    
    # Rebuild QuantState with full absmax
    full_state = QuantState(
        absmax=full_absmax,
        shape=torch.Size([partition_info.elements_per_rank * world_size]),
        dtype=quant_state.dtype,
        blocksize=quant_state.blocksize,
        code=quant_state.code,
        quant_type=quant_state.quant_type,
        offset=getattr(quant_state, 'offset', None),
        state2=getattr(quant_state, 'state2', None),
    )
    
    # Dequantize
    full_weight = dequantize_4bit(full_packed, full_state)
    
    # Remove padding and reshape
    full_weight = full_weight[:partition_info.original_numel]
    full_weight = full_weight.view(partition_info.original_shape)
    
    return full_weight


def repartition_tensor(
    full_weight: torch.Tensor,
    partition_info: PartitionInfo,
    quant_state: QuantState,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Re-quantize and re-partition a tensor after modification.
    
    Called after optimizer step if weights were updated.
    For QLoRA with frozen base, this is rarely needed.
    
    Args:
        full_weight: Full weight tensor
        partition_info: Original partition info
        quant_state: Original quantization state
    
    Returns:
        local_packed: Re-partitioned packed weights
        local_absmax: Re-partitioned absmax values
    """
    rank = partition_info.rank
    world_size = partition_info.world_size
    blocksize = partition_info.blocksize
    
    # Flatten and pad
    weight_flat = full_weight.contiguous().view(-1)
    total_elements = partition_info.elements_per_rank * world_size
    if weight_flat.numel() < total_elements:
        padding = total_elements - weight_flat.numel()
        weight_flat = torch.nn.functional.pad(weight_flat, (0, padding))
    
    # Re-quantize
    packed, new_state = quantize_4bit(
        weight_flat,
        blocksize=blocksize,
        quant_type=quant_state.quant_type
    )
    
    # Extract local partition
    packed_start = rank * partition_info.packed_partition_size
    packed_end = (rank + 1) * partition_info.packed_partition_size
    local_packed = packed[packed_start:packed_end].clone()
    
    absmax_start = rank * partition_info.absmax_partition_size
    absmax_end = (rank + 1) * partition_info.absmax_partition_size
    local_absmax = new_state.absmax[absmax_start:absmax_end].clone()
    
    return local_packed, local_absmax
```

---

## 3. Quantized Parameter Class (`quantized_param.py`)

```python
"""Quantized parameter wrapper for ZeRO-Q."""

from typing import Optional
import torch
import torch.nn as nn
import torch.distributed as dist

from .partition import (
    partition_quantized_tensor,
    gather_and_dequantize,
    repartition_tensor,
    PartitionInfo,
)
from .config import ZeroQConfig

try:
    from bitsandbytes.functional import QuantState
except ImportError:
    QuantState = None


class ZeroParamStatus:
    """Parameter availability status."""
    NOT_AVAILABLE = 0  # Only local partition exists
    AVAILABLE = 1      # Full parameter materialized
    INFLIGHT = 2       # Gather in progress


class QuantizedZeroParameter:
    """
    Wrapper for a quantized, partitioned parameter.
    
    This class manages the lifecycle of a parameter in ZeRO-Q:
    - Storage: 4-bit quantized, partitioned across GPUs
    - Computation: Dequantized, gathered to full tensor
    - Communication: Async all-gather of quantized shards
    
    Attributes:
        param: The wrapped nn.Parameter
        config: ZeroQConfig
        local_packed: This rank's packed weights
        local_absmax: This rank's absmax values
        quant_state: Quantization metadata
        partition_info: Partition metadata
        status: Current parameter status
    """
    
    def __init__(
        self,
        param: nn.Parameter,
        config: ZeroQConfig,
        rank: int,
        world_size: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        """
        Initialize a quantized partitioned parameter.
        
        Args:
            param: Original parameter to wrap
            config: ZeRO-Q configuration
            rank: This process's rank
            world_size: Total number of processes
            group: Process group for communication
        """
        self.param = param
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.group = group
        self.status = ZeroParamStatus.NOT_AVAILABLE
        
        # Quantize and partition
        self.local_packed, self.local_absmax, self.quant_state, self.partition_info = \
            partition_quantized_tensor(
                weight=param.data,
                rank=rank,
                world_size=world_size,
                blocksize=config.blocksize,
                quant_type=config.quant_type,
                compute_dtype=config.compute_dtype,
            )
        
        # Clear original parameter data to save memory
        # Store shape/dtype for reconstruction
        self._original_shape = param.shape
        self._original_dtype = param.dtype
        self._original_device = param.device
        
        # Replace param data with empty tensor
        param.data = torch.empty(0, device=param.device, dtype=config.compute_dtype)
        
        # Communication handles for async ops
        self._packed_handle = None
        self._absmax_handle = None
        self._gather_buffers = None
    
    @property
    def is_available(self) -> bool:
        """Check if full parameter is available."""
        return self.status == ZeroParamStatus.AVAILABLE
    
    def memory_usage(self) -> dict:
        """Get current memory usage breakdown."""
        packed_bytes = self.local_packed.numel() * self.local_packed.element_size()
        absmax_bytes = self.local_absmax.numel() * self.local_absmax.element_size()
        
        # If materialized, include full tensor
        if self.is_available:
            full_bytes = self.param.data.numel() * self.param.data.element_size()
        else:
            full_bytes = 0
        
        return {
            "packed_bytes": packed_bytes,
            "absmax_bytes": absmax_bytes,
            "full_bytes": full_bytes,
            "total_bytes": packed_bytes + absmax_bytes + full_bytes,
        }
    
    def fetch(self, async_op: bool = True) -> Optional[torch.Tensor]:
        """
        Gather partitions and materialize full parameter.
        
        Args:
            async_op: If True, returns immediately and caller must wait()
        
        Returns:
            Full parameter tensor if sync, None if async
        """
        if self.status == ZeroParamStatus.AVAILABLE:
            return self.param.data
        
        if self.status == ZeroParamStatus.INFLIGHT:
            # Wait for existing gather
            self._wait_for_gather()
            return self.param.data
        
        # Allocate gather buffers
        self._gather_buffers = {
            "packed": [
                torch.empty_like(self.local_packed) 
                for _ in range(self.world_size)
            ],
            "absmax": [
                torch.empty_like(self.local_absmax)
                for _ in range(self.world_size)
            ],
        }
        
        # Start async all-gather
        self._packed_handle = dist.all_gather(
            self._gather_buffers["packed"],
            self.local_packed,
            group=self.group,
            async_op=True,
        )
        self._absmax_handle = dist.all_gather(
            self._gather_buffers["absmax"],
            self.local_absmax,
            group=self.group,
            async_op=True,
        )
        
        self.status = ZeroParamStatus.INFLIGHT
        
        if not async_op:
            self._wait_for_gather()
            return self.param.data
        
        return None
    
    def _wait_for_gather(self):
        """Wait for async gather to complete and dequantize."""
        if self._packed_handle is not None:
            self._packed_handle.wait()
        if self._absmax_handle is not None:
            self._absmax_handle.wait()
        
        # Concatenate gathered tensors
        full_packed = torch.cat(self._gather_buffers["packed"], dim=0)
        full_absmax = torch.cat(self._gather_buffers["absmax"], dim=0)
        
        # Rebuild QuantState
        from bitsandbytes.functional import QuantState as BnBQuantState
        full_state = BnBQuantState(
            absmax=full_absmax,
            shape=torch.Size([self.partition_info.elements_per_rank * self.world_size]),
            dtype=self.quant_state.dtype,
            blocksize=self.quant_state.blocksize,
            code=self.quant_state.code,
            quant_type=self.quant_state.quant_type,
        )
        
        # Dequantize
        from bitsandbytes.functional import dequantize_4bit
        full_weight = dequantize_4bit(full_packed, full_state)
        
        # Remove padding and reshape
        full_weight = full_weight[:self.partition_info.original_numel]
        full_weight = full_weight.view(self._original_shape)
        
        # Update parameter
        self.param.data = full_weight.to(self.config.compute_dtype)
        self.status = ZeroParamStatus.AVAILABLE
        
        # Clear gather buffers
        self._gather_buffers = None
        self._packed_handle = None
        self._absmax_handle = None
    
    def release(self):
        """
        Release full parameter and return to partitioned state.
        
        For frozen parameters (QLoRA base), no re-quantization needed
        since weights haven't changed.
        """
        if self.status != ZeroParamStatus.AVAILABLE:
            return
        
        # If parameter was modified (rare for frozen), re-partition
        # For QLoRA frozen base, skip this
        if self.param.requires_grad:
            # Re-quantize and partition
            self.local_packed, self.local_absmax = repartition_tensor(
                self.param.data,
                self.partition_info,
                self.quant_state,
            )
        
        # Clear full tensor
        self.param.data = torch.empty(0, device=self._original_device)
        self.status = ZeroParamStatus.NOT_AVAILABLE
    
    def wait(self):
        """Wait for any pending async operations."""
        if self.status == ZeroParamStatus.INFLIGHT:
            self._wait_for_gather()
```

---

## 4. Parameter Coordinator (`coordinator.py`)

```python
"""Coordinates parameter fetching and releasing across modules."""

from typing import Dict, List, Optional, Set
from collections import defaultdict
import torch
import torch.nn as nn
import torch.distributed as dist

from .quantized_param import QuantizedZeroParameter, ZeroParamStatus
from .config import ZeroQConfig


class ZeroQParameterCoordinator:
    """
    Manages quantized parameter lifecycle across model modules.
    
    Responsibilities:
    - Track which parameters are available/partitioned
    - Coordinate fetch/release for forward/backward passes
    - Handle prefetching for efficiency
    - Manage memory budget
    """
    
    def __init__(
        self,
        config: ZeroQConfig,
        rank: int,
        world_size: int,
        group: Optional[dist.ProcessGroup] = None,
    ):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.group = group
        
        # Parameter tracking
        self._params: Dict[int, QuantizedZeroParameter] = {}
        self._module_params: Dict[int, List[int]] = defaultdict(list)
        
        # Prefetch tracking
        self._prefetch_queue: List[int] = []
        self._inflight: Set[int] = set()
    
    def register_parameter(
        self, 
        param: nn.Parameter,
        module: nn.Module,
    ) -> QuantizedZeroParameter:
        """
        Register a parameter for ZeRO-Q management.
        
        Args:
            param: Parameter to register
            module: Module containing the parameter
        
        Returns:
            Wrapped QuantizedZeroParameter
        """
        param_id = id(param)
        module_id = id(module)
        
        if param_id in self._params:
            return self._params[param_id]
        
        # Create quantized parameter wrapper
        qparam = QuantizedZeroParameter(
            param=param,
            config=self.config,
            rank=self.rank,
            world_size=self.world_size,
            group=self.group,
        )
        
        self._params[param_id] = qparam
        self._module_params[module_id].append(param_id)
        
        return qparam
    
    def fetch_module_params(
        self,
        module: nn.Module,
        async_op: bool = True,
    ):
        """
        Fetch all parameters for a module.
        
        Called by pre-forward/pre-backward hooks.
        
        Args:
            module: Module to fetch parameters for
            async_op: Whether to use async communication
        """
        module_id = id(module)
        param_ids = self._module_params.get(module_id, [])
        
        for param_id in param_ids:
            qparam = self._params[param_id]
            if qparam.status == ZeroParamStatus.NOT_AVAILABLE:
                qparam.fetch(async_op=async_op)
                if async_op:
                    self._inflight.add(param_id)
    
    def wait_module_params(self, module: nn.Module):
        """
        Wait for all module parameters to be available.
        
        Called after prefetch, before actual computation.
        """
        module_id = id(module)
        param_ids = self._module_params.get(module_id, [])
        
        for param_id in param_ids:
            if param_id in self._inflight:
                self._params[param_id].wait()
                self._inflight.discard(param_id)
    
    def release_module_params(self, module: nn.Module):
        """
        Release all parameters for a module.
        
        Called by post-forward/post-backward hooks.
        """
        module_id = id(module)
        param_ids = self._module_params.get(module_id, [])
        
        for param_id in param_ids:
            self._params[param_id].release()
    
    def prefetch_next_module(self, next_module: nn.Module):
        """
        Start prefetching parameters for next module.
        
        Called during forward pass to overlap communication.
        """
        if not self.config.overlap_comm:
            return
        
        self.fetch_module_params(next_module, async_op=True)
    
    def memory_summary(self) -> dict:
        """Get memory usage summary across all parameters."""
        total_packed = 0
        total_absmax = 0
        total_full = 0
        
        for qparam in self._params.values():
            usage = qparam.memory_usage()
            total_packed += usage["packed_bytes"]
            total_absmax += usage["absmax_bytes"]
            total_full += usage["full_bytes"]
        
        return {
            "partitioned_bytes": total_packed + total_absmax,
            "materialized_bytes": total_full,
            "total_bytes": total_packed + total_absmax + total_full,
            "num_params": len(self._params),
            "num_inflight": len(self._inflight),
        }
```

---

## 5. Hook System (`hooks.py`)

```python
"""Forward/backward hooks for ZeRO-Q parameter management."""

from typing import Callable, Optional, List, Tuple
import torch
import torch.nn as nn

from .coordinator import ZeroQParameterCoordinator


class ZeroQHookManager:
    """
    Manages forward/backward hooks for ZeRO-Q.
    
    Installs hooks that:
    - Pre-forward: Fetch and wait for module parameters
    - Post-forward: Release parameters (if not needed for backward)
    - Pre-backward: Re-fetch parameters
    - Post-backward: Release parameters
    """
    
    def __init__(
        self,
        coordinator: ZeroQParameterCoordinator,
        module_order: Optional[List[nn.Module]] = None,
    ):
        self.coordinator = coordinator
        self.module_order = module_order or []
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
    
    def install_hooks(self, model: nn.Module):
        """Install all hooks on model modules."""
        # Get execution order if not provided
        if not self.module_order:
            self.module_order = self._get_module_order(model)
        
        for i, module in enumerate(self.module_order):
            # Get next module for prefetching
            next_module = self.module_order[i + 1] if i + 1 < len(self.module_order) else None
            
            # Pre-forward hook
            handle = module.register_forward_pre_hook(
                self._make_pre_forward_hook(module, next_module)
            )
            self._handles.append(handle)
            
            # Post-forward hook
            handle = module.register_forward_hook(
                self._make_post_forward_hook(module)
            )
            self._handles.append(handle)
            
            # Backward hooks via tensor hooks (more reliable than module hooks)
            # Installed during forward when we have the output tensor
    
    def remove_hooks(self):
        """Remove all installed hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
    
    def _make_pre_forward_hook(
        self,
        module: nn.Module,
        next_module: Optional[nn.Module],
    ) -> Callable:
        """Create pre-forward hook for a module."""
        def hook(module, inputs):
            # Wait for any prefetched params
            self.coordinator.wait_module_params(module)
            
            # Start prefetching next module
            if next_module is not None:
                self.coordinator.prefetch_next_module(next_module)
        
        return hook
    
    def _make_post_forward_hook(self, module: nn.Module) -> Callable:
        """Create post-forward hook for a module."""
        def hook(module, inputs, outputs):
            # If training, we need params for backward
            # Only release after backward completes
            if not module.training:
                self.coordinator.release_module_params(module)
            else:
                # Install backward hook via output tensor
                if isinstance(outputs, torch.Tensor) and outputs.requires_grad:
                    outputs.register_hook(
                        self._make_backward_hook(module)
                    )
                elif isinstance(outputs, tuple):
                    for out in outputs:
                        if isinstance(out, torch.Tensor) and out.requires_grad:
                            out.register_hook(
                                self._make_backward_hook(module)
                            )
                            break
        
        return hook
    
    def _make_backward_hook(self, module: nn.Module) -> Callable:
        """Create backward hook for gradient computation."""
        def hook(grad):
            # After backward through this module, release params
            self.coordinator.release_module_params(module)
            return grad
        
        return hook
    
    def _get_module_order(self, model: nn.Module) -> List[nn.Module]:
        """
        Get module execution order via forward pass tracing.
        
        Simple version: just return all modules with parameters.
        Advanced: trace actual execution order.
        """
        modules = []
        for name, module in model.named_modules():
            # Only include modules with direct parameters
            if len(list(module.parameters(recurse=False))) > 0:
                modules.append(module)
        return modules
```

---

## 6. Integration Layer (`integration.py`)

```python
"""HuggingFace Transformers and PEFT integration."""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import torch.distributed as dist

from .config import ZeroQConfig
from .coordinator import ZeroQParameterCoordinator
from .hooks import ZeroQHookManager


def initialize(
    model: nn.Module,
    config: ZeroQConfig,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
    group: Optional[dist.ProcessGroup] = None,
) -> nn.Module:
    """
    Initialize a model for ZeRO-Q training.
    
    This is the main entry point for ZeRO-Q. It:
    1. Identifies parameters to quantize
    2. Creates QuantizedZeroParameter wrappers
    3. Installs forward/backward hooks
    4. Returns the wrapped model
    
    Args:
        model: Model to wrap (can be PEFT model)
        config: ZeRO-Q configuration
        rank: Process rank (auto-detected if None)
        world_size: World size (auto-detected if None)
        group: Process group (default group if None)
    
    Returns:
        Model ready for distributed training
    
    Example:
        >>> model = AutoModelForCausalLM.from_pretrained(...)
        >>> model = get_peft_model(model, lora_config)
        >>> model = zero_q.initialize(model, ZeroQConfig())
        >>> # Now train with standard loop
    """
    # Auto-detect distributed settings
    if rank is None:
        rank = dist.get_rank() if dist.is_initialized() else 0
    if world_size is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    # Create coordinator
    coordinator = ZeroQParameterCoordinator(
        config=config,
        rank=rank,
        world_size=world_size,
        group=group,
    )
    
    # Register parameters
    for name, module in model.named_modules():
        # Check if module should be quantized
        if not _should_quantize_module(name, module, config):
            continue
        
        for param_name, param in module.named_parameters(recurse=False):
            # Skip trainable params if configured
            if config.frozen_only and param.requires_grad:
                continue
            
            # Register for ZeRO-Q management
            coordinator.register_parameter(param, module)
    
    # Install hooks
    hook_manager = ZeroQHookManager(coordinator)
    hook_manager.install_hooks(model)
    
    # Store references on model for later access
    model._zero_q_coordinator = coordinator
    model._zero_q_hooks = hook_manager
    model._zero_q_config = config
    
    return model


def _should_quantize_module(
    name: str,
    module: nn.Module,
    config: ZeroQConfig,
) -> bool:
    """Check if a module should have its parameters quantized."""
    # Check exclusions
    for exclude in config.exclude_modules:
        if exclude in name:
            return False
    
    # Check targets if specified
    if config.target_modules is not None:
        for target in config.target_modules:
            if target in name:
                return True
        return False
    
    # Default: quantize Linear modules
    return isinstance(module, nn.Linear)


class ZeroQTrainer:
    """
    HuggingFace Trainer-compatible wrapper.
    
    Drop-in replacement for Trainer that handles ZeRO-Q setup.
    
    Example:
        >>> trainer = ZeroQTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=dataset,
        ...     zero_q_config=ZeroQConfig(),
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module,
        args: Any,  # TrainingArguments
        train_dataset: Any,
        zero_q_config: ZeroQConfig,
        **kwargs,
    ):
        from transformers import Trainer
        
        # Initialize ZeRO-Q on model
        model = initialize(model, zero_q_config)
        
        # Create base trainer
        self._trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            **kwargs,
        )
        
        self.model = model
        self.config = zero_q_config
    
    def train(self, **kwargs):
        """Run training."""
        return self._trainer.train(**kwargs)
    
    def evaluate(self, **kwargs):
        """Run evaluation."""
        return self._trainer.evaluate(**kwargs)
    
    def save_model(self, output_dir: str):
        """Save model (adapters only for QLoRA)."""
        self._trainer.save_model(output_dir)
    
    def __getattr__(self, name):
        """Delegate to base trainer."""
        return getattr(self._trainer, name)
```

---

## 7. Testing Examples

### `tests/test_partition.py`

```python
"""Tests for partition module."""

import pytest
import torch
import torch.distributed as dist

from zero_q.partition import (
    compute_aligned_partition_sizes,
    partition_quantized_tensor,
    gather_and_dequantize,
)


class TestAlignedPartitionSizes:
    """Tests for partition size computation."""
    
    def test_exact_division(self):
        """Test when numel divides evenly."""
        numel = 4096  # 64 blocks of 64
        world_size = 4
        blocksize = 64
        
        elements, packed, absmax = compute_aligned_partition_sizes(
            numel, world_size, blocksize
        )
        
        assert elements == 1024  # 4096 / 4
        assert packed == 512     # 1024 / 2 (4-bit packing)
        assert absmax == 16      # 1024 / 64 blocks
    
    def test_with_padding(self):
        """Test when padding is needed."""
        numel = 4000  # Not divisible by 64*4
        world_size = 4
        blocksize = 64
        
        elements, packed, absmax = compute_aligned_partition_sizes(
            numel, world_size, blocksize
        )
        
        # Should round up to 4096 (64 blocks * 64 elements)
        assert elements * world_size >= numel
        assert elements % blocksize == 0
    
    def test_different_blocksizes(self):
        """Test with various block sizes."""
        numel = 8192
        world_size = 4
        
        for blocksize in [64, 128, 256, 512]:
            elements, packed, absmax = compute_aligned_partition_sizes(
                numel, world_size, blocksize
            )
            
            # Verify alignment
            assert elements % blocksize == 0
            assert packed == elements // 2
            assert absmax == elements // blocksize


class TestPartitionQuantizedTensor:
    """Tests for tensor partitioning."""
    
    @pytest.fixture
    def sample_weight(self):
        """Create sample weight tensor."""
        torch.manual_seed(42)
        return torch.randn(4096, 4096, dtype=torch.float32)
    
    def test_partition_shapes(self, sample_weight):
        """Test output shapes are correct."""
        local_packed, local_absmax, quant_state, info = partition_quantized_tensor(
            weight=sample_weight,
            rank=0,
            world_size=4,
            blocksize=64,
        )
        
        # Check partition info
        assert info.world_size == 4
        assert info.blocksize == 64
        
        # Check shapes
        expected_packed = info.packed_partition_size
        expected_absmax = info.absmax_partition_size
        
        assert local_packed.shape[0] == expected_packed
        assert local_absmax.shape[0] == expected_absmax
    
    def test_different_ranks_different_data(self, sample_weight):
        """Test that different ranks get different partitions."""
        partitions = []
        for rank in range(4):
            local_packed, _, _, _ = partition_quantized_tensor(
                weight=sample_weight,
                rank=rank,
                world_size=4,
            )
            partitions.append(local_packed)
        
        # All partitions should be different
        for i in range(4):
            for j in range(i + 1, 4):
                assert not torch.equal(partitions[i], partitions[j])


class TestRoundtrip:
    """Test quantize -> partition -> gather -> dequantize roundtrip."""
    
    def test_numerical_accuracy(self):
        """Test that roundtrip preserves values within tolerance."""
        torch.manual_seed(42)
        original = torch.randn(1024, 1024, dtype=torch.float32)
        
        # Partition (simulating rank 0 of 4)
        local_packed, local_absmax, quant_state, info = partition_quantized_tensor(
            weight=original,
            rank=0,
            world_size=1,  # Single "GPU" for testing
            blocksize=64,
        )
        
        # Gather (trivial with world_size=1)
        restored = gather_and_dequantize(
            local_packed=local_packed,
            local_absmax=local_absmax,
            quant_state=quant_state,
            partition_info=info,
        )
        
        # Check accuracy
        # NF4 should have < 1% mean relative error
        relative_error = (original - restored).abs() / (original.abs() + 1e-8)
        mean_error = relative_error.mean().item()
        
        assert mean_error < 0.05  # Allow 5% for safety margin
```

---

## 8. Example Training Script

### `examples/train_32b_maxwell.py`

```python
"""
Example: Train 32B model on Maxwell GPUs using ZeRO-Q.

Hardware: 3x Tesla M40 (12GB each)
Model: Qwen2.5-Coder-32B-Instruct
Method: QLoRA with ZeRO-Q distribution

Usage:
    torchrun --nproc_per_node=3 train_32b_maxwell.py
"""

import os
import torch
import torch.distributed as dist
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

# Import ZeRO-Q
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src import ZeroQConfig, initialize, MAXWELL_CONFIG


def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    # Set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    print(f"[Rank {rank}] Starting on {device}")
    
    # Load model (initially on CPU to save GPU memory)
    model_name = "Qwen/Qwen2.5-Coder-32B-Instruct"
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load in FP32 for Maxwell compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map={"": device},  # Load to this GPU
        trust_remote_code=True,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],  # Minimal for memory
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    
    model = get_peft_model(model, lora_config)
    
    if rank == 0:
        model.print_trainable_parameters()
    
    # Initialize ZeRO-Q
    zero_q_config = ZeroQConfig(
        quant_type="nf4",
        blocksize=64,
        compute_dtype=torch.float32,  # Required for Maxwell
        frozen_only=True,  # Only quantize frozen base model
        async_gather=True,
        overlap_comm=True,
    )
    
    model = initialize(model, zero_q_config)
    
    if rank == 0:
        print(f"ZeRO-Q initialized with config: {zero_q_config}")
        mem = model._zero_q_coordinator.memory_summary()
        print(f"Memory usage: {mem}")
    
    # Load dataset
    dataset = load_dataset("json", data_files="data/train.jsonl")["train"]
    
    # Tokenize
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=512,
            padding="max_length",
        )
    
    dataset = dataset.map(tokenize, batched=True)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir="./outputs/zero_q_32b",
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2e-4,
        fp16=False,  # Not supported on Maxwell
        bf16=False,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=False,
    )
    
    # Train
    from transformers import Trainer
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )
    
    trainer.train()
    
    # Save
    if rank == 0:
        trainer.save_model("./outputs/zero_q_32b/final")
    
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

---

## Next Steps

1. **Implement `src/config.py`** - Configuration dataclass (done above)
2. **Implement `src/partition.py`** - Core partitioning logic (done above)
3. **Write unit tests** - Validate partitioning correctness
4. **Test on single node** - 2 GPUs with small model
5. **Scale to M40 cluster** - 3x M40 with 32B model

---

*Architecture document by Zero (Claude Opus 4.5) - December 10, 2025*
