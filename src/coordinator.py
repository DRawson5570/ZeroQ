"""
ZeroQ Parameter Coordinator

Manages quantized parameter gathering and release for distributed training.
This is the core component that enables ZeRO-style parameter partitioning
with 4-bit quantized weights.
"""

from enum import Enum
from typing import Optional, Dict, List, Tuple, Any
import torch
import torch.nn.functional as F
import torch.distributed as dist
from bitsandbytes.functional import quantize_4bit, dequantize_4bit, QuantState

try:
    from .config import ZeroQConfig
except ImportError:
    from config import ZeroQConfig

try:
    from .gradient_sync import reduce_scatter_grads
except ImportError:
    try:
        from gradient_sync import reduce_scatter_grads
    except ImportError:
        reduce_scatter_grads = None


class ZeroQParamStatus(Enum):
    """Status of a ZeroQ parameter."""
    NOT_AVAILABLE = 0  # Only local partition stored
    AVAILABLE = 1      # Full parameter available for compute
    INFLIGHT = 2       # Currently being gathered


class ZeroQParameter:
    """
    Wrapper for a quantized, partitioned parameter.
    
    Stores the local partition of packed weights and absmax values,
    and manages gathering/releasing the full parameter for compute.
    """
    
    def __init__(
        self,
        param: torch.nn.Parameter,
        rank: int,
        world_size: int,
        config: ZeroQConfig,
        param_id: int,
        module: Optional[torch.nn.Module] = None,
        param_name: Optional[str] = None,
    ):
        self.param = param
        self.rank = rank
        self.world_size = world_size
        self.config = config
        self.param_id = param_id

        # Where this Parameter lives (needed to safely replace meta params).
        self.module = module
        self.param_name = param_name
        
        # Original shape and numel
        self.original_shape = param.data.shape
        self.original_numel = param.data.numel()
        self.original_dtype = param.data.dtype
        
        # Status tracking
        self.status = ZeroQParamStatus.NOT_AVAILABLE
        
        # Local partitions (initialized in partition())
        self.local_packed: Optional[torch.Tensor] = None
        self.local_absmax: Optional[torch.Tensor] = None
        # Store only lightweight quant metadata; absmax is partitioned per-rank.
        self._quant_meta: Optional[Dict[str, Any]] = None
        
        # Communication handles
        self._gather_handles: Optional[Tuple[Any, Any]] = None
        
        # Pre-allocated buffers for gathering
        self._packed_buffers: Optional[List[torch.Tensor]] = None
        self._absmax_buffers: Optional[List[torch.Tensor]] = None

        # Partition sizing (set during partition)
        self.packed_per_rank: int = 0
        self.absmax_per_rank: int = 0
        self._packed_remainder: int = 0
        self._absmax_remainder: int = 0
        self._packed_total: int = 0
        self._absmax_total: int = 0

        # Optional contiguous gather buffers (all_gather_into_tensor)
        self._gathered_packed: Optional[torch.Tensor] = None
        self._gathered_absmax: Optional[torch.Tensor] = None
        self._send_packed: Optional[torch.Tensor] = None
        self._send_absmax: Optional[torch.Tensor] = None
        self._assembled_packed: Optional[torch.Tensor] = None
        self._assembled_absmax: Optional[torch.Tensor] = None

        # Training-from-scratch mode: fp32 master shard (no quantization)
        self.master_shard: Optional[torch.nn.Parameter] = None
        self._fp32_chunk_size: int = 0
        self._fp32_gather_buffers: Optional[List[torch.Tensor]] = None
        self._fp32_gather_handle: Optional[Any] = None
    
    def partition(self):
        """Quantize and partition the parameter."""
        if self.status != ZeroQParamStatus.NOT_AVAILABLE:
            return
        if self.local_packed is not None or self.master_shard is not None:
            return  # Already partitioned

        self.partition_from_full_precision(self.param.data)


    def partition_from_full_precision(self, weight: torch.Tensor):
        """Quantize and partition from a provided full-precision tensor.

        This enables streamed checkpoint loading: a loader can materialize one
        tensor at a time (per-layer), call this method, and never keep the full
        model resident on a single GPU.

        Args:
            weight: Full-precision parameter tensor (any device). Will be moved
                to the parameter's device (CUDA recommended).
        """

        if self.status != ZeroQParamStatus.NOT_AVAILABLE:
            return
        if self.local_packed is not None or self.master_shard is not None:
            return

        if getattr(self.config, "training_mode", False):
            self._partition_fp32_training(weight)
            return

        if weight.numel() != self.original_numel:
            raise ValueError(
                f"Weight numel mismatch for param_id={self.param_id}: "
                f"expected {self.original_numel}, got {weight.numel()}"
            )

        # Choose device for quant/partitions: prefer the parameter's device if it's CUDA,
        # otherwise fall back to the current CUDA device.
        target_device = self.param.device
        if target_device.type != "cuda" and torch.cuda.is_available():
            target_device = torch.device("cuda")

        # Meta params (from init_empty_weights) cannot have .data assigned to a real tensor.
        # Ensure the live model Parameter is a concrete placeholder on the target device.
        if self.param.device.type == "meta" or self.param.device != target_device:
            if self.module is None or self.param_name is None:
                raise RuntimeError(
                    "ZeroQParameter needs module+param_name to replace meta/cpu parameters safely. "
                    "Register parameters via ZeroQModuleWrapper so this info is available."
                )
            new_param = torch.nn.Parameter(
                torch.empty(0, device=target_device, dtype=self.original_dtype),
                requires_grad=self.param.requires_grad,
            )
            setattr(self.module, self.param_name, new_param)
            self.param = new_param

        weight = weight.contiguous().to(device=target_device)
        if weight.dtype != torch.float16:
            weight = weight.to(torch.float16)

        # Sync CUDA before quantization (avoid concurrent kernel issues)
        if target_device.type == "cuda":
            torch.cuda.synchronize()

        packed, quant_state = quantize_4bit(
            weight,
            blocksize=self.config.blocksize,
            quant_type=self.config.quant_type,
        )

        # Normalize to 1D for partitioning/gather. quant_state.shape preserves the
        # original logical shape for dequantization.
        packed = packed.contiguous().view(-1)
        absmax = quant_state.absmax.contiguous().view(-1)
        
        # Calculate partition sizes
        packed_size = packed.numel()
        absmax_size = absmax.numel()
        self._packed_total = packed_size
        self._absmax_total = absmax_size
        
        self.packed_per_rank = packed_size // self.world_size
        self.absmax_per_rank = absmax_size // self.world_size
        
        # Handle remainder (last rank gets extra)
        packed_remainder = packed_size % self.world_size
        absmax_remainder = absmax_size % self.world_size
        self._packed_remainder = packed_remainder
        self._absmax_remainder = absmax_remainder
        
        # Extract local partition
        packed_start = self.rank * self.packed_per_rank
        packed_end = packed_start + self.packed_per_rank
        if self.rank == self.world_size - 1:
            packed_end += packed_remainder
        
        absmax_start = self.rank * self.absmax_per_rank
        absmax_end = absmax_start + self.absmax_per_rank
        if self.rank == self.world_size - 1:
            absmax_end += absmax_remainder
        
        self.local_packed = packed[packed_start:packed_end].clone()
        self.local_absmax = absmax[absmax_start:absmax_end].clone()

        # Store only lightweight quant metadata; do NOT keep full absmax.
        self._quant_meta = {
            "shape": quant_state.shape,
            "dtype": quant_state.dtype,
            "blocksize": quant_state.blocksize,
            "code": quant_state.code,
            "quant_type": quant_state.quant_type,
        }
        
        # Clear the original parameter data (placeholder stays on target_device)
        self.param.data = torch.empty(0, device=target_device, dtype=self.original_dtype)
        
        self.status = ZeroQParamStatus.NOT_AVAILABLE

    def _partition_fp32_training(self, weight: torch.Tensor):
        """Shard weight into fp32 master shards (no quantization)."""
        target_device = self.param.device
        if target_device.type != "cuda" and torch.cuda.is_available():
            target_device = torch.device("cuda")

        if self.param.device.type == "meta" or self.param.device != target_device:
            if self.module is None or self.param_name is None:
                raise RuntimeError(
                    "ZeroQParameter needs module+param_name to replace meta/cpu parameters."
                )
            new_param = torch.nn.Parameter(
                torch.empty(0, device=target_device, dtype=torch.float32),
                requires_grad=True,
            )
            setattr(self.module, self.param_name, new_param)
            self.param = new_param

        flat = weight.detach().contiguous().view(-1).to(dtype=torch.float32, device=target_device)
        chunk_size = (flat.numel() + self.world_size - 1) // self.world_size
        start = self.rank * chunk_size
        end = min(start + chunk_size, flat.numel())
        shard = flat[start:end].clone()
        if shard.numel() < chunk_size:
            shard = F.pad(shard, (0, chunk_size - shard.numel()))

        self.master_shard = torch.nn.Parameter(shard, requires_grad=True)
        self._fp32_chunk_size = chunk_size

        self.param.data = torch.empty(0, device=target_device, dtype=torch.float32)
        self.status = ZeroQParamStatus.NOT_AVAILABLE

    def start_gather(self, group: Optional[dist.ProcessGroup] = None, async_op: bool = True):
        """
        Start async all-gather of quantized partitions.
        
        Args:
            group: Process group for communication
            async_op: Whether to use async operation
        
        Returns:
            Tuple of gather handles if async_op=True, else None
        """
        if self.status == ZeroQParamStatus.AVAILABLE:
            return None
        
        if self.status == ZeroQParamStatus.INFLIGHT:
            return self._gather_handles

        if getattr(self.config, "training_mode", False):
            return self._start_gather_fp32(group, async_op)

        # Prefer contiguous gather to reduce allocator fragmentation and peak memory.
        use_into_tensor = hasattr(dist, "all_gather_into_tensor") and self.world_size > 1
        if use_into_tensor:
            packed_stride = self.packed_per_rank + self._packed_remainder
            absmax_stride = self.absmax_per_rank + self._absmax_remainder

            if self._send_packed is None or self._send_packed.numel() != packed_stride:
                self._send_packed = torch.empty(
                    packed_stride,
                    dtype=self.local_packed.dtype,
                    device=self.local_packed.device,
                )
            if self._send_absmax is None or self._send_absmax.numel() != absmax_stride:
                self._send_absmax = torch.empty(
                    absmax_stride,
                    dtype=self.local_absmax.dtype,
                    device=self.local_absmax.device,
                )

            # Pad sends to the per-rank stride.
            self._send_packed.zero_()
            self._send_packed[: self.local_packed.numel()].copy_(self.local_packed)
            self._send_absmax.zero_()
            self._send_absmax[: self.local_absmax.numel()].copy_(self.local_absmax)

            if self._gathered_packed is None or self._gathered_packed.numel() != packed_stride * self.world_size:
                self._gathered_packed = torch.empty(
                    packed_stride * self.world_size,
                    dtype=self.local_packed.dtype,
                    device=self.local_packed.device,
                )
            if self._gathered_absmax is None or self._gathered_absmax.numel() != absmax_stride * self.world_size:
                self._gathered_absmax = torch.empty(
                    absmax_stride * self.world_size,
                    dtype=self.local_absmax.dtype,
                    device=self.local_absmax.device,
                )

            packed_handle = dist.all_gather_into_tensor(
                self._gathered_packed,
                self._send_packed,
                group=group,
                async_op=async_op,
            )
            absmax_handle = dist.all_gather_into_tensor(
                self._gathered_absmax,
                self._send_absmax,
                group=group,
                async_op=async_op,
            )
        else:
            # Fallback: list-based gather.
            if self._packed_buffers is None:
                self._packed_buffers = [
                    torch.empty(
                        self.packed_per_rank + (self._packed_remainder if i == self.world_size - 1 else 0),
                        dtype=self.local_packed.dtype,
                        device=self.local_packed.device,
                    )
                    for i in range(self.world_size)
                ]
            if self._absmax_buffers is None:
                self._absmax_buffers = [
                    torch.empty(
                        self.absmax_per_rank + (self._absmax_remainder if i == self.world_size - 1 else 0),
                        dtype=self.local_absmax.dtype,
                        device=self.local_absmax.device,
                    )
                    for i in range(self.world_size)
                ]

            packed_handle = dist.all_gather(
                self._packed_buffers,
                self.local_packed,
                group=group,
                async_op=async_op,
            )
            absmax_handle = dist.all_gather(
                self._absmax_buffers,
                self.local_absmax,
                group=group,
                async_op=async_op,
            )
        
        if async_op:
            self._gather_handles = (packed_handle, absmax_handle)
            self.status = ZeroQParamStatus.INFLIGHT
            return self._gather_handles
        else:
            # Synchronous - complete immediately
            self._complete_gather()
            return None
    
    def wait_gather(self):
        """Wait for async gather to complete."""
        if self.status != ZeroQParamStatus.INFLIGHT:
            return

        if getattr(self.config, "training_mode", False):
            if self._fp32_gather_handle is not None:
                self._fp32_gather_handle.wait()
            self._complete_gather_fp32()
            return

        if self._gather_handles is not None:
            self._gather_handles[0].wait()
            self._gather_handles[1].wait()

        self._complete_gather()
    
    def _complete_gather(self):
        """Complete the gather operation and dequantize."""
        # Concatenate gathered data (prefer contiguous gather buffers if available).
        if self._gathered_packed is not None and self._gathered_absmax is not None:
            packed_stride = self.packed_per_rank + self._packed_remainder
            absmax_stride = self.absmax_per_rank + self._absmax_remainder

            if self._assembled_packed is None or self._assembled_packed.numel() != self._packed_total:
                self._assembled_packed = torch.empty(
                    self._packed_total,
                    dtype=self._gathered_packed.dtype,
                    device=self._gathered_packed.device,
                )
            if self._assembled_absmax is None or self._assembled_absmax.numel() != self._absmax_total:
                self._assembled_absmax = torch.empty(
                    self._absmax_total,
                    dtype=self._gathered_absmax.dtype,
                    device=self._gathered_absmax.device,
                )

            # Unpad each rank's segment into a contiguous packed/absmax vector.
            off_p, off_a = 0, 0
            for i in range(self.world_size):
                seg_p = self._gathered_packed[i * packed_stride : (i + 1) * packed_stride]
                seg_a = self._gathered_absmax[i * absmax_stride : (i + 1) * absmax_stride]

                seg_p_len = self.packed_per_rank + (self._packed_remainder if i == self.world_size - 1 else 0)
                seg_a_len = self.absmax_per_rank + (self._absmax_remainder if i == self.world_size - 1 else 0)

                self._assembled_packed[off_p : off_p + seg_p_len].copy_(seg_p[:seg_p_len])
                self._assembled_absmax[off_a : off_a + seg_a_len].copy_(seg_a[:seg_a_len])
                off_p += seg_p_len
                off_a += seg_a_len

            gathered_packed = self._assembled_packed
            gathered_absmax = self._assembled_absmax

            # Free padded comm buffers before the large dequant alloc.
            # This matters most for huge layers (e.g., MLP projections) on 11GB GPUs.
            if self.full_memory_bytes >= 128 * 1024 * 1024:
                self._gathered_packed = None
                self._gathered_absmax = None
                self._send_packed = None
                self._send_absmax = None
        else:
            gathered_packed = torch.cat(self._packed_buffers, dim=0)
            gathered_absmax = torch.cat(self._absmax_buffers, dim=0)
        
        # Rebuild QuantState with gathered absmax
        if self._quant_meta is None:
            raise RuntimeError(f"Missing quant metadata for param_id={self.param_id}")

        gathered_state = QuantState(
            absmax=gathered_absmax,
            shape=self._quant_meta["shape"],
            dtype=self._quant_meta["dtype"],
            blocksize=self._quant_meta["blocksize"],
            code=self._quant_meta["code"],
            quant_type=self._quant_meta["quant_type"],
        )
        
        # Dequantize to compute dtype. Try to reduce allocator fragmentation before big allocs.
        if gathered_packed.is_cuda:
            torch.cuda.empty_cache()
        restored = dequantize_4bit(gathered_packed, gathered_state)
        
        # Convert back to original dtype (model's dtype) for forward pass
        if self.original_dtype != restored.dtype:
            restored = restored.to(self.original_dtype)

        restored = restored.view(self.original_shape)

        # IMPORTANT (checkpoint safety): avoid mutating `Parameter.data` in-place.
        # `torch.utils.checkpoint` records metadata for tensors saved for backward;
        # if we later set `param.data = empty(0)`, it mutates the same Tensor object
        # and checkpointing will fail with metadata mismatches. Replacing the module
        # attribute keeps previously-saved tensor objects intact.
        if self.module is not None and self.param_name is not None:
            new_param = torch.nn.Parameter(restored, requires_grad=self.param.requires_grad)
            setattr(self.module, self.param_name, new_param)
            self.param = new_param
        else:
            self.param.data = restored

        # Drop temporary gather/assembly buffers so forward has more headroom.
        self._packed_buffers = None
        self._absmax_buffers = None
        self._assembled_packed = None
        self._assembled_absmax = None
        self.status = ZeroQParamStatus.AVAILABLE
        self._gather_handles = None

    def _start_gather_fp32(self, group, async_op):
        """All-gather fp32 master shards across ranks."""
        if not dist.is_initialized() or self.world_size == 1:
            self._complete_gather_fp32()
            return None

        if self._fp32_gather_buffers is None:
            self._fp32_gather_buffers = [
                torch.empty_like(self.master_shard.data) for _ in range(self.world_size)
            ]

        handle = dist.all_gather(
            self._fp32_gather_buffers,
            self.master_shard.data,
            group=group,
            async_op=async_op,
        )

        if async_op:
            self._fp32_gather_handle = handle
            self.status = ZeroQParamStatus.INFLIGHT
            return handle
        else:
            self._complete_gather_fp32()
            return None

    def _complete_gather_fp32(self):
        """Concatenate gathered fp32 shards and set on module."""
        if self._fp32_gather_buffers is not None:
            full = torch.cat(self._fp32_gather_buffers)[:self.original_numel]
        else:
            full = self.master_shard.data[:self.original_numel]

        full = full.view(self.original_shape)

        new_param = torch.nn.Parameter(full, requires_grad=True)

        if self.master_shard is not None and reduce_scatter_grads is not None:
            zq_self = self
            def _grad_hook(grad):
                local_grad = reduce_scatter_grads(
                    grad, zq_self.world_size, zq_self.rank, group=None,
                )
                zq_self.master_shard.grad = local_grad
            new_param.register_hook(_grad_hook)

        if self.module is not None and self.param_name is not None:
            setattr(self.module, self.param_name, new_param)
            self.param = new_param
        else:
            self.param = new_param

        self._fp32_gather_buffers = None
        self._fp32_gather_handle = None
        self.status = ZeroQParamStatus.AVAILABLE

    def release(self):
        """Release full parameter, keeping only local partition."""
        if self.status == ZeroQParamStatus.NOT_AVAILABLE:
            return

        if getattr(self.config, "training_mode", False):
            self._release_fp32()
            return

        if self.module is not None and self.param_name is not None:
            placeholder = torch.nn.Parameter(
                torch.empty(0, device=self.param.device, dtype=self.original_dtype),
                requires_grad=self.param.requires_grad,
            )
            setattr(self.module, self.param_name, placeholder)
            self.param = placeholder
        else:
            self.param.data = torch.empty(0, device=self.param.device, dtype=self.original_dtype)
        self._packed_buffers = None
        self._absmax_buffers = None
        self._gathered_packed = None
        self._gathered_absmax = None
        self._send_packed = None
        self._send_absmax = None
        self._assembled_packed = None
        self._assembled_absmax = None
        self.status = ZeroQParamStatus.NOT_AVAILABLE

    def _release_fp32(self):
        """Release gathered param in training mode (master shard stays)."""
        device = self.master_shard.device
        if self.module is not None and self.param_name is not None:
            placeholder = torch.nn.Parameter(
                torch.empty(0, device=device, dtype=torch.float32),
                requires_grad=True,
            )
            setattr(self.module, self.param_name, placeholder)
            self.param = placeholder
        else:
            self.param.data = torch.empty(0, device=device, dtype=torch.float32)
        self._fp32_gather_buffers = None
        self.status = ZeroQParamStatus.NOT_AVAILABLE
    
    @property
    def local_memory_bytes(self) -> int:
        """Memory used by local partition."""
        if self.master_shard is not None:
            return self.master_shard.numel() * self.master_shard.element_size()
        if self.local_packed is None:
            return 0
        return (
            self.local_packed.numel() * self.local_packed.element_size() +
            self.local_absmax.numel() * self.local_absmax.element_size()
        )
    
    @property
    def full_memory_bytes(self) -> int:
        """Memory that would be used by full FP16 parameter."""
        return self.original_numel * 2  # float16 = 2 bytes


class ZeroQCoordinator:
    """
    Coordinates parameter gathering and releasing for ZeroQ training.
    
    Manages a collection of ZeroQParameter objects and provides hooks
    for pre/post forward and backward passes.
    """
    
    def __init__(
        self,
        config: ZeroQConfig,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        self.config = config
        self.process_group = process_group
        
        # Get rank and world size
        if dist.is_initialized():
            self.rank = dist.get_rank(process_group)
            self.world_size = dist.get_world_size(process_group)
        else:
            self.rank = 0
            self.world_size = 1
        
        # Parameter registry
        self._params: Dict[int, ZeroQParameter] = {}
        self._params_by_obj_id: Dict[int, ZeroQParameter] = {}
        self._param_to_module: Dict[int, torch.nn.Module] = {}
        self._next_param_id = 0
    
    def register_parameter(
        self,
        param: Any,
        module: Optional[torch.nn.Module] = None,
        param_name: Optional[str] = None,
    ) -> ZeroQParameter:
        """Register a parameter for ZeroQ management."""
        # Back-compat: some callers use register_parameter(name, param)
        if isinstance(param, str) and isinstance(module, torch.nn.Parameter):
            param_name = param
            param = module
            module = None

        param_id = self._next_param_id
        self._next_param_id += 1
        
        zq_param = ZeroQParameter(
            param=param,
            rank=self.rank,
            world_size=self.world_size,
            config=self.config,
            param_id=param_id,
            module=module,
            param_name=param_name,
        )
        
        self._params[param_id] = zq_param
        self._params_by_obj_id[id(param)] = zq_param
        if module is not None:
            self._param_to_module[param_id] = module
        
        return zq_param

    def get_param_for_tensor(self, param: torch.nn.Parameter) -> Optional[ZeroQParameter]:
        """Lookup the ZeRO-Q wrapper for a given Parameter object."""
        return self._params_by_obj_id.get(id(param))
    
    def partition_all(self):
        """Partition all registered parameters."""
        for zq_param in self._params.values():
            zq_param.partition()
    
    def fetch_params(self, param_ids: List[int], async_op: bool = True):
        """
        Fetch (gather) specified parameters.
        
        Args:
            param_ids: IDs of parameters to fetch
            async_op: Whether to use async operations
        """
        for param_id in param_ids:
            if param_id in self._params:
                self._params[param_id].start_gather(
                    group=self.process_group,
                    async_op=async_op,
                )
    
    def wait_params(self, param_ids: List[int]):
        """Wait for specified parameters to be available."""
        for param_id in param_ids:
            if param_id in self._params:
                self._params[param_id].wait_gather()
    
    def release_params(self, param_ids: List[int]):
        """Release specified parameters."""
        for param_id in param_ids:
            if param_id in self._params:
                self._params[param_id].release()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        local_bytes = sum(p.local_memory_bytes for p in self._params.values())
        full_bytes = sum(p.full_memory_bytes for p in self._params.values())
        
        return {
            "local_memory_mb": local_bytes / 1024**2,
            "full_fp16_memory_mb": full_bytes / 1024**2,
            "compression_ratio": full_bytes / max(local_bytes, 1),
            "num_params": len(self._params),
        }
    
    def trainable_master_params(self) -> List[torch.nn.Parameter]:
        """Return master-shard parameters for optimizer construction (training mode)."""
        return [p.master_shard for p in self._params.values() if p.master_shard is not None]

    def gather_full_state_dict(self) -> Dict[str, torch.Tensor]:
        """Gather all shards and return a full-model state dict.

        Only meaningful on rank 0 — other ranks return an empty dict.
        Uses the same all-gather path as the forward hook.
        """
        state = {}
        for zq_param in self._params.values():
            if zq_param.master_shard is None:
                continue
            zq_param.start_gather(group=self.process_group, async_op=False)
            if self.rank == 0:
                key = f"{id(zq_param.module)}.{zq_param.param_name}"
                state[key] = zq_param.param.data.clone()
            zq_param.release()
        return state

    def __len__(self):
        return len(self._params)


class ZeroQModuleWrapper:
    """
    Wraps a module to automatically gather/release parameters.
    
    Registers hooks for pre/post forward and backward to manage
    parameter availability.
    """
    
    def __init__(
        self,
        module: torch.nn.Module,
        coordinator: ZeroQCoordinator,
        trainable_only: bool = False,
    ):
        self.module = module
        self.coordinator = coordinator
        self.trainable_only = trainable_only
        
        # Map module -> parameter IDs
        self._module_param_ids: Dict[torch.nn.Module, List[int]] = {}
        
        # Register all parameters
        self._register_parameters(module)
        
        # Install hooks
        self._install_hooks(module)
    
    def _register_parameters(self, module: torch.nn.Module):
        """Register all parameters in module tree."""
        for name, child in module.named_modules():
            param_ids = []
            for param_name, param in child.named_parameters(recurse=False):
                # Respect wrapper-level filter
                if self.trainable_only and not param.requires_grad:
                    continue

                # Respect config-level partitioning rules
                # Default behavior for QLoRA: partition/freeze base weights only.
                if param.requires_grad and not self.coordinator.config.partition_trainable:
                    continue
                if self.coordinator.config.frozen_only and param.requires_grad:
                    continue

                # Reuse existing registrations when possible (important when a model
                # has been structurally modified after an initial registration, e.g.
                # PEFT replacing modules). This preserves original shapes even if
                # param.data has been cleared after partitioning.
                existing = self.coordinator.get_param_for_tensor(param)
                if existing is not None:
                    param_ids.append(existing.param_id)
                else:
                    zq_param = self.coordinator.register_parameter(param, child, param_name=param_name)
                    param_ids.append(zq_param.param_id)
            
            if param_ids:
                self._module_param_ids[child] = param_ids
    
    def _install_hooks(self, module: torch.nn.Module):
        """Install forward/backward hooks on all modules."""
        for child in module.modules():
            if child in self._module_param_ids:
                child.register_forward_pre_hook(self._pre_forward_hook)
                child.register_forward_hook(self._post_forward_hook)
                # For training we must re-gather parameters before the module's backward runs,
                # because we release them immediately after forward to keep memory bounded.
                if hasattr(child, "register_full_backward_pre_hook"):
                    child.register_full_backward_pre_hook(self._pre_backward_hook)
                child.register_full_backward_hook(self._post_backward_hook)
    
    def _pre_forward_hook(self, module: torch.nn.Module, inputs):
        """Gather parameters before forward pass."""
        if module in self._module_param_ids:
            param_ids = self._module_param_ids[module]
            self.coordinator.fetch_params(param_ids, async_op=False)
    
    def _post_forward_hook(self, module: torch.nn.Module, inputs, outputs):
        """Release gathered parameters after forward to keep peak memory bounded."""
        if module in self._module_param_ids:
            param_ids = self._module_param_ids[module]
            self.coordinator.release_params(param_ids)

    def _pre_backward_hook(self, module: torch.nn.Module, grad_output):
        """Re-gather parameters right before backward for this module."""
        if module in self._module_param_ids:
            param_ids = self._module_param_ids[module]
            self.coordinator.fetch_params(param_ids, async_op=False)

    def _post_backward_hook(self, module: torch.nn.Module, grad_input, grad_output):
        """Release parameters after backward pass to maintain memory savings during training."""
        if module in self._module_param_ids:
            param_ids = self._module_param_ids[module]
            self.coordinator.release_params(param_ids)
    
    def partition(self):
        """Partition all parameters."""
        self.coordinator.partition_all()
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return self.coordinator.get_memory_stats()
