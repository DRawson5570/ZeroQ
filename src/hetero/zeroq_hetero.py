"""
Heterogeneous ZeRO-Q: Weighted Partitioning for Mixed-VRAM GPUs.

Drop-in replacement for ZeroQCoordinator/ZeroQModuleWrapper that
distributes weight shards proportional to each GPU's VRAM capacity.

Key difference vs uniform ZeRO-Q:
  - Local shard size varies per rank based on rank_weights
  - All-gather uses padded sends (pad to max shard, gather, unpad)
  - Reassembly uses per-rank offsets/lengths from ShardPlan

Example with 24GB + 12GB + 24GB GPUs:
  - rank_weights = [24576.0, 11520.0, 24576.0]  (auto-detected)
  - 7B model quantized to ~3.5GB total packed data
  - GPU 0: ~1.4GB, GPU 1: ~0.7GB, GPU 2: ~1.4GB

Ported from Mnemosyne hetero-train codebase with improvements for
Maxwell architecture (fp16 compute, eager cache clearing).
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Optional, Dict, List, Tuple

import torch
import torch.distributed as dist

from bitsandbytes.functional import quantize_4bit, dequantize_4bit, QuantState
import bitsandbytes as bnb

from .shard_plan import make_plan, ShardPlan


def _cfg_attr(cfg: Any, name: str, default: Any) -> Any:
    """Safely get config attribute with default."""
    return getattr(cfg, name, default)


def _parse_rank_weights_env(world_size: int) -> list[float] | None:
    """Parse ZEROQ_HETERO_RANK_WEIGHTS environment variable.

    Format: comma-separated floats, one per rank.
    Example: ZEROQ_HETERO_RANK_WEIGHTS=24576,11520,24576
    """
    raw = os.environ.get("ZEROQ_HETERO_RANK_WEIGHTS")
    if not raw:
        return None
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if len(parts) != world_size:
        raise ValueError(
            f"ZEROQ_HETERO_RANK_WEIGHTS expects {world_size} comma-separated values; got {len(parts)}"
        )
    return [float(x) for x in parts]


def discover_rank_weights(
    *,
    group: Optional[dist.ProcessGroup] = None,
    activation_reserve_mb: float = 0.0,
) -> list[float]:
    """Auto-detect per-rank capacity weights via CUDA device properties.

    Preference order:
    1) ZEROQ_HETERO_RANK_WEIGHTS env var (comma-separated list)
    2) CUDA total_memory per rank (all_gather across ranks)
    3) Uniform weights (fallback)

    If activation_reserve_mb > 0, that amount is subtracted from each
    GPU's raw VRAM before computing weights. This accounts for the
    per-rank activation memory that does NOT shard — every rank pays
    the same activation tax, so smaller GPUs need proportionally more
    headroom reserved.

    Example without reserve:
      [24576, 11520] → ratio 2.13:1
    Example with 8192 MB reserve:
      [24576-8192, 11520-8192] = [16384, 3328] → ratio 4.92:1

    Returns:
        List of float weights, one per rank. Larger = more shards.
    """
    if not dist.is_initialized():
        return [1.0]

    ws = dist.get_world_size(group)

    # Check env override first
    env = _parse_rank_weights_env(ws)
    if env is not None:
        return env

    # Auto-detect from CUDA device properties
    w = 1.0
    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        try:
            props = torch.cuda.get_device_properties(local_rank)
            w = float(getattr(props, "total_memory", 0) or 0) / (1024**2)
        except Exception:
            w = 1.0

    # All-gather weights from all ranks (must be on CUDA for NCCL backend)
    gather_device = torch.device(f"cuda:{int(os.environ.get('LOCAL_RANK', '0'))}")
    t = torch.tensor([w], dtype=torch.float64, device=gather_device)
    out = [torch.empty_like(t) for _ in range(ws)]
    dist.all_gather(out, t, group=group)
    raw_weights = [float(x.item()) for x in out]

    if not any(v > 0 for v in raw_weights):
        return [1.0 for _ in raw_weights]

    # Subtract activation reserve from each GPU's capacity
    if activation_reserve_mb > 0:
        weights = [max(w - activation_reserve_mb, 1.0) for w in raw_weights]
        rank0 = dist.get_rank(group) if dist.is_initialized() else 0
        if rank0 == 0:
            print(f"[ZeRO-Q Hetero] Activation reserve: {activation_reserve_mb:.0f} MB per rank")
            print(f"[ZeRO-Q Hetero] Raw VRAM (MB):  {[f'{w:.0f}' for w in raw_weights]}")
            print(f"[ZeRO-Q Hetero] Adjusted weights: {[f'{w:.0f}' for w in weights]}")
    else:
        weights = raw_weights

    return weights


@dataclass
class _VarShardMeta:
    """Metadata for variable-length packed/absmax shards."""
    packed: ShardPlan
    absmax: ShardPlan


class HeteroZeroQParamStatus:
    NOT_AVAILABLE = 0
    AVAILABLE = 1
    INFLIGHT = 2


class HeteroZeroQParameter:
    """A hetero-sharded variant of ZeroQParameter.

    Key difference vs uniform: local shard size varies per rank based on
    rank_weights. Larger GPUs hold proportionally more quantized data.
    """

    def __init__(
        self,
        *,
        param: torch.nn.Parameter,
        rank: int,
        world_size: int,
        config: Any,
        param_id: int,
        rank_weights: list[float],
        module: Optional[torch.nn.Module] = None,
        param_name: Optional[str] = None,
        process_group: Optional[dist.ProcessGroup] = None,
    ):
        self.param = param
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.config = config
        self.param_id = int(param_id)
        self.rank_weights = list(rank_weights)
        self.module = module
        self.param_name = param_name
        self.process_group = process_group

        self.original_shape = param.data.shape
        self.original_numel = param.data.numel()
        self.original_dtype = param.data.dtype
        self.compute_in_4bit = bool(_cfg_attr(config, 'compute_in_4bit', False))

        self.status = HeteroZeroQParamStatus.NOT_AVAILABLE

        self.local_packed: Optional[torch.Tensor] = None
        self.local_absmax: Optional[torch.Tensor] = None
        self._quant_meta: Optional[Dict[str, Any]] = None

        self._gather_handles: Optional[Tuple[Any, Any]] = None

        self._gathered_packed: Optional[torch.Tensor] = None
        self._gathered_absmax: Optional[torch.Tensor] = None
        self._send_packed: Optional[torch.Tensor] = None
        self._send_absmax: Optional[torch.Tensor] = None
        self._assembled_packed: Optional[torch.Tensor] = None
        self._assembled_absmax: Optional[torch.Tensor] = None

        self._packed_total: int = 0
        self._absmax_total: int = 0
        self._shards: Optional[_VarShardMeta] = None

    def partition(self):
        """Partition from the param's current data (must not be meta)."""
        if self.status != HeteroZeroQParamStatus.NOT_AVAILABLE or self.local_packed is not None:
            return
        self.partition_from_full_precision(self.param.data)

    def partition_from_full_precision(self, weight: torch.Tensor):
        """Quantize a full-precision weight and partition across ranks by VRAM weight.

        This is the core hetero operation:
        1. Move weight to CUDA
        2. Quantize to 4-bit
        3. Split packed/absmax by weighted ShardPlan
        4. Keep only this rank's shard, discard the rest
        """
        if self.status != HeteroZeroQParamStatus.NOT_AVAILABLE or self.local_packed is not None:
            return

        if weight.numel() != self.original_numel:
            raise ValueError(
                f"Weight numel mismatch for param_id={self.param_id}: "
                f"expected {self.original_numel}, got {weight.numel()}"
            )

        target_device = self.param.device
        if target_device.type != "cuda" and torch.cuda.is_available():
            target_device = torch.device("cuda")

        # Replace meta/CPU params with empty CUDA placeholders
        if self.param.device.type == "meta" or self.param.device != target_device:
            if self.module is None or self.param_name is None:
                raise RuntimeError(
                    "HeteroZeroQParameter needs module+param_name to replace meta/CPU parameters safely."
                )
            new_param = torch.nn.Parameter(
                torch.empty(0, device=target_device, dtype=self.original_dtype),
                requires_grad=self.param.requires_grad,
            )
            setattr(self.module, self.param_name, new_param)
            self.param = new_param

        # Move weight to CUDA for quantization
        weight = weight.contiguous().to(device=target_device)
        if weight.dtype != torch.float16:
            weight = weight.to(torch.float16)

        if target_device.type == "cuda":
            torch.cuda.synchronize()

        # Quantize to 4-bit
        packed, quant_state = quantize_4bit(
            weight,
            blocksize=int(_cfg_attr(self.config, "blocksize", 64)),
            quant_type=str(_cfg_attr(self.config, "quant_type", "nf4")),
        )

        packed = packed.contiguous().view(-1)
        absmax = quant_state.absmax.contiguous().view(-1)

        self._packed_total = int(packed.numel())
        self._absmax_total = int(absmax.numel())

        # Create weighted shard plans
        packed_plan = make_plan(self._packed_total, self.rank_weights)
        absmax_plan = make_plan(self._absmax_total, self.rank_weights)
        self._shards = _VarShardMeta(packed=packed_plan, absmax=absmax_plan)

        # Extract this rank's shard
        p0 = int(packed_plan.offsets[self.rank])
        pN = p0 + int(packed_plan.lengths[self.rank])
        a0 = int(absmax_plan.offsets[self.rank])
        aN = a0 + int(absmax_plan.lengths[self.rank])

        self.local_packed = packed[p0:pN].clone()
        self.local_absmax = absmax[a0:aN].clone()

        # Save quantization metadata for reconstruction
        self._quant_meta = {
            "shape": quant_state.shape,
            "dtype": quant_state.dtype,
            "blocksize": quant_state.blocksize,
            "code": quant_state.code,
            "quant_type": quant_state.quant_type,
        }

        # Immediately free full-size tensors to minimize peak VRAM
        del weight, packed, absmax, quant_state
        torch.cuda.empty_cache()

        # Replace param with empty placeholder (free memory)
        self.param.data = torch.empty(0, device=target_device, dtype=self.original_dtype)
        self.status = HeteroZeroQParamStatus.NOT_AVAILABLE

    def start_gather(self, *, async_op: bool = True):
        """Begin all-gather of this parameter's shards from all ranks.

        Since shards are variable-length, we pad each rank's send buffer
        to max(shard_lengths), all-gather the padded buffers, then reassemble.
        """
        if self.status == HeteroZeroQParamStatus.AVAILABLE:
            return None
        if self.status == HeteroZeroQParamStatus.INFLIGHT:
            return self._gather_handles
        if self.local_packed is None or self.local_absmax is None or self._shards is None:
            raise RuntimeError("Parameter is not partitioned")

        # Single-GPU: skip all-gather, just dequantize directly
        if self.world_size <= 1:
            self._gathered_packed = None
            self._gathered_absmax = None
            self._assembled_packed = self.local_packed
            self._assembled_absmax = self.local_absmax
            self._complete_gather()
            return None

        # Stride = max shard size (all ranks send this many bytes, padded)
        packed_stride = max(self._shards.packed.lengths) if self._shards.packed.lengths else 0
        absmax_stride = max(self._shards.absmax.lengths) if self._shards.absmax.lengths else 0

        # Allocate/reuse send buffers (padded to stride)
        if self._send_packed is None or int(self._send_packed.numel()) != int(packed_stride):
            self._send_packed = torch.empty(
                int(packed_stride),
                dtype=self.local_packed.dtype,
                device=self.local_packed.device,
            )
        if self._send_absmax is None or int(self._send_absmax.numel()) != int(absmax_stride):
            self._send_absmax = torch.empty(
                int(absmax_stride),
                dtype=self.local_absmax.dtype,
                device=self.local_absmax.device,
            )

        # Copy local shard into padded send buffer
        self._send_packed.zero_()
        if int(self.local_packed.numel()) > 0:
            self._send_packed[: int(self.local_packed.numel())].copy_(self.local_packed)

        self._send_absmax.zero_()
        if int(self.local_absmax.numel()) > 0:
            self._send_absmax[: int(self.local_absmax.numel())].copy_(self.local_absmax)

        # All-gather padded buffers
        if hasattr(dist, "all_gather_into_tensor") and self.world_size > 1:
            if self._gathered_packed is None or int(self._gathered_packed.numel()) != int(packed_stride) * self.world_size:
                self._gathered_packed = torch.empty(
                    int(packed_stride) * self.world_size,
                    dtype=self.local_packed.dtype,
                    device=self.local_packed.device,
                )
            if self._gathered_absmax is None or int(self._gathered_absmax.numel()) != int(absmax_stride) * self.world_size:
                self._gathered_absmax = torch.empty(
                    int(absmax_stride) * self.world_size,
                    dtype=self.local_absmax.dtype,
                    device=self.local_absmax.device,
                )

            packed_handle = dist.all_gather_into_tensor(
                self._gathered_packed,
                self._send_packed,
                group=self.process_group,
                async_op=async_op,
            )
            absmax_handle = dist.all_gather_into_tensor(
                self._gathered_absmax,
                self._send_absmax,
                group=self.process_group,
                async_op=async_op,
            )
        else:
            # Fallback: list gather of padded sends
            packed_list = [torch.empty_like(self._send_packed) for _ in range(self.world_size)]
            absmax_list = [torch.empty_like(self._send_absmax) for _ in range(self.world_size)]

            packed_handle = dist.all_gather(packed_list, self._send_packed, group=self.process_group, async_op=async_op)
            absmax_handle = dist.all_gather(absmax_list, self._send_absmax, group=self.process_group, async_op=async_op)

            if not async_op:
                self._gathered_packed = torch.cat(packed_list, dim=0)
                self._gathered_absmax = torch.cat(absmax_list, dim=0)
            else:
                self._packed_list_tmp = packed_list
                self._absmax_list_tmp = absmax_list

        if async_op:
            self._gather_handles = (packed_handle, absmax_handle)
            self.status = HeteroZeroQParamStatus.INFLIGHT
            return self._gather_handles

        self._complete_gather()
        return None

    def wait_gather(self):
        """Wait for async gather to complete, then dequantize."""
        if self.status != HeteroZeroQParamStatus.INFLIGHT:
            return
        if self._gather_handles is not None:
            self._gather_handles[0].wait()
            self._gather_handles[1].wait()

        # If we used list-gather async, materialize gathered tensors now
        packed_list = getattr(self, "_packed_list_tmp", None)
        absmax_list = getattr(self, "_absmax_list_tmp", None)
        if packed_list is not None and absmax_list is not None:
            self._gathered_packed = torch.cat(packed_list, dim=0)
            self._gathered_absmax = torch.cat(absmax_list, dim=0)
            delattr(self, "_packed_list_tmp")
            delattr(self, "_absmax_list_tmp")

        self._complete_gather()

    def _complete_gather(self):
        """Reassemble variable-length shards and dequantize to full precision.

        After all-gather, each rank has all padded shards. We extract the
        actual data from each rank's shard using the ShardPlan offsets/lengths,
        then dequantize the reassembled packed tensor.
        """
        if self._quant_meta is None:
            raise RuntimeError(f"Missing quant metadata for param_id={self.param_id}")

        # Single-GPU path: _assembled buffers were set directly from local shards
        if self.world_size > 1:
            if self._gathered_packed is None or self._gathered_absmax is None or self._shards is None:
                raise RuntimeError("Gather buffers missing")

            packed_stride = max(self._shards.packed.lengths) if self._shards.packed.lengths else 0
            absmax_stride = max(self._shards.absmax.lengths) if self._shards.absmax.lengths else 0

            # Allocate/reuse assembly buffers (full tensor size)
            if self._assembled_packed is None or int(self._assembled_packed.numel()) != int(self._packed_total):
                self._assembled_packed = torch.empty(
                    int(self._packed_total),
                    dtype=self._gathered_packed.dtype,
                    device=self._gathered_packed.device,
                )
            if self._assembled_absmax is None or int(self._assembled_absmax.numel()) != int(self._absmax_total):
                self._assembled_absmax = torch.empty(
                    int(self._absmax_total),
                    dtype=self._gathered_absmax.dtype,
                    device=self._gathered_absmax.device,
                )

            # Reassemble: copy each rank's actual data (unpadded) into position
            for i in range(self.world_size):
                lp = int(self._shards.packed.lengths[i])
                la = int(self._shards.absmax.lengths[i])
                op = int(self._shards.packed.offsets[i])
                oa = int(self._shards.absmax.offsets[i])

                seg_p = self._gathered_packed[i * int(packed_stride) : (i + 1) * int(packed_stride)]
                seg_a = self._gathered_absmax[i * int(absmax_stride) : (i + 1) * int(absmax_stride)]

                if lp > 0:
                    self._assembled_packed[op : op + lp].copy_(seg_p[:lp])
                if la > 0:
                    self._assembled_absmax[oa : oa + la].copy_(seg_a[:la])

            # Drop large gather buffers before dequantizing to reduce peak VRAM
            self._gathered_packed = None
            self._gathered_absmax = None
            self._send_packed = None
            self._send_absmax = None

        out_dtype = self.original_dtype
        # Maxwell doesn't support BF16; prefer FP16 to avoid extra allocation
        if self._assembled_packed.is_cuda and out_dtype == torch.bfloat16:
            out_dtype = torch.float16

        gathered_state = QuantState(
            absmax=self._assembled_absmax,
            shape=self._quant_meta["shape"],
            dtype=out_dtype,
            blocksize=self._quant_meta["blocksize"],
            code=self._quant_meta["code"],
            quant_type=self._quant_meta["quant_type"],
        )

        if self._assembled_packed.is_cuda:
            torch.cuda.empty_cache()

        # ── 4-bit compute mode: keep as Params4bit, no fp16 dequant ──
        if self.compute_in_4bit and self.module is not None and self.param_name is not None:
            # Create a bnb Params4bit — bnb's Linear4bit.forward() will use
            # its fused matmul_4bit kernel, never materializing an fp16 copy
            param_4bit = bnb.nn.Params4bit(
                self._assembled_packed.clone(),
                requires_grad=False,
                quant_state=gathered_state,
                quant_type=self._quant_meta["quant_type"],
                blocksize=self._quant_meta["blocksize"],
            )
            setattr(self.module, self.param_name, param_4bit)
            self.param = param_4bit
            self.status = HeteroZeroQParamStatus.AVAILABLE
            self._gather_handles = None
            return

        # ── Standard mode: dequantize to fp16/fp32 ──
        restored = dequantize_4bit(self._assembled_packed, gathered_state)

        target_dtype = self.original_dtype
        # Keep FP16 on Maxwell to reduce peak VRAM
        if restored.is_cuda and target_dtype == torch.bfloat16 and restored.dtype == torch.float16:
            target_dtype = restored.dtype
        if target_dtype != restored.dtype:
            restored = restored.to(target_dtype)

        restored = restored.view(self.original_shape)

        # Replace the module's param with the restored tensor
        if self.module is not None and self.param_name is not None:
            new_param = torch.nn.Parameter(restored, requires_grad=self.param.requires_grad)
            setattr(self.module, self.param_name, new_param)
            self.param = new_param
        else:
            self.param.data = restored

        self.status = HeteroZeroQParamStatus.AVAILABLE
        self._gather_handles = None

    def release(self):
        """Release the dequantized parameter, keeping only the local shard."""
        if self.status == HeteroZeroQParamStatus.NOT_AVAILABLE:
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

        self._gathered_packed = None
        self._gathered_absmax = None
        self._send_packed = None
        self._send_absmax = None
        self._assembled_packed = None
        self._assembled_absmax = None

        self.status = HeteroZeroQParamStatus.NOT_AVAILABLE

    @property
    def local_memory_bytes(self) -> int:
        """Memory used by this rank's local shard."""
        if self.local_packed is None or self.local_absmax is None:
            return 0
        return (
            int(self.local_packed.numel()) * int(self.local_packed.element_size()) +
            int(self.local_absmax.numel()) * int(self.local_absmax.element_size())
        )

    @property
    def full_memory_bytes(self) -> int:
        """Memory that would be needed for the full FP16 tensor."""
        return int(self.original_numel) * 2


class HeteroZeroQCoordinator:
    """Coordinator for heterogeneous ZeRO-Q parameters.

    Drop-in replacement for ZeroQCoordinator that uses weighted
    partitioning based on per-rank VRAM capacity.
    """

    def __init__(self, config: Any, process_group: Optional[dist.ProcessGroup] = None):
        self.config = config
        self.process_group = process_group

        if dist.is_initialized():
            self.rank = dist.get_rank(process_group)
            self.world_size = dist.get_world_size(process_group)
        else:
            self.rank = 0
            self.world_size = 1

        activation_reserve = float(_cfg_attr(config, 'activation_reserve_mb', 0.0))
        self.rank_weights = discover_rank_weights(
            group=process_group,
            activation_reserve_mb=activation_reserve,
        )

        self._params: Dict[int, HeteroZeroQParameter] = {}
        self._params_by_obj_id: Dict[int, HeteroZeroQParameter] = {}
        self._next_param_id = 0
        self._collective_seq = 0  # Monotonic counter for collective ordering

    def register_parameter(
        self,
        param: torch.nn.Parameter,
        module: Optional[torch.nn.Module] = None,
        param_name: Optional[str] = None,
    ) -> HeteroZeroQParameter:
        """Register a parameter for heterogeneous partitioning."""
        pid = self._next_param_id
        self._next_param_id += 1

        zq = HeteroZeroQParameter(
            param=param,
            rank=self.rank,
            world_size=self.world_size,
            config=self.config,
            param_id=pid,
            rank_weights=self.rank_weights,
            module=module,
            param_name=param_name,
            process_group=self.process_group,
        )

        self._params[pid] = zq
        self._params_by_obj_id[id(param)] = zq
        return zq

    def get_param_for_tensor(self, param: torch.nn.Parameter) -> Optional[HeteroZeroQParameter]:
        """Look up a HeteroZeroQParameter by its torch.nn.Parameter object."""
        return self._params_by_obj_id.get(id(param))

    def partition_all(self):
        """Partition all registered parameters."""
        for p in self._params.values():
            p.partition()

    def verify_registration_consistency(self):
        """Cross-rank verification that all ranks registered identical params.

        Computes a hash of the (param_id, original_shape, original_numel) tuples
        and all-reduces to verify every rank agrees.  Catches registration ordering
        bugs at startup instead of a silent NCCL deadlock during training.
        """
        if self.world_size <= 1:
            return  # Nothing to verify

        # Build a deterministic fingerprint of the registration order
        entries = []
        for pid in sorted(self._params.keys()):
            p = self._params[pid]
            entries.append(f"{pid}:{tuple(p.original_shape)}:{p.original_numel}")
        fingerprint = "|".join(entries)
        # Use 15 hex chars (60 bits) to stay within signed int64 range for torch.long
        local_hash = int(hashlib.sha256(fingerprint.encode()).hexdigest()[:15], 16)

        # All-gather hashes from every rank
        local_tensor = torch.tensor([local_hash], dtype=torch.long,
                                     device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}")
        gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_tensor, group=self.process_group)

        hashes = [int(t.item()) for t in gathered]
        if len(set(hashes)) != 1:
            # Find which rank diverged
            mismatched = [r for r, h in enumerate(hashes) if h != hashes[0]]
            raise RuntimeError(
                f"ZeRO-Q FATAL: Registration order mismatch across ranks!\n"
                f"  Rank 0 hash: {hashes[0]}\n"
                f"  Mismatched ranks: {mismatched} (hashes: {[hashes[r] for r in mismatched]})\n"
                f"  Rank {self.rank} registered {len(self._params)} params.\n"
                f"  This would cause a silent NCCL deadlock during training.\n"
                f"  Check that all ranks load the same model with the same PEFT config."
            )

        if self.rank == 0:
            print(f"[ZeRO-Q] Registration consistency verified: {len(self._params)} params, "
                  f"all {self.world_size} ranks agree (hash={local_hash:#x})")

    @staticmethod
    def _is_partitioned(p: HeteroZeroQParameter) -> bool:
        return p.local_packed is not None and p.local_absmax is not None and p._shards is not None

    def fetch_params(self, param_ids: List[int], *, async_op: bool = True):
        """Begin gathering params by ID."""
        for pid in param_ids:
            if pid in self._params:
                p = self._params[pid]
                if self._is_partitioned(p):
                    p.start_gather(async_op=async_op)
                    self._collective_seq += 1

    def wait_params(self, param_ids: List[int]):
        """Wait for async gathers to complete."""
        for pid in param_ids:
            if pid in self._params:
                p = self._params[pid]
                if self._is_partitioned(p):
                    p.wait_gather()

    def release_params(self, param_ids: List[int]):
        """Release gathered params back to shard-only state."""
        for pid in param_ids:
            if pid in self._params:
                p = self._params[pid]
                if self._is_partitioned(p):
                    p.release()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics for this rank."""
        local_bytes = sum(p.local_memory_bytes for p in self._params.values())
        full_bytes = sum(p.full_memory_bytes for p in self._params.values())
        return {
            "local_memory_mb": float(local_bytes) / 1024**2,
            "full_fp16_memory_mb": float(full_bytes) / 1024**2,
            "compression_ratio": float(full_bytes) / max(float(local_bytes), 1.0),
            "num_params": float(len(self._params)),
            "rank_weight": float(self.rank_weights[self.rank]) if self.rank < len(self.rank_weights) else 0.0,
            "collective_seq": float(self._collective_seq),
        }

    def debug_collective_seq(self) -> str:
        """Diagnostic: compare collective sequence counters across ranks.

        Call this when you suspect an NCCL hang to see which rank fell behind.
        Uses all_gather so all ranks must call it together (e.g. inside a
        signal handler or timeout watchdog).

        Returns a human-readable summary string.
        """
        if self.world_size <= 1:
            return f"[ZeRO-Q] Single rank, seq={self._collective_seq}"

        local_tensor = torch.tensor(
            [self._collective_seq], dtype=torch.long,
            device=f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}"
        )
        gathered = [torch.zeros_like(local_tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered, local_tensor, group=self.process_group)

        seqs = [int(t.item()) for t in gathered]
        if len(set(seqs)) == 1:
            return (f"[ZeRO-Q] All {self.world_size} ranks at collective_seq={seqs[0]} ✓")
        else:
            lines = [f"[ZeRO-Q] COLLECTIVE SEQUENCE MISMATCH — potential deadlock source:"]
            for r, s in enumerate(seqs):
                marker = " ← BEHIND" if s < max(seqs) else ""
                lines.append(f"  Rank {r}: seq={s}{marker}")
            return "\n".join(lines)


class HeteroZeroQModuleWrapper:
    """Install forward/backward hooks for automatic gather/release.

    Drop-in replacement for ZeroQModuleWrapper that uses
    HeteroZeroQCoordinator for weighted partitioning.
    """

    def __init__(self, module: torch.nn.Module, coordinator: HeteroZeroQCoordinator, trainable_only: bool = False):
        self.module = module
        self.coordinator = coordinator
        self.trainable_only = trainable_only
        self._module_param_ids: Dict[torch.nn.Module, List[int]] = {}

        self._register_parameters(module)
        self._install_hooks(module)

    def _register_parameters(self, module: torch.nn.Module):
        """Register frozen parameters from all submodules.

        Deterministic ordering is critical: ranks must execute the same
        collectives in the same order.
        """
        for _name, child in sorted(module.named_modules(), key=lambda t: t[0]):
            param_ids: list[int] = []
            for param_name, param in sorted(child.named_parameters(recurse=False), key=lambda t: t[0]):
                if self.trainable_only and not param.requires_grad:
                    continue

                # Config-level rules (mirrors vendor/ZeroQ behavior)
                if param.requires_grad and not bool(_cfg_attr(self.coordinator.config, "partition_trainable", False)):
                    continue
                if bool(_cfg_attr(self.coordinator.config, "frozen_only", False)) and param.requires_grad:
                    continue

                existing = self.coordinator.get_param_for_tensor(param)
                if existing is not None:
                    param_ids.append(existing.param_id)
                else:
                    zq_param = self.coordinator.register_parameter(param, child, param_name=param_name)
                    param_ids.append(zq_param.param_id)

            if param_ids:
                self._module_param_ids[child] = param_ids

    def _install_hooks(self, module: torch.nn.Module):
        """Install forward/backward hooks on modules with registered params."""
        for child in module.modules():
            if child in self._module_param_ids:
                child.register_forward_pre_hook(self._pre_forward_hook)
                child.register_forward_hook(self._post_forward_hook)
                if hasattr(child, "register_full_backward_pre_hook"):
                    child.register_full_backward_pre_hook(self._pre_backward_hook)
                child.register_full_backward_hook(self._post_backward_hook)

    def _pre_forward_hook(self, module: torch.nn.Module, inputs):
        if module in self._module_param_ids:
            self.coordinator.fetch_params(self._module_param_ids[module], async_op=False)

    def _post_forward_hook(self, module: torch.nn.Module, inputs, outputs):
        if module in self._module_param_ids:
            self.coordinator.release_params(self._module_param_ids[module])

    def _pre_backward_hook(self, module: torch.nn.Module, grad_output):
        if module in self._module_param_ids:
            self.coordinator.fetch_params(self._module_param_ids[module], async_op=False)

    def _post_backward_hook(self, module: torch.nn.Module, grad_input, grad_output):
        if module in self._module_param_ids:
            self.coordinator.release_params(self._module_param_ids[module])

    def partition(self):
        """Partition all registered parameters."""
        self.coordinator.partition_all()

    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory usage statistics."""
        return self.coordinator.get_memory_stats()
