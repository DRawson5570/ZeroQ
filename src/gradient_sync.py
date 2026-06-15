"""
ZeroQ Gradient Synchronization

Implements efficient gradient synchronization algorithms for
distributed training across multiple nodes.

Algorithms:
- Ring All-Reduce: Bandwidth-optimal for large tensors
- Naive All-Reduce: Simple but less efficient
- Async SGD: Non-blocking gradient updates

Author: Zero (Claude Opus 4.5) in collaboration with Douglas Rawson
Created: December 2025
"""

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, List, Callable, Any, Union
import torch
import torch.nn.functional as F
import torch.distributed as dist

try:
    from .transport import TransportWorker, serialize_tensor, deserialize_tensor
except ImportError:
    from transport import TransportWorker, serialize_tensor, deserialize_tensor


class SyncMode(Enum):
    """Gradient synchronization modes."""
    RING_ALLREDUCE = "ring"      # Bandwidth-optimal ring all-reduce
    NAIVE_ALLREDUCE = "naive"    # Simple all-to-all reduce
    ASYNC_SGD = "async"          # Asynchronous SGD
    LOCAL_SGD = "local_sgd"      # Periodic sync (local SGD)


@dataclass
class GradSyncConfig:
    """Configuration for gradient synchronization."""
    mode: SyncMode = SyncMode.RING_ALLREDUCE
    
    # Ring all-reduce settings
    num_chunks: int = 4  # Number of chunks for pipelining
    
    # Async SGD settings
    async_buffer_size: int = 16  # Max pending gradients
    staleness_threshold: int = 4  # Max gradient staleness
    
    # Local SGD settings
    local_steps: int = 4  # Steps between syncs
    
    # Compression
    compress: bool = True
    compression_ratio: float = 0.01  # Top-k ratio
    
    # Timeout
    timeout_sec: float = 60.0


class RingAllReduce:
    """
    Ring All-Reduce implementation for bandwidth-optimal gradient sync.
    
    The ring algorithm:
    1. Split tensor into world_size chunks
    2. Reduce-scatter: Each rank sends chunk i to rank (i+1) % world_size
       and accumulates received chunk. After world_size-1 steps,
       each rank has complete sum for one chunk.
    3. All-gather: Each rank sends its complete chunk around the ring.
       After world_size-1 steps, all ranks have all chunks.
    
    Total data transferred: 2 * (world_size - 1) / world_size * tensor_size
    This is optimal for large tensors.
    """
    
    def __init__(
        self,
        transport: TransportWorker,
        config: Optional[GradSyncConfig] = None,
    ):
        self.transport = transport
        self.config = config or GradSyncConfig()
        
        self.rank = transport.rank
        self.world_size = transport.world_size
        
        # Ring topology
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank - 1 + self.world_size) % self.world_size
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        param_id: int,
    ) -> torch.Tensor:
        """
        Perform ring all-reduce on a tensor.
        
        Args:
            tensor: Gradient tensor to reduce
            param_id: Parameter ID for tracking
            
        Returns:
            Reduced tensor (sum across all ranks)
        """
        if self.world_size == 1:
            return tensor
        
        # Split tensor into chunks
        chunks = self._split_tensor(tensor)
        num_chunks = len(chunks)
        
        # Phase 1: Reduce-scatter
        for step in range(self.world_size - 1):
            # Determine which chunk to send/receive
            send_chunk_idx = (self.rank - step + self.world_size) % self.world_size
            recv_chunk_idx = (self.rank - step - 1 + self.world_size) % self.world_size
            
            # Send our chunk
            send_chunk_idx = send_chunk_idx % num_chunks
            self._send_chunk(chunks[send_chunk_idx], param_id, send_chunk_idx)
            
            # Receive and accumulate
            recv_chunk_idx = recv_chunk_idx % num_chunks
            received = self._recv_chunk(param_id, recv_chunk_idx)
            if received is not None:
                chunks[recv_chunk_idx] += received
        
        # Phase 2: All-gather
        for step in range(self.world_size - 1):
            send_chunk_idx = (self.rank - step + 1 + self.world_size) % self.world_size
            recv_chunk_idx = (self.rank - step + self.world_size) % self.world_size
            
            send_chunk_idx = send_chunk_idx % num_chunks
            self._send_chunk(chunks[send_chunk_idx], param_id, send_chunk_idx)
            
            recv_chunk_idx = recv_chunk_idx % num_chunks
            received = self._recv_chunk(param_id, recv_chunk_idx)
            if received is not None:
                chunks[recv_chunk_idx] = received
        
        # Reconstruct tensor
        return self._merge_chunks(chunks, tensor.shape)
    
    def _split_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Split tensor into world_size chunks."""
        flat = tensor.view(-1)
        chunk_size = math.ceil(flat.numel() / self.world_size)
        
        chunks = []
        for i in range(self.world_size):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, flat.numel())
            if start < flat.numel():
                chunks.append(flat[start:end].clone())
            else:
                # Pad with zeros for uneven splits
                chunks.append(torch.zeros(chunk_size, device=tensor.device, dtype=tensor.dtype))
        
        return chunks
    
    def _merge_chunks(self, chunks: List[torch.Tensor], shape: torch.Size) -> torch.Tensor:
        """Merge chunks back into original shape."""
        flat = torch.cat(chunks)
        numel = torch.tensor(shape).prod().item()
        return flat[:numel].view(shape)
    
    def _send_chunk(self, chunk: torch.Tensor, param_id: int, chunk_id: int):
        """Send a chunk to the next rank in the ring."""
        import pickle
        
        msg = {
            "type": "ring_send",
            "param_id": param_id,
            "chunk_id": chunk_id,
            "src_rank": self.rank,
            "data": serialize_tensor(chunk),
        }
        
        if self.send_rank in self.transport.peer_sockets:
            self.transport.peer_sockets[self.send_rank].send(pickle.dumps(msg))
    
    def _recv_chunk(
        self,
        param_id: int,
        chunk_id: int,
        timeout_sec: float = 30.0,
    ) -> Optional[torch.Tensor]:
        """Receive a chunk from the previous rank in the ring."""
        key = (param_id, chunk_id)
        start_time = time.time()
        
        while time.time() - start_time < timeout_sec:
            with self.transport._lock:
                if key in self.transport._recv_buffer:
                    if self.recv_rank in self.transport._recv_buffer[key]:
                        tensor = self.transport._recv_buffer[key].pop(self.recv_rank)
                        if not self.transport._recv_buffer[key]:
                            del self.transport._recv_buffer[key]
                        return tensor
            
            time.sleep(0.001)
        
        return None


class TorchDistributedSync:
    """
    Gradient synchronization using torch.distributed.
    
    This is for single-node multi-GPU training where all GPUs
    can use NCCL directly.
    """
    
    def __init__(
        self,
        process_group: Optional[dist.ProcessGroup] = None,
        config: Optional[GradSyncConfig] = None,
    ):
        self.process_group = process_group
        self.config = config or GradSyncConfig()
        
        if dist.is_initialized():
            self.rank = dist.get_rank(process_group)
            self.world_size = dist.get_world_size(process_group)
        else:
            self.rank = 0
            self.world_size = 1
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """
        Perform all-reduce using torch.distributed.
        
        Args:
            tensor: Gradient tensor (modified in-place)
            async_op: Whether to perform async operation
            
        Returns:
            Work handle if async_op=True, else None
        """
        if self.world_size == 1:
            return None
        
        return dist.all_reduce(
            tensor,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=async_op,
        )
    
    def all_reduce_coalesced(
        self,
        tensors: List[torch.Tensor],
        async_op: bool = False,
    ) -> Optional[dist.Work]:
        """
        Perform coalesced all-reduce for multiple tensors.
        
        More efficient than separate all-reduces.
        """
        if self.world_size == 1:
            return None
        
        # Flatten and reduce
        flat_tensors = [t.view(-1) for t in tensors]
        coalesced = torch.cat(flat_tensors)
        
        handle = dist.all_reduce(
            coalesced,
            op=dist.ReduceOp.SUM,
            group=self.process_group,
            async_op=async_op,
        )
        
        if not async_op:
            # Copy back to original tensors
            offset = 0
            for t in tensors:
                numel = t.numel()
                t.copy_(coalesced[offset:offset + numel].view(t.shape))
                offset += numel
        
        return handle


class GradientAccumulator:
    """
    Manages gradient accumulation with periodic synchronization.
    
    Useful for:
    - Gradient accumulation (effective batch size)
    - Local SGD (periodic sync)
    - Async SGD (bounded staleness)
    """
    
    def __init__(
        self,
        sync_backend: Union["TorchDistributedSync", "RingAllReduce"],
        config: Optional[GradSyncConfig] = None,
    ):
        self.sync = sync_backend
        self.config = config or GradSyncConfig()
        
        # Accumulated gradients
        self._accum_grads: Dict[int, torch.Tensor] = {}
        self._accum_count = 0
        
        # For async SGD
        self._pending_syncs: List[Any] = []
        self._step_counter = 0
    
    def accumulate(self, param_id: int, grad: torch.Tensor):
        """Accumulate a gradient."""
        if param_id not in self._accum_grads:
            self._accum_grads[param_id] = grad.clone()
        else:
            self._accum_grads[param_id] += grad
        
        self._accum_count += 1
    
    def sync_and_clear(self) -> Dict[int, torch.Tensor]:
        """
        Synchronize accumulated gradients and clear buffer.
        
        Returns:
            Dictionary of synchronized gradients
        """
        synced_grads = {}
        
        for param_id, grad in self._accum_grads.items():
            if isinstance(self.sync, TorchDistributedSync):
                self.sync.all_reduce(grad)
                synced_grads[param_id] = grad / self.sync.world_size
            elif isinstance(self.sync, RingAllReduce):
                synced = self.sync.all_reduce(grad, param_id)
                synced_grads[param_id] = synced / self.sync.world_size
        
        self._accum_grads.clear()
        self._accum_count = 0
        
        return synced_grads
    
    def should_sync(self) -> bool:
        """Check if we should sync based on config."""
        if self.config.mode == SyncMode.LOCAL_SGD:
            self._step_counter += 1
            return self._step_counter >= self.config.local_steps
        
        return True  # Sync every step for other modes


class HierarchicalSync:
    """
    Hierarchical gradient synchronization for multi-node multi-GPU.
    
    Strategy:
    1. Intra-node: Use NCCL for fast GPU-GPU communication
    2. Inter-node: Use ZeroMQ transport for node-node communication
    
    This is optimal when:
    - Intra-node bandwidth >> Inter-node bandwidth
    - Multiple GPUs per node
    """
    
    def __init__(
        self,
        local_sync: TorchDistributedSync,
        global_sync: RingAllReduce,
        local_rank: int,
        local_size: int,
    ):
        self.local_sync = local_sync
        self.global_sync = global_sync
        self.local_rank = local_rank
        self.local_size = local_size

        # Determine the (global) leader rank for this process group.
        # `dist.broadcast(..., group=...)` expects `src` as a global rank.
        if dist.is_initialized():
            global_rank = dist.get_rank()
            if self.local_size > 1:
                backend = dist.get_backend(self.local_sync.process_group)
                device = torch.device("cuda") if backend == "nccl" else torch.device("cpu")
                leader = torch.tensor(global_rank, dtype=torch.int64, device=device)
                dist.all_reduce(leader, op=dist.ReduceOp.MIN, group=self.local_sync.process_group)
                self._leader_global_rank = int(leader.item())
            else:
                self._leader_global_rank = global_rank
            self.is_local_leader = (global_rank == self._leader_global_rank)
        else:
            self._leader_global_rank = 0
            self.is_local_leader = True
    
    def all_reduce(
        self,
        tensor: torch.Tensor,
        param_id: int,
    ) -> torch.Tensor:
        """
        Perform hierarchical all-reduce.
        
        Args:
            tensor: Gradient tensor
            param_id: Parameter ID
            
        Returns:
            Reduced tensor
        """
        # Step 1: Local all-reduce (NCCL)
        if self.local_size > 1:
            self.local_sync.all_reduce(tensor)
        
        # Step 2: Global all-reduce (ring over ZeroMQ)
        # Only local leaders participate
        if self.is_local_leader:
            tensor = self.global_sync.all_reduce(tensor, param_id)
        
        # Step 3: Broadcast from local leader to local workers
        if self.local_size > 1:
            dist.broadcast(
                tensor,
                src=self._leader_global_rank,
                group=self.local_sync.process_group,
            )
        
        return tensor


def create_gradient_sync(
    mode: str = "auto",
    transport: Optional[TransportWorker] = None,
    process_group: Optional[dist.ProcessGroup] = None,
    config: Optional[GradSyncConfig] = None,
) -> Union["TorchDistributedSync", "RingAllReduce", "HierarchicalSync"]:
    """
    Factory function to create appropriate gradient sync backend.
    
    Args:
        mode: "auto", "nccl", "ring", or "hierarchical"
        transport: ZeroMQ transport (for ring/hierarchical)
        process_group: torch.distributed process group (for nccl)
        config: Sync configuration
        
    Returns:
        Gradient sync backend
    """
    config = config or GradSyncConfig()
    
    if mode == "auto":
        # Auto-detect best backend
        if transport is not None and transport.world_size > 1:
            mode = "ring"
        elif dist.is_initialized() and dist.get_world_size() > 1:
            mode = "nccl"
        else:
            mode = "nccl"  # Single process, no-op
    
    if mode == "nccl":
        return TorchDistributedSync(process_group, config)
    
    elif mode == "ring":
        if transport is None:
            raise ValueError("Ring mode requires TransportWorker")
        return RingAllReduce(transport, config)
    
    elif mode == "hierarchical":
        if transport is None or not dist.is_initialized():
            raise ValueError("Hierarchical mode requires both transport and torch.distributed")
        
        local_sync = TorchDistributedSync(process_group, config)
        global_sync = RingAllReduce(transport, config)
        
        local_rank = dist.get_rank(process_group)
        local_size = dist.get_world_size(process_group)
        
        return HierarchicalSync(local_sync, global_sync, local_rank, local_size)
    
    else:
        raise ValueError(f"Unknown mode: {mode}")


def reduce_scatter_grads(
    param_full_grad: torch.Tensor,
    world_size: int,
    rank: int,
    group: Optional[dist.ProcessGroup] = None,
) -> torch.Tensor:
    """
    Reduce-scatter a full gradient so each rank gets its averaged shard.

    Used in ZeRO-Q training-from-scratch mode: after ``loss.backward()``
    produces a full-sized gradient, this function reduces across ranks and
    returns only the local shard (matching the fp32 master weight shard).

    If NCCL ``reduce_scatter`` is unavailable (e.g. Maxwell SM 5.2), falls
    back to ``all_reduce`` + local slice.

    Args:
        param_full_grad: Full gradient tensor (same shape as the gathered param).
        world_size: Number of ranks.
        rank: This rank's index.
        group: Process group (``None`` = default).

    Returns:
        Local gradient shard of size ``chunk_size``, averaged across ranks.
    """
    if not dist.is_initialized() or world_size == 1:
        return param_full_grad.contiguous().view(-1).clone()

    flat = param_full_grad.contiguous().view(-1)
    chunk_size = (flat.numel() + world_size - 1) // world_size

    if flat.numel() % world_size != 0:
        flat = F.pad(flat, (0, chunk_size * world_size - flat.numel()))

    output = torch.empty(chunk_size, device=flat.device, dtype=flat.dtype)

    try:
        dist.reduce_scatter_tensor(output, flat, op=dist.ReduceOp.SUM, group=group)
    except (RuntimeError, AttributeError):
        dist.all_reduce(flat, op=dist.ReduceOp.SUM, group=group)
        output.copy_(flat[rank * chunk_size : (rank + 1) * chunk_size])

    output.div_(world_size)
    return output


# Convenience exports
__all__ = [
    "SyncMode",
    "GradSyncConfig",
    "RingAllReduce",
    "TorchDistributedSync",
    "GradientAccumulator",
    "HierarchicalSync",
    "create_gradient_sync",
    "reduce_scatter_grads",
]
