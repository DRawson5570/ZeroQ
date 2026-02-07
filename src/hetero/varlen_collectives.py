"""
Variable-Length Collectives for Heterogeneous All-Gather.

Standard all_gather requires equal-sized tensors on all ranks.
These helpers handle the case where each rank holds a different
number of elements (because bigger GPUs get bigger shards).

Ported from Mnemosyne hetero-train codebase.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.distributed as dist


def _all_gather_int(
    value: int,
    *,
    group: Optional[dist.ProcessGroup] = None,
    device: torch.device | None = None,
) -> list[int]:
    """All-gather a single integer from each rank."""
    t = torch.tensor([int(value)], dtype=torch.int64, device=device)
    out = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(out, t, group=group)
    return [int(x.item()) for x in out]


def all_gather_varlen_1d(
    local: torch.Tensor,
    *,
    group: Optional[dist.ProcessGroup] = None,
) -> tuple[list[int], torch.Tensor]:
    """All-gather 1D tensors with per-rank variable lengths.

    Implementation: all_gather lengths -> pad to max -> all_gather padded -> unpad.

    Args:
        local: This rank's 1D tensor (may differ in size across ranks)
        group: Process group

    Returns:
        (lengths, concatenated) where lengths[i] is rank i's contribution
    """
    if not dist.is_initialized():
        return [int(local.numel())], local.contiguous()

    if local.dim() != 1:
        raise ValueError("all_gather_varlen_1d requires a 1D tensor")

    local = local.contiguous()
    lengths = _all_gather_int(int(local.numel()), group=group, device=local.device)
    max_len = max(lengths) if lengths else 0

    if max_len == 0:
        return lengths, local.new_empty((0,))

    send = local
    if int(send.numel()) != int(max_len):
        padded = local.new_zeros((int(max_len),))
        if int(local.numel()) > 0:
            padded[: int(local.numel())].copy_(local)
        send = padded

    ws = dist.get_world_size(group)

    if hasattr(dist, "all_gather_into_tensor"):
        gathered = local.new_empty((int(ws) * int(max_len),))
        dist.all_gather_into_tensor(gathered, send, group=group)
        chunks: list[torch.Tensor] = []
        for i, ln in enumerate(lengths):
            if ln <= 0:
                continue
            seg = gathered[i * int(max_len) : (i + 1) * int(max_len)]
            chunks.append(seg[: int(ln)])
        if not chunks:
            return lengths, local.new_empty((0,))
        return lengths, torch.cat(chunks, dim=0)

    # Fallback: list-based gather (requires equal shapes; we already padded)
    out = [local.new_empty((int(max_len),)) for _ in range(ws)]
    dist.all_gather(out, send, group=group)

    chunks = []
    for i, ln in enumerate(lengths):
        if ln <= 0:
            continue
        chunks.append(out[i][: int(ln)])

    if not chunks:
        return lengths, local.new_empty((0,))
    return lengths, torch.cat(chunks, dim=0)
