"""
Weighted Shard Plan for Heterogeneous GPU Partitioning.

Distributes tensor elements across ranks proportional to their
VRAM capacity weights. A 24GB GPU gets 2x the shards of a 12GB GPU.

Ported from Mnemosyne hetero-train codebase.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class ShardPlan:
    """Describes how a tensor is split across ranks with variable lengths."""
    lengths: list[int]   # Elements per rank
    offsets: list[int]   # Start offset per rank


def prefix_sums(lengths: list[int]) -> list[int]:
    """Compute exclusive prefix sums (offsets) from lengths."""
    offsets: list[int] = [0]
    total = 0
    for n in lengths[:-1]:
        total += int(n)
        offsets.append(total)
    return offsets


def compute_weighted_shard_lengths(total: int, weights: list[float]) -> list[int]:
    """Allocate `total` items across ranks proportional to `weights`.

    Deterministic across ranks as long as `weights` are identical.

    Args:
        total: Total number of items to distribute
        weights: Per-rank capacity weights (e.g. VRAM in MB)

    Returns:
        List of per-rank allocation sizes that sum to `total`

    Example:
        >>> compute_weighted_shard_lengths(1000, [24.0, 12.0, 24.0])
        [400, 200, 400]  # 24GB GPUs get 2x the 12GB GPU
    """
    total = int(total)
    if total < 0:
        raise ValueError("total must be >= 0")
    if not weights:
        raise ValueError("weights must be non-empty")

    cleaned: list[float] = []
    for w in weights:
        wf = float(w)
        if wf < 0:
            raise ValueError("weights must be >= 0")
        cleaned.append(wf)

    wsum = sum(cleaned)
    if wsum <= 0:
        # Fallback to uniform
        cleaned = [1.0 for _ in cleaned]
        wsum = float(len(cleaned))

    raw = [total * (w / wsum) for w in cleaned]
    base = [int(x) for x in raw]
    remainder = total - sum(base)

    # Distribute remainder to largest fractional parts
    fracs = [(raw[i] - base[i], i) for i in range(len(base))]
    fracs.sort(key=lambda t: (t[0], -t[1]), reverse=True)
    for k in range(remainder):
        base[fracs[k][1]] += 1

    return base


def make_plan(total: int, weights: list[float]) -> ShardPlan:
    """Create a ShardPlan distributing `total` items by `weights`.

    Args:
        total: Total items to distribute
        weights: Per-rank capacity weights

    Returns:
        ShardPlan with lengths and offsets
    """
    lengths = compute_weighted_shard_lengths(total, weights)
    offsets = prefix_sums(lengths)
    if sum(lengths) != int(total):
        raise AssertionError("lengths do not sum to total")
    return ShardPlan(lengths=lengths, offsets=offsets)
