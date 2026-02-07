"""
ZeRO-Q Heterogeneous GPU Support

Enables weighted partitioning across GPUs of different VRAM sizes.
Larger GPUs hold proportionally more weight shards.

Ported from Mnemosyne hetero-train codebase.
"""

from .shard_plan import ShardPlan, make_plan, compute_weighted_shard_lengths
from .zeroq_hetero import (
    HeteroZeroQParameter,
    HeteroZeroQCoordinator,
    HeteroZeroQModuleWrapper,
    HeteroZeroQParamStatus,
    discover_rank_weights,
)
from .varlen_collectives import all_gather_varlen_1d

__all__ = [
    "ShardPlan",
    "make_plan",
    "compute_weighted_shard_lengths",
    "HeteroZeroQParameter",
    "HeteroZeroQCoordinator",
    "HeteroZeroQModuleWrapper",
    "HeteroZeroQParamStatus",
    "discover_rank_weights",
    "all_gather_varlen_1d",
]
