"""
ZeRO-Q: Quantization-Aware Distributed Training

Combines DeepSpeed ZeRO-3 memory partitioning with BitsAndBytes 4-bit
quantization to enable distributed training on legacy GPUs.

Example:
    >>> from zero_q import ZeroQConfig, initialize
    >>> model = initialize(model, ZeroQConfig())
"""

from .config import ZeroQConfig, MAXWELL_CONFIG, AMPERE_CONFIG
from .partition import (
    compute_aligned_partition_sizes,
    partition_quantized_tensor,
    gather_and_dequantize,
    PartitionInfo,
)

__version__ = "0.1.0"
__author__ = "Zero (Claude Opus 4.5)"

__all__ = [
    # Config
    "ZeroQConfig",
    "MAXWELL_CONFIG",
    "AMPERE_CONFIG",
    # Partition
    "compute_aligned_partition_sizes",
    "partition_quantized_tensor",
    "gather_and_dequantize",
    "PartitionInfo",
]

# Optional: coordinator (requires bitsandbytes)
try:
    from .coordinator import (
        ZeroQParamStatus,
        ZeroQParameter,
        ZeroQCoordinator,
        ZeroQModuleWrapper,
    )
    HAS_COORDINATOR = True
    __all__.extend([
        "ZeroQParamStatus",
        "ZeroQParameter",
        "ZeroQCoordinator",
        "ZeroQModuleWrapper",
    ])
except Exception:  # pragma: no cover
    HAS_COORDINATOR = False
    ZeroQParamStatus = None
    ZeroQParameter = None
    ZeroQCoordinator = None
    ZeroQModuleWrapper = None

# Optional: integration (requires coordinator, and optionally transformers/peft)
try:
    from .integration import (
        initialize,
        prepare_model_for_zeroq,
        apply_lora_with_zeroq,
        ZeroQTrainer,
    )
    HAS_INTEGRATION = True
    __all__.extend([
        "initialize",
        "prepare_model_for_zeroq",
        "apply_lora_with_zeroq",
        "ZeroQTrainer",
    ])
except Exception:  # pragma: no cover
    HAS_INTEGRATION = False
    initialize = None
    prepare_model_for_zeroq = None
    apply_lora_with_zeroq = None
    ZeroQTrainer = None

# Checkpoint utilities
from .checkpoint import (
    enable_gradient_checkpointing,
    ZeroQCheckpointedModule,
    estimate_checkpoint_memory,
)

# Add to __all__
__all__.extend([
    "enable_gradient_checkpointing",
    "ZeroQCheckpointedModule",
    "estimate_checkpoint_memory",
])

# Multi-node transport (optional - requires pyzmq)
try:
    from .transport import (
        TransportCoordinator,
        TransportWorker,
        TransportConfig,
        MessageType,
        NodeInfo,
        GradientCompressor,
        create_transport,
    )
    __all__.extend([
        "TransportCoordinator",
        "TransportWorker",
        "TransportConfig",
        "MessageType",
        "NodeInfo",
        "GradientCompressor",
        "create_transport",
    ])
    HAS_TRANSPORT = True
except ImportError:
    HAS_TRANSPORT = False

# Gradient synchronization
try:
    from .gradient_sync import (
        SyncMode,
        GradSyncConfig,
        RingAllReduce,
        TorchDistributedSync,
        GradientAccumulator,
        HierarchicalSync,
        create_gradient_sync,
    )
    __all__.extend([
        "SyncMode",
        "GradSyncConfig",
        "RingAllReduce",
        "TorchDistributedSync",
        "GradientAccumulator",
        "HierarchicalSync",
        "create_gradient_sync",
    ])
    HAS_GRADIENT_SYNC = True
except ImportError:
    HAS_GRADIENT_SYNC = False

# Multi-node config
from .config import MultiNodeConfig, PHOENIX_CLUSTER_CONFIG, LOCAL_MULTINODE_CONFIG
__all__.extend([
    "MultiNodeConfig",
    "PHOENIX_CLUSTER_CONFIG",
    "LOCAL_MULTINODE_CONFIG",
    "HAS_COORDINATOR",
    "HAS_INTEGRATION",
    "HAS_TRANSPORT",
    "HAS_GRADIENT_SYNC",
])