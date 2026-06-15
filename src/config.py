"""
ZeRO-Q Configuration Module.

Defines configuration dataclasses for ZeRO-Q quantized distributed training.
"""

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
    
    Example:
        >>> config = ZeroQConfig(
        ...     quant_type="nf4",
        ...     compute_dtype=torch.float32,  # For Maxwell GPUs
        ...     blocksize=64,
        ... )
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
    activation_reserve_mb: float = 0.0  # Subtract from VRAM before computing shard weights
    
    # 4-bit compute: keep weights as bnb Params4bit during forward (no fp16 dequant)
    # Requires model Linear layers to be bnb.nn.Linear4bit
    compute_in_4bit: bool = False
    
    # Training-from-scratch: shard all params as fp32 (no quantization)
    training_mode: bool = False
    compress_between_steps: bool = False
    
    # Module targeting
    target_modules: Optional[List[str]] = None
    exclude_modules: List[str] = field(default_factory=lambda: ["lm_head"])
    
    def __post_init__(self):
        """Validate configuration."""
        valid_quant_types = ("nf4", "fp4")
        if self.quant_type not in valid_quant_types:
            raise ValueError(
                f"quant_type must be one of {valid_quant_types}, got {self.quant_type}"
            )
        
        valid_blocksizes = (64, 128, 256, 512, 1024, 2048, 4096)
        if self.blocksize not in valid_blocksizes:
            raise ValueError(
                f"blocksize must be one of {valid_blocksizes}, got {self.blocksize}"
            )
        
        valid_dtypes = (torch.float32, torch.float16, torch.bfloat16)
        if self.compute_dtype not in valid_dtypes:
            raise ValueError(
                f"compute_dtype must be one of {valid_dtypes}, got {self.compute_dtype}"
            )
    
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
        d = d.copy()  # Don't modify original
        
        # Handle compute_dtype string conversion
        if isinstance(d.get("compute_dtype"), str):
            dtype_map = {
                "torch.float32": torch.float32,
                "torch.float16": torch.float16,
                "torch.bfloat16": torch.bfloat16,
            }
            d["compute_dtype"] = dtype_map.get(d["compute_dtype"], torch.float32)
        
        return cls(**d)
    
    def __repr__(self) -> str:
        """Human-readable representation."""
        return (
            f"ZeroQConfig(\n"
            f"  quant_type={self.quant_type!r},\n"
            f"  blocksize={self.blocksize},\n"
            f"  compute_dtype={self.compute_dtype},\n"
            f"  frozen_only={self.frozen_only},\n"
            f"  async_gather={self.async_gather},\n"
            f")"
        )


# Pre-defined configurations for common hardware

# Configuration optimized for Maxwell GPUs (SM 5.2, e.g., Tesla M40)
MAXWELL_CONFIG = ZeroQConfig(
    compute_dtype=torch.float32,  # Required - no FP16 hardware
    double_quant=True,            # Save memory where possible
    blocksize=64,                 # Standard blocksize
    async_gather=True,            # Overlap communication
)

# Configuration optimized for Ampere GPUs (SM 8.0+, e.g., A100)
AMPERE_CONFIG = ZeroQConfig(
    compute_dtype=torch.bfloat16,  # BF16 supported
    double_quant=True,
    blocksize=64,
    async_gather=True,
)

# Configuration for consumer GPUs (RTX 30xx, 40xx)
CONSUMER_CONFIG = ZeroQConfig(
    compute_dtype=torch.float16,  # FP16 supported on Turing+
    double_quant=True,
    blocksize=64,
    async_gather=True,
)


@dataclass
class ZeroQTrainConfig(ZeroQConfig):
    """
    Configuration for training from scratch with ZeRO-Q fp32 sharding.

    All parameters are sharded across GPUs as fp32 master weights (no
    quantization).  Gradients are reduce-scattered so each rank updates
    only its local shard.

    Set ``compress_between_steps=True`` to NF4-compress master shards
    between optimizer steps (saves ~3.5x weight memory, adds quant noise).
    """

    training_mode: bool = True
    frozen_only: bool = False
    partition_trainable: bool = True
    cpu_offload: bool = False
    optimizer_cls: str = "AdamW"
    optimizer_kwargs: dict = field(default_factory=lambda: {"lr": 3e-4})

    def __post_init__(self):
        if self.compress_between_steps:
            super().__post_init__()


TRAIN_FROM_SCRATCH_CONFIG = ZeroQTrainConfig(
    compute_dtype=torch.float32,
    training_mode=True,
    frozen_only=False,
    partition_trainable=True,
)


@dataclass
class MultiNodeConfig:
    """
    Configuration for multi-node distributed training.
    
    Attributes:
        enabled: Whether multi-node mode is enabled
        coordinator_host: Hostname of the coordinator node
        coordinator_port: Port for coordinator communication
        node_id: Unique identifier for this node
        local_hostname: This node's hostname for peer connections
        
        heartbeat_interval_sec: Interval between heartbeats
        heartbeat_timeout_sec: Timeout before node is considered dead
        
        use_hierarchical_sync: Use hierarchical all-reduce
        gradient_compression: Enable gradient compression
        compression_ratio: Top-K compression ratio (0.01 = 1%)
        
        data_port_start: Starting port for peer data connections
        tcp_buffer_size: TCP buffer size in bytes
    """
    
    enabled: bool = False
    coordinator_host: str = "localhost"
    coordinator_port: int = 5555
    node_id: str = ""
    local_hostname: str = "localhost"
    
    heartbeat_interval_sec: float = 5.0
    heartbeat_timeout_sec: float = 15.0
    
    use_hierarchical_sync: bool = True
    gradient_compression: bool = True
    compression_ratio: float = 0.01
    
    data_port_start: int = 5560
    tcp_buffer_size: int = 4 * 1024 * 1024  # 4MB
    
    def __post_init__(self):
        """Generate node_id if not provided."""
        if not self.node_id:
            import socket
            import os
            self.node_id = f"{socket.gethostname()}-{os.getpid()}"
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "enabled": self.enabled,
            "coordinator_host": self.coordinator_host,
            "coordinator_port": self.coordinator_port,
            "node_id": self.node_id,
            "local_hostname": self.local_hostname,
            "heartbeat_interval_sec": self.heartbeat_interval_sec,
            "heartbeat_timeout_sec": self.heartbeat_timeout_sec,
            "use_hierarchical_sync": self.use_hierarchical_sync,
            "gradient_compression": self.gradient_compression,
            "compression_ratio": self.compression_ratio,
            "data_port_start": self.data_port_start,
            "tcp_buffer_size": self.tcp_buffer_size,
        }
    
    @classmethod
    def from_dict(cls, d: dict) -> "MultiNodeConfig":
        """Create from dictionary."""
        return cls(**d)


# Pre-defined multi-node configurations

# Configuration for PE1 + PE2 cluster (Phoenix lab)
PHOENIX_CLUSTER_CONFIG = MultiNodeConfig(
    enabled=True,
    coordinator_host="pe1",
    coordinator_port=5555,
    use_hierarchical_sync=True,
    gradient_compression=True,
    compression_ratio=0.01,
)

# Configuration for local testing (single machine)
LOCAL_MULTINODE_CONFIG = MultiNodeConfig(
    enabled=True,
    coordinator_host="localhost",
    coordinator_port=5555,
    use_hierarchical_sync=False,
    gradient_compression=False,
)
