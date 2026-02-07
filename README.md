# ZeRO-Q: Quantization-Aware Distributed Training

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Douglas Rawson & Zero (Claude)
**Status:** ✅ Production — Multi-node heterogeneous training validated
**Hardware:** Tesla M40 cluster (mixed 24GB/12GB) over TCP/NCCL

---

## What It Does

ZeRO-Q combines **ZeRO-3 parameter partitioning** with **4-bit NF4 quantization** to train large models on GPUs that individually can't hold them.

The **hetero** module extends this to **mixed-VRAM clusters** — a 24GB GPU takes 2x the shard of a 12GB GPU, automatically.

```
Standard ZeRO-3:           ZeRO-Q:
├─ Shard FP16 weights      ├─ Quantize to 4-bit NF4
├─ All-gather FP16         ├─ Shard packed uint8 + absmax (weighted by VRAM)
├─ 2 bytes/param comm      ├─ All-gather 4-bit (~0.5 bytes/param)
└─ Compute in FP16         └─ Dequantize locally → compute in FP16
```

**Result:** Train a 14GB fp16 model across GPUs that individually have 12GB, over regular Ethernet.

---

## Proven Results

**Qwen2.5-7B-Instruct** with LoRA on 3x Tesla M40 (1×24GB + 2×12GB), two physical servers:

| Metric | Value |
|--------|-------|
| Model size (fp16) | ~14 GB |
| Per-GPU memory | 1.1–2.1 GB (quantized shards) |
| GPU utilization | 98-100% compute |
| NCCL throughput | 300 MiB/s (TCP, Gen2 PCIe) |
| Shard distribution | Weighted proportional to VRAM |

---

## Architecture

```
src/
├── __init__.py            # Public API
├── config.py              # ZeroQConfig, MAXWELL_CONFIG preset
├── coordinator.py         # Uniform ZeRO-Q coordinator (equal shards)
├── partition.py           # Quantized tensor partitioning with aligned splits
├── prefetch.py            # Async parameter prefetching
├── gradient_sync.py       # Distributed gradient synchronization
├── transport.py           # ZeroMQ multi-node transport layer
├── checkpoint.py          # Gradient checkpointing utilities
├── integration.py         # HuggingFace/PEFT integration
└── hetero/                # Heterogeneous GPU support
    ├── __init__.py
    ├── zeroq_hetero.py    # HeteroZeroQCoordinator, HeteroZeroQParameter,
    │                      #   HeteroZeroQModuleWrapper, discover_rank_weights()
    ├── shard_plan.py      # Weighted shard length computation
    └── varlen_collectives.py  # Variable-length all-gather for uneven shards
```

### Core Concepts

**Gather/Release Lifecycle:** Pre-forward hook gathers (dequantizes full param from all shards). Post-forward hook releases (drops back to local shard). Same for backward. At any moment, only ~1 layer's worth of full-precision params is materialized.

**Hetero Weighted Partitioning:** `discover_rank_weights()` auto-detects each GPU's VRAM via `torch.cuda.get_device_properties()`, then `make_plan()` computes shard lengths proportional to capacity. A 24GB card holds 2x the shard of a 12GB card.

**Safety:**
- **Registration consistency verification** — after partition, each rank hashes its `param_id → shape` mapping and all-gathers to verify all ranks agree. Catches ordering bugs before they become silent NCCL deadlocks.
- **Collective sequence counter** — monotonic counter tracks collective operations per rank. `debug_collective_seq()` compares across ranks to pinpoint divergence.

---

## Quick Start (Hetero)

```python
import torch
from src.hetero.zeroq_hetero import (
    HeteroZeroQCoordinator,
    HeteroZeroQModuleWrapper,
    discover_rank_weights,
)
from src.config import ZeroQConfig

# Configure
config = ZeroQConfig(
    compute_dtype=torch.float16,
    double_quant=True,
    blocksize=64,
    async_gather=True,
    prefetch_count=0,
)

# Create coordinator (auto-discovers GPU VRAM weights)
coordinator = HeteroZeroQCoordinator(config)

# Wrap model — registers frozen params, installs gather/release hooks
wrapper = HeteroZeroQModuleWrapper(model, coordinator, trainable_only=False)

# Stream checkpoint and partition across ranks
partition_from_checkpoint(coordinator, model, model_dir, device, rank)

# Verify all ranks agree on registration order
coordinator.verify_registration_consistency()

# Train with standard loop — hooks handle gather/release automatically
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()
    optimizer.step()
```

---

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `ZEROQ_HETERO_RANK_WEIGHTS` | Override auto-detected VRAM weights | `24576,11520,11520` |
| `NCCL_SOCKET_IFNAME` | Network interface for NCCL | `eno` |
| `NCCL_BUFFSIZE` | NCCL buffer size (bytes) | `2097152` |

---

## Requirements

```
torch >= 2.0
bitsandbytes == 0.43.2    # Critical: last version with Maxwell (SM 5.2) support
transformers
peft
safetensors
```

> **Note:** bitsandbytes 0.48+ dropped Maxwell GPU support. Use 0.43.2 for Tesla M40 / GTX 9xx / GTX 10xx.

---

## Hardware Tested

| GPU | Memory | SM | PCIe | Status |
|-----|--------|----|------|--------|
| Tesla M40 | 24 GB | 5.2 | Gen3 x16 | ✅ Primary target |
| Tesla M40 | 12 GB | 5.2 | Gen3 x16 | ✅ Hetero secondary |

Multi-node validated over standard 1GbE TCP with NCCL backend.

---

## License

MIT

---

## Acknowledgments

- **DeepSpeed** — ZeRO architecture inspiration
- **bitsandbytes** — 4-bit NF4 quantization primitives
- **Mnemosyne** — Original codebase where hetero module was developed
