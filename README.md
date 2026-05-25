# ZeRO-Q: Quantization-Aware Distributed Training

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Author:** Douglas Rawson
**Status:** ✅ Production — Custom 4.7B decoder transformer at 88% GPU util, 8.3× faster with 4-bit compute
**Hardware:** Tesla M40 cluster (mixed 24GB/12GB) over TCP/NCCL, now integrated with compiled-hybrid-lm

---

## What It Does

ZeRO-Q combines **ZeRO-3 parameter partitioning** with **4-bit NF4 quantization** to train large models on GPUs that individually can't hold them.

The **hetero** module extends this to **mixed-VRAM clusters** — a 24GB GPU takes 2x the shard of a 12GB GPU, automatically. The **4-bit compute** mode eliminates per-layer gather/release entirely, converting `nn.Linear` to `bnb.nn.Linear4bit` for fused `matmul_4bit` — 8× faster on SYS-topology GPUs.

```
Standard ZeRO-3:           ZeRO-Q (gather/release):    ZeRO-Q (4-bit compute):
├─ Shard FP16 weights      ├─ Quantize to 4-bit NF4    ├─ Quantize to 4-bit NF4
├─ All-gather FP16         ├─ All-gather per layer     ├─ Gather full weights ONCE
├─ 2 bytes/param comm      ├─ Dequantize → compute     ├─ Convert to bnb.Linear4bit
└─ Compute in FP16         └─ Release after forward    └─ Native matmul_4bit, no hooks
```

**Result:** Train a 4.7B-param custom transformer at 88% GPU util on 2× Tesla M40, or a 14GB fp16 model across GPUs that individually have 12GB, over regular Ethernet.

---

## Proven Results

### Custom 4.7B Decoder Transformer (compiled-hybrid-lm)

**DeepSeekForCausalLM** (d=3072, 40 layers, explicit Q/K/V/O/FFN, GPT-2 BPE) on 2× Tesla M40 24GB:

| Metric | Before (gather/release) | After (4-bit compute) |
|--------|-------------------------|-----------------------|
| Epoch time (50 steps) | 1,118s | **135s** |
| Training throughput | 2.86 tok/s | **23.7 tok/s** |
| GPU utilization | 13% | **88%** |
| Peak VRAM per GPU | 17.7 GB (cycling) | **5.4 GB** (steady) |
| Speedup | — | **8.3×** |

With compiled priors (21-channel n-gram, topic, KV-cache, POS) + SuperpositionSteererV3 (65K params, 9 hooks).

### Qwen2.5-7B-Instruct (LoRA, Hetero)

On 3× Tesla M40 (1×24GB + 2×12GB), two physical servers:

| Metric | Value |
|--------|-------|
| Model size (fp16) | ~14 GB |
| Per-GPU memory | 1.1–2.1 GB (quantized shards) |
| GPU utilization | 98-100% compute |
| NCCL throughput | 300 MiB/s (TCP, Gen2 PCIe) |
| Shard distribution | Weighted proportional to VRAM |

### Projected Cluster Capacity (5× M40 24GB)

| Model Size | 4-bit Weight/GPU | Feasibility |
|------------|------------------|-------------|
| 4.7B | 1.2 GB | ✅ Running |
| 10B | 2.5 GB | ✅ Comfortable |
| 20B | 5.0 GB | ✅ Fit |
| 30B | 7.5 GB | ✅ With checkpointing |
| 35B | 8.8 GB | ✅ Tight, activation-bound |

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

**4-bit Compute Mode:** After partition, gathers full 4-bit weights once per `nn.Linear` layer, converts to `bnb.nn.Linear4bit` with `Params4bit`, then removes ZeroQ hooks entirely. Linear4bit uses bitsandbytes' fused `matmul_4bit` kernel — no per-step gather/release, no dequantization overhead. Eliminates NCCL all-gather bottleneck on SYS-topology GPUs (8.3× speedup observed).

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

### ZeroQConfig Options

| Field | Default | Description |
|-------|---------|-------------|
| `quant_type` | `"nf4"` | Quantization type (`nf4` or `fp4`) |
| `blocksize` | `64` | Elements per quantization block |
| `double_quant` | `True` | Double quantization for absmax |
| `compute_dtype` | `float32` | Compute dtype (`float32` for Maxwell) |
| `async_gather` | `True` | Async all-gather operations |
| `prefetch_count` | `1` | Layers to prefetch ahead |
| `activation_reserve_mb` | `0.0` | MB to subtract from VRAM before shard weighting |
| `compute_in_4bit` | `False` | Keep weights as Params4bit (no fp16 dequant) |

---

## Requirements

```
torch >= 2.0
bitsandbytes == 0.41.3    # CRITICAL: last version with Maxwell (SM 5.2) support. MUST PIN.
triton == 3.3.1            # Required by bitsandbytes 0.41.3 nn module imports
transformers
peft
safetensors
```

> **Note:** bitsandbytes 0.46.1+ dropped Maxwell GPU support entirely. Use 0.41.3 EXACTLY for Tesla M40 / GTX 9xx / GTX 10xx. Version must be pinned — any other version will fail to import on Maxwell hardware.

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
