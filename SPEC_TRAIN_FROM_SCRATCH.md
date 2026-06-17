# SPEC: ZeroQ Training From Scratch

> 2026-06-15. Feature spec for training transformer models from random init
> using ZeroQ's existing sharding infrastructure. Written by DeepSeek v4 Pro.

## 1. Objective

Enable training transformer models **from scratch** (random init, all params
trainable) using ZeroQ's existing sharding infrastructure. The current ZeroQ
freezes the base model at 4-bit and only trains LoRA/surface params. This
extension shards **all** parameters across GPUs with fp32 master weights,
enabling 5B+ models on 5x M40 24GB with DDP-level throughput.

## 2. What Already Exists (Do Not Rebuild)

| Component | File | Status |
|-----------|------|--------|
| `repartition_tensor()` | `src/partition.py:356-407` | Coded, never called in hot path |
| `ZeroQParameter` gather/release lifecycle | `src/coordinator.py` | Working for frozen params |
| `ZeroQModuleWrapper` forward/backward hooks | `src/coordinator.py` | Working |
| `RingAllReduce`, `HierarchicalSync` | `src/gradient_sync.py` | Working |
| `ZeroMQ multi-node transport` | `src/transport.py` | Working |
| `HeteroZeroQCoordinator` | `src/hetero/` | Working (mixed VRAM) |
| `enable_gradient_checkpointing()` | `src/checkpoint.py` | Working |
| Steerer hook compatibility | Proven in production | DDP v5 running |

## 3. Architecture

### 3.1 Parameter Lifecycle (Training From Scratch)

```
         +---------------------------------------------+
         |           GPU 0 (shard 0/5)                 |
         |                                             |
STORAGE  |  fp32 master shard: P*4/N bytes             |
         |  AdamW exp_avg shard: P*4/N bytes           |
         |  AdamW exp_avg_sq shard: P*4/N bytes        |
         |                                             |
         +----------------------+-----------------------+
                                |
    +------ PRE-FORWARD --------+
    |                           |
    |  all_gather(shards)       |  <-- NCCL, collects fp32 shards from all GPUs
    |  full_param = cat()       |  <-- temporary: lives only during fwd+bwd
    |  module.weight = full     |
    |                           |
    +------ FORWARD ------------+
    |                           |
    |  output = layer(x)        |  <-- standard forward, requires_grad=True
    |                           |
    +------ BACKWARD -----------+
    |                           |
    |  loss.backward()          |  <-- autograd computes grad w.r.t. full_param
    |  reduce_scatter(grad)     |  <-- each GPU gets 1/N of the gradient
    |  local_grad = shard       |
    |                           |
    +------ OPTIMIZER ----------+
    |                           |
    |  adam.step(local_grad)    |  <-- update local fp32 master shard only
    |  zero_grad()              |
    |                           |
    +------ RELEASE ------------+
                                |
       module.weight = empty    |  <-- free the gathered full param
       (master shard stays)     |
```

### 3.2 Optional 4-bit Inter-Step Compression

Between steps, the fp32 master shard can be compressed to 4-bit NF4 for
storage, freeing 3.5x memory. On the next forward, decompress before
all-gather. This is **optional** -- fp32 sharding alone fits 5.25B on
5x M40 24GB. Enable for 7B+ or when VRAM is tight.

```
After optimizer step:
  compressed = quantize_nf4(master_shard)    # 4x smaller
  del master_shard                            # free fp32

Before next forward:
  master_shard = dequantize_nf4(compressed)  # restore fp32
  all_gather(master_shards)                   # gather across GPUs
```

### 3.3 Memory Budget (5.25B on 5x M40 24GB)

| Component | Per GPU (fp32 only) | Per GPU (+ 4-bit compress) |
|-----------|--------------------|-----------------------------|
| Master weight shard | 4.2 GB | 1.05 GB (4-bit) + 4.2 GB (restore) |
| Optimizer exp_avg | 4.2 GB | 4.2 GB |
| Optimizer exp_avg_sq | 4.2 GB | 4.2 GB |
| Temp gathered layer | ~0.35 GB | ~0.35 GB |
| Gradients (local shard) | 4.2 GB | 4.2 GB |
| Activations (grad ckpt) | ~0.5 GB | ~0.5 GB |
| **Total** | **~17.7 GB** | **~14.5 GB** |
| **Headroom (of 24 GB)** | **6.3 GB** | **9.5 GB** |

## 4. Files to Modify

### 4.1 `src/coordinator.py` -- Core Changes

**Change 1**: Make `frozen_only` configurable with a new `training_mode` flag.

```python
class ZeroQCoordinator:
    def __init__(self, ..., training_mode=False):
        self.training_mode = training_mode
        # When training_mode=True:
        #   - ALL params registered (not just frozen)
        #   - requires_grad=True preserved
        #   - release() keeps master shard instead of empty(0)
```

**Change 2**: `ZeroQParameter` -- add fp32 master shard storage.

```python
class ZeroQParameter:
    def __init__(self, ...):
        self.master_shard = None  # fp32 shard for training mode

    def gather(self):
        if self.coordinator.training_mode:
            # All-gather fp32 master shards -> full param
            all_shards = all_gather(self.master_shard)
            full_param = torch.cat(all_shards)
            return full_param.view(self.original_shape).requires_grad_(True)
        else:
            # Existing frozen path: dequantize from 4-bit
            ...

    def release(self):
        if self.coordinator.training_mode:
            # DON'T discard -- keep master shard
            # Just free the gathered full param
            self.param.data = torch.empty(0, device=self.device)
        else:
            # Existing frozen path
            ...
```

**Change 3**: `ZeroQModuleWrapper._register_parameters` -- remove frozen_only gate.

```python
def _register_parameters(self):
    for name, param in self.module.named_parameters():
        if self.coordinator.config.frozen_only and param.requires_grad:
            continue  # EXISTING: skip trainable in frozen mode
        # In training_mode, register ALL params
        if self.coordinator.training_mode and param.requires_grad:
            # Partition into fp32 shards (no quantization)
            self._register_trainable_param(name, param)
        else:
            # Existing 4-bit path for frozen params
            self._register_frozen_param(name, param)
```

### 4.2 `src/partition.py` -- Add fp32 Sharding

Add `partition_fp32()` alongside existing `partition_quantized_tensor()`:

```python
def partition_fp32(tensor: torch.Tensor, world_size: int, rank: int):
    """Shard an fp32 tensor evenly across ranks. No quantization."""
    flat = tensor.contiguous().view(-1)
    chunk_size = (flat.numel() + world_size - 1) // world_size
    start = rank * chunk_size
    end = min(start + chunk_size, flat.numel())
    return flat[start:end].clone(), tensor.shape
```

Add `gather_fp32()`:

```python
def gather_fp32(local_shard, original_shape, world_size):
    """All-gather fp32 shards -> reconstruct full tensor."""
    gathered = [torch.empty_like(local_shard) for _ in range(world_size)]
    dist.all_gather(gathered, local_shard)
    return torch.cat(gathered)[:original_shape.numel()].view(original_shape)
```

### 4.3 `src/gradient_sync.py` -- Add Reduce-Scatter

Add gradient reduce-scatter for sharded training:

```python
def reduce_scatter_grads(param_full_grad, world_size, rank):
    """Reduce-scatter: each GPU gets its shard of the averaged gradient."""
    flat = param_full_grad.contiguous().view(-1)
    chunk_size = (flat.numel() + world_size - 1) // world_size

    # Pad if needed
    if flat.numel() % world_size != 0:
        flat = F.pad(flat, (0, chunk_size * world_size - flat.numel()))

    output = torch.empty(chunk_size, device=flat.device, dtype=flat.dtype)
    dist.reduce_scatter(output, list(flat.chunk(world_size)))
    output /= world_size  # average

    return output
```

### 4.4 `src/config.py` -- Add Training Mode Config

```python
@dataclass
class ZeroQTrainConfig(ZeroQConfig):
    training_mode: bool = True
    frozen_only: bool = False
    compress_between_steps: bool = False  # optional 4-bit inter-step
    optimizer_cls: str = 'AdamW'
    optimizer_kwargs: dict = field(default_factory=lambda: {'lr': 3e-4})
```

### 4.5 Integration: `steered_trainer.py` -- New `--backend zeroq-train` Mode

Add a third backend choice alongside `dense` and `zeroq`:

```python
parser.add_argument('--backend', type=str, default='dense',
                    choices=['dense', 'zeroq', 'zeroq-train'],
                    ...)
```

The `zeroq-train` backend:
1. Initializes NCCL process group (like DDP)
2. Creates model on CPU
3. Wraps with `ZeroQCoordinator(training_mode=True)`
4. Shards all params into fp32 shards across GPUs
5. Creates per-shard AdamW optimizer (only holds local shard's state)
6. Training loop: gather -> forward -> backward -> reduce-scatter -> optimizer step -> release
7. Steerer stays outside ZeroQ (same as steerer stays outside DDP) -- manually allreduce its gradients

## 5. API (User-Facing)

### Launch Command

```bash
python steered_trainer.py \
  --variant baseline --d_model 4096 --n_layers 32 --d_ff 11008 --n_heads 32 \
  --gpu_ids 0,1,2,3,4 --backend zeroq-train \
  --batch 1 --seq_len 256 --gradient-checkpointing \
  --lr 3e-4 --steer-lr 1e-3 --lr-warmup 200 --lr-decay --steps 3000000 \
  --data_dir /mnt/models/c4_tokenized --ckpt_dir ~/steered_5b_proof \
  --use_rope --window 64 --force-best
```

### Checkpoint Format

```python
{
    'model_state': { ... },           # full fp32 state_dict (gathered from shards)
    'optimizer_shards': { ... },      # per-rank optimizer state (saved by rank 0 after gather)
    'steerer_state': { ... },
    'step': N,
    'config': { ..., 'backend': 'zeroq-train' },
}
```

## 6. Testing Strategy

### Unit Tests

| Test | What It Verifies |
|------|-----------------|
| `test_partition_fp32` | Shard -> gather round-trip is bit-exact |
| `test_reduce_scatter_grads` | Gradient sharding produces correct local grads |
| `test_trainable_param_lifecycle` | gather -> forward -> backward -> optimizer -> release cycle |
| `test_checkpoint_save_load` | Gathered state_dict matches pre-sharding model |

### Integration Tests

| Test | What It Verifies |
|------|-----------------|
| `test_2gpu_training_from_scratch` | 2x GPU, 10-step training, loss decreases |
| `test_steerer_hooks_with_zeroq_train` | Steerer hooks fire correctly with gathered params |
| `test_gradient_checkpointing_compat` | Grad ckpt + ZeroQ train mode -- no OOM, correct grads |
| `test_resume_from_checkpoint` | Save at step 5, resume at step 6, loss continues decreasing |

### Smoke Test on M40

```bash
# Tiny model, 2 GPUs, 10 steps -- verify the full pipeline
python steered_trainer.py \
  --variant baseline --d_model 256 --n_layers 4 --d_ff 1024 --n_heads 4 \
  --gpu_ids 0,1 --backend zeroq-train \
  --batch 2 --seq_len 64 --steps 10 --gradient-checkpointing \
  --dataset wikitext --ckpt_dir /tmp/zeroq_train_smoke
```

## 7. Implementation Order

| Step | What | Est. Time | Dependencies |
|------|------|-----------|-------------|
| 1 | `partition_fp32()` + `gather_fp32()` in partition.py | 15 min | None |
| 2 | `reduce_scatter_grads()` in gradient_sync.py | 15 min | None |
| 3 | `ZeroQParameter` training mode in coordinator.py | 20 min | Steps 1-2 |
| 4 | `ZeroQModuleWrapper` trainable registration | 10 min | Step 3 |
| 5 | `ZeroQTrainConfig` in config.py | 5 min | None |
| 6 | Unit tests for steps 1-4 | 15 min | Steps 1-4 |
| 7 | `--backend zeroq-train` in steered_trainer.py | 20 min | Steps 1-5 |
| 8 | Integration test on 2x GPU | 15 min | Step 7 |
| 9 | Smoke test on M40 hardware | 10 min | Step 8 |
| **Total** | | **~2 hours** | |

## 8. Risks and Mitigations

| Risk | Mitigation |
|------|------------|
| NCCL reduce_scatter unsupported on M40 | Fallback: all_reduce + local slice (slower but works) |
| Memory spike during gather (full layer temporarily materialized) | Per-layer gather/release (already the ZeroQ pattern) |
| Steerer hooks + gathered params device mismatch | Steerer stays outside ZeroQ, hooks on the model directly (same as DDP pattern) |
| Gradient accumulation needed for large effective batch | Standard: accumulate N micro-batches, then reduce_scatter once |
| Checkpoint too large (all shards gathered to rank 0) | Stream-save: rank 0 gathers one param group at a time |
| 4-bit inter-step compression adds quantization noise | OFF by default. Only enable with `--compress-between-steps` when VRAM-constrained |
| NCCL device collision on concurrent single-GPU runs | `_ensure_single_rank_pg` now calls `torch.cuda.set_device(device)` before NCCL init (§8.3) |
| OOM from allocator fragmentation on M40 (12GB) | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (§8.4) |

### 8.1 Fused Per-Layer Lion Optimizer

**Added 2026-06-15. Gate 0 (bnb Lion8bit on Maxwell SM 5.2) and Gate 1 (loss 1.04→0.80) verified PASS.**

The `--fused-lion` flag replaces AdamW with a per-layer fused Lion optimizer:

```python
config = ZeroQTrainConfig(
    master_dtype=torch.float16,      # fp16 master shards
    fused_shard_step=True,            # inline Lion step in _post_backward_hook
    lion_lr=args.lr,                  # Lion learning rate
    ...
)
```

Key behaviors:
- **fp16 master shards** with stochastic rounding (SR) — halves optimizer memory vs fp32
- **Inline step**: Lion update happens immediately after gradient reduce_scatter in the `_post_backward_hook`, before the layer is released — eliminates a separate optimizer loop
- **Excluded params** (tok_emb, pos_emb, norms, ln_f) stay fp32 with external AdamW optimizer — they're tiny and weight-tied, so the memory cost is negligible

Lion requires only 1 momentum state (vs Adam's 2), halving optimizer VRAM when combined with fp16 masters. Tested on M40 SM 5.2 with bitsandbytes 0.41.3 Lion8bit — passes Gate 0 (runs at all) and Gate 1 (loss 1.04→0.80 in first few steps).

### 8.2 Single-GPU Mode with CPU Offload

**Added 2026-06-15. Enables 5.25B training on a single RTX 3080 10GB.**

The `prepare_zeroq_train` function auto-detects single-GPU vs multi-GPU:

```python
cpu_offload = not _dist.is_initialized() or _dist.get_world_size() == 1
```

On single GPU:
- **4-bit weights cached on GPU** for forward passes
- **fp32 master weights + optimizer state on CPU** — avoids blowing VRAM
- **Per-layer gather/release** keeps only one layer materialised at a time
- **Fused Lion disabled**: master shards are on CPU, so inline GPU Lion step doesn't apply

On multi-GPU:
- **fp32 master shards on GPU** (ZeRO-3 partitioning)
- **No CPU offload**: optimizer state lives on GPU
- **Fused Lion enabled**: inline updates in `_post_backward_hook`

### 8.3 Single-GPU NCCL Device Pinning

**Added 2026-06-17. Prevents `cuda:0` / `cuda:1` device mismatch on concurrent single-GPU runs.**

When running two independent ZeroQ-train processes on separate GPUs (e.g., GPU 0 and GPU 1 concurrently), `_ensure_single_rank_pg` creates a single-rank NCCL process group. Without explicit device pinning, NCCL defaults to `cuda:0`, placing tensors on the wrong device.

Fix in `steered_trainer.py:_ensure_single_rank_pg`:

```python
if device.type == 'cuda':
    torch.cuda.set_device(device)   # Pin NCCL to the target GPU
```

Multi-GPU DDP is unaffected — `init_ddp` already sets the device, and `_ensure_single_rank_pg` exits early when `dist.is_initialized()` is True.

### 8.4 Memory Fragmentation Mitigation (M40)

**Added 2026-06-17. Prevents OOM from CUDA allocator fragmentation at batch=3/4 on 12GB M40.**

The M40's CUDA allocator fragments memory under repeated per-layer gather/release cycles. Setting the environment variable:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Allows the allocator to defragment by expanding reserved segments, avoiding "Tried to allocate 18MiB but 756MiB reserved" OOM failures. All pe3 launch scripts include this env var.

### 8.5 Project Self-Containment

**Added 2026-06-17. The `backends.py` module (TrainableSurface, ZeroQPartitionedBackend) previously lived in `~/deepseek_experiments/hybrid/` — an external dependency. Copied into `~/compiled_priors/backends.py` so the project is fully self-contained. No external repo dependencies.**

### 8.6 Compiled Priors CLI Flag

**Added 2026-06-17. `--priors-dir` replaces the hardcoded module-level `PRIOR_DIR` constant. Defaults to `compiled_priors_v3/` in the project directory. Priors are now project-local — no symlinks, no external drives, no silent failures.**

```bash
python steered_trainer.py ... --priors-dir ~/my_priors
```

### 8.7 Inject Layer Override

**Added 2026-06-17. `--inject-layers` overrides the auto-computed `build_inject_layers()` for explicit control. Pinned-layers in `layer_split` mode now respect the override, keeping all inject layers on GPU 0 to prevent device mismatches with the steerer.**

```bash
python steered_trainer.py ... --inject-layers "5,10,20,24,28,32,36,46,53"
```

## 9. Non-Goals (Explicitly Out of Scope)

- **STE backward through quantization** -- we use fp32 master weights, not QAT
- **Modifying existing frozen-backbone ZeroQ path** -- `training_mode=False` preserves all existing behavior
- **Multi-node for training mode** -- single-node NCCL first, multi-node later
- **Tensor parallelism** -- parameter sharding only, no intra-layer splitting
- **Mixed precision** -- fp32 only on M40 (Rule 33). Ampere+ users can add AMP later

## 10. Success Criteria

1. `steered_trainer.py --backend zeroq-train` trains a 5.25B model on 5x M40 24GB
2. Memory per GPU < 18 GB (with gradient checkpointing)
3. Throughput within 70% of equivalent DDP (communication overhead < 30%)
4. Loss curve matches `--backend dense --multi_gpu_mode layer_split` within 5% at step 1000
5. Steerer hooks produce identical modulation as DDP mode
6. Checkpoint save/load round-trips correctly
