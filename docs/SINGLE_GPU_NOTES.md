# Single-GPU Notes (read before adding a "single-GPU mode")

_Last updated: 2026-05-29 (Opus 4.8), from the Qwen3.5-4B cartridge-training work on a 10GB RTX 3080._

## TL;DR

**On a single GPU, do not use ZeroQ — use `transformers.BitsAndBytesConfig` 4-bit directly.**
ZeroQ adds nothing on one GPU, and a "single-GPU ZeroQ mode" would be pure theater.

## Why ZeroQ has no single-GPU benefit

ZeroQ's value is **multi-GPU**:

- **ZeRO-3 packed-weight partitioning** — each rank holds a shard of the packed NF4 weights and
  gathers/reconstructs on demand (`src/hetero/zeroq_hetero.py`). This only matters when no single
  GPU can hold the model.
- **Maxwell (sm_52) support** — for the M40 cluster, where modern `bitsandbytes` (>= 0.46) dropped
  Maxwell kernels.
- **`compute_in_4bit`** — after gather, it rebuilds a `bnb.nn.Params4bit` so `Linear4bit.forward()`
  uses the fused `matmul_4bit` kernel instead of materializing an fp16 weight copy
  (`zeroq_hetero.py` ~L478-500).

On a **single Ampere+ GPU (e.g. RTX 3080, sm_86)** all three collapse:

- There is no gather/release — the whole model already lives on the one GPU.
- `bitsandbytes` Ampere kernels work fine; no Maxwell workaround needed.
- `BitsAndBytesConfig(load_in_4bit=True, ...)` **already** produces `bnb.nn.Linear4bit` layers whose
  `forward()` calls the fused `matmul_4bit`. That is exactly what `compute_in_4bit` reconstructs.
  So ZeroQ's fused-4-bit path and the stock bnb path are **the same kernel** — no memory or speed delta.

Measured: Qwen3.5-4B loads at **3.12 GB** in 4-bit NF4 via plain `BitsAndBytesConfig`
(`nf4`, `bnb_4bit_compute_dtype=float16`, `bnb_4bit_use_double_quant=True`). The 8.3x speedup /
"5.4 GB steady vs 17.7 GB cycling" in the README is the multi-GPU gather/release win — it does not
apply when there is nothing to gather.

### If a single-GPU entry point is ever still wanted

Keep it a thin, non-distributed helper that just:
1. loads the HF model with `BitsAndBytesConfig` 4-bit,
2. asserts the `Linear` layers are `bnb.nn.Linear4bit` (so `matmul_4bit` is in use),
3. returns the model.

It must **not** touch the coordinator / hetero / `dist.*` paths (those require
`dist.is_initialized()` and a rank topology). Treat it as a convenience wrapper around bnb, not a
new execution mode, and document that it carries no perf/memory benefit over stock bnb.

## The bnb 4-bit gotcha that actually bit us: gradient-checkpointing no-op

When fine-tuning a **frozen** 4-bit base with a small trainable adapter (QLoRA pattern), the OOM was
**not** a ZeroQ/bnb memory problem — it was gradient checkpointing silently doing nothing.

### Symptom

Training OOM'd on the 3080 at sequence length > 256, and a memory probe showed
**GC-on and GC-off produced identical peaks** (8.20 GB @ seq256, OOM beyond). Identical peaks with and
without checkpointing is the diagnostic signature of a checkpointing no-op.

### Root cause

`model.gradient_checkpointing_enable()` only sets a flag. Each decoder layer gates the actual
checkpoint call on:

```python
if self.gradient_checkpointing and self.training:
    ...  # checkpointed path
```

The QLoRA pattern loads the base with `.eval()` and freezes params (`requires_grad=False`) but
**never puts the base in train mode**. With `self.training == False`, the checkpoint branch never
fires, so the full autograd graph is retained. (This is especially expensive for linear/hybrid
attention like Qwen3.5's `chunk_gated_delta_rule`, whose recurrent state is activation-heavy.)

Freezing params does **not** set `training=True`. They are independent flags.

### Fix

- During the **training step**, call `model.train()` on the base. Params stay frozen via
  `requires_grad=False`. If the model's dropout config is all-zero (verify — Qwen3.5 has no dropout
  fields), train-mode outputs are identical to eval-mode for a frozen base, so this is safe.
- During **generation**, call `model.eval()` (and disable GC + re-enable the KV cache, since
  checkpointing is incompatible with `use_cache`).

### Result

Qwen3.5-4B (4-bit base + 8.5M-param adapter) on the 10GB 3080:

| seq len | GC no-op (eval base) | GC engaged (base.train()) |
|---|---|---|
| 256 | 8.20 GB | 4.21 GB |
| 320 | OOM | 4.54 GB |
| 384 | OOM | 4.81 GB |
| 448 | OOM | 5.07 GB |
| 512 | OOM | 5.34 GB |

seq 512 now trains at 5.34 GB with ~4 GB headroom. No quantization or partitioning change was needed
— the fix was one `model.train()` call.

### Lessons

- A frozen base used only for feature extraction must still be in `.train()` for HF gradient
  checkpointing to engage.
- "Enabling GC didn't lower peak memory" ⇒ the gate isn't firing; check `model.training`.
- Don't conflate "OOM on this GPU" with "architecturally untrainable" — prove the memory accounting
  (per-seq / per-layer peaks) before declaring a dead end.
