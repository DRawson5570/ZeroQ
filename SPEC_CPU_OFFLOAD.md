# SPEC: CPU-Offloaded ZeRO-Q Training

> 2026-06-15. 4-bit GPU cache + fp32 CPU master weights + CPU optimizer.
> Enables 5.25B training on a single 10GB GPU (RTX 3080).

## 1. Objective

Train 5.25B params from scratch on a single 10GB GPU by combining ZeroQ's
existing per-layer 4-bit dequantize/release lifecycle with fp32 master
weights and optimizer state on CPU.

## 2. Design

Reuse the existing frozen-backbone 4-bit lifecycle (dequantize -> forward ->
release) but add: requires_grad=True on dequantized tensors, grad hooks that
move gradients to CPU, CPU-side optimizer, and re-quantization after updates.

```
GPU (3080, 10 GB)                          CPU (125 GB RAM)
+----------------------+                   +--------------------------+
| 4-bit weights (all   |                   | fp32 master weights      |
| layers): 2.6 GB      |                   | (full model): 21 GB     |
|                      |                   |                          |
| One layer dequant'd: |  <- dequantize -- | (not used for forward -- |
| ~180 MB (transient)  |                   |  4-bit cache is faster)  |
|                      |                   |                          |
| Activations: ~0.5 GB |                   | AdamW exp_avg: 21 GB     |
|                      |  grad -> cpu ----> | AdamW exp_avg_sq: 21 GB  |
|                      |                   |                          |
|                      |  <- re-quantize -- | optimizer.step() here    |
|                      |  (update 4-bit)   |                          |
+----------------------+                   +--------------------------+
     ~3.5 GB peak                               ~63 GB
```

## 3. Why 4-bit GPU Cache Beats CPU->GPU Streaming

Dequantizing 4-bit->fp32 on GPU: ~0.2 ms per layer (GPU compute).
Moving fp32 from CPU->GPU via PCIe Gen4: ~5.6 ms per layer (180 MB @ 32 GB/s).
The 4-bit weights serve as a GPU-side cache. 28x faster access than PCIe.

## 4. Per-Step Data Flow

**Forward (GPU, per layer):**
1. Pre-forward hook: dequantize 4-bit -> fp32 on GPU, requires_grad=True, register grad hook
2. Forward: compute output
3. Post-forward hook: release dequantized fp32 (4-bit stays on GPU)
4. Gradient checkpointing: intermediates discarded, recomputed during backward

**Backward (GPU, per layer):**
1. Gradient checkpointing re-runs forward (hooks dequantize again)
2. Backward: compute gradients w.r.t. dequantized fp32
3. Grad hook: master_shard.grad = grad.to('cpu')
4. Post-backward hook: release dequantized fp32

**Optimizer (CPU):**
1. optimizer.step() on CPU fp32 master shards
2. optimizer.zero_grad()

**Re-quantize (GPU, per layer):**
1. Move updated fp32 master shard CPU->GPU
2. quantize_4bit() on GPU
3. Replace old 4-bit shard
4. Free GPU copy of master shard

## 5. Memory Budget

**GPU (RTX 3080, 10 GB):**
| Component | Size |
|-----------|------|
| 4-bit weights (all layers) | 2.6 GB |
| One layer dequantized | ~180 MB |
| Activations (grad ckpt) | ~500 MB |
| Re-quantization buffer | ~180 MB |
| PyTorch/CUDA overhead | ~500 MB |
| **Peak** | **~4.0 GB** |

**CPU (125 GB RAM):**
| Component | Size |
|-----------|------|
| fp32 master weights | 21 GB |
| AdamW exp_avg | 21 GB |
| AdamW exp_avg_sq | 21 GB |
| **Total** | **~63 GB** |

## 6. Files to Modify

1. `src/config.py`: Add `cpu_offload: bool = False` to ZeroQTrainConfig
2. `src/coordinator.py`:
   - `_partition_cpu_offload()`: 4-bit on GPU + fp32 master on CPU
   - `_start_gather_4bit_trainable()`: frozen-path dequantize + requires_grad + grad hook
   - CPU grad hook in `_complete_gather`: grad.to('cpu')
   - `update_4bit_from_masters()`: re-quantize CPU masters -> GPU 4-bit after optimizer step
3. `steered_trainer.py`:
   - Remove single-GPU validation error for zeroq-train
   - Auto-detect cpu_offload for single-GPU
   - Call update_4bit_from_masters() after optimizer.step()

## 7. Success Criteria

1. 5.25B model trains on local 3080 (single GPU, 10 GB)
2. GPU peak memory < 5 GB
3. Loss decreases over 10 steps
4. No NaN, no OOM
