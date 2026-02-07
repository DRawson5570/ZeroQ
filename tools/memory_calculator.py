"""
ZeRO-Q Memory Calculator

Calculates expected memory savings for ZeRO-Q distributed training
at various model sizes and GPU counts.
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class ModelConfig:
    """Model configuration for memory calculation."""
    name: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    num_attention_heads: int
    intermediate_size: Optional[int] = None  # Defaults to 4 * hidden_size
    
    @property
    def total_params(self) -> int:
        """Calculate approximate total parameters."""
        h = self.hidden_size
        L = self.num_layers
        V = self.vocab_size
        
        # Per layer
        qkvo = 4 * h * h  # Q, K, V, O projections
        mlp = 3 * h * (self.intermediate_size or 4 * h)  # gate, up, down
        norms = 2 * h  # RMSNorm per layer (weight only)
        
        per_layer = qkvo + mlp + norms
        
        # Global
        embeddings = V * h  # Token embeddings
        lm_head = V * h  # Usually tied with embeddings but counted separately
        final_norm = h
        
        total = L * per_layer + embeddings + lm_head + final_norm
        return int(total)


@dataclass
class ZeroQMemoryCalc:
    """ZeRO-Q memory calculator."""
    
    @staticmethod
    def fp16_memory_mb(params: int) -> float:
        """FP16 memory in MB."""
        return params * 2 / (1024 * 1024)
    
    @staticmethod
    def fp32_memory_mb(params: int) -> float:
        """FP32 memory in MB."""
        return params * 4 / (1024 * 1024)
    
    @staticmethod
    def quant_4bit_memory_mb(params: int, blocksize: int = 64) -> float:
        """4-bit quantized memory in MB (including quant state)."""
        # 4-bit = 0.5 bytes per param
        base = params * 0.5
        # Quant state: absmax per block (FP16 = 2 bytes per block)
        num_blocks = params / blocksize
        quant_state = num_blocks * 2
        return (base + quant_state) / (1024 * 1024)
    
    @staticmethod  
    def zeroq_memory_per_gpu(
        params: int,
        num_gpus: int,
        lora_params: int = 0,
        blocksize: int = 64,
    ) -> dict:
        """
        Calculate ZeRO-Q memory per GPU.
        
        Returns:
            dict with memory breakdown in MB
        """
        # Base model (frozen) - stored in 4-bit, partitioned across GPUs
        frozen_params = params - lora_params
        frozen_quant = ZeroQMemoryCalc.quant_4bit_memory_mb(frozen_params, blocksize)
        frozen_per_gpu = frozen_quant / num_gpus
        
        # LoRA/trainable params - stored in FP16, replicated
        lora_memory = ZeroQMemoryCalc.fp16_memory_mb(lora_params)
        
        # Communication buffer - need space for one layer's params during all-gather
        # Estimate: largest layer is usually MLP, ~8*h^2 params
        # In FP16 (dequantized for compute)
        max_layer_params = params / 32  # Rough estimate
        comm_buffer = ZeroQMemoryCalc.fp16_memory_mb(max_layer_params)
        
        # Optimizer states for LoRA (Adam: 2x for m and v in FP32)
        optimizer_states = ZeroQMemoryCalc.fp32_memory_mb(lora_params * 2)
        
        # Gradients for LoRA (FP16)
        gradients = ZeroQMemoryCalc.fp16_memory_mb(lora_params)
        
        total = frozen_per_gpu + lora_memory + comm_buffer + optimizer_states + gradients
        
        return {
            "frozen_quantized_mb": frozen_per_gpu,
            "lora_params_mb": lora_memory,
            "comm_buffer_mb": comm_buffer,
            "optimizer_states_mb": optimizer_states,
            "gradients_mb": gradients,
            "total_per_gpu_mb": total,
        }
    
    @staticmethod
    def baseline_memory_per_gpu(
        params: int,
        num_gpus: int,
        lora_params: int = 0,
    ) -> dict:
        """
        Calculate baseline ZeRO-3 memory per GPU (FP16 weights).
        
        Returns:
            dict with memory breakdown in MB
        """
        # ZeRO-3 partitions params across GPUs (in FP16)
        frozen_params = params - lora_params
        frozen_per_gpu = ZeroQMemoryCalc.fp16_memory_mb(frozen_params) / num_gpus
        
        # LoRA replicated
        lora_memory = ZeroQMemoryCalc.fp16_memory_mb(lora_params)
        
        # Communication buffer (FP16)
        max_layer_params = params / 32
        comm_buffer = ZeroQMemoryCalc.fp16_memory_mb(max_layer_params)
        
        # Optimizer states
        optimizer_states = ZeroQMemoryCalc.fp32_memory_mb(lora_params * 2)
        
        # Gradients
        gradients = ZeroQMemoryCalc.fp16_memory_mb(lora_params)
        
        total = frozen_per_gpu + lora_memory + comm_buffer + optimizer_states + gradients
        
        return {
            "frozen_fp16_mb": frozen_per_gpu,
            "lora_params_mb": lora_memory,
            "comm_buffer_mb": comm_buffer,
            "optimizer_states_mb": optimizer_states,
            "gradients_mb": gradients,
            "total_per_gpu_mb": total,
        }


# Common model configurations
MODELS = {
    "llama-7b": ModelConfig("Llama 7B", 4096, 32, 32000, 32),
    "llama-13b": ModelConfig("Llama 13B", 5120, 40, 32000, 40),
    "llama-70b": ModelConfig("Llama 70B", 8192, 80, 32000, 64, 28672),
    "qwen-7b": ModelConfig("Qwen 7B", 4096, 32, 152064, 32),
    "qwen-32b": ModelConfig("Qwen 32B", 5120, 64, 152064, 40, 27648),
    "qwen-72b": ModelConfig("Qwen 72B", 8192, 80, 152064, 64, 29568),
    "deepseek-7b": ModelConfig("DeepSeek 7B", 4096, 30, 102400, 32),
}


def compare_memory(
    model_name: str,
    gpu_counts: List[int] = [2, 4, 8],
    lora_rank: int = 64,
    lora_targets: int = 4,  # q, k, v, o projections
):
    """
    Compare ZeRO-Q vs baseline memory for a given model.
    
    Args:
        model_name: Key from MODELS dict
        gpu_counts: List of GPU counts to compare
        lora_rank: LoRA rank for trainable params
        lora_targets: Number of LoRA target modules per layer
    """
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}")
        print(f"Available: {list(MODELS.keys())}")
        return
    
    model = MODELS[model_name]
    total_params = model.total_params
    
    # Calculate LoRA params: rank * (in + out) per target per layer
    lora_per_target = lora_rank * model.hidden_size * 2
    lora_params = lora_per_target * lora_targets * model.num_layers
    
    print(f"\n{'='*70}")
    print(f"{model.name} Memory Comparison")
    print(f"{'='*70}")
    print(f"Total Parameters: {total_params:,} ({total_params / 1e9:.2f}B)")
    print(f"LoRA Parameters:  {lora_params:,} ({lora_params / 1e6:.2f}M)")
    print(f"{'='*70}")
    
    print(f"\n{'GPUs':<6} {'Baseline (MB)':<15} {'ZeRO-Q (MB)':<15} {'Savings':<10} {'Fits M40?':<10}")
    print(f"{'-'*56}")
    
    m40_vram = 11264  # 11GB in MB
    
    for num_gpus in gpu_counts:
        baseline = ZeroQMemoryCalc.baseline_memory_per_gpu(total_params, num_gpus, lora_params)
        zeroq = ZeroQMemoryCalc.zeroq_memory_per_gpu(total_params, num_gpus, lora_params)
        
        savings = baseline["total_per_gpu_mb"] / zeroq["total_per_gpu_mb"]
        fits_baseline = "✓" if baseline["total_per_gpu_mb"] < m40_vram else "✗"
        fits_zeroq = "✓" if zeroq["total_per_gpu_mb"] < m40_vram else "✗"
        
        print(f"{num_gpus:<6} {baseline['total_per_gpu_mb']:<15.1f} {zeroq['total_per_gpu_mb']:<15.1f} {savings:<10.2f}x B:{fits_baseline} Z:{fits_zeroq}")
    
    # Detailed breakdown for 2 GPUs
    print(f"\n{'='*70}")
    print(f"Detailed Breakdown (2 GPUs)")
    print(f"{'='*70}")
    
    baseline = ZeroQMemoryCalc.baseline_memory_per_gpu(total_params, 2, lora_params)
    zeroq = ZeroQMemoryCalc.zeroq_memory_per_gpu(total_params, 2, lora_params)
    
    print(f"\n{'Component':<25} {'Baseline (MB)':<15} {'ZeRO-Q (MB)':<15}")
    print(f"{'-'*55}")
    print(f"{'Frozen Params':<25} {baseline['frozen_fp16_mb']:<15.1f} {zeroq['frozen_quantized_mb']:<15.1f}")
    print(f"{'LoRA Params':<25} {baseline['lora_params_mb']:<15.1f} {zeroq['lora_params_mb']:<15.1f}")
    print(f"{'Comm Buffer':<25} {baseline['comm_buffer_mb']:<15.1f} {zeroq['comm_buffer_mb']:<15.1f}")
    print(f"{'Optimizer States':<25} {baseline['optimizer_states_mb']:<15.1f} {zeroq['optimizer_states_mb']:<15.1f}")
    print(f"{'Gradients':<25} {baseline['gradients_mb']:<15.1f} {zeroq['gradients_mb']:<15.1f}")
    print(f"{'-'*55}")
    print(f"{'TOTAL':<25} {baseline['total_per_gpu_mb']:<15.1f} {zeroq['total_per_gpu_mb']:<15.1f}")


def find_trainable_config(model_name: str, num_gpus: int, target_vram_mb: int = 10000):
    """
    Find what batch size / sequence length is trainable.
    
    Args:
        model_name: Key from MODELS dict
        num_gpus: Number of GPUs
        target_vram_mb: Target VRAM per GPU in MB
    """
    if model_name not in MODELS:
        return
    
    model = MODELS[model_name]
    total_params = model.total_params
    
    # LoRA params (rank 64)
    lora_rank = 64
    lora_params = lora_rank * model.hidden_size * 2 * 4 * model.num_layers
    
    zeroq = ZeroQMemoryCalc.zeroq_memory_per_gpu(total_params, num_gpus, lora_params)
    base_memory = zeroq["total_per_gpu_mb"]
    
    available = target_vram_mb - base_memory
    
    print(f"\n{model.name} with {num_gpus} GPUs (ZeRO-Q)")
    print(f"Base memory: {base_memory:.1f} MB")
    print(f"Available for activations: {available:.1f} MB")
    
    # Activation memory estimate: batch * seq * hidden * num_layers * 2 (for attention) * 2 (bytes)
    # This is a rough estimate
    h = model.hidden_size
    L = model.num_layers
    
    print(f"\nEstimated trainable configs:")
    for seq_len in [512, 1024, 2048, 4096]:
        for batch in [1, 2, 4, 8]:
            # Very rough activation estimate (actual depends on gradient checkpointing)
            act_mb = batch * seq_len * h * L * 4 / (1024 * 1024)  # Conservative
            if act_mb < available:
                print(f"  Batch {batch}, Seq {seq_len}: ~{act_mb:.1f} MB activations")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "qwen-32b"
    
    print("\n" + "="*70)
    print("ZeRO-Q Memory Analysis")
    print("="*70)
    
    # Compare all models
    for model_name in ["llama-7b", "qwen-32b", "llama-70b"]:
        compare_memory(model_name, gpu_counts=[2, 4, 6, 8])
    
    # Show what's trainable on M40 cluster
    print("\n" + "="*70)
    print("Phoenix Cluster (5x M40 @ 11GB each)")
    print("="*70)
    
    find_trainable_config("qwen-32b", 5, 10000)
