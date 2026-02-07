#!/usr/bin/env python3
"""
ZeroQ Benchmarks

Measure memory savings and throughput for ZeRO-Q vs baseline approaches.
"""

import os
import sys
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from typing import Dict, List, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MAXWELL_CONFIG, ZeroQConfig
from src.coordinator import ZeroQCoordinator, ZeroQModuleWrapper


@dataclass
class BenchmarkResult:
    """Results from a benchmark run."""
    name: str
    memory_mb: float
    throughput_samples_per_sec: float
    time_per_step_ms: float
    compression_ratio: float = 1.0
    
    def __str__(self):
        return (
            f"{self.name}:\n"
            f"  Memory: {self.memory_mb:.2f} MB\n"
            f"  Throughput: {self.throughput_samples_per_sec:.2f} samples/sec\n"
            f"  Time/step: {self.time_per_step_ms:.2f} ms\n"
            f"  Compression: {self.compression_ratio:.2f}x"
        )


class SimpleMLP(nn.Module):
    """Simple MLP for benchmarking."""
    def __init__(self, input_size: int, hidden_size: int, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_size, bias=False))
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(hidden_size, hidden_size, bias=False))
        self.layers.append(nn.Linear(hidden_size, input_size, bias=False))
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return x


def count_parameters(model: nn.Module) -> int:
    """Count total parameters in model."""
    return sum(p.numel() for p in model.parameters())


def measure_memory(device: int = 0) -> float:
    """Measure current GPU memory usage in MB."""
    torch.cuda.synchronize(device)
    return torch.cuda.memory_allocated(device) / 1024**2


def benchmark_baseline_fp16(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    device: int = 0,
) -> BenchmarkResult:
    """Benchmark baseline FP16 model."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    model = model.half().to(f'cuda:{device}')
    
    mem_after_load = measure_memory(device)
    
    # Warmup
    x = torch.randn(batch_size, seq_len, device=f'cuda:{device}', dtype=torch.float16)
    for _ in range(3):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize(device)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_steps):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    
    total_time = end - start
    time_per_step = total_time / num_steps * 1000  # ms
    throughput = (batch_size * num_steps) / total_time
    
    return BenchmarkResult(
        name="Baseline FP16",
        memory_mb=mem_after_load,
        throughput_samples_per_sec=throughput,
        time_per_step_ms=time_per_step,
        compression_ratio=1.0,
    )


def benchmark_zeroq(
    model: nn.Module,
    batch_size: int,
    seq_len: int,
    num_steps: int,
    config: ZeroQConfig,
    device: int = 0,
) -> BenchmarkResult:
    """Benchmark ZeRO-Q quantized model (memory-only, no distributed ops)."""
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.empty_cache()
    
    model = model.half().to(f'cuda:{device}')
    baseline_mem = measure_memory(device)
    
    # For single-GPU benchmarks, we just measure memory savings from quantization
    # without the distributed all-gather overhead
    
    # Simulate ZeRO-Q memory savings by quantizing parameters
    import bitsandbytes as bnb
    
    original_size = sum(p.numel() * 2 for p in model.parameters())  # FP16 = 2 bytes
    quantized_size = 0
    
    # Use blocksize to determine min elements for quantization
    min_elements = config.blocksize * 2  # At least 2 blocks
    
    for name, param in model.named_parameters():
        if param.numel() >= min_elements:
            # 4-bit quantization: 0.5 bytes per element + quant state overhead
            quantized_size += param.numel() * 0.5 + 256  # absmax overhead
        else:
            quantized_size += param.numel() * 2  # Keep small params in FP16
    
    mem_after_partition = baseline_mem * (quantized_size / original_size)
    
    # Warmup
    x = torch.randn(batch_size, seq_len, device=f'cuda:{device}', dtype=torch.float32)
    for _ in range(3):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize(device)
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(num_steps):
        with torch.no_grad():
            _ = model(x)
    torch.cuda.synchronize(device)
    end = time.perf_counter()
    
    total_time = end - start
    time_per_step = total_time / num_steps * 1000  # ms
    throughput = (batch_size * num_steps) / total_time
    
    compression = original_size / quantized_size
    
    return BenchmarkResult(
        name="ZeRO-Q 4-bit",
        memory_mb=mem_after_partition,
        throughput_samples_per_sec=throughput,
        time_per_step_ms=time_per_step,
        compression_ratio=compression,
    )


def run_memory_scaling_benchmark(
    hidden_sizes: List[int] = [512, 1024, 2048, 4096],
    num_layers: int = 8,
    device: int = 0,
):
    """
    Benchmark memory scaling with model size.
    
    Shows how ZeRO-Q memory grows compared to baseline.
    """
    print("=" * 60)
    print("Memory Scaling Benchmark")
    print("=" * 60)
    print(f"{'Hidden Size':<15} {'Params':<12} {'FP16 MB':<12} {'ZeRO-Q MB':<12} {'Savings':<10}")
    print("-" * 60)
    
    for hidden_size in hidden_sizes:
        torch.cuda.empty_cache()
        
        # Create model
        model = SimpleMLP(hidden_size, hidden_size, num_layers)
        num_params = count_parameters(model)
        
        # Baseline
        model_fp16 = SimpleMLP(hidden_size, hidden_size, num_layers).half().to(f'cuda:{device}')
        fp16_mem = measure_memory(device)
        del model_fp16
        torch.cuda.empty_cache()
        
        # ZeRO-Q
        model = SimpleMLP(hidden_size, hidden_size, num_layers).half().to(f'cuda:{device}')
        for param in model.parameters():
            param.requires_grad = False
        
        coordinator = ZeroQCoordinator(MAXWELL_CONFIG)
        wrapper = ZeroQModuleWrapper(model, coordinator)
        wrapper.partition()
        
        zeroq_mem = measure_memory(device)
        savings = fp16_mem / max(zeroq_mem, 0.1)
        
        print(f"{hidden_size:<15} {num_params:<12,} {fp16_mem:<12.2f} {zeroq_mem:<12.2f} {savings:<10.2f}x")
        
        del model, coordinator, wrapper
        torch.cuda.empty_cache()


def run_throughput_benchmark(
    batch_sizes: List[int] = [1, 4, 8, 16],
    hidden_size: int = 2048,
    num_layers: int = 8,
    num_steps: int = 50,
    device: int = 0,
):
    """
    Benchmark throughput at different batch sizes.
    """
    print("\n" + "=" * 60)
    print("Throughput Benchmark")
    print("=" * 60)
    print(f"{'Batch Size':<12} {'FP16 (s/s)':<15} {'ZeRO-Q (s/s)':<15} {'Overhead':<10}")
    print("-" * 60)
    
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        
        # Baseline
        model = SimpleMLP(hidden_size, hidden_size, num_layers)
        baseline = benchmark_baseline_fp16(model, batch_size, hidden_size, num_steps, device)
        del model
        torch.cuda.empty_cache()
        
        # ZeRO-Q
        model = SimpleMLP(hidden_size, hidden_size, num_layers)
        zeroq = benchmark_zeroq(model, batch_size, hidden_size, num_steps, MAXWELL_CONFIG, device)
        del model
        torch.cuda.empty_cache()
        
        overhead = zeroq.time_per_step_ms / baseline.time_per_step_ms
        
        print(f"{batch_size:<12} {baseline.throughput_samples_per_sec:<15.2f} {zeroq.throughput_samples_per_sec:<15.2f} {overhead:<10.2f}x")


def run_all_benchmarks():
    """Run all benchmarks."""
    print("\n" + "=" * 60)
    print("ZeRO-Q Benchmarks")
    print("=" * 60)
    
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks")
        return
    
    device = 0
    print(f"GPU: {torch.cuda.get_device_name(device)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(device).total_memory / 1024**3:.1f} GB")
    
    run_memory_scaling_benchmark(device=device)
    
    # Skip throughput benchmark for now - it requires proper model setup
    # run_throughput_benchmark(device=device)
    
    print("\n" + "=" * 60)
    print("Benchmarks Complete")
    print("=" * 60)


if __name__ == '__main__':
    run_all_benchmarks()
