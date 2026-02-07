"""
Gradient checkpointing utilities for ZeRO-Q.

Enables training with minimal activation memory by recomputing
activations during backward pass.
"""

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from typing import List, Optional, Tuple, Callable


def enable_gradient_checkpointing(
    model: nn.Module,
    checkpoint_layers: Optional[List[str]] = None,
    use_reentrant: bool = False,
):
    """
    Enable gradient checkpointing on a model.
    
    For transformer models, this typically checkpoints each
    decoder/encoder layer.
    
    Args:
        model: The model to enable checkpointing on
        checkpoint_layers: List of layer names to checkpoint
                          (default: auto-detect transformer layers)
        use_reentrant: Whether to use reentrant checkpointing
                      (False recommended for better memory)
    """
    # Try to auto-detect the model architecture
    if checkpoint_layers is None:
        checkpoint_layers = _auto_detect_layers(model)
    
    if not checkpoint_layers:
        # Fall back to model's built-in method if available
        if hasattr(model, 'gradient_checkpointing_enable'):
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": use_reentrant}
            )
            return
        else:
            print("Warning: Could not auto-detect layers for checkpointing")
            return
    
    # Apply checkpointing to specified layers
    for name in checkpoint_layers:
        layer = _get_layer_by_name(model, name)
        if layer is not None:
            _wrap_layer_with_checkpoint(layer, use_reentrant)


def _auto_detect_layers(model: nn.Module) -> List[str]:
    """Auto-detect transformer layers for checkpointing."""
    layer_patterns = [
        "model.layers",      # Llama, Qwen, Mistral
        "transformer.h",     # GPT-2, GPT-Neo
        "encoder.layer",     # BERT, RoBERTa
        "decoder.layers",    # T5 decoder
    ]
    
    for pattern in layer_patterns:
        parts = pattern.split(".")
        current = model
        valid = True
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                valid = False
                break
        
        if valid and isinstance(current, (nn.ModuleList, list)):
            return [f"{pattern}.{i}" for i in range(len(current))]
    
    return []


def _get_layer_by_name(model: nn.Module, name: str) -> Optional[nn.Module]:
    """Get a layer by its dotted name."""
    parts = name.split(".")
    current = model
    for part in parts:
        if hasattr(current, part):
            current = getattr(current, part)
        elif isinstance(current, (nn.ModuleList, list)) and part.isdigit():
            current = current[int(part)]
        else:
            return None
    return current


def _wrap_layer_with_checkpoint(layer: nn.Module, use_reentrant: bool):
    """Wrap a layer's forward method with checkpointing."""
    original_forward = layer.forward
    
    def checkpointed_forward(*args, **kwargs):
        return checkpoint(
            original_forward,
            *args,
            use_reentrant=use_reentrant,
            **kwargs,
        )
    
    layer.forward = checkpointed_forward


class ZeroQCheckpointedModule(nn.Module):
    """
    A module wrapper that combines ZeRO-Q with gradient checkpointing.
    
    This provides maximum memory efficiency by:
    1. Storing frozen params in 4-bit (ZeRO-Q)
    2. Recomputing activations during backward (checkpointing)
    
    Example:
        >>> from zero_q import ZeroQConfig, ZeroQCoordinator
        >>> from zero_q.checkpoint import ZeroQCheckpointedModule
        >>> 
        >>> config = ZeroQConfig()
        >>> coordinator = ZeroQCoordinator(config)
        >>> model = ZeroQCheckpointedModule(
        ...     base_model,
        ...     coordinator,
        ...     checkpoint_layers=["model.layers.0", "model.layers.1", ...]
        ... )
    """
    
    def __init__(
        self,
        module: nn.Module,
        coordinator,  # ZeroQCoordinator
        checkpoint_layers: Optional[List[str]] = None,
        use_reentrant: bool = False,
    ):
        super().__init__()
        self.module = module
        self.coordinator = coordinator
        self.checkpoint_layers = checkpoint_layers
        self.use_reentrant = use_reentrant
        
        # Enable gradient checkpointing
        enable_gradient_checkpointing(
            self.module, 
            checkpoint_layers, 
            use_reentrant
        )
    
    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
    
    @property
    def memory_estimate(self) -> dict:
        """Estimate memory usage with checkpointing."""
        # Base model params (quantized)
        quantized_params = sum(
            p.numel() * 0.5 for p in self.module.parameters() 
            if not p.requires_grad
        )
        
        # Trainable params (FP16/32)
        trainable_params = sum(
            p.numel() * p.element_size() for p in self.module.parameters()
            if p.requires_grad
        )
        
        # With checkpointing, activation memory is O(sqrt(L)) instead of O(L)
        # where L is number of layers
        num_layers = len(self.checkpoint_layers) if self.checkpoint_layers else 32
        activation_factor = num_layers ** 0.5 / num_layers
        
        return {
            "quantized_params_mb": quantized_params / (1024 * 1024),
            "trainable_params_mb": trainable_params / (1024 * 1024),
            "activation_reduction": f"{1/activation_factor:.1f}x fewer stored activations",
        }


def estimate_checkpoint_memory(
    hidden_size: int,
    num_layers: int,
    batch_size: int,
    seq_len: int,
    with_checkpointing: bool = True,
) -> dict:
    """
    Estimate activation memory with/without checkpointing.
    
    Args:
        hidden_size: Model hidden dimension
        num_layers: Number of transformer layers
        batch_size: Training batch size
        seq_len: Sequence length
        with_checkpointing: Whether checkpointing is enabled
    
    Returns:
        dict with memory estimates
    """
    # Per-layer activation memory (rough estimate)
    # Attention: Q, K, V, attention weights, etc.
    # MLP: intermediate activations
    bytes_per_element = 2  # FP16
    
    # Attention activations per layer
    attn_per_layer = batch_size * seq_len * hidden_size * 4  # Q, K, V, output
    attn_per_layer += batch_size * seq_len * seq_len  # attention weights
    
    # MLP activations per layer  
    mlp_per_layer = batch_size * seq_len * hidden_size * 3  # gate, up, down
    
    per_layer = (attn_per_layer + mlp_per_layer) * bytes_per_element
    
    if with_checkpointing:
        # Only store ~sqrt(L) activations
        effective_layers = num_layers ** 0.5
    else:
        effective_layers = num_layers
    
    total_bytes = per_layer * effective_layers
    
    return {
        "per_layer_mb": per_layer / (1024 * 1024),
        "total_mb": total_bytes / (1024 * 1024),
        "effective_layers": effective_layers,
        "savings": num_layers / effective_layers if with_checkpointing else 1.0,
    }


if __name__ == "__main__":
    # Example memory estimates
    print("Activation Memory Estimates (batch=2, seq=512)")
    print("=" * 60)
    
    configs = [
        ("Llama 7B", 4096, 32),
        ("Qwen 32B", 5120, 64),
        ("Llama 70B", 8192, 80),
    ]
    
    for name, hidden, layers in configs:
        no_ckpt = estimate_checkpoint_memory(hidden, layers, 2, 512, False)
        with_ckpt = estimate_checkpoint_memory(hidden, layers, 2, 512, True)
        
        print(f"\n{name}:")
        print(f"  Without checkpointing: {no_ckpt['total_mb']:.1f} MB")
        print(f"  With checkpointing:    {with_ckpt['total_mb']:.1f} MB")
        print(f"  Savings:               {no_ckpt['total_mb'] / with_ckpt['total_mb']:.1f}x")
