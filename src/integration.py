"""
ZeroQ Integration with HuggingFace Transformers and PEFT

Provides seamless integration with the HuggingFace ecosystem,
allowing ZeRO-Q to work with standard training pipelines.
"""

from typing import Optional, Dict, Any, List, Union
import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from transformers import PreTrainedModel, TrainingArguments, Trainer
    from transformers.modeling_utils import unwrap_model
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

try:
    from peft import PeftModel, LoraConfig, get_peft_model
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False

# Handle both relative and absolute imports
try:
    from .config import ZeroQConfig, MAXWELL_CONFIG
    from .coordinator import ZeroQCoordinator, ZeroQModuleWrapper, ZeroQParamStatus
except ImportError:
    from config import ZeroQConfig, MAXWELL_CONFIG
    from coordinator import ZeroQCoordinator, ZeroQModuleWrapper, ZeroQParamStatus


def prepare_model_for_zeroq(
    model: nn.Module,
    config: Optional[ZeroQConfig] = None,
    freeze_base: bool = True,
) -> nn.Module:
    """
    Prepare a model for ZeRO-Q distributed training.
    
    This function:
    1. Optionally freezes base model parameters
    2. Sets up ZeRO-Q coordinator
    3. Wraps model with ZeRO-Q hooks
    4. Partitions quantized weights across GPUs
    
    Args:
        model: The model to prepare (can be HuggingFace or plain PyTorch)
        config: ZeRO-Q configuration (defaults to MAXWELL_CONFIG)
        freeze_base: Whether to freeze non-LoRA parameters
    
    Returns:
        The wrapped model ready for distributed training
    """
    if config is None:
        config = MAXWELL_CONFIG
    
    # Freeze base model if requested (typical for QLoRA)
    if freeze_base:
        for param in model.parameters():
            param.requires_grad = False
        
        # Unfreeze LoRA params if present
        if HAS_PEFT and isinstance(model, PeftModel):
            for name, param in model.named_parameters():
                if 'lora_' in name.lower():
                    param.requires_grad = True
    
    # Create coordinator
    coordinator = ZeroQCoordinator(config)
    
    # Wrap model - only register frozen params with ZeRO-Q
    # (trainable params stay unpartitioned for gradient handling)
    wrapper = ZeroQModuleWrapper(
        model, 
        coordinator, 
        trainable_only=False  # Register all params
    )
    
    # Partition quantized weights
    wrapper.partition()
    
    # Store wrapper reference on model for later access
    model._zeroq_wrapper = wrapper
    model._zeroq_coordinator = coordinator
    model._zeroq_config = config
    
    return model


def get_zeroq_memory_stats(model: nn.Module) -> Dict[str, float]:
    """Get memory statistics for a ZeRO-Q wrapped model."""
    if hasattr(model, '_zeroq_wrapper'):
        return model._zeroq_wrapper.get_memory_stats()
    return {}


class ZeroQTrainer(Trainer if HAS_TRANSFORMERS else object):
    """
    HuggingFace Trainer with ZeRO-Q support.
    
    Drop-in replacement for the standard Trainer that automatically
    handles ZeRO-Q parameter gathering/releasing.
    
    Example:
        >>> from zero_q.integration import ZeroQTrainer
        >>> trainer = ZeroQTrainer(
        ...     model=model,
        ...     args=training_args,
        ...     train_dataset=dataset,
        ...     zeroq_config=ZeroQConfig(),
        ... )
        >>> trainer.train()
    """
    
    def __init__(
        self,
        model: nn.Module = None,
        args: 'TrainingArguments' = None,
        zeroq_config: Optional[ZeroQConfig] = None,
        **kwargs
    ):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "ZeroQTrainer requires transformers. "
                "Install with: pip install transformers"
            )
        
        self.zeroq_config = zeroq_config or MAXWELL_CONFIG
        
        # Prepare model for ZeRO-Q before passing to parent
        if model is not None:
            model = prepare_model_for_zeroq(model, self.zeroq_config)
        
        super().__init__(model=model, args=args, **kwargs)
    
    def training_step(
        self, 
        model: nn.Module, 
        inputs: Dict[str, Union[torch.Tensor, Any]]
    ) -> torch.Tensor:
        """
        Training step with ZeRO-Q parameter management.
        
        Note: The hooks installed by ZeroQModuleWrapper handle
        parameter gathering automatically during forward pass.
        """
        # Standard training step - hooks handle parameter gathering
        return super().training_step(model, inputs)
    
    def log_zeroq_stats(self):
        """Log ZeRO-Q memory statistics."""
        stats = get_zeroq_memory_stats(self.model)
        if stats:
            self.log({
                "zeroq/local_memory_mb": stats.get("local_memory_mb", 0),
                "zeroq/compression_ratio": stats.get("compression_ratio", 1),
                "zeroq/num_params": stats.get("num_params", 0),
            })


def apply_lora_with_zeroq(
    model: nn.Module,
    lora_config: 'LoraConfig',
    zeroq_config: Optional[ZeroQConfig] = None,
) -> nn.Module:
    """
    Apply LoRA to a model and prepare for ZeRO-Q training.
    
    This is the recommended way to set up QLoRA-style training
    with ZeRO-Q distributed memory optimization.
    
    Args:
        model: Base model (will be quantized and partitioned)
        lora_config: PEFT LoRA configuration
        zeroq_config: ZeRO-Q configuration
    
    Returns:
        Model ready for distributed QLoRA training
    
    Example:
        >>> from peft import LoraConfig
        >>> from zero_q.integration import apply_lora_with_zeroq
        >>> 
        >>> lora_config = LoraConfig(
        ...     r=8,
        ...     lora_alpha=16,
        ...     target_modules=["q_proj", "v_proj"],
        ... )
        >>> model = apply_lora_with_zeroq(model, lora_config)
    """
    if not HAS_PEFT:
        raise ImportError(
            "apply_lora_with_zeroq requires peft. "
            "Install with: pip install peft"
        )
    
    if zeroq_config is None:
        zeroq_config = MAXWELL_CONFIG
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    
    # Prepare for ZeRO-Q (freezes base, partitions quantized weights)
    model = prepare_model_for_zeroq(model, zeroq_config, freeze_base=True)
    
    return model


# Convenience function for quick setup
def initialize(
    model: nn.Module,
    config: Optional[ZeroQConfig] = None,
    lora_config: Optional['LoraConfig'] = None,
) -> nn.Module:
    """
    One-stop initialization for ZeRO-Q.
    
    Args:
        model: Model to initialize
        config: ZeRO-Q config (optional)
        lora_config: LoRA config to apply (optional)
    
    Returns:
        Model ready for ZeRO-Q training
    
    Example:
        >>> import zero_q
        >>> model = zero_q.initialize(model)  # Simple
        >>> model = zero_q.initialize(model, config, lora_config)  # Full
    """
    if lora_config is not None:
        return apply_lora_with_zeroq(model, lora_config, config)
    else:
        return prepare_model_for_zeroq(model, config)
