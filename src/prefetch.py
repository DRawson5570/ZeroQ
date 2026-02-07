"""
ZeroQ Prefetch Manager

Implements async prefetching of quantized parameters to overlap
communication with computation. This is critical for performance
in distributed training.
"""

from typing import Optional, Dict, List, Set, Deque
from collections import deque
import torch
import torch.distributed as dist

# Handle both relative and absolute imports
try:
    from .config import ZeroQConfig
    from .coordinator import ZeroQCoordinator, ZeroQParameter, ZeroQParamStatus
except ImportError:
    from config import ZeroQConfig
    from coordinator import ZeroQCoordinator, ZeroQParameter, ZeroQParamStatus


class PrefetchManager:
    """
    Manages async prefetching of quantized parameters.
    
    The idea: While computing on layer N, start gathering layer N+1.
    This hides communication latency behind computation.
    
    Example:
        >>> prefetch = PrefetchManager(coordinator, prefetch_count=2)
        >>> 
        >>> # During forward pass
        >>> for layer_idx, layer in enumerate(model.layers):
        ...     # Start prefetching next layers
        ...     prefetch.prefetch_for_layer(layer_idx + 1)
        ...     prefetch.prefetch_for_layer(layer_idx + 2)
        ...     
        ...     # Wait for current layer's params
        ...     prefetch.wait_for_layer(layer_idx)
        ...     
        ...     # Compute
        ...     output = layer(input)
        ...     
        ...     # Release previous layer
        ...     if layer_idx > 0:
        ...         prefetch.release_layer(layer_idx - 1)
    """
    
    def __init__(
        self,
        coordinator: ZeroQCoordinator,
        prefetch_count: int = 2,
    ):
        self.coordinator = coordinator
        self.prefetch_count = prefetch_count
        
        # Track which layers are being prefetched
        self._inflight: Set[int] = set()
        self._prefetch_queue: Deque[int] = deque()
        
        # Layer -> param_ids mapping (must be registered)
        self._layer_params: Dict[int, List[int]] = {}
    
    def register_layer(self, layer_idx: int, param_ids: List[int]):
        """Register which parameters belong to a layer."""
        self._layer_params[layer_idx] = param_ids
    
    def prefetch_for_layer(self, layer_idx: int):
        """
        Start async prefetch for a layer's parameters.
        
        Non-blocking - returns immediately.
        """
        if layer_idx not in self._layer_params:
            return
        
        if layer_idx in self._inflight:
            return  # Already prefetching
        
        param_ids = self._layer_params[layer_idx]
        
        # Start async gather for all params in this layer
        for param_id in param_ids:
            if param_id in self.coordinator._params:
                param = self.coordinator._params[param_id]
                if param.status == ZeroQParamStatus.NOT_AVAILABLE:
                    param.start_gather(
                        group=self.coordinator.process_group,
                        async_op=True
                    )
        
        self._inflight.add(layer_idx)
        self._prefetch_queue.append(layer_idx)
    
    def wait_for_layer(self, layer_idx: int):
        """
        Wait for a layer's parameters to be available.
        
        Blocking - ensures params are ready for compute.
        """
        if layer_idx not in self._layer_params:
            return
        
        param_ids = self._layer_params[layer_idx]
        
        # Wait for all params in this layer
        for param_id in param_ids:
            if param_id in self.coordinator._params:
                param = self.coordinator._params[param_id]
                param.wait_gather()
        
        if layer_idx in self._inflight:
            self._inflight.remove(layer_idx)
    
    def release_layer(self, layer_idx: int):
        """
        Release a layer's parameters after use.
        
        Frees memory for subsequent layers.
        """
        if layer_idx not in self._layer_params:
            return
        
        param_ids = self._layer_params[layer_idx]
        self.coordinator.release_params(param_ids)
    
    def prefetch_ahead(self, current_layer: int):
        """
        Prefetch the next N layers based on prefetch_count.
        
        Convenience method for standard forward pass pattern.
        """
        for i in range(1, self.prefetch_count + 1):
            self.prefetch_for_layer(current_layer + i)
    
    def get_stats(self) -> Dict[str, int]:
        """Get prefetch statistics."""
        return {
            "inflight_layers": len(self._inflight),
            "registered_layers": len(self._layer_params),
            "prefetch_count": self.prefetch_count,
        }


class LayerWiseZeroQ:
    """
    Layer-wise ZeRO-Q wrapper with automatic prefetching.
    
    Wraps a model and automatically manages parameter prefetching
    during forward and backward passes.
    
    Example:
        >>> wrapper = LayerWiseZeroQ(model, config)
        >>> wrapper.partition()
        >>> 
        >>> # Forward pass automatically prefetches
        >>> output = wrapper.forward(input_ids)
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        config: Optional[ZeroQConfig] = None,
        layer_attr: str = "layers",  # Attribute name for layer list
    ):
        from .config import MAXWELL_CONFIG
        
        self.model = model
        self.config = config or MAXWELL_CONFIG
        self.layer_attr = layer_attr
        
        # Create coordinator
        self.coordinator = ZeroQCoordinator(self.config)
        
        # Create prefetch manager
        self.prefetch = PrefetchManager(
            self.coordinator,
            prefetch_count=2,
        )
        
        # Register parameters by layer
        self._setup_layers()
    
    def _setup_layers(self):
        """Register parameters organized by layer."""
        # Get the layers module list
        layers = getattr(self.model, self.layer_attr, None)
        if layers is None:
            # Try common alternatives
            for attr in ["layers", "h", "block", "decoder.layers", "encoder.layers"]:
                try:
                    layers = eval(f"self.model.{attr}")
                    if layers is not None:
                        self.layer_attr = attr
                        break
                except:
                    continue
        
        if layers is None:
            raise ValueError(
                f"Could not find layer list. "
                f"Tried: layers, h, block, decoder.layers, encoder.layers. "
                f"Please specify layer_attr manually."
            )
        
        # Register each layer's parameters
        for layer_idx, layer in enumerate(layers):
            param_ids = []
            for param in layer.parameters():
                if not param.requires_grad:  # Only quantize frozen params
                    zq_param = self.coordinator.register_parameter(param, layer)
                    param_ids.append(zq_param.param_id)
            
            self.prefetch.register_layer(layer_idx, param_ids)
        
        # Also register embedding and output layers
        self._register_non_layer_params()
    
    def _register_non_layer_params(self):
        """Register params that aren't in the main layer stack."""
        # Find params not in layers
        layer_params = set()
        layers = getattr(self.model, self.layer_attr)
        for layer in layers:
            for param in layer.parameters():
                layer_params.add(id(param))
        
        # Register remaining params as "layer -1" (embedding) and "layer N" (output)
        embed_params = []
        output_params = []
        
        for name, param in self.model.named_parameters():
            if id(param) not in layer_params and not param.requires_grad:
                zq_param = self.coordinator.register_parameter(param)
                if 'embed' in name.lower():
                    embed_params.append(zq_param.param_id)
                else:
                    output_params.append(zq_param.param_id)
        
        if embed_params:
            self.prefetch.register_layer(-1, embed_params)  # Embedding layer
        if output_params:
            num_layers = len(getattr(self.model, self.layer_attr))
            self.prefetch.register_layer(num_layers, output_params)  # Output layer
    
    def partition(self):
        """Partition all registered parameters."""
        self.coordinator.partition_all()
    
    def forward_with_prefetch(self, *args, **kwargs):
        """
        Forward pass with automatic prefetching.
        
        Note: This is a demonstration. Real integration would
        require hooking into the model's forward method.
        """
        # Fetch embedding layer
        self.prefetch.wait_for_layer(-1)
        
        # Start prefetching first transformer layers
        self.prefetch.prefetch_ahead(0)
        
        # The actual forward would happen here with hooks
        # managing prefetch/release per layer
        
        return self.model(*args, **kwargs)
    
    def get_memory_stats(self) -> Dict[str, float]:
        """Get memory statistics."""
        return self.coordinator.get_memory_stats()


def create_prefetch_hooks(
    model: torch.nn.Module,
    prefetch_manager: PrefetchManager,
    layer_attr: str = "layers",
):
    """
    Create forward hooks that implement prefetching.
    
    Returns a list of hook handles that can be removed later.
    """
    handles = []
    layers = getattr(model, layer_attr)
    num_layers = len(layers)
    
    for layer_idx, layer in enumerate(layers):
        def make_pre_hook(idx):
            def pre_hook(module, inputs):
                # Wait for this layer's params
                prefetch_manager.wait_for_layer(idx)
                # Start prefetching next layers
                prefetch_manager.prefetch_ahead(idx)
            return pre_hook
        
        def make_post_hook(idx):
            def post_hook(module, inputs, outputs):
                # Release this layer if not needed for backward
                if not torch.is_grad_enabled():
                    prefetch_manager.release_layer(idx)
            return post_hook
        
        handles.append(layer.register_forward_pre_hook(make_pre_hook(layer_idx)))
        handles.append(layer.register_forward_hook(make_post_hook(layer_idx)))
    
    return handles
