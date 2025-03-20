"""Models package for qllama."""

from importlib import import_module
from typing import Dict, Any, Optional
import logging

from .base import BaseModelHandler

_logger = logging.getLogger(__name__)

MODEL_REGISTRY = {
    "smolvlm2": "qllama.models.vision.smolvlm.SmolVLMHandler",
    "mistral": "qllama.models.text.mistral.MistralHandler",
    # Add more models here as they're implemented
}

def get_model_handler(model_name: str, **kwargs) -> BaseModelHandler:
    """Get a model handler for the specified model.
    
    Args:
        model_name: The name of the model to load
        **kwargs: Additional model-specific arguments
        
    Returns:
        An instance of a model handler
    """
    model_key = model_name.lower().replace("-", "").replace("_", "")
    
    if model_key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {', '.join(MODEL_REGISTRY.keys())}")
    
    handler_path = MODEL_REGISTRY[model_key]
    module_path, class_name = handler_path.rsplit(".", 1)
    
    try:
        module = import_module(module_path)
        handler_class = getattr(module, class_name)
        return handler_class(model_name=model_name, **kwargs)
    except (ImportError, AttributeError) as e:
        _logger.error(f"Failed to load handler for model {model_name}: {e}")
        raise ImportError(f"Could not load handler for model {model_name}") from e
