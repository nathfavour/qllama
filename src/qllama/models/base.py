"""Base model handler for qllama."""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union

_logger = logging.getLogger(__name__)

class BaseModelHandler(ABC):
    """Base class for all model handlers in qllama."""
    
    def __init__(self, model_name: str, **kwargs):
        """Initialize the model handler.
        
        Args:
            model_name: The name of the model to load
            **kwargs: Additional model-specific arguments
        """
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = kwargs.get("device", "cuda" if self._is_cuda_available() else "cpu")
        _logger.info(f"Initializing {self.__class__.__name__} for model {model_name} on {self.device}")
    
    @staticmethod
    def _is_cuda_available() -> bool:
        """Check if CUDA is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor."""
        pass
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate a response from the model.
        
        Args:
            messages: A list of message dictionaries
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text response
        """
        pass
    
    def process_messages(self, messages: List[Dict[str, Any]]) -> Any:
        """Process messages into model inputs.
        
        Args:
            messages: A list of message dictionaries
            
        Returns:
            Processed inputs ready for the model
        """
        raise NotImplementedError("Subclasses must implement process_messages")
