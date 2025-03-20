"""Handler for SmolVLM2 model."""

import logging
from typing import Dict, List, Any, Optional
import os
import sys

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

from ..base import BaseModelHandler
from ...utils import load_image, load_video

_logger = logging.getLogger(__name__)

class SmolVLMHandler(BaseModelHandler):
    """Handler for SmolVLM2 model."""
    
    def __init__(self, model_name: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct", **kwargs):
        """Initialize the SmolVLM handler.
        
        Args:
            model_name: The name or path of the SmolVLM model
            **kwargs: Additional model-specific arguments
        """
        if "HuggingFaceTB/SmolVLM" not in model_name:
            model_name = "HuggingFaceTB/SmolVLM2-2.2B-Instruct"
        
        super().__init__(model_name=model_name, **kwargs)
        self.torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        self.attn_implementation = kwargs.get("attn_implementation", "flash_attention_2")
        
        # Check if PIL is properly installed
        try:
            from PIL import Image
        except ImportError as e:
            if "_imaging" in str(e):
                _logger.error("PIL is not properly installed. This is usually caused by missing dependencies.")
                _logger.error("Try reinstalling pillow with: pip uninstall -y pillow && pip install --no-cache-dir pillow")
                raise ImportError("PIL is not properly installed. Try 'pip uninstall -y pillow && pip install --no-cache-dir pillow'") from e
            raise
        
    def load_model(self) -> None:
        """Load the SmolVLM model and processor."""
        _logger.info(f"Loading SmolVLM model: {self.model_name}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                _attn_implementation=self.attn_implementation
            ).to(self.device)
            _logger.info("Successfully loaded SmolVLM model and processor")
        except Exception as e:
            _logger.error(f"Failed to load SmolVLM model: {e}")
            raise
    
    def process_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process messages for SmolVLM.
        
        Args:
            messages: A list of message dictionaries
            
        Returns:
            Processed inputs ready for the model
        """
        # Process images and videos in the messages
        for message in messages:
            if "content" in message and isinstance(message["content"], list):
                for i, content_item in enumerate(message["content"]):
                    if isinstance(content_item, dict):
                        # Handle image URLs or paths
                        if content_item.get("type") == "image" and "url" in content_item:
                            image = load_image(content_item["url"])
                            message["content"][i]["image"] = image
                            # Keep the URL for reference but remove it from processing
                            if "url" in content_item:
                                content_item.pop("url")
                        
                        # Handle video paths
                        elif content_item.get("type") == "video" and "path" in content_item:
                            video_frames = load_video(content_item["path"])
                            message["content"][i]["frames"] = video_frames
                            if "path" in content_item:
                                content_item.pop("path")
        
        # Apply chat template and tokenize
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device, dtype=self.torch_dtype)
        
        return inputs
    
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate a response from the SmolVLM model.
        
        Args:
            messages: A list of message dictionaries
            **kwargs: Additional generation parameters
            
        Returns:
            The generated text response
        """
        if self.model is None:
            self.load_model()
        
        inputs = self.process_messages(messages)
        
        # Set generation parameters with sensible defaults
        generation_kwargs = {
            "do_sample": kwargs.get("do_sample", False),
            "max_new_tokens": kwargs.get("max_new_tokens", 64),
            "temperature": kwargs.get("temperature", 1.0),
            "top_p": kwargs.get("top_p", 1.0),
        }
        
        _logger.debug(f"Generating with parameters: {generation_kwargs}")
        
        # Generate text
        generated_ids = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode the generated ids
        generated_texts = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
        )
        
        return generated_texts[0] if generated_texts else ""
