"""Handler for Mistral models."""

import logging
from typing import Dict, List, Any, Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from ..base import BaseModelHandler

_logger = logging.getLogger(__name__)

class MistralHandler(BaseModelHandler):
    """Handler for Mistral models."""
    
    def __init__(self, model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", **kwargs):
        """Initialize the Mistral handler.
        
        Args:
            model_name: The name or path of the Mistral model
            **kwargs: Additional model-specific arguments
        """
        super().__init__(model_name=model_name, **kwargs)
        self.torch_dtype = kwargs.get("torch_dtype", torch.bfloat16)
        
    def load_model(self) -> None:
        """Load the Mistral model and tokenizer."""
        _logger.info(f"Loading Mistral model: {self.model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=self.torch_dtype,
                device_map=self.device
            )
            self.processor = self.tokenizer  # Alias for consistency
            _logger.info("Successfully loaded Mistral model and tokenizer")
        except Exception as e:
            _logger.error(f"Failed to load Mistral model: {e}")
            raise
    
    def process_messages(self, messages: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process messages for Mistral.
        
        Args:
            messages: A list of message dictionaries
            
        Returns:
            Processed inputs ready for the model
        """
        # Convert to the format expected by the model
        conversation = []
        
        for message in messages:
            role = message.get("role", "user")
            
            if role not in ["user", "assistant", "system"]:
                role = "user"
            
            content = ""
            
            # Extract text from content
            if "content" in message:
                if isinstance(message["content"], str):
                    content = message["content"]
                elif isinstance(message["content"], list):
                    for item in message["content"]:
                        if isinstance(item, dict) and item.get("type") == "text":
                            content += item.get("text", "") + " "
            
            conversation.append({"role": role, "content": content.strip()})
        
        # Format the conversation
        prompt = self.tokenizer.apply_chat_template(
            conversation, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        return inputs
    
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """Generate a response from the Mistral model.
        
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
            "do_sample": kwargs.get("do_sample", True),
            "max_new_tokens": kwargs.get("max_new_tokens", 128),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9),
        }
        
        _logger.debug(f"Generating with parameters: {generation_kwargs}")
        
        # Generate text
        with torch.no_grad():
            output = self.model.generate(**inputs, **generation_kwargs)
        
        # Decode the generated ids
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the newly generated part
        response = generated_text[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):]
        
        return response.strip()
