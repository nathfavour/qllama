"""Terminal interface for qllama."""

import logging
import os
import sys
import re
from typing import Dict, List, Any, Optional, Tuple
import readline
import shlex

from qllama.models import get_model_handler

_logger = logging.getLogger(__name__)

class QllamaTerminal:
    """Terminal interface for qllama."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        temperature: float = 1.0,
        max_tokens: int = 64,
    ):
        """Initialize the QllamaTerminal.
        
        Args:
            model_name: The name of the model to use
            device: The device to run the model on
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # History of user inputs and model responses
        self.history = []
        
        # Pattern to match image or video attachments
        self.attachment_pattern = re.compile(r'<(?:image|video):([^>]+)>')
        
        print(f"Initializing qllama with model: {model_name}")
        try:
            self.model_handler = get_model_handler(
                model_name=model_name,
                device=device,
            )
            print(f"Loading {model_name}...")
            self.model_handler.load_model()
            print(f"{model_name} loaded successfully!")
        except Exception as e:
            _logger.error(f"Failed to initialize model {model_name}: {e}")
            print(f"Error: {e}")
            sys.exit(1)
    
    def parse_user_input(self, user_input: str) -> Tuple[List[Dict[str, Any]], bool]:
        """Parse user input to extract commands and attachments.
        
        Args:
            user_input: The user's input string
            
        Returns:
            A tuple of (formatted messages, continue_flag)
        """
        # Check for exit command
        if user_input.strip().lower() in ("exit", "quit", "/exit", "/quit"):
            return [], False
        
        # Extract attachments (images and videos)
        content = []
        text = user_input
        
        # Find all attachment tags
        attachments = self.attachment_pattern.findall(text)
        
        # If attachments are found, convert them to content items
        if attachments:
            # Replace attachment tags with placeholders
            for attachment_path in attachments:
                # Determine if it's an image or video based on extension
                ext = os.path.splitext(attachment_path)[1].lower()
                if ext in ('.jpg', '.jpeg', '.png', '.gif', '.bmp'):
                    content.append({
                        "type": "image",
                        "url": attachment_path
                    })
                    text = text.replace(f"<image:{attachment_path}>", "")
                elif ext in ('.mp4', '.avi', '.mov', '.mkv'):
                    content.append({
                        "type": "video",
                        "path": attachment_path
                    })
                    text = text.replace(f"<video:{attachment_path}>", "")
        
        # Add remaining text if not empty
        text = text.strip()
        if text:
            content.append({
                "type": "text",
                "text": text
            })
        
        # Format the message in the expected structure
        messages = [{"role": "user", "content": content}]
        
        # Include history in messages
        full_messages = self.history + messages
        
        return full_messages, True
    
    def run(self) -> None:
        """Run the terminal interface."""
        print(f"\nqllama chat with {self.model_name} 🦙")
        print("Type 'exit' or 'quit' to end the conversation.")
        print("To include images: <image:/path/to/image.jpg> or <image:https://example.com/image.jpg>")
        print("To include videos: <video:/path/to/video.mp4>")
        print("\n")
        
        # Setup command history
        histfile = os.path.join(os.path.expanduser("~"), ".qllama_history")
        try:
            readline.read_history_file(histfile)
            readline.set_history_length(1000)
        except FileNotFoundError:
            pass
        
        try:
            while True:
                try:
                    user_input = input("\nUser: ")
                    readline.write_history_file(histfile)
                    
                    if not user_input.strip():
                        continue
                    
                    messages, continue_flag = self.parse_user_input(user_input)
                    
                    if not continue_flag:
                        print("Exiting qllama. Goodbye!")
                        break
                    
                    if not messages:
                        continue
                    
                    print("\nqllama: ", end="", flush=True)
                    
                    response = self.model_handler.generate(
                        messages,
                        temperature=self.temperature,
                        max_new_tokens=self.max_tokens,
                    )
                    
                    print(response)
                    
                    # Update history with user input and model response
                    # Extract only the last user message
                    self.history.append(messages[-1])
                    # Add the model's response to history
                    self.history.append({
                        "role": "assistant",
                        "content": [{"type": "text", "text": response}]
                    })
                    
                except KeyboardInterrupt:
                    print("\nOperation interrupted. Type 'exit' to quit.")
                except Exception as e:
                    _logger.error(f"Error during conversation: {e}")
                    print(f"\nError: {e}")
        
        finally:
            # Clean up
            try:
                readline.write_history_file(histfile)
            except Exception as e:
                _logger.warning(f"Could not save history file: {e}")
