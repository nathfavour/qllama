"""Example of basic qllama usage."""

from qllama.models import get_model_handler

def main():
    """Demo basic usage of qllama API."""
    print("Initializing SmolVLM2 model...")
    
    # Get a model handler
    handler = get_model_handler("smolvlm2")
    
    # Load the model
    handler.load_model()
    
    # Create a message with an image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image", 
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                },
                {
                    "type": "text", 
                    "text": "Can you describe this image in detail?"
                },
            ]
        }
    ]
    
    # Generate a response
    print("Generating response...")
    response = handler.generate(messages, max_new_tokens=100)
    
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
