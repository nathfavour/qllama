"""Example of comparing multiple images with qllama."""

from qllama.models import get_model_handler

def main():
    """Demo comparing multiple images with qllama."""
    print("Initializing SmolVLM2 model...")
    
    # Get a model handler
    handler = get_model_handler("smolvlm2")
    
    # Load the model
    handler.load_model()
    
    # Create a message with multiple images
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text", 
                    "text": "What is the difference between these two images?"
                },
                {
                    "type": "image", 
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/bee.jpg"
                },
                {
                    "type": "image", 
                    "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/0052a70beed5bf71b92610a43a52df6d286cd5f3/diffusers/rabbit.jpg"
                },
            ]
        }
    ]
    
    # Generate a response
    print("Generating response...")
    response = handler.generate(messages, max_new_tokens=150)
    
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
