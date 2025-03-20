"""Example of using a text-only model with qllama."""

from qllama.models import get_model_handler

def main():
    """Demo basic usage of text model in qllama."""
    print("Initializing Mistral model...")
    
    # Get a model handler
    handler = get_model_handler("mistral")
    
    # Load the model
    handler.load_model()
    
    # Create a simple text message
    messages = [
        {
            "role": "system",
            "content": "You are a helpful AI assistant."
        },
        {
            "role": "user",
            "content": "Explain quantum computing in simple terms."
        }
    ]
    
    # Generate a response
    print("Generating response...")
    response = handler.generate(messages, max_new_tokens=150)
    
    print("\nResponse:")
    print(response)

if __name__ == "__main__":
    main()
