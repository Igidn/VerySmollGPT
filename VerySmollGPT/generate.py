"""
Inference script for VerySmollGPT
Generate text using a trained model
"""

import torch
from model import VerySmollGPT
from tokenizer import CharTokenizer


def load_model(checkpoint_path, device='cpu'):
    """
    Load a trained model from checkpoint
    
    Args:
        checkpoint_path: path to checkpoint file
        device: 'cpu' or 'cuda'
    
    Returns:
        model: loaded model
        config: training configuration
    """
    print(f"Loading model from {checkpoint_path}...")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # Create model
    model = VerySmollGPT(
        vocab_size=config['vocab_size'],
        d_model=config.get('d_model', 128),
        n_layers=config.get('n_layers', 4),
        n_heads=config.get('n_heads', 4),
        d_ff=config.get('d_ff', 512),
        max_seq_len=config.get('max_seq_len', 128),
        dropout=config.get('dropout', 0.1)
    )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Trained for {checkpoint['epoch'] + 1} epochs")
    print(f"Best validation loss: {checkpoint.get('best_val_loss', 'N/A')}")
    
    return model, config


def generate_text(
    model,
    tokenizer,
    prompt="Once upon a time",
    max_new_tokens=200,
    temperature=0.8,
    top_k=40,
    device='cpu'
):
    """
    Generate text from a prompt
    
    Args:
        model: trained VerySmollGPT model
        tokenizer: CharTokenizer instance
        prompt: starting text
        max_new_tokens: number of tokens to generate
        temperature: sampling temperature (higher = more random)
        top_k: only sample from top k tokens
        device: 'cpu' or 'cuda'
    
    Returns:
        generated_text: generated text string
    """
    model.eval()
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)
    
    print(f"Prompt: {prompt}")
    print(f"Generating {max_new_tokens} tokens...")
    print("-" * 70)
    
    # Generate
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k
        )
    
    # Decode
    generated_ids = generated_ids[0].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    return generated_text


def interactive_mode(model, tokenizer, device='cpu'):
    """
    Interactive generation mode
    """
    print("\n" + "=" * 70)
    print("Interactive Text Generation Mode")
    print("=" * 70)
    print("Commands:")
    print("  - Type a prompt and press Enter to generate")
    print("  - Type 'quit' or 'exit' to exit")
    print("  - Type 'help' for options")
    print("=" * 70 + "\n")
    
    # Default settings
    max_tokens = 200
    temperature = 0.8
    top_k = 40
    
    while True:
        try:
            user_input = input("\nPrompt > ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            
            if user_input.lower() == 'help':
                print("\nCurrent settings:")
                print(f"  Max tokens: {max_tokens}")
                print(f"  Temperature: {temperature}")
                print(f"  Top-k: {top_k}")
                print("\nTo change settings, use:")
                print("  set max_tokens <value>")
                print("  set temperature <value>")
                print("  set top_k <value>")
                continue
            
            # Check for settings commands
            if user_input.lower().startswith('set '):
                parts = user_input.split()
                if len(parts) == 3:
                    param, value = parts[1], parts[2]
                    try:
                        if param == 'max_tokens':
                            max_tokens = int(value)
                            print(f"✓ Max tokens set to {max_tokens}")
                        elif param == 'temperature':
                            temperature = float(value)
                            print(f"✓ Temperature set to {temperature}")
                        elif param == 'top_k':
                            top_k = int(value)
                            print(f"✓ Top-k set to {top_k}")
                        else:
                            print(f"Unknown parameter: {param}")
                    except ValueError:
                        print(f"Invalid value: {value}")
                continue
            
            # Generate text
            generated = generate_text(
                model,
                tokenizer,
                prompt=user_input,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                device=device
            )
            
            print(generated)
            print("-" * 70)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


def main():
    """Main inference function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate text with VerySmollGPT')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='checkpoints/best_model.pt',
        help='Path to model checkpoint'
    )
    parser.add_argument(
        '--tokenizer',
        type=str,
        default='Data/tokenizer',
        help='Path to tokenizer (without extension)'
    )
    parser.add_argument(
        '--prompt',
        type=str,
        default=None,
        help='Text prompt for generation'
    )
    parser.add_argument(
        '--max-tokens',
        type=int,
        default=200,
        help='Maximum number of tokens to generate'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.8,
        help='Sampling temperature'
    )
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='Top-k sampling parameter'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to use for inference'
    )
    
    args = parser.parse_args()
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.load(args.tokenizer)
    
    # Load model
    model, config = load_model(args.checkpoint, device=args.device)
    
    # Interactive or single generation
    if args.interactive:
        interactive_mode(model, tokenizer, device=args.device)
    else:
        if args.prompt is None:
            args.prompt = "Once upon a time, there was a"
        
        generated = generate_text(
            model,
            tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=args.device
        )
        
        print(generated)


if __name__ == "__main__":
    main()
