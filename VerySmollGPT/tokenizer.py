"""
Character-level tokenizer for VerySmollGPT
Simple and efficient character-level tokenization
"""

import os
import json
import pickle
from typing import List, Dict, Optional


class CharTokenizer:
    """
    Character-level tokenizer that maps characters to integers and vice versa.
    Includes special tokens for padding, unknown characters, etc.
    """
    
    def __init__(self):
        self.char_to_idx: Dict[str, int] = {}
        self.idx_to_char: Dict[int, str] = {}
        self.vocab_size: int = 0
        
        # Special tokens
        self.PAD_TOKEN = '<PAD>'
        self.UNK_TOKEN = '<UNK>'
        self.BOS_TOKEN = '<BOS>'  # Beginning of sequence
        self.EOS_TOKEN = '<EOS>'  # End of sequence
        
        self.special_tokens = [
            self.PAD_TOKEN,
            self.UNK_TOKEN,
            self.BOS_TOKEN,
            self.EOS_TOKEN
        ]
        
    def build_vocab(self, text_path: str) -> None:
        """
        Build vocabulary from a text file by collecting all unique characters.
        
        Args:
            text_path: Path to the text file to build vocabulary from
        """
        print(f"Building vocabulary from {text_path}...")
        
        # Collect all unique characters
        unique_chars = set()
        
        with open(text_path, 'r', encoding='utf-8') as f:
            for line in f:
                unique_chars.update(line)
        
        # Sort characters for consistent ordering
        sorted_chars = sorted(list(unique_chars))
        
        # Add special tokens first
        all_tokens = self.special_tokens + sorted_chars
        
        # Build mapping dictionaries
        self.char_to_idx = {char: idx for idx, char in enumerate(all_tokens)}
        self.idx_to_char = {idx: char for idx, char in enumerate(all_tokens)}
        self.vocab_size = len(all_tokens)
        
        print(f"Vocabulary built! Size: {self.vocab_size}")
        print(f"Special tokens: {self.special_tokens}")
        print(f"Sample characters: {sorted_chars[:20]}...")
        
    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text into a list of token IDs.
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS and EOS tokens
            
        Returns:
            List of token IDs
        """
        # Get unknown token index
        unk_idx = self.char_to_idx.get(self.UNK_TOKEN, 0)
        
        # Encode each character
        token_ids = [self.char_to_idx.get(char, unk_idx) for char in text]
        
        # Add special tokens if requested
        if add_special_tokens:
            bos_idx = self.char_to_idx[self.BOS_TOKEN]
            eos_idx = self.char_to_idx[self.EOS_TOKEN]
            token_ids = [bos_idx] + token_ids + [eos_idx]
        
        return token_ids
    
    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode a list of token IDs back into text.
        
        Args:
            token_ids: List of token IDs
            skip_special_tokens: Whether to skip special tokens in output
            
        Returns:
            Decoded text string
        """
        chars = []
        
        for idx in token_ids:
            if idx in self.idx_to_char:
                char = self.idx_to_char[idx]
                
                # Skip special tokens if requested
                if skip_special_tokens and char in self.special_tokens:
                    continue
                    
                chars.append(char)
            else:
                # Unknown token ID
                if not skip_special_tokens:
                    chars.append(self.UNK_TOKEN)
        
        return ''.join(chars)
    
    def save(self, save_path: str) -> None:
        """
        Save tokenizer to disk.
        
        Args:
            save_path: Path to save the tokenizer (without extension)
        """
        # Save as pickle
        tokenizer_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': self.idx_to_char,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        
        pickle_path = f"{save_path}.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        print(f"Tokenizer saved to {pickle_path}")
        
        # Also save as JSON for human readability
        json_path = f"{save_path}.json"
        # Convert integer keys to strings for JSON
        json_data = {
            'char_to_idx': self.char_to_idx,
            'idx_to_char': {str(k): v for k, v in self.idx_to_char.items()},
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False)
        print(f"Tokenizer also saved to {json_path} (human-readable)")
    
    def load(self, load_path: str) -> None:
        """
        Load tokenizer from disk.
        
        Args:
            load_path: Path to load the tokenizer from (without extension)
        """
        pickle_path = f"{load_path}.pkl"
        
        if not os.path.exists(pickle_path):
            raise FileNotFoundError(f"Tokenizer file not found: {pickle_path}")
        
        with open(pickle_path, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        self.char_to_idx = tokenizer_data['char_to_idx']
        self.idx_to_char = tokenizer_data['idx_to_char']
        self.vocab_size = tokenizer_data['vocab_size']
        self.special_tokens = tokenizer_data['special_tokens']
        
        print(f"Tokenizer loaded from {pickle_path}")
        print(f"Vocabulary size: {self.vocab_size}")
    
    def __len__(self) -> int:
        """Return vocabulary size."""
        return self.vocab_size
    
    def get_vocab(self) -> Dict[str, int]:
        """Return the vocabulary dictionary."""
        return self.char_to_idx.copy()


def main():
    """
    Build tokenizer and tokenize the entire dataset
    """
    import numpy as np
    
    # Create Data folder
    data_folder = "Data"
    os.makedirs(data_folder, exist_ok=True)
    print(f"Created/verified folder: {data_folder}/")
    
    # Initialize tokenizer
    tokenizer = CharTokenizer()
    
    # Build vocabulary from data
    data_path = "data.txt"
    if os.path.exists(data_path):
        tokenizer.build_vocab(data_path)
        
        # Save tokenizer to Data folder
        tokenizer_path = os.path.join(data_folder, "tokenizer")
        tokenizer.save(tokenizer_path)
        
        # Test encoding and decoding
        test_text = "Once upon a time, there was a little girl named Lily."
        print(f"\nTest text: {test_text}")
        
        # Encode
        encoded = tokenizer.encode(test_text, add_special_tokens=True)
        print(f"Encoded: {encoded[:20]}... (showing first 20 tokens)")
        print(f"Total tokens: {len(encoded)}")
        
        # Decode
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
        print(f"Decoded: {decoded}")
        
        # Verify
        print(f"Match: {test_text == decoded}")
        
        # Tokenize entire dataset
        print(f"\n{'='*60}")
        print("Tokenizing entire dataset...")
        print(f"{'='*60}")
        
        with open(data_path, 'r', encoding='utf-8') as f:
            full_text = f.read()
        
        print(f"Total characters in dataset: {len(full_text):,}")
        
        # Encode the full text
        print("Encoding text...")
        tokenized_data = tokenizer.encode(full_text, add_special_tokens=False)
        print(f"Total tokens: {len(tokenized_data):,}")
        
        # Save tokenized data as numpy array
        tokenized_path = os.path.join(data_folder, "tokenized_data.npy")
        np.save(tokenized_path, np.array(tokenized_data, dtype=np.int32))
        print(f"\nTokenized data saved to: {tokenized_path}")
        
        # Get file size
        file_size = os.path.getsize(tokenized_path) / (1024 * 1024)
        print(f"Tokenized data size: {file_size:.2f} MB")
        
        print(f"\n{'='*60}")
        print("Tokenization complete!")
        print(f"{'='*60}")
        print(f"Files created in {data_folder}/:")
        print(f"  - tokenizer.pkl (binary tokenizer)")
        print(f"  - tokenizer.json (human-readable tokenizer)")
        print(f"  - tokenized_data.npy (tokenized dataset)")
        
    else:
        print(f"Data file not found: {data_path}")
        print("Please make sure data.txt is in the same directory.")


if __name__ == "__main__":
    main()
