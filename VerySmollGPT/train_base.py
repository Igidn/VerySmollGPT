"""
Training script for VerySmollGPT-Base
Trains a character-level transformer on TinyStories dataset
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from model import create_model
from tokenizer import CharTokenizer


class TextDataset(Dataset):
    """
    Dataset for character-level language modeling
    Creates sliding windows of text for training
    """
    def __init__(self, data, block_size):
        """
        Args:
            data: numpy array of token IDs
            block_size: context window size (sequence length)
        """
        self.data = data
        self.block_size = block_size
    
    def __len__(self):
        # Number of possible windows
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        # Get a chunk of data
        chunk = self.data[idx:idx + self.block_size + 1]
        
        # Input is all tokens except the last
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        
        # Target is all tokens except the first (shifted by 1)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y


class Trainer:
    """
    Trainer class for VerySmollGPT
    """
    def __init__(
        self,
        model,
        train_dataset,
        val_dataset,
        config
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config['weight_decay']
        )
        
        # Setup learning rate scheduler (cosine decay)
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['min_learning_rate']
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config['batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Create checkpoint directory
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            # Move to device
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            loss, _ = self.model(inputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Print progress
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                avg_loss = total_loss / (batch_idx + 1)
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]['lr']
                
                print(f"Epoch: {self.current_epoch + 1}/{self.config['num_epochs']} | "
                      f"Batch: {batch_idx + 1}/{num_batches} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.6f} | "
                      f"Time: {elapsed:.1f}s")
            
            # Limit batches per epoch if specified
            if 'max_batches_per_epoch' in self.config and (batch_idx + 1) >= self.config['max_batches_per_epoch']:
                print(f"Reached max batches per epoch ({self.config['max_batches_per_epoch']}). Stopping epoch.")
                break
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        
        for inputs, targets in self.val_loader:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            loss, _ = self.model(inputs, targets)
            total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def save_checkpoint(self, filepath, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
        
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint_dir'],
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            print(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
    
    def train(self):
        """Main training loop"""
        print("\n" + "=" * 70)
        print("Starting Training")
        print("=" * 70)
        print(f"Device: {self.device}")
        print(f"Training samples: {len(self.train_dataset):,}")
        print(f"Validation samples: {len(self.val_dataset):,}")
        print(f"Batch size: {self.config['batch_size']}")
        print(f"Number of epochs: {self.config['num_epochs']}")
        print(f"Initial learning rate: {self.config['learning_rate']}")
        print("=" * 70 + "\n")
        
        for epoch in range(self.config['num_epochs']):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_loss = self.train_epoch()
            
            # Validate
            val_loss = self.validate()
            
            # Update learning rate
            self.scheduler.step()
            
            # Print epoch summary
            print("\n" + "-" * 70)
            print(f"Epoch {epoch + 1}/{self.config['num_epochs']} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print("-" * 70 + "\n")
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                self.config['checkpoint_dir'],
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
            
            self.save_checkpoint(checkpoint_path, is_best=is_best)
        
        print("\n" + "=" * 70)
        print("Training Complete!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        print("=" * 70)


def load_data(data_path, train_split=0.9):
    """
    Load and split tokenized data
    
    Args:
        data_path: path to tokenized_data.npy
        train_split: fraction of data for training
    
    Returns:
        train_data, val_data
    """
    print(f"Loading data from {data_path}...")
    data = np.load(data_path)
    print(f"Total tokens: {len(data):,}")
    
    # Split into train and validation
    split_idx = int(len(data) * train_split)
    train_data = data[:split_idx]
    val_data = data[split_idx:]
    
    print(f"Training tokens: {len(train_data):,}")
    print(f"Validation tokens: {len(val_data):,}")
    
    return train_data, val_data


def main():
    """Main training function"""
    
    # Configuration
    config = {
        'vocab_size': 102,  
        'd_model': 256,      
        'n_layers': 6,      
        'n_heads': 8,      
        'd_ff': 1024,     
        'max_seq_len': 128,
        'dropout': 0.1,
        
        # Training
        'num_epochs': 5,
        'batch_size': 16,   
        'learning_rate': 3e-4,
        'min_learning_rate': 1e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'max_batches_per_epoch': 130_000,
        
        # Data
        'block_size': 128,  # Context window
        'train_split': 0.9,
        
        # Logging
        'log_interval': 100,
        'checkpoint_dir': 'checkpoints',
        
        # System
        'num_workers': 2,
    }
    
    # Paths
    data_path = "Data/tokenized_data.npy"
    tokenizer_path = "Data/tokenizer"
    
    # Load tokenizer to verify vocab size
    print("Loading tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.load(tokenizer_path)
    config['vocab_size'] = tokenizer.vocab_size
    print(f"Vocabulary size: {config['vocab_size']}")
    
    # Load data
    train_data, val_data = load_data(data_path, config['train_split'])
    
    # Create datasets
    print("\nCreating datasets...")
    train_dataset = TextDataset(train_data, config['block_size'])
    val_dataset = TextDataset(val_data, config['block_size'])
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
    # Create model
    print("\nCreating model...")
    print("\nCreating model...")
    model = create_model(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        n_layers=config['n_layers'],
        n_heads=config['n_heads'],
        d_ff=config['d_ff'],
        max_seq_len=config['max_seq_len'],
        dropout=config['dropout']
    )
    
    # Create trainer
    trainer = Trainer(model, train_dataset, val_dataset, config)
    
    # Train
    trainer.train()
    
    print("\n✓ Training completed successfully!")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}/")


if __name__ == "__main__":
    main()
