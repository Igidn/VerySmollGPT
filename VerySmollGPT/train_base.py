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
from tqdm import tqdm

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
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.block_size + 1]
        
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        
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
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            betas=(0.9, 0.95),
            weight_decay=config['weight_decay']
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config['num_epochs'],
            eta_min=config['min_learning_rate']
        )
        
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
        
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        os.makedirs(config['checkpoint_dir'], exist_ok=True)
        
        self.log_file = os.path.join(config['checkpoint_dir'], 'training_log.txt')
        self.log(f"Logging to {self.log_file}")
        
    def log(self, message):
        """Log message to console and file"""
        print(message)
        with open(self.log_file, "a") as f:
            f.write(message + "\n")
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = len(self.train_loader)
        if 'max_batches_per_epoch' in self.config:
            num_batches = min(num_batches, self.config['max_batches_per_epoch'])
        
        start_time = time.time()
        
        pbar = tqdm(
            enumerate(self.train_loader),
            total=num_batches,
            desc=f"Epoch {self.current_epoch + 1}/{self.config['num_epochs']} [Train]",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
        for batch_idx, (inputs, targets) in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            loss, _ = self.model(inputs, targets)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['grad_clip']
            )
            
            self.optimizer.step()
            
            total_loss += loss.item()
            self.global_step += 1
            avg_loss = total_loss / (batch_idx + 1)
            
            lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}',
                'lr': f'{lr:.6f}'
            })
            
            if (batch_idx + 1) % self.config['log_interval'] == 0:
                elapsed = time.time() - start_time
                self.log(f"Epoch: {self.current_epoch + 1}/{self.config['num_epochs']} | "
                      f"Batch: {batch_idx + 1}/{num_batches} | "
                      f"Loss: {avg_loss:.4f} | "
                      f"LR: {lr:.6f} | "
                      f"Time: {elapsed:.1f}s")
            
            if 'max_batches_per_epoch' in self.config and (batch_idx + 1) >= self.config['max_batches_per_epoch']:
                self.log(f"Reached max batches per epoch ({self.config['max_batches_per_epoch']}). Stopping epoch.")
                pbar.close()
                break
        
        pbar.close()
        avg_loss = total_loss / num_batches
        return avg_loss
    
    @torch.no_grad()
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = len(self.val_loader)
        if 'max_val_batches' in self.config:
            num_batches = min(num_batches, self.config['max_val_batches'])
        
        pbar = tqdm(
            enumerate(self.val_loader),
            total=num_batches,
            desc=f"Epoch {self.current_epoch + 1}/{self.config['num_epochs']} [Val]",
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        )
        
        for batch_idx, (inputs, targets) in pbar:
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            loss, _ = self.model(inputs, targets)
            total_loss += loss.item()
            avg_loss = total_loss / (batch_idx + 1)
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{avg_loss:.4f}'
            })
            
            if 'max_val_batches' in self.config and (batch_idx + 1) >= self.config['max_val_batches']:
                self.log(f"Reached max validation batches ({self.config['max_val_batches']}). Stopping validation.")
                pbar.close()
                break
        
        pbar.close()
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
        torch.save(checkpoint, filepath)
        self.log(f"Checkpoint saved to {filepath}")
        
        if is_best:
            best_path = os.path.join(
                self.config['checkpoint_dir'],
                'best_model.pt'
            )
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, best_path)
            self.log(f"Best model saved to {best_path}")
    
    def load_checkpoint(self, filepath):
        """Load model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch'] + 1 
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {filepath}")
        print(f"Resuming from epoch {self.current_epoch + 1}")
    
    def train(self):
        """Main training loop"""
        self.log("\n" + "=" * 70)
        self.log("Starting Training")
        self.log("=" * 70)
        self.log(f"Device: {self.device}")
        self.log(f"Training samples: {len(self.train_dataset):,}")
        self.log(f"Validation samples: {len(self.val_dataset):,}")
        self.log(f"Batch size: {self.config['batch_size']}")
        self.log(f"Number of epochs: {self.config['num_epochs']}")
        self.log(f"Initial learning rate: {self.config['learning_rate']}")
        self.log(f"Mixed Precision: {self.config.get('use_amp', False)}")
        self.log("=" * 70 + "\n")
        
        start_epoch = self.current_epoch
        for epoch in range(start_epoch, self.config['num_epochs']):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch()
            
            checkpoint_path = os.path.join(
                self.config['checkpoint_dir'],
                f'checkpoint_epoch_{epoch + 1}.pt'
            )
            self.save_checkpoint(checkpoint_path, is_best=False)
            
            val_loss = self.validate()
            
            self.scheduler.step()
            
            self.log("\n" + "-" * 70)
            self.log(f"Epoch {epoch + 1}/{self.config['num_epochs']} Summary:")
            self.log(f"  Train Loss: {train_loss:.4f}")
            self.log(f"  Val Loss:   {val_loss:.4f}")
            self.log("-" * 70 + "\n")
            
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                best_path = os.path.join(
                    self.config['checkpoint_dir'],
                    'best_model.pt'
                )
                checkpoint = {
                    'epoch': self.current_epoch,
                    'global_step': self.global_step,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'best_val_loss': self.best_val_loss,
                    'config': self.config
                }
                torch.save(checkpoint, best_path)
                self.log(f"Best model saved to {best_path}")
        
        self.log("\n" + "=" * 70)
        self.log("Training Complete!")
        self.log(f"Best validation loss: {self.best_val_loss:.4f}")
        self.log("=" * 70)


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


def find_latest_checkpoint(checkpoint_dir):
    """
    Find the latest checkpoint file in the checkpoint directory
    
    Args:
        checkpoint_dir: directory containing checkpoints
    
    Returns:
        path to latest checkpoint or None if no checkpoints found
    """
    if not os.path.exists(checkpoint_dir):
        return None
    
    import glob
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, 'checkpoint_epoch_*.pt'))
    
    if not checkpoint_files:
        return None
    
    epoch_numbers = []
    for ckpt in checkpoint_files:
        try:
            basename = os.path.basename(ckpt)
            epoch_str = basename.replace('checkpoint_epoch_', '').replace('.pt', '')
            epoch_numbers.append((int(epoch_str), ckpt))
        except ValueError:
            continue
    
    if not epoch_numbers:
        return None
    
    epoch_numbers.sort(key=lambda x: x[0], reverse=True)
    return epoch_numbers[0][1]


def main():
    """Main training function"""
    
    config = {
        'vocab_size': 102,  
        'd_model': 256,      
        'n_layers': 6,      
        'n_heads': 8,      
        'd_ff': 1024,     
        'max_seq_len': 128,
        'dropout': 0.1,
        
        
        'num_epochs': 3,
        'batch_size': 16,   
        'learning_rate': 3e-4,
        'min_learning_rate': 1e-4,
        'weight_decay': 0.01,
        'grad_clip': 1.0,
        'max_batches_per_epoch': 130_000,
        'max_val_batches': 10_000, 
        
        'block_size': 128, 
        'train_split': 0.9,
        
        'log_interval': 100,
        'checkpoint_dir': 'checkpoints',
        
        'num_workers': 2,
    }
    
    data_path = "Data/tokenized_data.npy"
    tokenizer_path = "Data/tokenizer"
    
    print("Loading tokenizer...")
    tokenizer = CharTokenizer()
    tokenizer.load(tokenizer_path)
    config['vocab_size'] = tokenizer.vocab_size
    print(f"Vocabulary size: {config['vocab_size']}")
    
    train_data, val_data = load_data(data_path, config['train_split'])
    
    print("\nCreating datasets...")
    train_dataset = TextDataset(train_data, config['block_size'])
    val_dataset = TextDataset(val_data, config['block_size'])
    
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    
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
    
    trainer = Trainer(model, train_dataset, val_dataset, config)
    
    latest_checkpoint = find_latest_checkpoint(config['checkpoint_dir'])
    if latest_checkpoint:
        print(f"\n{'='*70}")
        print(f"Found existing checkpoint: {latest_checkpoint}")
        print(f"Resuming training...")
        print(f"{'='*70}\n")
        trainer.load_checkpoint(latest_checkpoint)
    
    trainer.train()
    
    print("\n✓ Training completed successfully!")
    print(f"Checkpoints saved to: {config['checkpoint_dir']}/")


if __name__ == "__main__":
    main()
