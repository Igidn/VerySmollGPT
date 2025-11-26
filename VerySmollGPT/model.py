"""
VerySmollGPT: A small decoder-only transformer model
Architecture based on GPT-2 with reduced parameters for character-level language modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention with causal masking for autoregressive generation
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Linear projections for Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Scaling factor for dot product attention
        self.scale = math.sqrt(self.head_dim)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) or None
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project and reshape to (batch_size, n_heads, seq_len, head_dim)
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        # (batch_size, n_heads, seq_len, head_dim) @ (batch_size, n_heads, head_dim, seq_len)
        # -> (batch_size, n_heads, seq_len, seq_len)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        
        # Apply causal mask
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch_size, n_heads, seq_len, seq_len) @ (batch_size, n_heads, seq_len, head_dim)
        # -> (batch_size, n_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reshape back to (batch_size, seq_len, d_model)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Final output projection
        output = self.out_proj(attn_output)
        
        return output


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed-Forward Network with GELU activation
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = F.gelu(x)  # GELU activation
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    """
    Single Transformer Decoder Block with:
    - Multi-Head Self-Attention (with causal mask)
    - Feed-Forward Network
    - Residual connections
    - Layer Normalization
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        
        # Layer normalization (pre-norm architecture like GPT-2)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Multi-head attention
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        
        # Feed-forward network
        self.ffn = FeedForwardNetwork(d_model, d_ff, dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (batch_size, seq_len, d_model)
            mask: (batch_size, seq_len, seq_len) or None
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        # Pre-norm + attention + residual
        attn_output = self.attention(self.ln1(x), mask)
        x = x + self.dropout(attn_output)
        
        # Pre-norm + FFN + residual
        ffn_output = self.ffn(self.ln2(x))
        x = x + self.dropout(ffn_output)
        
        return x


class VerySmollGPT(nn.Module):
    """
    VerySmollGPT: Small decoder-only transformer for character-level language modeling
    
    Architecture:
    - 4 transformer layers
    - 4 attention heads
    - 128 embedding dimension
    - 512 feed-forward dimension
    - 128 token context window
    - ~3M parameters
    """
    def __init__(
        self,
        vocab_size,
        d_model=128,
        n_layers=4,
        n_heads=4,
        d_ff=512,
        max_seq_len=128,
        dropout=0.1
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.dropout_rate = dropout
        
        # Token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Learned positional encoding
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection to vocabulary
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Weight tying (share embeddings with output layer)
        self.lm_head.weight = self.token_embedding.weight
        
        # Initialize weights
        self.apply(self._init_weights)
        
        # Report parameter count
        self.print_model_info()
    
    def _init_weights(self, module):
        """Initialize weights with small values"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def get_causal_mask(self, seq_len, device):
        """
        Create causal mask for autoregressive generation
        Returns lower triangular matrix of ones
        """
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
    
    def forward(self, input_ids, targets=None):
        """
        Args:
            input_ids: (batch_size, seq_len) - token indices
            targets: (batch_size, seq_len) - target token indices for training
        Returns:
            if targets is None:
                logits: (batch_size, seq_len, vocab_size)
            else:
                loss: scalar tensor
                logits: (batch_size, seq_len, vocab_size)
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Get token embeddings
        token_emb = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Get position embeddings
        positions = torch.arange(0, seq_len, dtype=torch.long, device=device)
        positions = positions.unsqueeze(0).expand(batch_size, seq_len)
        pos_emb = self.position_embedding(positions)  # (batch_size, seq_len, d_model)
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb)
        
        # Create causal mask
        mask = self.get_causal_mask(seq_len, device)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Project to vocabulary
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # Calculate loss if targets are provided
        loss = None
        if targets is not None:
            # Flatten for cross-entropy
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1  # Ignore padding tokens if any
            )
        
        if loss is not None:
            return loss, logits
        else:
            return logits
    
    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """
        Generate new tokens autoregressively
        
        Args:
            input_ids: (batch_size, seq_len) - starting tokens
            max_new_tokens: number of tokens to generate
            temperature: sampling temperature (higher = more random)
            top_k: if set, only sample from top k most likely tokens
        Returns:
            generated: (batch_size, seq_len + max_new_tokens)
        """
        self.eval()
        
        for _ in range(max_new_tokens):
            # Crop to max sequence length
            input_ids_cond = input_ids if input_ids.size(1) <= self.max_seq_len else input_ids[:, -self.max_seq_len:]
            
            # Forward pass
            logits = self(input_ids_cond)
            
            # Get logits for last token
            logits = logits[:, -1, :] / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample from distribution
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return input_ids
    
    def print_model_info(self):
        """Print model architecture and parameter count"""
        n_params = sum(p.numel() for p in self.parameters())
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print("=" * 60)
        print("VerySmollGPT Model Architecture")
        print("=" * 60)
        print(f"Vocabulary size:      {self.vocab_size:,}")
        print(f"Embedding dimension:  {self.d_model}")
        print(f"Number of layers:     {self.n_layers}")
        print(f"Attention heads:      {self.n_heads}")
        print(f"Feed-forward dim:     {self.d_ff}")
        print(f"Max sequence length:  {self.max_seq_len}")
        print(f"Dropout rate:         {self.dropout_rate}")
        print("-" * 60)
        print(f"Total parameters:     {n_params:,} ({n_params/1e6:.2f}M)")
        print(f"Trainable parameters: {n_trainable:,} ({n_trainable/1e6:.2f}M)")
        print("=" * 60)


def create_model(vocab_size):
    model = VerySmollGPT(
        vocab_size=vocab_size,
        d_model=256,      
        n_layers=6,
        n_heads=8,
        d_ff=1024,
        max_seq_len=128,
        dropout=0.1
    )
    return model
