# VerySmollGPT

A lightweight character-level GPT model designed to train on Raspberry Pi 

[Huggingface](https://huggingface.co/Kittykat924/VerySmollGPT-5M-Base)

## Architecture

- **Model**: Decoder-only Transformer (GPT-style)
- **Parameters**: ~3-5M
- **Layers**: 6
- **Attention Heads**: 8
- **Embedding Dimension**: 256
- **Feed-forward Dimension**: 1024
- **Context Window**: 128 tokens
- **Tokenizer**: Character-level
- **Vocabulary Size**: 102 unique characters

## Project Structure

```
VerySmollGPT/
├── Data/
│   ├── tokenized_data.npy    # Preprocessed training data
│   ├── tokenizer.pkl          # Tokenizer (binary)
│   └── tokenizer.json         # Tokenizer (readable)
└── VerySmollGPT/
    ├── model.py              # Model architecture
    ├── tokenizer.py          # Character-level tokenizer
    ├── train_base.py         # Training script
    └── generate.py           # Inference script
```

## Requirements

```bash
pip install torch numpy
```

For CPU-only (Raspberry Pi):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```

## Dataset

The model is trained on the [TinyStories dataset](https://huggingface.co/datasets/roneneldan/TinyStories) - a collection of short stories generated to help small language models learn coherent text generation.

- **Dataset size**: 25MB (optimized for Raspberry Pi training time)
- **Total tokens**: ~25M characters

## Training

### On Raspberry Pi

```bash
cd VerySmollGPT
python3 train_base.py
```

### Training Configuration

Edit the `config` dictionary in `train_base.py`:

```python
config = {
    'num_epochs': 5,           # 2-5 epochs recommended
    'batch_size': 16,          # Adjusted for larger model
    'learning_rate': 3e-4,     # Initial LR
    'min_learning_rate': 1e-4, # Final LR (cosine decay)
    'max_seq_len': 128,        # Context window
}
```

### Training Time Estimation

On Raspberry Pi 4 (4GB RAM):
- **Per epoch**: ~2-4 hours (CPU only)
- **Total (5 epochs)**: ~10-20 hours

On a modern GPU:
- **Per epoch**: ~5-10 minutes
- **Total (5 epochs)**: ~30-50 minutes

Checkpoints are saved after each epoch in `checkpoints/`:
- `checkpoint_epoch_X.pt` - Each epoch
- `best_model.pt` - Best validation loss

## Inference

### Generate text from a prompt

```bash
python3 generate.py --prompt "Once upon a time" --max-tokens 200
```

### Interactive mode

```bash
python3 generate.py --interactive
```

Commands in interactive mode:
- Type a prompt and press Enter to generate
- `set max_tokens <value>` - Change generation length
- `set temperature <value>` - Change randomness (0.1-2.0)
- `set top_k <value>` - Change sampling diversity
- `quit` or `exit` - Exit

### CLI Options

```bash
python3 generate.py \
  --checkpoint checkpoints/best_model.pt \
  --prompt "Your prompt here" \
  --max-tokens 200 \
  --temperature 0.8 \
  --top-k 40 \
  --device cpu
```

## Model Performance

The model generates simple, coherent short stories in the style of the TinyStories dataset. Examples:

**Prompt**: "Once upon a time, there was a little"

**Generated**: "Once upon a time, there was a little girl named Lily. She loved to play with her toys. One day, she found a big red ball..."

## Raspberry Pi Optimization

### Memory Usage
- **Model size**: ~12-16 MB (float32)
- **Training RAM**: ~1-2 GB (batch_size=16)
- **Inference RAM**: ~200-300 MB

### Tips for Raspberry Pi
1. **Reduce batch size** if out of memory: `batch_size=8` or `batch_size=16`
2. **Close other applications** during training
3. **Use swap space** if RAM limited
4. **Train overnight** - it takes time on CPU
5. **Monitor temperature**: `vcgencmd measure_temp`

### CPU vs GPU
- Training on CPU: Required for Raspberry Pi
- Training on GPU: 50-100x faster if available
- Model works identically on both

## Development Timeline

1. ✓ Data collection (TinyStories, 20MB)
2. ✓ Character-level tokenizer
3. ✓ Model architecture implementation
4. ✓ Training script with checkpointing
5. ✓ Inference script
6. ⏳ Training on device
7. ⏳ Fine-tuning (optional)

## Troubleshooting

### Out of Memory
```python
# Reduce batch size in train_base.py
config['batch_size'] = 8
```

### Slow Training
- Expected on Raspberry Pi (CPU-only)
- Consider training on a faster machine first
- Transfer the checkpoint to Raspberry Pi for inference

### Model not generating coherent text
- Train for more epochs
- Increase model size (if you have more RAM)
- Check validation loss is decreasing

## Future Improvements

- [ ] Byte-pair encoding (BPE) tokenizer for better compression
- [ ] Model quantization (int8) for smaller size
- [ ] ONNX export for optimized inference
- [ ] Fine-tuning on custom datasets
- [ ] Web interface for generation

## License

MIT

## Credits

- Architecture inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- Dataset: [TinyStories by Ronen Eldan and Yuanzhi Li](https://huggingface.co/datasets/roneneldan/TinyStories)
