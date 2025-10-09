# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run script: `python draw-poly-while-training.py --input <image_path> --shape "<layer_sizes>" --epochs <number>`
- Example: `python draw-poly-while-training.py --input centered_ring.png --shape "[10]*8" --epochs 100`
- Run with acceleration: Add `--use-compile` flag to use torch.compile (requires PyTorch 2.0+)
- Control visualization frequency: Add `--save-interval <N>` to save visualizations every N epochs (default: 1)
- Manage memory usage: Add `--chunk-size <size>` to control number of points processed at once (default: 131072)
- Configure logging: Add `--log-file <path>` to save logs to a file (default: console only)
- Enable debug logging: Add `--debug` flag to increase logging verbosity

### Resume Training
- Resume from checkpoint: `--resume <checkpoint_path>` to continue training from a saved checkpoint
- Save periodic checkpoints: `--checkpoint-interval <N>` to save checkpoints every N epochs
- Specify checkpoint directory: `--checkpoint-dir <directory>` to save checkpoints to a different directory from output
- Switch optimizer when resuming: `--resume-optimizer <adam|sgd|sgd_momentum|rmsprop>` to change optimizer
- Change learning rate when resuming: `--resume-lr <rate>` to override the learning rate
- Example: Resume training and switch from ADAM to SGD:
  ```bash
  python draw-poly-while-training.py --input centered_ring.png --shape "[10]*8" --epochs 200 \
    --resume results/checkpoint_*.pt --resume-optimizer sgd --resume-lr 0.01
  ```
- Example: Save checkpoints to a separate directory:
  ```bash
  python draw-poly-while-training.py --input centered_ring.png --shape "[10]*8" --epochs 200 \
    --checkpoint-interval 10 --checkpoint-dir checkpoints/
  ```

### Kolmogorov Regularization (Experimental)
The codebase supports Kolmogorov complexity regularization via a secondary "weight predictor" network that learns to predict the main network's weights. This encourages simpler, more compressible weight configurations.

**Basic Usage:**
```bash
python draw-poly-while-training.py --input centered_ring.png --shape "[10]*8" --epochs 100 \
  --kolmogorov-shape "[5, 5]" --kolmogorov-weight 0.01
```

**Key Parameters:**
- `--kolmogorov-shape`: Architecture of weight predictor network (e.g., `"[5, 5]"` for two hidden layers of 5 neurons)
- `--kolmogorov-weight`: Regularization strength (default: 0.0, disabled)
- `--kolmogorov-loss-type`: Loss function - `cross_entropy` (default), `mse`, `gaussian_nll`, or `laplacian_nll`
- `--kolmogorov-bins`: Number of bins for cross-entropy quantization (default: 256)
- `--kolmogorov-weight-min/max`: Weight range for quantization (default: -3.0 to 3.0)
- `--kolmogorov-lr`: Learning rate for weight predictor (default: same as main network)

**Loss Types Explained:**
- `cross_entropy`: Quantizes weights into bins, measures encoding cost (most theoretically sound)
- `mse`: Simple L2 regression (baseline)
- `gaussian_nll`: Assumes Gaussian weight distribution
- `laplacian_nll`: Assumes sparse/peaked weight distribution

**Example with Cross-Entropy:**
```bash
python draw-poly-while-training.py --input rgb_ring.png --shape "[10]*8" --epochs 200 \
  --kolmogorov-shape "[8, 8]" --kolmogorov-weight 0.05 \
  --kolmogorov-loss-type cross_entropy --kolmogorov-bins 128
```

## Logging Features
- Timestamped logs with proper log levels (INFO, WARNING, ERROR, DEBUG)
- Performance timing for visualization steps
- Optional file logging with the `--log-file` parameter

## Code Style Guidelines
- Imports: Group standard library, third-party, and local imports in that order
- Formatting: Use 4 spaces for indentation
- Types: Use type hints where appropriate
- Variable names: Use snake_case for variables and functions
- Classes: Use CamelCase for class names
- Error handling: Use try/except blocks with specific exceptions
- Comments: Use docstrings for functions and classes
- Keep code modular and functions focused on single responsibilities
- Follow PyTorch conventions when working with neural networks