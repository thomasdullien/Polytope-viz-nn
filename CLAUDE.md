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
- Switch optimizer when resuming: `--resume-optimizer <adam|sgd|sgd_momentum|rmsprop>` to change optimizer
- Change learning rate when resuming: `--resume-lr <rate>` to override the learning rate
- Example: Resume training and switch from ADAM to SGD:
  ```bash
  python draw-poly-while-training.py --input centered_ring.png --shape "[10]*8" --epochs 200 \
    --resume results/checkpoint_*.pt --resume-optimizer sgd --resume-lr 0.01
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