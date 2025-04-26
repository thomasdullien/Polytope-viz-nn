# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands
- Run script: `python draw-poly-while-training.py --input <image_path> --shape "<layer_sizes>" --epochs <number>`
- Example: `python draw-poly-while-training.py --input centered_ring.png --shape "[10]*8" --epochs 100`
- Run with acceleration: Add `--use-compile` flag to use torch.compile (requires PyTorch 2.0+)
- Control visualization frequency: Add `--save-interval <N>` to save visualizations every N epochs (default: 1)

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