# Polytope-viz-nn
A repository for experiments related to visualizing training dynamics in neural networks based on drawing the polytope boundaries

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

The scripts also use `ffmpeg` to assemble snapshot PNGs into videos.

## Commands

Grayscale regression:

```bash
.venv/bin/python draw-poly-while-training.py --input centered_ring.png --shape "[10]*8" --epochs 100
```

RGB classification:

```bash
.venv/bin/python draw-poly-classifier.py --input centered_ring_rgb.png --shape "[10]*8" --epochs 100
```

Optimizer stability options:

```bash
.venv/bin/python draw-poly-while-training.py --input centered_ring.png --shape "[10]*8" --epochs 100 \
  --optimizer adamw --adam-eps 1e-6 --amsgrad --grad-clip-norm 1.0
```

Muon optimizer:

```bash
.venv/bin/python draw-poly-while-training.py --input centered_ring.png --shape "[10]*8" --epochs 100 \
  --optimizer muon --muon-eps 1e-6 --muon-ns-steps 5 --grad-clip-norm 1.0
```

Muon is applied to 2D weight matrices. Biases and other non-2D parameters are
optimized with AdamW fallback so every trainable parameter is still updated.

For movie-heavy runs with frequent snapshots, JPEG snapshots are much faster to
encode than PNG snapshots:

```bash
.venv/bin/python draw-poly-while-training.py --input centered_ring.png --shape "[4096]" --epochs 500000 \
  --points 4096 --batch-size 4096 --save-interval 10 --snapshot-format jpg --jpeg-quality 90
```

Visualization caches fixed full-image tensors and pixel coordinate maps between
frames. If a larger experiment shows CUDA memory fragmentation, add
`--empty-cache-each-viz-chunk` to restore the older behavior of clearing the CUDA
cache after every visualization chunk.

## Structure

- `draw-poly-while-training.py` and `draw-poly-classifier.py`: compatibility CLI wrappers.
- `polytope_viz/cli.py`: command-line argument definitions.
- `polytope_viz/pipeline.py`: shared experiment pipeline.
- `polytope_viz/models.py`: MLP and polytope hash model.
- `polytope_viz/kolmogorov.py`: Kolmogorov weight-predictor regularization.
- `polytope_viz/training.py`: optimizer creation and training loop.
- `polytope_viz/data.py`: grayscale, RGB, and video preprocessing.
- `polytope_viz/coordinates.py`: centered image coordinate conversion.
- `polytope_viz/visualization.py`: snapshot rendering.
- `polytope_viz/checkpointing.py`: checkpoint save/load.
- `polytope_viz/shape.py`: safe shape parsing for inputs such as `"[10]*8"`.

## Coordinate Convention

Model inputs use centered image coordinates: the image center is `(0, 0)`,
the upper-left corner is approximately `(-0.5, -0.5)`, and the lower-right
corner is approximately `(0.5, 0.5)`. Snapshot rendering maps these centered
coordinates back to the original pixel grid.
