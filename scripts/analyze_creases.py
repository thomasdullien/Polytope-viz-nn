#!/usr/bin/env python3
import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def load_first_layer(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = checkpoint["model_state_dict"]
    weight = state["network.0.weight"].detach().cpu().numpy()
    bias = state["network.0.bias"].detach().cpu().numpy()
    if weight.ndim != 2 or weight.shape[1] != 2:
        raise ValueError(f"Expected first-layer weight shape (N, 2), got {weight.shape}")
    return weight, bias, checkpoint


def compute_creases(weight, bias):
    wx = weight[:, 0]
    wy = weight[:, 1]
    norm = np.linalg.norm(weight, axis=1)
    eps = 1e-12
    valid = norm > eps

    # Crease orientation is modulo pi: normals w and -w define the same line direction.
    normal_angle = np.mod(np.arctan2(wy, wx), math.pi)
    line_angle = np.mod(normal_angle + math.pi / 2.0, math.pi)
    signed_offset = np.zeros_like(bias)
    signed_offset[valid] = -bias[valid] / norm[valid]

    unit_normals = np.zeros_like(weight)
    unit_normals[valid] = weight[valid] / norm[valid, None]

    return {
        "wx": wx,
        "wy": wy,
        "bias": bias,
        "norm": norm,
        "normal_angle": normal_angle,
        "line_angle": line_angle,
        "signed_offset": signed_offset,
        "unit_nx": unit_normals[:, 0],
        "unit_ny": unit_normals[:, 1],
        "valid": valid,
    }


def circular_pairwise_angle_dist(angles):
    delta = np.abs(angles[:, None] - angles[None, :])
    return np.minimum(delta, math.pi - delta)


def nearest_neighbor_stats(angles, offsets, angle_scale=1.0, offset_scale=None):
    if offset_scale is None:
        offset_scale = np.std(offsets)
        if offset_scale <= 1e-12:
            offset_scale = 1.0
    angle_dist = circular_pairwise_angle_dist(angles) / angle_scale
    offset_dist = np.abs(offsets[:, None] - offsets[None, :]) / offset_scale
    dist = np.sqrt(angle_dist**2 + offset_dist**2)
    np.fill_diagonal(dist, np.inf)
    nearest = dist.min(axis=1)
    nearest_idx = dist.argmin(axis=1)
    return nearest, nearest_idx, offset_scale


def write_csv(path, creases):
    fields = [
        "index",
        "wx",
        "wy",
        "bias",
        "norm",
        "normal_angle",
        "line_angle",
        "signed_offset",
        "unit_nx",
        "unit_ny",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i in range(len(creases["wx"])):
            writer.writerow({field: (i if field == "index" else creases[field][i]) for field in fields})


def save_plots(out_dir, label, creases, nearest):
    out_dir.mkdir(parents=True, exist_ok=True)
    valid = creases["valid"]
    angles = creases["line_angle"][valid]
    offsets = creases["signed_offset"][valid]
    norms = creases["norm"][valid]

    plt.figure(figsize=(10, 4))
    plt.hist(angles, bins=180, range=(0, math.pi), color="#3b82f6")
    plt.xlabel("crease line angle modulo pi")
    plt.ylabel("count")
    plt.title(f"{label}: crease angle histogram")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_angle_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.hist(offsets, bins=180, color="#10b981")
    plt.xlabel("signed offset -b / ||w||")
    plt.ylabel("count")
    plt.title(f"{label}: crease offset histogram")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_offset_hist.png", dpi=160)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.scatter(angles, offsets, s=3, alpha=0.35, c=norms, cmap="viridis")
    plt.colorbar(label="||w||")
    plt.xlabel("crease line angle modulo pi")
    plt.ylabel("signed offset -b / ||w||")
    plt.title(f"{label}: crease angle/offset scatter")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_angle_offset_scatter.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.hist(nearest, bins=120, color="#ef4444")
    plt.xlabel("nearest-neighbor distance in normalized (angle, offset)")
    plt.ylabel("count")
    plt.title(f"{label}: nearest crease distance")
    plt.tight_layout()
    plt.savefig(out_dir / f"{label}_nearest_neighbor_hist.png", dpi=160)
    plt.close()


def summarize(label, creases):
    valid = creases["valid"]
    angles = creases["line_angle"][valid]
    offsets = creases["signed_offset"][valid]
    norms = creases["norm"][valid]
    nearest, nearest_idx, offset_scale = nearest_neighbor_stats(angles, offsets, angle_scale=math.pi)

    angle_bins = np.histogram(angles, bins=180, range=(0, math.pi))[0]
    offset_bins = np.histogram(offsets, bins=180)[0]

    summary = {
        "label": label,
        "n_creases": int(valid.sum()),
        "dead_or_near_zero_norm": int((~valid).sum()),
        "norm_mean": float(np.mean(norms)),
        "norm_median": float(np.median(norms)),
        "offset_mean": float(np.mean(offsets)),
        "offset_std": float(np.std(offsets)),
        "angle_hist_max_bin": int(angle_bins.max()),
        "angle_hist_mean_bin": float(angle_bins.mean()),
        "angle_hist_max_over_mean": float(angle_bins.max() / angle_bins.mean()),
        "offset_hist_max_bin": int(offset_bins.max()),
        "offset_hist_mean_bin": float(offset_bins.mean()),
        "offset_hist_max_over_mean": float(offset_bins.max() / offset_bins.mean()),
        "nearest_median": float(np.median(nearest)),
        "nearest_p10": float(np.quantile(nearest, 0.10)),
        "nearest_p01": float(np.quantile(nearest, 0.01)),
        "offset_scale_for_nn": float(offset_scale),
    }
    return summary, nearest, nearest_idx


def main():
    parser = argparse.ArgumentParser(description="Analyze first-layer crease geometry from a checkpoint.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", default="analysis/creases")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weight, bias, _checkpoint = load_first_layer(args.checkpoint)
    creases = compute_creases(weight, bias)
    summary, nearest, nearest_idx = summarize(args.label, creases)

    write_csv(out_dir / f"{args.label}_creases.csv", creases)
    save_plots(out_dir, args.label, creases, nearest)

    with open(out_dir / f"{args.label}_summary.txt", "w") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    print(f"Wrote {out_dir / (args.label + '_creases.csv')}")
    print(f"Wrote plots and summary under {out_dir}")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
