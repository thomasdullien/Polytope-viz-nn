import numpy as np


def pixel_to_model_coords(x_pixels, y_pixels, width: int, height: int):
    """Map pixel indices to centered model coordinates."""
    return (x_pixels / width) - 0.5, (y_pixels / height) - 0.5


def model_to_pixel_coords(x_coords, y_coords, width: int, height: int):
    """Map centered model coordinates back to pixel indices."""
    x_pixels = np.round((x_coords + 0.5) * width).astype(int)
    y_pixels = np.round((y_coords + 0.5) * height).astype(int)
    return x_pixels, y_pixels


def valid_pixel_mask(x_pixels, y_pixels, width: int, height: int):
    return (0 <= x_pixels) & (x_pixels < width) & (0 <= y_pixels) & (y_pixels < height)
