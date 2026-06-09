import os
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

from . import diagnostics
from .coordinates import model_to_pixel_coords, valid_pixel_mask
from .diagnostics import CONSTANT_OUTPUT_THRESHOLD, dump_network_weights

_GRID_CACHE = {}
_PIXEL_COORD_CACHE = {}


def _empty_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _cached_grid_inputs(data, device):
    key = (id(data), data.shape, str(device))
    cached = _GRID_CACHE.get(key)
    if cached is None:
        cached = torch.as_tensor(data[:, :-1], dtype=torch.float32, device=device)
        _GRID_CACHE[key] = cached
    return cached


def _run_full_grid(network, data, chunk_size, logger, empty_cache_each_chunk=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    if chunk_size is None:
        chunk_size = 128 * 1024
    total_points = data.shape[0]
    num_chunks = (total_points + chunk_size - 1) // chunk_size
    logger.debug(f"Processing {total_points} points in {num_chunks} chunks of {chunk_size}")
    all_outputs = []
    all_activation_hashes = []
    grid_inputs = _cached_grid_inputs(data, device)
    try:
        with torch.no_grad():
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_points)
                chunk_inputs = grid_inputs[start_idx:end_idx]
                chunk_outputs, chunk_activation_hashes = network(chunk_inputs)
                all_outputs.append(chunk_outputs.detach().cpu().numpy())
                all_activation_hashes.append(chunk_activation_hashes.detach().cpu().numpy())
                del chunk_outputs, chunk_activation_hashes
                if empty_cache_each_chunk:
                    _empty_cuda_cache()
    except Exception as e:
        logger.error(f"Error during network forward pass: {str(e)}")
        logger.error(f"Chunk index: {i if 'i' in locals() else 'N/A'}, Chunk size: {chunk_size}")
        raise ValueError(f"Network forward pass failed: {str(e)}") from e
    return np.concatenate(all_outputs), np.concatenate(all_activation_hashes)


def _activation_boundary_map(activation_map):
    height, width = activation_map.shape
    boundary_map = np.zeros((height, width), dtype=np.uint8)
    center = activation_map[1:-1, 1:-1]
    boundary_mask = np.zeros((height - 2, width - 2), dtype=bool)
    for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor = activation_map[1 + dy:height - 1 + dy, 1 + dx:width - 1 + dx]
        boundary_mask = boundary_mask | (neighbor != center)
    boundary_map[1:-1, 1:-1][boundary_mask] = 255
    return boundary_map


def _valid_pixel_coords(data, width, height):
    key = (id(data), data.shape, width, height)
    cached = _PIXEL_COORD_CACHE.get(key)
    if cached is None:
        x_coords, y_coords = model_to_pixel_coords(data[:, 0], data[:, 1], width, height)
        valid = valid_pixel_mask(x_coords, y_coords, width, height)
        cached = (x_coords, y_coords, valid)
        _PIXEL_COORD_CACHE[key] = cached
    x_coords, y_coords, valid = cached
    if not np.any(valid):
        raise ValueError('No valid points were found for processing. Check data format and dimensions.')
    return x_coords[valid], y_coords[valid], valid


def _mark_points(image, points, width, height, color):
    x, y = model_to_pixel_coords(points[:, 0], points[:, 1], width, height)
    valid = valid_pixel_mask(x, y, width, height)
    if np.any(valid):
        image[y[valid], x[valid]] = color


def _epoch_from_path(output_path):
    return int(output_path.split('_epoch_')[1].split('.')[0])


def _handle_constant_hash(activation_hashes, logger, warning, output_path='weights.txt'):
    logger.warning(warning)
    current_activation_hash = hash(activation_hashes.tobytes())
    if diagnostics.constant_activation_map is None:
        diagnostics.constant_activation_map = current_activation_hash
        diagnostics.constant_output_counter = 1
        logger.warning(f"First constant output detected. Counter: {diagnostics.constant_output_counter}")
    elif diagnostics.constant_activation_map == current_activation_hash:
        diagnostics.constant_output_counter += 1
        logger.warning(f"Same constant output detected. Counter: {diagnostics.constant_output_counter}")
        if diagnostics.constant_output_counter >= CONSTANT_OUTPUT_THRESHOLD:
            logger.error(f"Network has produced the same constant output for {CONSTANT_OUTPUT_THRESHOLD} consecutive iterations.")
            logger.error('Aborting due to network failure.')
            raise RuntimeError('Network produced repeated constant output')
    else:
        diagnostics.constant_output_counter = 1
        diagnostics.constant_activation_map = current_activation_hash
        logger.warning('Constant output but different activation map. Resetting counter.')


def visualize_grayscale(network, data, train_data, val_data, image_shape, output_path, target_image_path, logger, train_loss=None, val_loss=None, network_shape_str=None, random_seed=None, epoch=None, num_points=None, learning_rate=None, optimizer=None, momentum=None, chunk_size=None, final_activation=None, image_write_params=None, empty_cache_each_chunk=False):
    height, width = image_shape
    outputs, activation_hashes = _run_full_grid(network, data, chunk_size, logger, empty_cache_each_chunk=empty_cache_each_chunk)
    outputs = outputs.flatten()
    logger.debug(f"Network outputs (len {len(outputs)}) range: min={outputs.min():.6f}, max={outputs.max():.6f}")

    is_constant = outputs.min() == outputs.max()
    if is_constant:
        _handle_constant_hash(activation_hashes, logger, 'Network outputs are exactly constant (min == max). This indicates a serious problem with the network.')

    activation_map = np.zeros((height, width), dtype=np.uint64)
    prediction_map = np.zeros((height, width), dtype=np.float32)
    valid_x, valid_y, valid = _valid_pixel_coords(data, width, height)
    activation_map[valid_y, valid_x] = activation_hashes[valid]
    prediction_map[valid_y, valid_x] = np.clip(outputs[valid] * 255, 0, 255)
    boundary_map = _activation_boundary_map(activation_map)

    prediction_map = prediction_map.astype(np.uint8)
    rgb_prediction = cv2.cvtColor(prediction_map, cv2.COLOR_GRAY2RGB)
    rgb_prediction_no_boundaries = rgb_prediction.copy()
    rgb_prediction[boundary_map == 255] = [255, 0, 0]

    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.resize(target_image, (width, height), interpolation=cv2.INTER_NEAREST)
    target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_GRAY2RGB).astype(np.uint8)
    _mark_points(target_image_rgb, train_data, width, height, (255, 0, 0))
    _mark_points(target_image_rgb, val_data, width, height, (0, 0, 255))

    combined_image = np.hstack((rgb_prediction, rgb_prediction_no_boundaries, target_image_rgb))
    text_x = 2 * width + 10
    text_y = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_lines = [
        f'Epoch: {_epoch_from_path(output_path)}',
        f'Train Loss: {train_loss:.4f}' if train_loss is not None else None,
        f'Val Loss: {val_loss:.4f}' if val_loss is not None else None,
        f'Shape: {network_shape_str}' if network_shape_str is not None else None,
        f'Seed: {random_seed}' if random_seed is not None else None,
        f'Points: {num_points}' if num_points is not None else None,
        f'LR: {learning_rate}' if learning_rate is not None else None,
    ]
    if optimizer is not None:
        text_lines.append(str(optimizer))
    current_y = text_y
    for line in filter(None, text_lines):
        cv2.putText(combined_image, line, (text_x, current_y), font, 0.9, (0, 0, 255), 1, cv2.LINE_AA)
        current_y += 35
    if is_constant:
        cv2.putText(combined_image, f'WARNING: Constant output ({outputs.min():.6f})', (text_x, current_y), font, 0.9, (0, 0, 255), 1, cv2.LINE_AA)
    if image_write_params is None:
        cv2.imwrite(output_path, combined_image)
    else:
        cv2.imwrite(output_path, combined_image, image_write_params)


def generate_kernel_smoothed_image(train_data, image_shape, sigma=3.0):
    height, width = image_shape
    density = np.zeros((height, width), dtype=np.float32)
    values = np.zeros((height, width), dtype=np.float32)
    for x_norm, y_norm, value in train_data:
        x, y = model_to_pixel_coords(x_norm, y_norm, width, height)
        if 0 <= x < width and 0 <= y < height:
            density[y, x] += 1
            values[y, x] += value
    mask = density > 0
    values[mask] /= density[mask]
    smoothed_density = gaussian_filter(density, sigma=sigma)
    smoothed_values = gaussian_filter(values, sigma=sigma)
    significant_density = smoothed_density > 0.01 * smoothed_density.max()
    result = np.zeros_like(smoothed_values)
    result[significant_density] = smoothed_values[significant_density] / smoothed_density[significant_density]
    if result.max() > result.min():
        result = (result - result.min()) / (result.max() - result.min())
    return result


def save_kernel_smoothed_image(train_data, image_shape, output_path, original_image_path, sigma=3.0, logger=None):
    smoothed_image = generate_kernel_smoothed_image(train_data, image_shape, sigma)
    smoothed_image_8bit = (smoothed_image * 255).astype(np.uint8)
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.resize(original_image, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    smoothed_rgb = cv2.cvtColor(smoothed_image_8bit, cv2.COLOR_GRAY2RGB)
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    density = np.zeros(image_shape, dtype=np.float32)
    for x_norm, y_norm, _ in train_data:
        x, y = model_to_pixel_coords(x_norm, y_norm, image_shape[1], image_shape[0])
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            density[y, x] += 1
    smoothed_density = gaussian_filter(density, sigma=sigma)
    threshold = 0.005 * smoothed_density.max() if smoothed_density.max() > 0 else 0
    initialized_mask = smoothed_density > threshold
    smoothed_rgb[~initialized_mask] = [180, 255, 255]
    combined_image = np.hstack((smoothed_rgb, original_rgb))
    plt.figure(figsize=(12, 6))
    plt.imshow(combined_image)
    plt.text(image_shape[1] // 2, 20, 'Kernel Smoothed', color='white', ha='center', fontsize=12)
    plt.text(image_shape[1] + image_shape[1] // 2, 20, 'Original Image', color='white', ha='center', fontsize=12)
    plt.text(image_shape[1] // 2, image_shape[0] - 20, f'Sigma: {sigma}', color='white', ha='center', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    if logger:
        logger.debug(f"Kernel smoothed image saved to: {output_path}")


def visualize_classifier(network, data, train_data, val_data, image_shape, output_path, target_image_path, logger, train_loss=None, val_loss=None, network_shape_str=None, random_seed=None, epoch=None, num_points=None, learning_rate=None, optimizer=None, momentum=None, chunk_size=None, final_activation=None, image_write_params=None, empty_cache_each_chunk=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    train_inputs = torch.tensor(train_data[:, :-1], dtype=torch.float32).to(device)
    train_targets = train_data[:, -1].astype(int)
    with torch.no_grad():
        train_outputs, _ = network(train_inputs)
        train_predictions = torch.argmax(train_outputs, dim=1).cpu().numpy()
    misclassified_count = int(np.sum(train_predictions != train_targets))
    total_train = len(train_targets)

    height, width = image_shape
    outputs, activation_hashes = _run_full_grid(network, data, chunk_size, logger, empty_cache_each_chunk=empty_cache_each_chunk)
    class_predictions = np.argmax(outputs, axis=1)
    is_constant = len(np.unique(class_predictions)) == 1
    if is_constant:
        _handle_constant_hash(activation_hashes, logger, 'Network outputs are exactly constant (all same class). This indicates a serious problem with the network.')

    activation_map = np.zeros((height, width), dtype=np.uint64)
    red_map = np.zeros((height, width), dtype=np.float32)
    green_map = np.zeros((height, width), dtype=np.float32)
    blue_map = np.zeros((height, width), dtype=np.float32)
    class_map = np.zeros((height, width), dtype=np.int32) - 1
    valid_x, valid_y, valid = _valid_pixel_coords(data, width, height)
    valid_outputs = outputs[valid]
    activation_map[valid_y, valid_x] = activation_hashes[valid]
    red_map[valid_y, valid_x] = np.clip(valid_outputs[:, 0] * 255, 0, 255)
    green_map[valid_y, valid_x] = np.clip(valid_outputs[:, 1] * 255, 0, 255)
    blue_map[valid_y, valid_x] = np.clip(valid_outputs[:, 2] * 255, 0, 255)
    class_map[valid_y, valid_x] = class_predictions[valid]

    red_neuron_rgb = cv2.cvtColor(red_map.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    green_neuron_rgb = cv2.cvtColor(green_map.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    blue_neuron_rgb = cv2.cvtColor(blue_map.astype(np.uint8), cv2.COLOR_GRAY2RGB)
    class_prediction_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    class_prediction_rgb[class_map == 0] = [255, 0, 0]
    class_prediction_rgb[class_map == 1] = [0, 255, 0]
    class_prediction_rgb[class_map == 2] = [0, 0, 255]

    boundary_map = _activation_boundary_map(activation_map)
    class_with_boundaries = class_prediction_rgb.copy()
    class_with_boundaries[boundary_map == 255] = [255, 255, 255]

    target_image = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    target_image_rgb = cv2.resize(target_image, (width, height), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
    _mark_points(target_image_rgb, train_data, width, height, (255, 0, 255))
    _mark_points(target_image_rgb, val_data, width, height, (0, 255, 255))
    if misclassified_count == 0:
        frame_thickness = 5
        target_image_rgb[:frame_thickness, :] = [0, 255, 0]
        target_image_rgb[-frame_thickness:, :] = [0, 255, 0]
        target_image_rgb[:, :frame_thickness] = [0, 255, 0]
        target_image_rgb[:, -frame_thickness:] = [0, 255, 0]

    top_row = np.hstack((red_neuron_rgb, blue_neuron_rgb, class_with_boundaries))
    bottom_row = np.hstack((green_neuron_rgb, class_prediction_rgb, target_image_rgb))
    combined_image = np.vstack((top_row, bottom_row))

    font = cv2.FONT_HERSHEY_SIMPLEX
    for label_text, x, y in [
        ('Red Neuron', width // 2, 15),
        ('Blue Neuron', width + width // 2, 15),
        ('Predictions + Boundaries', 2 * width + width // 2, 15),
        ('Green Neuron', width // 2, height + 15),
        ('Class Predictions', width + width // 2, height + 15),
        ('Original + Training', 2 * width + width // 2, height + 15),
    ]:
        text_size = cv2.getTextSize(label_text, font, 0.5, 1)[0]
        cv2.putText(combined_image, label_text, (x - text_size[0] // 2, y), font, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    text_x = 2 * width + 10
    current_y = height + 40
    text_lines = [
        f'Epoch: {_epoch_from_path(output_path)}',
        f'Train Loss: {train_loss:.4f}' if train_loss is not None else None,
        f'Val Loss: {val_loss:.4f}' if val_loss is not None else None,
        f'Misclassified: {misclassified_count}/{total_train}',
        f'Shape: {network_shape_str}' if network_shape_str is not None else None,
        f'Seed: {random_seed}' if random_seed is not None else None,
        f'Points: {num_points}' if num_points is not None else None,
        f'LR: {learning_rate}' if learning_rate is not None else None,
        f'Final Act: {final_activation}' if final_activation is not None else None,
    ]
    if optimizer is not None:
        text_lines.append(str(optimizer))
    for line in filter(None, text_lines):
        cv2.putText(combined_image, line, (text_x, current_y), font, 0.9, (255, 255, 0), 1, cv2.LINE_AA)
        current_y += 25
    if is_constant:
        cv2.putText(combined_image, f'WARNING: Constant class ({class_predictions[0]})', (text_x, current_y), font, 0.9, (255, 255, 0), 1, cv2.LINE_AA)

    combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)
    if image_write_params is None:
        cv2.imwrite(output_path, combined_image_bgr)
    else:
        cv2.imwrite(output_path, combined_image_bgr, image_write_params)
    return misclassified_count, total_train
