import os
import imageio
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import math
import cv2
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import random
import base64
import sys
import argparse
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
import logging
from datetime import datetime
import time
import subprocess

# Configure logging
def setup_logger(log_file=None):
    """Set up and configure the logger with optional file output.
    
    Args:
        log_file: Optional path to a log file. If provided, logs will be written to this file
                 in addition to the console output.
    
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger('polytope_nn')
    logger.setLevel(logging.INFO)
    
    # Prevent adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create console handler and set level
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Add file handler if log file is specified
    if log_file:
        try:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(logging.INFO)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.info(f"Logging to file: {log_file}")
        except Exception as e:
            logger.error(f"Failed to set up file logging to {log_file}: {str(e)}")
    
    return logger

# Initialize logger without file logging yet - we'll update it in parse_arguments()
logger = setup_logger()

### Neural Network Definition ###
class PolytopeNet(nn.Module):
    def __init__(self, input_dim, layer_sizes, debug=False):
        super(PolytopeNet, self).__init__()
        layers = []
        self.activation_layers = []
        self.debug = debug

        # Save the random state
        rng_state = torch.get_rng_state()
        
        prev_dim = input_dim
        for i, size in enumerate(layer_sizes):
            # Create linear layer
            linear_layer = nn.Linear(prev_dim, size)
            # Initialize weights using He initialization
            nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
            # Initialize biases to small positive values
            nn.init.constant_(linear_layer.bias, 0.1)
            layers.append(linear_layer)
            
            # Use LeakyReLU with a small negative slope (0.01 is the default)
            leaky_relu_layer = nn.LeakyReLU(negative_slope=0.01)
            layers.append(leaky_relu_layer)
            self.activation_layers.append(leaky_relu_layer)
            
            # Register a buffer for the random hash values for this layer's activations
            self.register_buffer(f'hash_coeffs_{i}', torch.randint(1, 2**31-1, (size,), dtype=torch.int64))
            prev_dim = size

        # Create and initialize output layer
        output_layer = nn.Linear(prev_dim, 1)
        nn.init.kaiming_normal_(output_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(output_layer.bias, 0.1)
        layers.append(output_layer)
        
        self.network = nn.Sequential(*layers)

        # Restore the random state
        torch.set_rng_state(rng_state)

    def forward(self, x):
        # Unified path for both training and visualization
        polytope_hash = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        current = x
        layer_idx = 0

        for i, layer in enumerate(self.network):
            current = layer(current)
            if isinstance(layer, nn.LeakyReLU):
                # --- Polytope Hash Calculation ---
                activation_pattern = (current > 0)
                hash_coeffs = getattr(self, f'hash_coeffs_{layer_idx}')
                polytope_hash += (activation_pattern * hash_coeffs).sum(dim=1)
                layer_idx += 1

                # --- Debug Logic (guarded to prevent graph breaks) ---
                if self.debug:
                    # This block does not run when debug is False,
                    # allowing torch.compile to ignore it.
                    dead_neurons = (current == 0).float().mean().item()
                    logger.debug(f"Layer {i//2} LeakyReLU: dead neurons = {dead_neurons:.2%}, range = [{current.min():.6f}, {current.max():.6f}]")
            
            elif self.debug and isinstance(layer, nn.Linear):
                 logger.debug(f"Layer {i//2} Linear: range = [{current.min():.6f}, {current.max():.6f}]")

        return current, polytope_hash


### Data Preprocessing ###
def preprocess_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not open image: {image_path}")
    image = image.astype(np.float32) / 255.0
    height, width = image.shape
    data = [(j / width, i / height, image[i, j]) for i in range(height) for j in range(width)]
    return np.array(data)

def preprocess_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    data = []
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        t_norm = frame_idx / frame_count
        for i in range(height):
            for j in range(width):
                data.append((j / width, i / height, t_norm, gray_frame[i, j]))
        frame_idx += 1
    cap.release()
    return np.array(data)

### Sampling ###
def sample_data(data, train_size, val_size):
    if train_size + val_size > len(data):
        raise ValueError("Requested training + validation size exceeds dataset size.")
    np.random.shuffle(data)
    return data[:train_size], data[train_size:train_size + val_size]


def train_network(network, optimizer, train_data, val_data, epochs, batch_size, learning_rate, output_dir, network_shape_b64, random_seed, network_shape_str, debug=False, args=None):
    """Train the network and save visualizations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    
    # Convert data to tensors and move to device
    train_inputs = torch.tensor(train_data[:, :-1], dtype=torch.float32).to(device)
    train_targets = torch.tensor(train_data[:, -1], dtype=torch.float32).unsqueeze(1).to(device)
    val_inputs = torch.tensor(val_data[:, :-1], dtype=torch.float32).to(device)
    val_targets = torch.tensor(val_data[:, -1], dtype=torch.float32).unsqueeze(1).to(device)
    
    # Create data loaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Define loss function
    criterion = nn.MSELoss()
    
    # Training loop
    for epoch in range(epochs):
        network.train()
        total_loss = 0
        num_batches = 0
        
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            outputs, _ = network(batch_inputs)
            loss = criterion(outputs, batch_targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        network.eval()
        with torch.no_grad():
            val_outputs, _ = network(val_inputs)
            val_loss = criterion(val_outputs, val_targets).item()
        
        if debug:
            logger.debug(f"Epoch {epoch+1}/{epochs}")
            logger.debug(f"Training Loss: {avg_train_loss:.6f}")
            logger.debug(f"Validation Loss: {val_loss:.6f}")
    
    # Return the final loss values
    return avg_train_loss, val_loss

# Global variables to track constant outputs
constant_activation_map = None
constant_output_counter = 0
CONSTANT_OUTPUT_THRESHOLD = 30  # Number of consecutive constant outputs before aborting

# Global variables to track network parameters and loss history
network_param_history = {}  # Dictionary to track network parameter hashes
loss_history = []  # List to track loss values over epochs
MAX_LOSS_HISTORY = 200  # Maximum number of epochs to track

def hash_network_params(network):
    """Create a hash of the network parameters."""
    param_hash = 0
    for param in network.parameters():
        # Convert parameters to bytes and hash them
        param_bytes = param.detach().cpu().numpy().tobytes()
        param_hash = hash((param_hash, hash(param_bytes)))
    return param_hash

def check_optimization_loop(network, epoch):
    """Check if the network parameters have been seen before."""
    global network_param_history
    
    # Hash the current network parameters
    param_hash = hash_network_params(network)
    
    # Check if we've seen these parameters before
    if param_hash in network_param_history:
        last_seen_epoch = network_param_history[param_hash]
        logger.warning(f"Network parameters at epoch {epoch} match those from epoch {last_seen_epoch}.")
        logger.warning("The optimization is looping. Aborting training.")
        return True
    
    # Update the history
    network_param_history[param_hash] = epoch
    return False

def check_loss_stagnation(current_loss, epoch, network_outputs=None):
    """Check if the loss has stagnated or is increasing AND network outputs are constant."""
    global loss_history
    
    # Add current loss to history
    loss_history.append(current_loss)
    
    # Keep only the last MAX_LOSS_HISTORY epochs
    if len(loss_history) > MAX_LOSS_HISTORY:
        loss_history = loss_history[-MAX_LOSS_HISTORY:]
    
    # Need at least 30 epochs of history to make a decision
    if len(loss_history) < 30:
        return False
    
    # Calculate average losses over different periods
    avg_30 = sum(loss_history[-30:]) / 30
    avg_100 = sum(loss_history[-100:]) / min(100, len(loss_history))
    avg_200 = sum(loss_history[-200:]) / min(200, len(loss_history))
    
    # Check if current loss is greater than the minimum achieved in the last 30, 100, and 200 epochs
    min_30 = min(loss_history[-30:])
    min_100 = min(loss_history[-100:]) if len(loss_history) >= 100 else min_30
    min_200 = min(loss_history[-200:]) if len(loss_history) >= 200 else min_100
    
    # Check if current loss is greater than the minimum achieved
    loss_stagnated = False
    if current_loss > min_30 and current_loss > min_100 and current_loss > min_200:
        # Check if any two of the three averages are below the current loss
        if (avg_30 < current_loss and avg_100 < current_loss) or \
           (avg_30 < current_loss and avg_200 < current_loss) or \
           (avg_100 < current_loss and avg_200 < current_loss):
            loss_stagnated = True
    
    # Check if network outputs are constant
    outputs_constant = False
    if network_outputs is not None:
        outputs_constant = network_outputs.min() == network_outputs.max()
    
    # Only abort if both conditions are met
    if loss_stagnated and outputs_constant:
        logger.warning(f"Loss has stagnated AND network outputs are constant.")
        logger.info(f"Current loss: {current_loss:.6f}")
        logger.info(f"Average loss (last 30 epochs): {avg_30:.6f}")
        logger.info(f"Average loss (last 100 epochs): {avg_100:.6f}")
        logger.info(f"Average loss (last 200 epochs): {avg_200:.6f}")
        logger.info(f"Minimum loss (last 30 epochs): {min_30:.6f}")
        logger.info(f"Minimum loss (last 100 epochs): {min_100:.6f}")
        logger.info(f"Minimum loss (last 200 epochs): {min_200:.6f}")
        logger.info(f"Network output range: [{network_outputs.min():.6f}, {network_outputs.max():.6f}]")
        logger.warning("Aborting training due to loss stagnation and constant outputs.")
        return True
    
    return False

def visualize_decision_boundary_with_predictions(network, data, train_data, val_data, image_shape, output_path, target_image_path, train_loss=None, val_loss=None, network_shape_str=None, random_seed=None, epoch=None, num_points=None, learning_rate=None, optimizer=None, momentum=None, chunk_size=None):
    global constant_activation_map, constant_output_counter
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)  # Ensure network is on GPU

    height, width = image_shape
    activation_map = np.zeros((height, width), dtype=np.uint64)
    prediction_map = np.zeros((height, width), dtype=np.float32)
    boundary_map = np.zeros((height, width), dtype=np.uint8)

    # Process data in chunks to avoid memory issues
    if chunk_size is None:
        chunk_size = 128 * 1024  # Default: 128k points per chunk
    total_points = data.shape[0]
    num_chunks = (total_points + chunk_size - 1) // chunk_size  # Ceiling division
    
    logger.debug(f"Processing {total_points} points in {num_chunks} chunks of {chunk_size}")
    
    # Initialize arrays to collect results from all chunks
    all_outputs = []
    all_activation_hashes = []
    
    try:
        with torch.no_grad():
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = min(start_idx + chunk_size, total_points)
                
                logger.debug(f"Processing chunk {i+1}/{num_chunks} (points {start_idx} to {end_idx})")
                
                # Process this chunk
                chunk_data = data[start_idx:end_idx]
                chunk_inputs = torch.tensor(chunk_data[:, :-1], dtype=torch.float32).to(device)
                
                # Forward pass for this chunk
                chunk_outputs, chunk_activation_hashes = network(chunk_inputs)
                
                # Collect results (move to CPU immediately to free GPU memory)
                all_outputs.append(chunk_outputs.detach().cpu().numpy())
                all_activation_hashes.append(chunk_activation_hashes.detach().cpu().numpy())
                
                # Explicitly free GPU memory
                del chunk_inputs, chunk_outputs, chunk_activation_hashes
                torch.cuda.empty_cache()
    
    except Exception as e:
        logger.error(f"Error during network forward pass: {str(e)}")
        logger.error(f"Chunk index: {i}, Chunk size: {chunk_size}")
        logger.error(f"Chunk input shape: {chunk_inputs.shape if 'chunk_inputs' in locals() else 'N/A'}")
        raise ValueError(f"Network forward pass failed: {str(e)}")
    
    # Combine results from all chunks
    outputs = np.concatenate(all_outputs).flatten()
    activation_hashes = np.concatenate(all_activation_hashes)
    
    # Debug network outputs
    logger.debug(f"Network outputs (len {len(outputs)}) range: min={outputs.min():.6f}, max={outputs.max():.6f}")
    
    # Check if outputs are constant
    is_constant = outputs.min() == outputs.max()
    if is_constant:
        logger.warning("Network outputs are exactly constant (min == max). This indicates a serious problem with the network.")
        
        # Create a hash of the activation map for comparison
        current_activation_hash = hash(activation_hashes.tobytes())
        
        # Check if this is the first constant output
        if constant_activation_map is None:
            constant_activation_map = current_activation_hash
            constant_output_counter = 1
            logger.warning(f"First constant output detected. Counter: {constant_output_counter}")
        # Check if activation map is the same as before
        elif constant_activation_map == current_activation_hash:
            constant_output_counter += 1
            logger.warning(f"Same constant output detected. Counter: {constant_output_counter}")
            
            # Abort if we've seen the same constant output too many times
            if constant_output_counter >= CONSTANT_OUTPUT_THRESHOLD:
                logger.error(f"Network has produced the same constant output for {CONSTANT_OUTPUT_THRESHOLD} consecutive iterations.")
                logger.error("Aborting due to network failure.")
                dump_network_weights(network, "weights.txt")
                exit(1)
        else:
            # Reset counter if activation map changed
            constant_output_counter = 1
            constant_activation_map = current_activation_hash
            logger.warning("Constant output but different activation map. Resetting counter.")

    
    # Vectorized point processing
    x_coords = np.round(data[:, 0] * (width - 1)).astype(int)
    y_coords = np.round(data[:, 1] * (height - 1)).astype(int)
    
    # Ensure coordinates are in bounds
    valid_indices = (0 <= x_coords) & (x_coords < width) & (0 <= y_coords) & (y_coords < height)
    if not np.any(valid_indices):
        logger.error("No valid points were found for processing. Check data format and dimensions.")
        raise ValueError("No valid points were found for processing. Check data format and dimensions.")
    
    # Filter valid data points
    valid_x = x_coords[valid_indices]
    valid_y = y_coords[valid_indices]
    valid_hashes = activation_hashes[valid_indices]
    valid_outputs = outputs[valid_indices]
    
    # Process valid points vectorized
    points_processed = len(valid_x)
    logger.debug(f"Processing {points_processed} valid data points")
    
    # Assign to activation map
    activation_map[valid_y, valid_x] = valid_hashes
    
    # Scale and assign to prediction map (vectorized)
    scaled_outputs = np.clip(valid_outputs * 255, 0, 255)
    prediction_map[valid_y, valid_x] = scaled_outputs

    logger.debug(f"Prediction map range before uint8: min={prediction_map.min():.6f}, max={prediction_map.max():.6f}")

    # Detect decision boundaries using vectorized operations
    # Create shifted versions of the activation map
    shifts = [
        (0, 1),   # right
        (0, -1),  # left
        (1, 0),   # down
        (-1, 0)   # up
    ]
    
    # Create a mask where any neighbor differs from the center
    center = activation_map[1:-1, 1:-1]
    boundary_mask = np.zeros((height-2, width-2), dtype=bool)
    
    for dy, dx in shifts:
        neighbor = activation_map[1+dy:height-1+dy, 1+dx:width-1+dx]
        boundary_mask = boundary_mask | (neighbor != center)
    
    # Apply boundary mask to the boundary map
    boundary_map[1:-1, 1:-1][boundary_mask] = 255

    # Convert prediction map to RGB
    prediction_map = prediction_map.astype(np.uint8)
    logger.debug(f"Prediction map range after uint8: min={prediction_map.min()}, max={prediction_map.max()}")

    rgb_prediction = cv2.cvtColor(prediction_map, cv2.COLOR_GRAY2RGB)
    
    # Create version without boundaries
    rgb_prediction_no_boundaries = rgb_prediction.copy()

    # Overlay decision boundaries in red
    rgb_prediction[boundary_map == 255] = [255, 0, 0]

    target_image = cv2.imread(target_image_path, cv2.IMREAD_GRAYSCALE)
    target_image = cv2.resize(target_image, (width, height), interpolation=cv2.INTER_NEAREST)  # Preserve sharp edges
    target_image_rgb = cv2.cvtColor(target_image, cv2.COLOR_GRAY2RGB).astype(np.uint8)  # Ensure correct format

    # Apply training points (pure red) - vectorized
    train_x = np.round(train_data[:, 0] * width).astype(int)
    train_y = np.round(train_data[:, 1] * height).astype(int)
    valid_train = (0 <= train_x) & (train_x < width) & (0 <= train_y) & (train_y < height)
    if np.any(valid_train):
        target_image_rgb[train_y[valid_train], train_x[valid_train]] = (255, 0, 0)
    
    # Apply validation points (pure blue) - vectorized
    val_x = np.round(val_data[:, 0] * width).astype(int)
    val_y = np.round(val_data[:, 1] * height).astype(int)
    valid_val = (0 <= val_x) & (val_x < width) & (0 <= val_y) & (val_y < height)
    if np.any(valid_val):
        target_image_rgb[val_y[valid_val], val_x[valid_val]] = (0, 0, 255)
    
    # Concatenate three images side by side
    combined_image = np.hstack((rgb_prediction, rgb_prediction_no_boundaries, target_image_rgb))
    
    # Get epoch number from filename
    epoch_num = int(output_path.split('_epoch_')[1].split('.')[0])
    
    # Add text using cv2
    text_x = 2 * width + 10
    text_y = 30
    vertical_spacing = 30
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    font_color = (0, 0, 255)  # Red in BGR
    thickness = 1

    # List of text lines to add
    text_lines = [
        f'Epoch: {epoch_num}',
        f'Train Loss: {train_loss:.4f}' if train_loss is not None else None,
        f'Val Loss: {val_loss:.4f}' if val_loss is not None else None,
        f'Shape: {network_shape_str}' if network_shape_str is not None else None,
        f'Seed: {random_seed}' if random_seed is not None else None,
        f'Points: {num_points}' if num_points is not None else None,
        f'LR: {learning_rate}' if learning_rate is not None else None,
    ]
    
    if optimizer is not None:
        optimizer_text = f'Optimizer: {optimizer}'
        if optimizer == 'sgd_momentum' and momentum is not None:
            optimizer_text += f' (mom={momentum})'
        text_lines.append(optimizer_text)

    # Add text to the image
    current_y = text_y
    # Filter out None values from text_lines
    for line in filter(None, text_lines):
        cv2.putText(combined_image, line, (text_x, current_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        current_y += vertical_spacing

    # Add warning if outputs are constant
    if is_constant:
        warning_text = f'WARNING: Constant output ({outputs.min():.6f})'
        cv2.putText(combined_image, warning_text, (text_x, current_y), font, font_scale, font_color, thickness, cv2.LINE_AA)
        current_y += vertical_spacing
        if constant_output_counter > 1:
            counter_text = f'Constant counter: {constant_output_counter}/{CONSTANT_OUTPUT_THRESHOLD}'
            cv2.putText(combined_image, counter_text, (text_x, current_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Save the image
    cv2.imwrite(output_path, combined_image)
    
    # No need for timing here, it's done at a higher level

def generate_kernel_smoothed_image(train_data, image_shape, sigma=3.0):
    """
    Generate a kernel-smoothed image from training data points.
    
    Args:
        train_data: Array of (x_norm, y_norm, value) training points
        image_shape: Tuple of (height, width) for the output image
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Smoothed image as a numpy array
    """
    height, width = image_shape
    # Initialize empty image
    density = np.zeros((height, width), dtype=np.float32)
    values = np.zeros((height, width), dtype=np.float32)
    
    # Place training points on the grid
    for x_norm, y_norm, value in train_data:
        x, y = int(np.round(x_norm * (width-1))), int(np.round(y_norm * (height-1)))
        if 0 <= x < width and 0 <= y < height:
            density[y, x] += 1
            values[y, x] += value
    
    # Normalize values where density > 0
    mask = density > 0
    values[mask] /= density[mask]
    
    # Apply Gaussian smoothing to both density and values
    smoothed_density = gaussian_filter(density, sigma=sigma)
    smoothed_values = gaussian_filter(values, sigma=sigma)
    
    # Normalize the result where density is significant
    significant_density = smoothed_density > 0.01 * smoothed_density.max()
    result = np.zeros_like(smoothed_values)
    result[significant_density] = smoothed_values[significant_density] / smoothed_density[significant_density]
    
    # Normalize to 0-1 range
    if result.max() > result.min():
        result = (result - result.min()) / (result.max() - result.min())
    
    return result

def save_kernel_smoothed_image(train_data, image_shape, output_path, original_image_path, sigma=3.0):
    """
    Generate and save a kernel-smoothed image from training data points.
    
    Args:
        train_data: Array of (x_norm, y_norm, value) training points
        image_shape: Tuple of (height, width) for the output image
        output_path: Path to save the output image
        original_image_path: Path to the original image for comparison
        sigma: Standard deviation for Gaussian kernel
    """
    # Generate the smoothed image
    smoothed_image = generate_kernel_smoothed_image(train_data, image_shape, sigma)
    
    # Convert to 8-bit image
    smoothed_image_8bit = (smoothed_image * 255).astype(np.uint8)
    
    # Read the original image for comparison
    original_image = cv2.imread(original_image_path, cv2.IMREAD_GRAYSCALE)
    original_image = cv2.resize(original_image, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    
    # Create RGB versions of both images
    smoothed_rgb = cv2.cvtColor(smoothed_image_8bit, cv2.COLOR_GRAY2RGB)
    original_rgb = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    
    # Let's directly use the approach from generate_kernel_smoothed_image
    # Create a density map from training points
    density = np.zeros(image_shape, dtype=np.float32)
    
    # Place training points on the density map
    for x_norm, y_norm, _ in train_data:
        x, y = int(np.round(x_norm * (image_shape[1]-1))), int(np.round(y_norm * (image_shape[0]-1)))
        if 0 <= x < image_shape[1] and 0 <= y < image_shape[0]:
            density[y, x] += 1
    
    # Apply Gaussian smoothing to get smooth transitions
    smoothed_density = gaussian_filter(density, sigma=sigma)
    
    # Create a mask for significant density (areas with enough data points)
    # Using a low threshold to include more of the smoothed area
    threshold = 0.005 * smoothed_density.max() if smoothed_density.max() > 0 else 0
    initialized_mask = smoothed_density > threshold
    
    # Create light cyan background for uninitialized areas
    cyan_bg = np.zeros_like(smoothed_rgb)
    cyan_bg[:,:,0] = 180  # Red channel (lighter)
    cyan_bg[:,:,1] = 255  # Green channel (full)
    cyan_bg[:,:,2] = 255  # Blue channel (full)
    
    # Apply blue to uninitialized areas while keeping the smoothed values for initialized areas
    for y in range(image_shape[0]):
        for x in range(image_shape[1]):
            if not initialized_mask[y, x]:
                smoothed_rgb[y, x] = cyan_bg[y, x]
    
    # Combine images side by side
    combined_image = np.hstack((smoothed_rgb, original_rgb))
    
    # Add titles using matplotlib
    plt.figure(figsize=(12, 6))
    plt.imshow(combined_image)
    plt.text(image_shape[1] // 2, 20, "Kernel Smoothed", color='white', ha='center', fontsize=12)
    plt.text(image_shape[1] + image_shape[1] // 2, 20, "Original Image", color='white', ha='center', fontsize=12)
    plt.text(image_shape[1] // 2, image_shape[0] - 20, f"Sigma: {sigma}", color='white', ha='center', fontsize=10)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    logger.debug(f"Kernel smoothed image saved to: {output_path}")

def save_checkpoint(network, optimizer, epoch, output_dir, network_shape_b64, random_seed, loss_history_data=None, network_param_history_data=None, args=None):
    """Save a complete training checkpoint.

    Args:
        network: The neural network model
        optimizer: The optimizer instance
        epoch: Current epoch number
        output_dir: Directory to save checkpoint
        network_shape_b64: Base64 encoded network shape string
        random_seed: Random seed used for training
        loss_history_data: Loss history list for stagnation detection
        network_param_history_data: Network parameter history dict
        args: Command line arguments
    """
    seed_str = str(random_seed) if random_seed is not None else "none"
    checkpoint_path = os.path.join(output_dir, f"checkpoint_{network_shape_b64}_{seed_str}_epoch_{epoch:06d}.pt")

    checkpoint = {
        'epoch': epoch,
        'model_state_dict': network.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_type': args.optimizer if args else 'adam',
        'learning_rate': args.learning_rate if args else 0.001,
        'momentum': args.momentum if args and hasattr(args, 'momentum') else None,
        'pytorch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'loss_history': loss_history_data if loss_history_data is not None else [],
        'network_param_history': network_param_history_data if network_param_history_data is not None else {},
        'network_shape_b64': network_shape_b64,
        'random_seed': random_seed,
        'args': {
            'shape': args.shape if args else None,
            'points': args.points if args else 5000,
            'batch_size': args.batch_size if args else 1024,
            'learning_rate': args.learning_rate if args else 0.001,
            'optimizer': args.optimizer if args else 'adam',
            'momentum': args.momentum if args and hasattr(args, 'momentum') else 0.9,
        }
    }

    # Add CUDA RNG state if available
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
        checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()

    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path

def load_checkpoint(checkpoint_path, network, optimizer=None, restore_rng=True, restore_optimizer=True):
    """Load a training checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        network: The neural network model to load weights into
        optimizer: Optional optimizer to load state into
        restore_rng: Whether to restore RNG states
        restore_optimizer: Whether to restore optimizer state

    Returns:
        Dictionary containing checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    # Load model state
    network.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Model state loaded from epoch {checkpoint['epoch']}")

    # Load optimizer state if requested and provided
    if restore_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Optimizer state restored ({checkpoint.get('optimizer_type', 'unknown')})")

    # Restore RNG states if requested
    if restore_rng:
        if 'pytorch_rng_state' in checkpoint:
            torch.set_rng_state(checkpoint['pytorch_rng_state'])
        if 'numpy_rng_state' in checkpoint:
            np.random.set_state(checkpoint['numpy_rng_state'])
        if 'python_rng_state' in checkpoint:
            random.setstate(checkpoint['python_rng_state'])
        if torch.cuda.is_available():
            if 'cuda_rng_state' in checkpoint:
                torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
            if 'cuda_rng_state_all' in checkpoint:
                torch.cuda.set_rng_state_all(checkpoint['cuda_rng_state_all'])
        logger.info("RNG states restored")

    return checkpoint

def dump_network_weights(network, filename):
    """Dump network weights to a file in a human-readable format."""
    with open(filename, 'w') as f:
        # Count the number of linear layers to determine network architecture
        linear_layers = [m for m in network.network if isinstance(m, nn.Linear)]
        f.write(f"Network architecture: {[l.in_features for l in linear_layers] + [linear_layers[-1].out_features]}\n")
        f.write(f"Number of layers: {len(linear_layers)}\n\n")

        # Iterate through the sequential layers
        for i, module in enumerate(network.network):
            if isinstance(module, nn.Linear):
                f.write(f"Layer {i} (Linear):\n")
                f.write(f"  Shape: {module.weight.shape}\n")
                f.write(f"  Weight range: min={module.weight.min():.6f}, max={module.weight.max():.6f}\n")
                f.write(f"  Weight mean: {module.weight.mean():.6f}, std: {module.weight.std():.6f}\n")
                f.write(f"  Bias range: min={module.bias.min():.6f}, max={module.bias.max():.6f}\n")
                f.write(f"  Bias mean: {module.bias.mean():.6f}, std: {module.bias.std():.6f}\n\n")

                # Write weight matrix
                weight_matrix = module.weight.detach().cpu().numpy()
                for row in weight_matrix:
                    f.write("    " + " ".join(f"{x:.6f}" for x in row) + "\n")

                # Write bias vector
                f.write("\n  Bias vector:\n")
                bias_vector = module.bias.detach().cpu().numpy()
                f.write("    " + " ".join(f"{x:.6f}" for x in bias_vector) + "\n\n")
            elif isinstance(module, nn.LeakyReLU):
                f.write(f"Layer {i} (LeakyReLU activation)\n\n")

### Full Training and Visualization Pipeline ###
def full_pipeline(
    input_path,
    is_video=False,
    train_size=5000,
    val_size=1000,
    layer_sizes=[10, 10, 10, 10, 10, 10, 10, 10],
    epochs=10,
    batch_size=1024,
    learning_rate=0.001,
    output_dir="results",
    random_seed=None,
    network_shape_b64=None,
    network_shape_str=None,
    debug=False,
    args=None
):
    global loss_history, network_param_history

    # Start timing the entire training process
    training_start_time = time.time()
    logger.info(f"Starting full training pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Preprocess data
    if is_video:
        data = preprocess_video(input_path)
        height, width = 64, 64  # Assuming a fixed resolution for visualization
    else:
        data = preprocess_image(input_path)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        height, width = img.shape

    # Step 2: Sample training and validation data
    train_data, val_data = sample_data(data, train_size, val_size)

    # Handle checkpoint resumption
    start_epoch = 0
    checkpoint_data = None
    if args and args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # We'll load the checkpoint after creating the network

    # Generate and save kernel smoothed image before training (skip if resuming)
    seed_str = str(random_seed) if random_seed is not None else "none"
    smoothed_image_path = os.path.join(output_dir, f"{os.path.basename(input_path)}_{network_shape_b64}_{seed_str}-kernel-smoothed.png")
    sigma = args.smoothing_sigma if args and hasattr(args, 'smoothing_sigma') else 3.0

    if not (args and args.resume):
        save_kernel_smoothed_image(train_data, (height, width), smoothed_image_path, input_path, sigma=sigma)

    # Step 3: Initialize the network
    input_dim = 3 if is_video else 2
    network = PolytopeNet(input_dim, layer_sizes, debug=debug)
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params}")
    from torchinfo import summary
    summary(network, input_size=(1024, 2))

    # Apply torch.compile for acceleration if requested and available (PyTorch 2.0+)
    if args.use_compile and hasattr(torch, 'compile'):
        logger.info("Using torch.compile for network acceleration")
        network = torch.compile(network)
    elif args.use_compile and not hasattr(torch, 'compile'):
        logger.warning("torch.compile requested but not available. Requires PyTorch 2.0+")
        logger.info("Continuing without compilation")

    # Step 3b: Create optimizer (potentially overridden by checkpoint or resume args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    optimizer = create_optimizer(network, args)

    # Step 3c: Load checkpoint if resuming
    # Track current optimizer type and learning rate for visualization
    current_optimizer_type = args.optimizer
    current_learning_rate = args.learning_rate
    current_momentum = args.momentum if args.optimizer == 'sgd_momentum' else None

    if args and args.resume:
        # Determine whether to restore optimizer state
        restore_optimizer = (args.resume_optimizer is None)  # Only restore if not switching optimizers

        checkpoint_data = load_checkpoint(args.resume, network, optimizer,
                                         restore_rng=True, restore_optimizer=restore_optimizer)
        start_epoch = checkpoint_data['epoch'] + 1  # Resume from next epoch

        # Restore loss history and network param history for stagnation detection
        if 'loss_history' in checkpoint_data:
            loss_history = checkpoint_data['loss_history']
        if 'network_param_history' in checkpoint_data:
            network_param_history = checkpoint_data['network_param_history']

        # Override optimizer if requested
        if args.resume_optimizer:
            logger.info(f"Switching optimizer from {checkpoint_data.get('optimizer_type', 'unknown')} to {args.resume_optimizer}")
            optimizer = create_optimizer(network, args)  # Create new optimizer with new type
            current_optimizer_type = args.resume_optimizer
            current_momentum = args.momentum if args.resume_optimizer == 'sgd_momentum' else None

        # Override learning rate if requested
        if args.resume_lr:
            logger.info(f"Changing learning rate from {checkpoint_data.get('learning_rate', 'unknown')} to {args.resume_lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.resume_lr
            current_learning_rate = args.resume_lr

        logger.info(f"Resuming training from epoch {start_epoch}")
    
    # Step 4: Train and visualize decision boundaries
    boundary_frames = []
    save_interval = args.save_interval if args and hasattr(args, 'save_interval') else 1
    checkpoint_interval = args.checkpoint_interval if args and hasattr(args, 'checkpoint_interval') else None

    for epoch in range(start_epoch, epochs):
        train_loss, val_loss = train_network(network, optimizer, train_data, val_data, epochs=1, batch_size=batch_size, learning_rate=learning_rate, output_dir=output_dir, network_shape_b64=network_shape_b64, random_seed=random_seed, network_shape_str=network_shape_str, debug=debug, args=args)
        logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")

        # Check optimization problems on every epoch
        network.eval()
        with torch.no_grad():
            # Create tensors for validation inputs if needed
            if 'val_inputs' not in locals():
                val_inputs = torch.tensor(val_data[:, :-1], dtype=torch.float32).to(device)
            val_outputs, _ = network(val_inputs)
        
        # Check for optimization loops on every epoch
        if check_optimization_loop(network, epoch):
            dump_network_weights(network, os.path.join(output_dir, "weights_optimization_loop.txt"))
            logger.warning("Optimization is looping. Aborting training.")
            break
            
        # Check for loss stagnation on every epoch
        if check_loss_stagnation(train_loss, epoch, val_outputs):
            dump_network_weights(network, os.path.join(output_dir, "weights_stagnation.txt"))
            logger.warning("Loss has stagnated and outputs are constant. Aborting training.")
            break
        
        # Only visualize at save_interval or at the final epoch
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            epoch_str = "%06d" % (epoch + 1)
            seed_str = str(random_seed) if random_seed is not None else "none"
            output_path = os.path.join(output_dir, f"{os.path.basename(input_path)}_{network_shape_b64}_{seed_str}_epoch_{epoch_str}.png")
            
            # Time the visualization step without extra logs
            start_time = time.time()
            
            # Temporarily change the logging level for the visualization duration
            original_level = logger.level
            if not debug:
                logger.setLevel(logging.WARNING)  # Only show warnings and errors
            
            visualize_decision_boundary_with_predictions(
              network, data, train_data, val_data,
              (height, width), output_path, input_path,
              train_loss=train_loss, val_loss=val_loss,
              network_shape_str=network_shape_str,
              random_seed=random_seed,
              epoch=epoch,
              num_points=args.points if args else None,
              learning_rate=current_learning_rate if args else None,
              optimizer=current_optimizer_type if args else None,
              momentum=current_momentum,
              chunk_size=args.chunk_size if args and hasattr(args, 'chunk_size') else None
            )
            
            # Restore original logging level
            logger.setLevel(original_level)
            
            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Visualization @ epoch {epoch + 1}: {duration:.2f}s")
            #boundary_frames.append(imageio.imread(output_path))

        # Save checkpoint periodically if checkpoint_interval is set
        if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(network, optimizer, epoch, output_dir, network_shape_b64,
                          random_seed, loss_history_data=loss_history,
                          network_param_history_data=network_param_history, args=args)

    # Step 5: Save final checkpoint
    final_checkpoint_path = save_checkpoint(network, optimizer, epochs - 1, output_dir,
                                           network_shape_b64, random_seed,
                                           loss_history_data=loss_history,
                                           network_param_history_data=network_param_history, args=args)
    logger.info(f"Final checkpoint saved: {final_checkpoint_path}")

    # Step 6: Save video if processing a video input
    if is_video:
        video_output_path = os.path.join(output_dir, "boundary_evolution.mp4")
        imageio.mimsave(video_output_path, boundary_frames, fps=5)
        logger.info(f"Boundary evolution video saved at: {video_output_path}")

    # Step 7: Dump network weights
    weights_filename = os.path.join(output_dir, f"weights_{network_shape_b64}_{seed_str}.txt")
    dump_network_weights(network, weights_filename)

    # Step 8: Generate video from PNGs
    logger.info("Starting video generation...")
    base_filename = f"{os.path.basename(input_path)}_{network_shape_b64}_{seed_str}"
    glob_pattern = os.path.join(output_dir, f"{base_filename}_epoch_*.png")
    video_output_path = os.path.join(output_dir, f"{base_filename}.mp4")

    ffmpeg_command = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', '24',
        '-pattern_type', 'glob',
        '-i', glob_pattern,
        '-c:v', 'libx264',
        '-crf', '18',
        '-pix_fmt', 'yuv420p',
        video_output_path
    ]

    try:
        # Using f-string for the command for clarity and safety with trusted inputs
        logger.info(f"Running ffmpeg command: {' '.join(ffmpeg_command)}")
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        logger.info("ffmpeg stdout:\n" + result.stdout)
        logger.info("ffmpeg stderr:\n" + result.stderr)
        logger.info(f"Video generated successfully: {video_output_path}")
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        logger.error("ffmpeg command failed.")
        logger.error("ffmpeg stdout:\n" + e.stdout)
        logger.error("ffmpeg stderr:\n" + e.stderr)

def parse_arguments():
    """Parse command line arguments using argparse."""
    parser = argparse.ArgumentParser(description='Neural network experiments for image processing.')
    
    # Required arguments
    parser.add_argument('--input', '-i', required=True, help='Path to the input image')
    parser.add_argument('--shape', '-s', required=True, help='Neural network shape (e.g., "[5]*40")')
    parser.add_argument('--epochs', '-e', type=int, required=True, help='Number of training epochs')
    
    # Optional arguments
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    parser.add_argument('--points', '-p', type=int, default=5000, help='Number of points to sample (default: 5000)')
    parser.add_argument('--batch-size', '-b', type=int, default=1024, help='Batch size for training (default: 1024)')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--output-dir', '-o', default='results', help='Output directory for results (default: results)')
    parser.add_argument('--optimizer', type=str, default='adam', 
                      choices=['adam', 'sgd', 'sgd_momentum', 'rmsprop'],
                      help='Optimizer to use (default: adam)')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum factor for SGD with momentum (default: 0.9)')
    parser.add_argument('--smoothing-sigma', type=float, default=3.0,
                      help='Sigma parameter for Gaussian kernel smoothing (default: 3.0)')
    parser.add_argument('--use-compile', action='store_true',
                      help='Use torch.compile to accelerate the neural network (requires PyTorch 2.0+)')
    parser.add_argument('--save-interval', type=int, default=1,
                      help='Save boundary visualization every N epochs (default: 1)')
    parser.add_argument('--chunk-size', type=int, default=128*1024,
                      help='Number of points to process at once during visualization (default: 128K)')
    parser.add_argument('--log-file', type=str, default=None,
                      help='Path to the log file (default: log to console only)')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint file to resume training from')
    parser.add_argument('--resume-optimizer', type=str, default=None,
                      choices=['adam', 'sgd', 'sgd_momentum', 'rmsprop'],
                      help='Override optimizer when resuming (default: use checkpoint optimizer)')
    parser.add_argument('--resume-lr', type=float, default=None,
                      help='Override learning rate when resuming (default: use checkpoint LR)')
    parser.add_argument('--checkpoint-interval', type=int, default=None,
                      help='Save checkpoint every N epochs (default: only save final checkpoint)')

    args = parser.parse_args()
    
    # Set up logging with file if specified
    global logger
    if args.log_file:
        logger = setup_logger(args.log_file)
        
    # Set debug level if requested
    if args.debug:
        logger.setLevel(logging.DEBUG)
        for handler in logger.handlers:
            handler.setLevel(logging.DEBUG)
    
    # Create a string that includes all relevant parameters for the filename
    params_str = f"{args.shape}_{args.points}_{args.batch_size}_{args.learning_rate}_{args.optimizer}"
    if args.optimizer == 'sgd_momentum':
        params_str += f"_mom{args.momentum}"
    
    # Encode the parameters string
    network_shape_b64 = base64.b64encode(params_str.encode()).decode()
    
    return args, network_shape_b64

def create_optimizer(network, args):
    """Create the specified optimizer with appropriate parameters.

    If resuming with a different optimizer, uses resume_optimizer instead of optimizer.
    """
    # Use resume_optimizer if specified (for switching optimizers), otherwise use optimizer
    optimizer_type = args.resume_optimizer if hasattr(args, 'resume_optimizer') and args.resume_optimizer else args.optimizer
    learning_rate = args.resume_lr if hasattr(args, 'resume_lr') and args.resume_lr else args.learning_rate

    if optimizer_type == 'adam':
        return optim.Adam(network.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd':
        return optim.SGD(network.parameters(), lr=learning_rate)
    elif optimizer_type == 'sgd_momentum':
        momentum = args.momentum if hasattr(args, 'momentum') else 0.9
        return optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(network.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

# Parse arguments
args, network_shape_b64 = parse_arguments()

# Log the start of execution with important parameters
logger.info("=== Starting polytope-viz-nn training ===")
logger.info(f"Input image: {args.input}")
logger.info(f"Network shape: {args.shape}")
logger.info(f"Training epochs: {args.epochs}")
logger.info(f"Points: {args.points}")
logger.info(f"Optimizer: {args.optimizer}")
logger.info(f"Learning rate: {args.learning_rate}")
logger.info(f"Output directory: {args.output_dir}")
if args.seed is not None:
    logger.info(f"Random seed: {args.seed}")

# Set random seed if provided
if args.seed is not None:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info("Random seed set for all libraries")

# Run the full pipeline
try:
    full_pipeline(
        input_path=args.input,
        is_video=False,
        train_size=args.points,
        layer_sizes=eval(args.shape),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=args.output_dir,
        random_seed=args.seed,
        network_shape_b64=network_shape_b64,
        network_shape_str=args.shape,
        debug=args.debug,
        args=args  # Pass the full args object to access optimizer settings
    )
    logger.info("=== Training completed successfully ===")
except Exception as e:
    logger.error(f"Training failed with error: {str(e)}")
    logger.exception("Exception details:")
    raise

