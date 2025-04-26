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

### Neural Network Definition ###
class ReLUNetwork(nn.Module):
    def __init__(self, input_dim, layer_sizes, debug=False):
        super(ReLUNetwork, self).__init__()
        layers = []
        self.activation_layers = []
        self.debug = debug

        prev_dim = input_dim
        for size in layer_sizes:
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
            prev_dim = size

        # Create and initialize output layer
        output_layer = nn.Linear(prev_dim, 1)
        nn.init.kaiming_normal_(output_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(output_layer.bias, 0.1)
        layers.append(output_layer)
        
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        activations = []
        # Debug: Track the range of values at each layer
        if self.debug:
            print("\nLayer activations:")
        current = x
        for i, layer in enumerate(self.network):
            current = layer(current)
            if isinstance(layer, nn.LeakyReLU):
                # Count dead neurons (output = 0)
                dead_neurons = (current == 0).float().mean().item()
                if self.debug:
                    print(f"Layer {i//2} LeakyReLU: dead neurons = {dead_neurons:.2%}, range = [{current.min():.6f}, {current.max():.6f}]")
                activations.append((current > 0).int())  # Activation tracking
            elif isinstance(layer, nn.Linear) and self.debug:
                print(f"Layer {i//2} Linear: range = [{current.min():.6f}, {current.max():.6f}]")

        activation_pattern = torch.cat(activations, dim=1) if activations else None
        return current, activation_pattern

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


def train_network(network, train_data, val_data, epochs, batch_size, learning_rate, output_dir, network_shape_b64, random_seed, network_shape_str, debug=False, args=None):
    """Train the network and save visualizations."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)
    
    # Create optimizer based on command line arguments
    optimizer = create_optimizer(network, args)
    
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
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Training Loss: {avg_train_loss:.6f}")
            print(f"Validation Loss: {val_loss:.6f}")
    
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
        print(f"WARNING: Network parameters at epoch {epoch} match those from epoch {last_seen_epoch}.")
        print("The optimization is looping. Aborting training.")
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
        print(f"WARNING: Loss has stagnated AND network outputs are constant.")
        print(f"Current loss: {current_loss:.6f}")
        print(f"Average loss (last 30 epochs): {avg_30:.6f}")
        print(f"Average loss (last 100 epochs): {avg_100:.6f}")
        print(f"Average loss (last 200 epochs): {avg_200:.6f}")
        print(f"Minimum loss (last 30 epochs): {min_30:.6f}")
        print(f"Minimum loss (last 100 epochs): {min_100:.6f}")
        print(f"Minimum loss (last 200 epochs): {min_200:.6f}")
        print(f"Network output range: [{network_outputs.min():.6f}, {network_outputs.max():.6f}]")
        print("Aborting training due to loss stagnation and constant outputs.")
        return True
    
    return False

def visualize_decision_boundary_with_predictions(network, data, train_data, val_data, image_shape, output_path, target_image_path, train_loss=None, val_loss=None, network_shape_str=None, random_seed=None, epoch=None, num_points=None, learning_rate=None, optimizer=None, momentum=None):
    global constant_activation_map, constant_output_counter
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)  # Ensure network is on GPU

    height, width = image_shape
    activation_map = np.zeros((height, width), dtype=np.uint64)
    prediction_map = np.zeros((height, width), dtype=np.float32)
    boundary_map = np.zeros((height, width), dtype=np.uint8)

    # Move input data to GPU
    inputs = torch.tensor(data[:, :-1], dtype=torch.float32).to(device)
    try:
        with torch.no_grad():
            outputs, activation_patterns = network(inputs)
    except Exception as e:
        print(f"Error during network forward pass: {str(e)}")
        print(f"Input shape: {inputs.shape}")
        print(f"Input range: min={inputs.min():.6f}, max={inputs.max():.6f}")
        print(f"Input dtype: {inputs.dtype}")
        raise ValueError(f"Network forward pass failed: {str(e)}")

    # Move back to CPU only when necessary
    outputs = outputs.detach().cpu().numpy().flatten()
    activation_patterns = activation_patterns.detach().cpu().numpy()

    # Debug network outputs
    print(f"Network outputs (len {len(outputs)}) range: min={outputs.min():.6f}, max={outputs.max():.6f}")
    
    # Check if outputs are constant
    is_constant = outputs.min() == outputs.max()
    if is_constant:
        print("WARNING: Network outputs are exactly constant (min == max). This indicates a serious problem with the network.")
        
        # Create a hash of the activation map for comparison
        current_activation_hash = hash(activation_patterns.tobytes())
        
        # Check if this is the first constant output
        if constant_activation_map is None:
            constant_activation_map = current_activation_hash
            constant_output_counter = 1
            print(f"First constant output detected. Counter: {constant_output_counter}")
        # Check if activation map is the same as before
        elif constant_activation_map == current_activation_hash:
            constant_output_counter += 1
            print(f"Same constant output detected. Counter: {constant_output_counter}")
            
            # Abort if we've seen the same constant output too many times
            if constant_output_counter >= CONSTANT_OUTPUT_THRESHOLD:
                print(f"ERROR: Network has produced the same constant output for {CONSTANT_OUTPUT_THRESHOLD} consecutive iterations.")
                print("Aborting due to network failure.")
                dump_network_weights(network, "weights.txt")
                exit(1)
        else:
            # Reset counter if activation map changed
            constant_output_counter = 1
            constant_activation_map = current_activation_hash
            print("Constant output but different activation map. Resetting counter.")

    # Vectorized point processing
    x_coords = np.round(data[:, 0] * (width - 1)).astype(int)
    y_coords = np.round(data[:, 1] * (height - 1)).astype(int)
    
    # Ensure coordinates are in bounds
    valid_indices = (0 <= x_coords) & (x_coords < width) & (0 <= y_coords) & (y_coords < height)
    if not np.any(valid_indices):
        raise ValueError("No valid points were found for processing. Check data format and dimensions.")
    
    # Filter valid data points
    valid_x = x_coords[valid_indices]
    valid_y = y_coords[valid_indices]
    valid_activations = activation_patterns[valid_indices]
    valid_outputs = outputs[valid_indices]
    
    # Process valid points vectorized
    points_processed = len(valid_x)
    print(f"Processing {points_processed} valid data points")
    
    # Create hashes for activation patterns (this part must still be done per-point)
    activation_hashes = np.zeros(points_processed, dtype=np.uint64)
    for i in range(points_processed):
        activation_hashes[i] = hash(tuple(valid_activations[i])) & 0xFFFFFFFFFFFFFFFF
    
    # Assign to activation map
    activation_map[valid_y, valid_x] = activation_hashes
    
    # Scale and assign to prediction map (vectorized)
    scaled_outputs = np.clip(valid_outputs * 255, 0, 255)
    prediction_map[valid_y, valid_x] = scaled_outputs

    print(f"Prediction map range before uint8: min={prediction_map.min():.6f}, max={prediction_map.max():.6f}")

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
    print(f"Prediction map range after uint8: min={prediction_map.min()}, max={prediction_map.max()}")

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
    
    # Create figure and display image
    plt.figure(figsize=(18, 6))
    plt.imshow(combined_image, interpolation='nearest', vmin=0, vmax=255)
    
    # Add text using matplotlib with increased vertical spacing
    # Position text at the start of the third image (2*width pixels from the left)
    text_x = 2 * width + 10  # Start 10 pixels into the third image
    text_y = 30  # Initial y position
    vertical_spacing = 30  # Spacing between lines
    plt.text(text_x, text_y, f'Epoch: {epoch_num}', color='red', fontsize=10)
    if train_loss is not None:
        plt.text(text_x, text_y + vertical_spacing, f'Train Loss: {train_loss:.4f}', color='red', fontsize=10)
    if val_loss is not None:
        plt.text(text_x, text_y + 2*vertical_spacing, f'Val Loss: {val_loss:.4f}', color='red', fontsize=10)
    if network_shape_str is not None:
        plt.text(text_x, text_y + 3*vertical_spacing, f'Shape: {network_shape_str}', color='red', fontsize=10)
    if random_seed is not None:
        plt.text(text_x, text_y + 4*vertical_spacing, f'Seed: {random_seed}', color='red', fontsize=10)
    
    # Add additional parameters
    if num_points is not None:
        plt.text(text_x, text_y + 5*vertical_spacing, f'Points: {num_points}', color='red', fontsize=10)
    if learning_rate is not None:
        plt.text(text_x, text_y + 6*vertical_spacing, f'LR: {learning_rate}', color='red', fontsize=10)
    if optimizer is not None:
        optimizer_text = f'Optimizer: {optimizer}'
        if optimizer == 'sgd_momentum' and momentum is not None:
            optimizer_text += f' (mom={momentum})'
        plt.text(text_x, text_y + 7*vertical_spacing, optimizer_text, color='red', fontsize=10)
    
    # Add warning if outputs are constant
    if is_constant:
        warning_text = f'WARNING: Constant output ({outputs.min():.6f})'
        plt.text(text_x, text_y + 8*vertical_spacing, warning_text, color='red', fontsize=10, fontweight='bold')
        if constant_output_counter > 1:
            counter_text = f'Constant counter: {constant_output_counter}/{CONSTANT_OUTPUT_THRESHOLD}'
            plt.text(text_x, text_y + 9*vertical_spacing, counter_text, color='red', fontsize=10)
    
    plt.axis("off")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.show()
    plt.close()

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
    
    print(f"Kernel smoothed image saved to: {output_path}")

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

    # Generate and save kernel smoothed image before training
    seed_str = str(random_seed) if random_seed is not None else "none"
    smoothed_image_path = os.path.join(output_dir, f"{os.path.basename(input_path)}_{network_shape_b64}_{seed_str}-kernel-smoothed.png")
    sigma = args.smoothing_sigma if args and hasattr(args, 'smoothing_sigma') else 3.0
    save_kernel_smoothed_image(train_data, (height, width), smoothed_image_path, input_path, sigma=sigma)
    
    # Step 3: Initialize the network
    input_dim = 3 if is_video else 2
    network = ReLUNetwork(input_dim, layer_sizes, debug=debug)
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")
    from torchinfo import summary
    summary(network, input_size=(1024, 2))
    
    # Apply torch.compile for acceleration if requested and available (PyTorch 2.0+)
    if args.use_compile and hasattr(torch, 'compile'):
        print("Using torch.compile for network acceleration")
        network = torch.compile(network)
    elif args.use_compile and not hasattr(torch, 'compile'):
        print("Warning: torch.compile requested but not available. Requires PyTorch 2.0+")
        print("Continuing without compilation")
    
    # Step 4: Train and visualize decision boundaries
    boundary_frames = []
    save_interval = args.save_interval if args and hasattr(args, 'save_interval') else 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        train_loss, val_loss = train_network(network, train_data, val_data, epochs=1, batch_size=batch_size, learning_rate=learning_rate, output_dir=output_dir, network_shape_b64=network_shape_b64, random_seed=random_seed, network_shape_str=network_shape_str, debug=debug, args=args)

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
            print("Optimization is looping. Aborting training.")
            break
            
        # Check for loss stagnation on every epoch
        if check_loss_stagnation(train_loss, epoch, val_outputs):
            dump_network_weights(network, os.path.join(output_dir, "weights_stagnation.txt"))
            print("Loss has stagnated and outputs are constant. Aborting training.")
            break
        
        # Only visualize at save_interval or at the final epoch
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            epoch_str = "%04d" % (epoch + 1)
            seed_str = str(random_seed) if random_seed is not None else "none"
            output_path = os.path.join(output_dir, f"{os.path.basename(input_path)}_{network_shape_b64}_{seed_str}_epoch_{epoch_str}.png")
            visualize_decision_boundary_with_predictions(
              network, data, train_data, val_data,
              (height, width), output_path, input_path,
              train_loss=train_loss, val_loss=val_loss,
              network_shape_str=network_shape_str,
              random_seed=random_seed,
              epoch=epoch,
              num_points=args.points if args else None,
              learning_rate=args.learning_rate if args else None,
              optimizer=args.optimizer if args else None,
              momentum=args.momentum if args and args.optimizer == 'sgd_momentum' else None
            )
            #boundary_frames.append(imageio.imread(output_path))

    # Step 5: Save video if processing a video input
    if is_video:
        video_output_path = os.path.join(output_dir, "boundary_evolution.mp4")
        imageio.mimsave(video_output_path, boundary_frames, fps=5)
        print(f"Boundary evolution video saved at: {video_output_path}")

    # Step 6: Dump network weights
    weights_filename = os.path.join(output_dir, f"weights_{network_shape_b64}_{seed_str}.txt")
    dump_network_weights(network, weights_filename)

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
    
    args = parser.parse_args()
    
    # Create a string that includes all relevant parameters for the filename
    params_str = f"{args.shape}_{args.points}_{args.learning_rate}_{args.optimizer}"
    if args.optimizer == 'sgd_momentum':
        params_str += f"_mom{args.momentum}"
    
    # Encode the parameters string
    network_shape_b64 = base64.b64encode(params_str.encode()).decode()
    
    return args, network_shape_b64

def create_optimizer(network, args):
    """Create the specified optimizer with appropriate parameters."""
    if args.optimizer == 'adam':
        return optim.Adam(network.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd':
        return optim.SGD(network.parameters(), lr=args.learning_rate)
    elif args.optimizer == 'sgd_momentum':
        return optim.SGD(network.parameters(), lr=args.learning_rate, momentum=args.momentum)
    elif args.optimizer == 'rmsprop':
        return optim.RMSprop(network.parameters(), lr=args.learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")

# Parse arguments
args, network_shape_b64 = parse_arguments()

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

# Run the full pipeline
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

