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
    logger = logging.getLogger('polytope_classifier')
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

### Kolmogorov Regularization Utilities ###

def quantize_weights(weights, num_bins, weight_range):
    """Quantize continuous weight values into discrete bins.

    Args:
        weights: Tensor of weight values
        num_bins: Number of bins for quantization
        weight_range: Tuple of (min_val, max_val) for clipping

    Returns:
        Tensor of bin indices (long type for cross-entropy)
    """
    min_val, max_val = weight_range
    # Clip weights to range
    clipped = torch.clamp(weights, min_val, max_val)
    # Normalize to [0, 1]
    normalized = (clipped - min_val) / (max_val - min_val)
    # Convert to bin indices [0, num_bins-1]
    bins = (normalized * (num_bins - 1)).long()
    return bins


def gaussian_nll_loss(actual, predicted_mean, predicted_log_var):
    """Compute Gaussian negative log-likelihood loss.

    This measures how likely the actual weights are under a predicted
    Gaussian distribution, which is equivalent to the encoding cost.

    Args:
        actual: Actual weight values
        predicted_mean: Predicted mean of Gaussian
        predicted_log_var: Predicted log variance

    Returns:
        Scalar loss value
    """
    # NLL = 0.5 * (log(2*pi*var) + (x - mu)^2 / var)
    # Using log_var for numerical stability
    var = torch.exp(predicted_log_var)
    loss = 0.5 * (predicted_log_var + ((actual - predicted_mean) ** 2) / var + math.log(2 * math.pi))
    return loss.mean()


def laplacian_nll_loss(actual, predicted_location, predicted_log_scale):
    """Compute Laplacian negative log-likelihood loss.

    Laplacian distribution is appropriate for sparse/peaked weight distributions.

    Args:
        actual: Actual weight values
        predicted_location: Predicted location parameter (like mean)
        predicted_log_scale: Predicted log scale parameter

    Returns:
        Scalar loss value
    """
    # NLL = log(2*scale) + |x - location| / scale
    scale = torch.exp(predicted_log_scale)
    loss = predicted_log_scale + math.log(2) + torch.abs(actual - predicted_location) / scale
    return loss.mean()


### Weight Predictor Network for Kolmogorov Regularization ###

class WeightPredictorNetwork(nn.Module):
    """A small network that predicts weights of the main network.

    This network takes normalized weight indices (layer, neuron, weight_position)
    and predicts the actual weight value. The prediction error serves as a proxy
    for Kolmogorov complexity - more predictable weights have lower complexity.

    Supports multiple loss types:
    - 'mse': Mean squared error (simple regression)
    - 'cross_entropy': Quantized cross-entropy (information-theoretic)
    - 'gaussian_nll': Gaussian negative log-likelihood
    - 'laplacian_nll': Laplacian negative log-likelihood
    """

    def __init__(self, layer_sizes, loss_type='mse', num_bins=256, debug=False):
        """Initialize weight predictor network.

        Args:
            layer_sizes: List of hidden layer sizes
            loss_type: Type of loss function to use
            num_bins: Number of bins for cross_entropy quantization
            debug: Enable debug logging
        """
        super(WeightPredictorNetwork, self).__init__()
        self.debug = debug
        self.loss_type = loss_type
        self.num_bins = num_bins

        layers = []

        # Input: 3 features (layer_id_norm, neuron_id_norm, weight_position_norm)
        input_dim = 3
        prev_dim = input_dim

        # Hidden layers
        for i, size in enumerate(layer_sizes):
            linear_layer = nn.Linear(prev_dim, size)
            nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.constant_(linear_layer.bias, 0.1)
            layers.append(linear_layer)
            layers.append(nn.LeakyReLU(negative_slope=0.01))
            prev_dim = size

        self.hidden_layers = nn.Sequential(*layers)

        # Output layer depends on loss type
        if loss_type == 'cross_entropy':
            # Output: probability distribution over bins
            self.output_layer = nn.Linear(prev_dim, num_bins)
            # LogSoftmax for numerical stability with NLLLoss
        elif loss_type in ['gaussian_nll', 'laplacian_nll']:
            # Output: [location/mean, log_scale/log_variance]
            self.output_layer = nn.Linear(prev_dim, 2)
        else:  # mse
            # Output: single predicted value
            self.output_layer = nn.Linear(prev_dim, 1)

        # Initialize output layer
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, weight_positions):
        """Predict weight values or distributions from normalized positions.

        Args:
            weight_positions: Tensor of shape (N, 3) with normalized indices

        Returns:
            Predictions (format depends on loss_type):
            - mse: (N, 1) predicted values
            - cross_entropy: (N, num_bins) log-probabilities
            - gaussian_nll/laplacian_nll: (N, 2) distribution parameters
        """
        hidden = self.hidden_layers(weight_positions)
        output = self.output_layer(hidden)

        if self.loss_type == 'cross_entropy':
            # Apply log_softmax for cross-entropy
            return F.log_softmax(output, dim=-1)
        else:
            return output


def compute_kolmogorov_loss(weight_predictor, main_network, device, weight_range=(-3.0, 3.0)):
    """Compute the Kolmogorov complexity proxy loss.

    This measures how well the weight predictor can predict the main network's
    weights, serving as a proxy for the compressibility/complexity of the weights.

    Args:
        weight_predictor: WeightPredictorNetwork instance
        main_network: Main PolytopeNet instance
        device: torch device for computation
        weight_range: Tuple of (min_val, max_val) for weight clipping in cross_entropy

    Returns:
        Scalar loss value
    """
    # Get weight enumeration from main network
    weight_positions, actual_weights = main_network.get_weight_enumeration()
    weight_positions = weight_positions.to(device)
    actual_weights = actual_weights.to(device)

    # Get predictions from weight predictor
    predictions = weight_predictor(weight_positions)

    loss_type = weight_predictor.loss_type

    if loss_type == 'mse':
        return F.mse_loss(predictions.squeeze(), actual_weights)

    elif loss_type == 'cross_entropy':
        # For cross-entropy with continuous targets, we need a differentiable approach
        # Option 1: Use soft targets (differentiable)
        min_val, max_val = weight_range
        clipped = torch.clamp(actual_weights, min_val, max_val)
        normalized = (clipped - min_val) / (max_val - min_val)
        # Continuous bin index (float, not long!)
        continuous_bins = normalized * (weight_predictor.num_bins - 1)

        # Create soft one-hot encoding using temperature-based softmax
        # This is differentiable w.r.t. actual_weights
        temperature = 0.1  # Lower = closer to hard assignment
        bin_indices = torch.arange(weight_predictor.num_bins, device=actual_weights.device, dtype=torch.float32)
        distances = -torch.abs(continuous_bins.unsqueeze(1) - bin_indices.unsqueeze(0)) / temperature
        soft_targets = F.softmax(distances, dim=1)

        # Compute cross-entropy with soft targets (differentiable!)
        # predictions are log-probabilities, soft_targets are probabilities
        return -(soft_targets * predictions).sum(dim=1).mean()

    elif loss_type == 'gaussian_nll':
        predicted_mean = predictions[:, 0]
        predicted_log_var = predictions[:, 1]
        return gaussian_nll_loss(actual_weights, predicted_mean, predicted_log_var)

    elif loss_type == 'laplacian_nll':
        predicted_location = predictions[:, 0]
        predicted_log_scale = predictions[:, 1]
        return laplacian_nll_loss(actual_weights, predicted_location, predicted_log_scale)

    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


### Neural Network Definition ###
class PolytopeNet(nn.Module):
    def __init__(self, input_dim, layer_sizes, final_activation='relu', debug=False):
        super(PolytopeNet, self).__init__()
        layers = []
        self.activation_layers = []
        self.debug = debug
        self.final_activation = final_activation

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

        # Create and initialize output layer (3 neurons for RGB)
        output_layer = nn.Linear(prev_dim, 3)
        nn.init.kaiming_normal_(output_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(output_layer.bias, 0.1)
        layers.append(output_layer)

        # Add final activation layer based on configuration
        if final_activation == 'relu':
            layers.append(nn.ReLU())
        elif final_activation == 'sigmoid':
            layers.append(nn.Sigmoid())
        elif final_activation == 'leaky_relu':
            layers.append(nn.LeakyReLU(negative_slope=0.01))
        else:
            raise ValueError(f"Unknown final activation: {final_activation}")

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
            if isinstance(layer, nn.LeakyReLU) and layer_idx < len(self.activation_layers):
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

    def get_weight_enumeration(self):
        """Enumerate all weights in the network with normalized positions.

        Returns:
            Tuple of (weight_positions, actual_weights) where:
            - weight_positions: Tensor of shape (N, 3) with normalized (layer, neuron, position)
            - actual_weights: Tensor of shape (N,) with actual weight values
        """
        # Check if we have cached position indices
        if not hasattr(self, '_weight_position_cache'):
            self._build_weight_position_cache()

        # Extract actual weight values efficiently (stays on GPU)
        actual_weights = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                # Flatten weight matrix and bias, keep on device
                actual_weights.append(module.weight.flatten())
                actual_weights.append(module.bias)

        # Concatenate all weights into single tensor (stays on GPU)
        actual_weights = torch.cat(actual_weights)

        return self._weight_position_cache.clone(), actual_weights

    def _build_weight_position_cache(self):
        """Build cached weight position indices (called once)."""
        weight_positions = []

        # Count total layers for normalization
        linear_layers = [m for m in self.network if isinstance(m, nn.Linear)]
        num_layers = len(linear_layers)

        layer_counter = 0
        for module in self.network:
            if isinstance(module, nn.Linear):
                # Normalize layer index
                layer_id_norm = layer_counter / max(num_layers - 1, 1)
                device = module.weight.device

                # Process weight matrix - vectorized
                out_features, in_features = module.weight.shape

                # Create meshgrid for neuron and weight indices
                neuron_indices = torch.arange(out_features, device=device, dtype=torch.float32)
                weight_indices = torch.arange(in_features, device=device, dtype=torch.float32)

                # Normalize indices
                neuron_norm = neuron_indices / max(out_features - 1, 1)
                weight_norm = weight_indices / max(in_features - 1, 1)

                # Create all combinations efficiently
                neuron_grid, weight_grid = torch.meshgrid(neuron_norm, weight_norm, indexing='ij')
                layer_grid = torch.full_like(neuron_grid, layer_id_norm)

                # Stack into (N, 3) tensor
                positions = torch.stack([
                    layer_grid.flatten(),
                    neuron_grid.flatten(),
                    weight_grid.flatten()
                ], dim=1)
                weight_positions.append(positions)

                # Process bias vector - vectorized
                bias_positions = torch.stack([
                    torch.full((out_features,), layer_id_norm, device=device),
                    neuron_norm,
                    torch.ones(out_features, device=device)  # Bias gets position 1.0
                ], dim=1)
                weight_positions.append(bias_positions)

                layer_counter += 1

        # Concatenate all positions and cache
        self._weight_position_cache = torch.cat(weight_positions, dim=0)
        logger.debug(f"Built weight position cache: {self._weight_position_cache.shape[0]} weights")


### Data Preprocessing ###
def preprocess_image(image_path):
    """Preprocess RGB classification image.

    Reads an RGB image and extracts pixels that are exactly red (#FF0000),
    green (#00FF00), or blue (#0000FF). Converts to normalized coordinates
    with class labels.

    Args:
        image_path: Path to RGB image file

    Returns:
        Array of (x_norm, y_norm, class_label) where class_label is 0=red, 1=green, 2=blue
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not open image: {image_path}")

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]

    data = []
    for i in range(height):
        for j in range(width):
            r, g, b = image[i, j]

            # Check for exact color matches
            if r == 255 and g == 0 and b == 0:
                # Red pixel -> class 0
                data.append((j / width, i / height, 0))
            elif r == 0 and g == 255 and b == 0:
                # Green pixel -> class 1
                data.append((j / width, i / height, 1))
            elif r == 0 and g == 0 and b == 255:
                # Blue pixel -> class 2
                data.append((j / width, i / height, 2))

    if len(data) == 0:
        raise ValueError(f"No red, green, or blue pixels found in image: {image_path}")

    logger.info(f"Found {len(data)} colored pixels in image")
    return np.array(data)

def preprocess_video(video_path):
    """Video processing not supported for classification task."""
    raise NotImplementedError("Video processing is not supported for RGB classification")

### Sampling ###
def sample_data(data, train_size, val_size):
    if train_size + val_size > len(data):
        raise ValueError("Requested training + validation size exceeds dataset size.")
    np.random.shuffle(data)
    return data[:train_size], data[train_size:train_size + val_size]


def train_network(network, optimizer, train_data, val_data, epochs, batch_size, learning_rate, output_dir, network_shape_b64, random_seed, network_shape_str, debug=False, args=None, weight_predictor=None, weight_predictor_optimizer=None):
    """Train the network and save visualizations.

    Args:
        weight_predictor: Optional WeightPredictorNetwork for Kolmogorov regularization
        weight_predictor_optimizer: Optional optimizer for weight predictor
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)

    # Move weight predictor to device if provided
    if weight_predictor is not None:
        weight_predictor.to(device)

    # Convert data to tensors and move to device
    train_inputs = torch.tensor(train_data[:, :-1], dtype=torch.float32).to(device)
    train_targets = torch.tensor(train_data[:, -1], dtype=torch.long).to(device)  # Long for classification
    val_inputs = torch.tensor(val_data[:, :-1], dtype=torch.float32).to(device)
    val_targets = torch.tensor(val_data[:, -1], dtype=torch.long).to(device)  # Long for classification

    # Create data loaders
    train_dataset = TensorDataset(train_inputs, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Define loss function - CrossEntropyLoss for classification
    criterion = nn.CrossEntropyLoss()

    # Get Kolmogorov regularization parameters
    kolmogorov_weight = args.kolmogorov_weight if args and hasattr(args, 'kolmogorov_weight') else 0.0
    use_kolmogorov = weight_predictor is not None and kolmogorov_weight > 0.0

    # Training loop
    for epoch in range(epochs):
        network.train()
        if weight_predictor is not None:
            weight_predictor.train()

        total_loss = 0
        total_task_loss = 0
        total_kolmogorov_loss = 0
        num_batches = 0

        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            if weight_predictor_optimizer is not None:
                weight_predictor_optimizer.zero_grad()

            # Forward pass for main task
            outputs, _ = network(batch_inputs)
            task_loss = criterion(outputs, batch_targets)

            # Compute combined loss
            if use_kolmogorov:
                # Compute Kolmogorov regularization loss
                weight_range = (args.kolmogorov_weight_min, args.kolmogorov_weight_max) if args and hasattr(args, 'kolmogorov_weight_min') else (-3.0, 3.0)
                kolmogorov_loss = compute_kolmogorov_loss(weight_predictor, network, device, weight_range)
                loss = task_loss + kolmogorov_weight * kolmogorov_loss
                total_kolmogorov_loss += kolmogorov_loss.item()
            else:
                loss = task_loss

            # Backward pass
            loss.backward()
            optimizer.step()
            if weight_predictor_optimizer is not None:
                weight_predictor_optimizer.step()

            total_loss += loss.item()
            total_task_loss += task_loss.item()
            num_batches += 1

        avg_combined_loss = total_loss / num_batches  # This is what SGD actually minimizes
        avg_task_loss = total_task_loss / num_batches
        avg_kolmogorov_loss = total_kolmogorov_loss / num_batches if use_kolmogorov else 0.0

        # Validation
        network.eval()
        if weight_predictor is not None:
            weight_predictor.eval()

        with torch.no_grad():
            val_outputs, _ = network(val_inputs)
            val_loss = criterion(val_outputs, val_targets).item()

        if debug:
            logger.debug(f"Epoch {epoch+1}/{epochs}")
            logger.debug(f"Combined Loss: {avg_combined_loss:.6f}")
            logger.debug(f"Task Loss: {avg_task_loss:.6f}")
            if use_kolmogorov:
                logger.debug(f"Kolmogorov Loss: {avg_kolmogorov_loss:.6f}")
            logger.debug(f"Validation Loss: {val_loss:.6f}")

    # Return all loss values for logging
    return avg_task_loss, val_loss, avg_kolmogorov_loss if use_kolmogorov else 0.0, avg_combined_loss

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

    # Check if network outputs are constant (all predictions the same class)
    outputs_constant = False
    if network_outputs is not None:
        # For classification, check if all predictions are the same
        predictions = torch.argmax(network_outputs, dim=1)
        outputs_constant = len(torch.unique(predictions)) == 1

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
        logger.warning("Aborting training due to loss stagnation and constant outputs.")
        return True

    return False

def visualize_decision_boundary_with_predictions(network, data, train_data, val_data, image_shape, output_path, target_image_path, train_loss=None, val_loss=None, network_shape_str=None, random_seed=None, epoch=None, num_points=None, learning_rate=None, optimizer=None, momentum=None, chunk_size=None, final_activation=None):
    """Visualize classification results in a 2x3 grid layout."""
    global constant_activation_map, constant_output_counter

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    network.to(device)  # Ensure network is on GPU

    # Calculate misclassifications on training data
    train_inputs = torch.tensor(train_data[:, :-1], dtype=torch.float32).to(device)
    train_targets = train_data[:, -1].astype(int)
    with torch.no_grad():
        train_outputs, _ = network(train_inputs)
        train_predictions = torch.argmax(train_outputs, dim=1).cpu().numpy()
    misclassified_count = np.sum(train_predictions != train_targets)
    total_train = len(train_targets)

    height, width = image_shape
    activation_map = np.zeros((height, width), dtype=np.uint64)

    # Three prediction maps for RGB neurons
    red_map = np.zeros((height, width), dtype=np.float32)
    green_map = np.zeros((height, width), dtype=np.float32)
    blue_map = np.zeros((height, width), dtype=np.float32)

    # Class prediction map
    class_map = np.zeros((height, width), dtype=np.int32) - 1  # -1 for uninitialized

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
    outputs = np.concatenate(all_outputs)  # Shape: (N, 3) for RGB outputs
    activation_hashes = np.concatenate(all_activation_hashes)

    # Debug network outputs
    logger.debug(f"Network outputs shape: {outputs.shape}, range: min={outputs.min():.6f}, max={outputs.max():.6f}")

    # Get class predictions
    class_predictions = np.argmax(outputs, axis=1)

    # Check if outputs are constant
    is_constant = len(np.unique(class_predictions)) == 1
    if is_constant:
        logger.warning("Network outputs are exactly constant (all same class). This indicates a serious problem with the network.")

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
    valid_class_predictions = class_predictions[valid_indices]

    # Process valid points vectorized
    points_processed = len(valid_x)
    logger.debug(f"Processing {points_processed} valid data points")

    # Assign to activation map
    activation_map[valid_y, valid_x] = valid_hashes

    # Assign individual neuron outputs (normalized to 0-255)
    red_map[valid_y, valid_x] = np.clip(valid_outputs[:, 0] * 255, 0, 255)
    green_map[valid_y, valid_x] = np.clip(valid_outputs[:, 1] * 255, 0, 255)
    blue_map[valid_y, valid_x] = np.clip(valid_outputs[:, 2] * 255, 0, 255)

    # Assign class predictions
    class_map[valid_y, valid_x] = valid_class_predictions

    # Convert neuron output maps to uint8
    red_map = red_map.astype(np.uint8)
    green_map = green_map.astype(np.uint8)
    blue_map = blue_map.astype(np.uint8)

    # Create RGB versions for neuron outputs (grayscale converted to RGB)
    red_neuron_rgb = cv2.cvtColor(red_map, cv2.COLOR_GRAY2RGB)
    green_neuron_rgb = cv2.cvtColor(green_map, cv2.COLOR_GRAY2RGB)
    blue_neuron_rgb = cv2.cvtColor(blue_map, cv2.COLOR_GRAY2RGB)

    # Create class prediction image (saturated colors only)
    class_prediction_rgb = np.zeros((height, width, 3), dtype=np.uint8)
    class_prediction_rgb[class_map == 0] = [255, 0, 0]  # Red
    class_prediction_rgb[class_map == 1] = [0, 255, 0]  # Green
    class_prediction_rgb[class_map == 2] = [0, 0, 255]  # Blue

    # Detect decision boundaries using vectorized operations
    boundary_map = np.zeros((height, width), dtype=np.uint8)
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

    # Create version with boundaries overlaid on class predictions
    class_with_boundaries = class_prediction_rgb.copy()
    class_with_boundaries[boundary_map == 255] = [255, 255, 255]  # White boundaries

    # Read and prepare original image
    target_image = cv2.imread(target_image_path, cv2.IMREAD_COLOR)
    target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB)
    target_image = cv2.resize(target_image, (width, height), interpolation=cv2.INTER_NEAREST)  # Preserve sharp edges
    target_image_rgb = target_image.astype(np.uint8)

    # Apply training points (magenta) - vectorized
    train_x = np.round(train_data[:, 0] * width).astype(int)
    train_y = np.round(train_data[:, 1] * height).astype(int)
    valid_train = (0 <= train_x) & (train_x < width) & (0 <= train_y) & (train_y < height)
    if np.any(valid_train):
        target_image_rgb[train_y[valid_train], train_x[valid_train]] = (255, 0, 255)  # Magenta

    # Apply validation points (cyan) - vectorized
    val_x = np.round(val_data[:, 0] * width).astype(int)
    val_y = np.round(val_data[:, 1] * height).astype(int)
    valid_val = (0 <= val_x) & (val_x < width) & (0 <= val_y) & (val_y < height)
    if np.any(valid_val):
        target_image_rgb[val_y[valid_val], val_x[valid_val]] = (0, 255, 255)  # Cyan

    # Add green frame to bottom-right image if no misclassifications
    frame_thickness = 5
    if misclassified_count == 0:
        # Add green frame around target_image_rgb
        target_image_rgb[:frame_thickness, :] = [0, 255, 0]  # Top
        target_image_rgb[-frame_thickness:, :] = [0, 255, 0]  # Bottom
        target_image_rgb[:, :frame_thickness] = [0, 255, 0]  # Left
        target_image_rgb[:, -frame_thickness:] = [0, 255, 0]  # Right

    # Create 2x3 grid layout
    # Top row: red neuron, blue neuron, class with boundaries
    # Bottom row: green neuron, class predictions, original with training points
    top_row = np.hstack((red_neuron_rgb, blue_neuron_rgb, class_with_boundaries))
    bottom_row = np.hstack((green_neuron_rgb, class_prediction_rgb, target_image_rgb))
    combined_image = np.vstack((top_row, bottom_row))

    # Get epoch number from filename
    epoch_num = int(output_path.split('_epoch_')[1].split('.')[0])

    # Add text labels to each panel
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (255, 255, 255)  # White
    thickness = 1

    # Labels for each panel
    labels = [
        ("Red Neuron", width // 2, 15),
        ("Blue Neuron", width + width // 2, 15),
        ("Predictions + Boundaries", 2 * width + width // 2, 15),
        ("Green Neuron", width // 2, height + 15),
        ("Class Predictions", width + width // 2, height + 15),
        ("Original + Training", 2 * width + width // 2, height + 15),
    ]

    for label_text, x, y in labels:
        text_size = cv2.getTextSize(label_text, font, font_scale, thickness)[0]
        text_x = x - text_size[0] // 2
        cv2.putText(combined_image, label_text, (text_x, y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Add metadata text in the bottom right area
    text_x = 2 * width + 10
    text_y = height + 40
    vertical_spacing = 25
    font_scale_meta = 0.9
    font_color_meta = (255, 255, 0)  # Yellow

    # List of text lines to add
    text_lines = [
        f'Epoch: {epoch_num}',
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
        optimizer_text = f'Optimizer: {optimizer}'
        if optimizer == 'sgd_momentum' and momentum is not None:
            optimizer_text += f' (mom={momentum})'
        text_lines.append(optimizer_text)

    # Add text to the image
    current_y = text_y
    # Filter out None values from text_lines
    for line in filter(None, text_lines):
        cv2.putText(combined_image, line, (text_x, current_y), font, font_scale_meta, font_color_meta, thickness, cv2.LINE_AA)
        current_y += vertical_spacing

    # Add warning if outputs are constant
    if is_constant:
        warning_text = f'WARNING: Constant class ({class_predictions[0]})'
        cv2.putText(combined_image, warning_text, (text_x, current_y), font, font_scale_meta, font_color_meta, thickness, cv2.LINE_AA)
        current_y += vertical_spacing
        if constant_output_counter > 1:
            counter_text = f'Constant counter: {constant_output_counter}/{CONSTANT_OUTPUT_THRESHOLD}'
            cv2.putText(combined_image, counter_text, (text_x, current_y), font, font_scale_meta, font_color_meta, thickness, cv2.LINE_AA)

    # Convert RGB to BGR for cv2.imwrite
    combined_image_bgr = cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR)

    # Save the image
    cv2.imwrite(output_path, combined_image_bgr)

    # Return misclassification info for logging
    return misclassified_count, total_train

def save_checkpoint(network, optimizer, epoch, output_dir, network_shape_b64, random_seed, loss_history_data=None, network_param_history_data=None, args=None):
    """Save a complete training checkpoint.

    Args:
        network: The neural network model
        optimizer: The optimizer instance
        epoch: Current epoch number
        output_dir: Directory to save checkpoint (used as default if checkpoint_dir not specified)
        network_shape_b64: Base64 encoded network shape string
        random_seed: Random seed used for training
        loss_history_data: Loss history list for stagnation detection
        network_param_history_data: Network parameter history dict
        args: Command line arguments
    """
    # Use checkpoint_dir if specified, otherwise use output_dir
    checkpoint_dir = args.checkpoint_dir if args and hasattr(args, 'checkpoint_dir') and args.checkpoint_dir else output_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    seed_str = str(random_seed) if random_seed is not None else "none"
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{network_shape_b64}_{seed_str}_epoch_{epoch:06d}.pt")

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
        'network_shape_b64': network_shape_b64,
        'random_seed': random_seed,
        'final_activation': args.final_activation if args else 'relu',
        'args': {
            'shape': args.shape if args else None,
            'points': args.points if args else 5000,
            'batch_size': args.batch_size if args else 1024,
            'learning_rate': args.learning_rate if args else 0.001,
            'optimizer': args.optimizer if args else 'adam',
            'momentum': args.momentum if args and hasattr(args, 'momentum') else 0.9,
            'final_activation': args.final_activation if args else 'relu',
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
            elif isinstance(module, nn.ReLU):
                f.write(f"Layer {i} (ReLU activation)\n\n")
            elif isinstance(module, nn.Sigmoid):
                f.write(f"Layer {i} (Sigmoid activation)\n\n")

### Full Training and Visualization Pipeline ###
def full_pipeline(
    input_path,
    is_video=False,
    train_size=5000,
    val_size=None,  # Will default to batch_size if None
    layer_sizes=[10, 10, 10, 10, 10, 10, 10, 10],
    epochs=10,
    batch_size=1024,
    learning_rate=0.001,
    output_dir="results",
    snapshot_dir="snapshots",
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
    os.makedirs(snapshot_dir, exist_ok=True)

    # Step 1: Preprocess data
    if is_video:
        raise NotImplementedError("Video processing is not supported for RGB classification")
    else:
        data = preprocess_image(input_path)
        img = cv2.imread(input_path, cv2.IMREAD_COLOR)
        height, width = img.shape[:2]

    # Step 2: Sample training and validation data
    # Use batch_size as val_size if not specified for fair loss comparison
    if val_size is None:
        val_size = batch_size
    train_data, val_data = sample_data(data, train_size, val_size)

    # Handle checkpoint resumption
    start_epoch = 0
    checkpoint_data = None
    if args and args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        # We'll load the checkpoint after creating the network

    # Step 3: Initialize the network
    input_dim = 2  # x, y coordinates
    final_activation = args.final_activation if args else 'relu'
    network = PolytopeNet(input_dim, layer_sizes, final_activation=final_activation, debug=debug)
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

    # Step 3c: Create weight predictor network if Kolmogorov regularization is enabled
    weight_predictor = None
    weight_predictor_optimizer = None
    if args and hasattr(args, 'kolmogorov_shape') and args.kolmogorov_shape and args.kolmogorov_weight > 0.0:
        logger.info("Creating weight predictor network for Kolmogorov regularization")
        kolmogorov_layer_sizes = eval(args.kolmogorov_shape)
        loss_type = args.kolmogorov_loss_type if hasattr(args, 'kolmogorov_loss_type') else 'mse'
        num_bins = args.kolmogorov_bins if hasattr(args, 'kolmogorov_bins') else 256
        weight_predictor = WeightPredictorNetwork(kolmogorov_layer_sizes, loss_type=loss_type, num_bins=num_bins, debug=debug)
        weight_predictor.to(device)

        # Create optimizer for weight predictor (use same type as main network)
        weight_predictor_lr = args.kolmogorov_lr if hasattr(args, 'kolmogorov_lr') and args.kolmogorov_lr else learning_rate
        if args.optimizer == 'adam':
            weight_predictor_optimizer = optim.Adam(weight_predictor.parameters(), lr=weight_predictor_lr)
        elif args.optimizer == 'sgd':
            weight_predictor_optimizer = optim.SGD(weight_predictor.parameters(), lr=weight_predictor_lr)
        elif args.optimizer == 'sgd_momentum':
            weight_predictor_optimizer = optim.SGD(weight_predictor.parameters(), lr=weight_predictor_lr, momentum=args.momentum)
        elif args.optimizer == 'rmsprop':
            weight_predictor_optimizer = optim.RMSprop(weight_predictor.parameters(), lr=weight_predictor_lr)

        # Count parameters
        wp_params = sum(p.numel() for p in weight_predictor.parameters() if p.requires_grad)
        logger.info(f"Weight predictor parameters: {wp_params}")
        logger.info(f"Kolmogorov loss type: {loss_type}")
        logger.info(f"Kolmogorov weight: {args.kolmogorov_weight}")
        if loss_type == 'cross_entropy':
            logger.info(f"Kolmogorov bins: {num_bins}")

    # Step 3d: Load checkpoint if resuming
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

        # Reset loss history and network param history for stagnation detection
        # (these are not saved in checkpoints to avoid unbounded growth)
        loss_history = []
        network_param_history = {}

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
        train_loss, val_loss, kolmogorov_loss, combined_loss = train_network(network, optimizer, train_data, val_data, epochs=1, batch_size=batch_size, learning_rate=learning_rate, output_dir=output_dir, network_shape_b64=network_shape_b64, random_seed=random_seed, network_shape_str=network_shape_str, debug=debug, args=args, weight_predictor=weight_predictor, weight_predictor_optimizer=weight_predictor_optimizer)
        if weight_predictor is not None and args.kolmogorov_weight > 0.0:
            logger.info(f"Epoch {epoch + 1}/{epochs} - Combined Loss: {combined_loss:.6f} (Task: {train_loss:.6f} + λ·K: {args.kolmogorov_weight * kolmogorov_loss:.6f}), Val Loss: {val_loss:.6f}, K-Loss: {kolmogorov_loss:.6f}")
        else:
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
            output_path = os.path.join(snapshot_dir, f"{os.path.basename(input_path)}_{network_shape_b64}_{seed_str}_epoch_{epoch_str}.png")

            # Time the visualization step without extra logs
            start_time = time.time()

            # Temporarily change the logging level for the visualization duration
            original_level = logger.level
            if not debug:
                logger.setLevel(logging.WARNING)  # Only show warnings and errors

            misclassified, total = visualize_decision_boundary_with_predictions(
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
              chunk_size=args.chunk_size if args and hasattr(args, 'chunk_size') else None,
              final_activation=final_activation
            )

            # Restore original logging level
            logger.setLevel(original_level)

            end_time = time.time()
            duration = end_time - start_time
            logger.info(f"Visualization @ epoch {epoch + 1}: {duration:.2f}s - Misclassified: {misclassified}/{total}")

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

    # Step 6: Dump network weights
    seed_str = str(random_seed) if random_seed is not None else "none"
    weights_filename = os.path.join(output_dir, f"weights_{network_shape_b64}_{seed_str}.txt")
    dump_network_weights(network, weights_filename)

    # Step 7: Generate video from PNGs
    logger.info("Starting video generation...")
    base_filename = f"{os.path.basename(input_path)}_{network_shape_b64}_{seed_str}"
    glob_pattern = os.path.join(snapshot_dir, f"{base_filename}_epoch_*.png")
    video_output_path = os.path.join(output_dir, f"{base_filename}.mp4")

    ffmpeg_command = [
        'ffmpeg',
        '-loglevel', 'quiet',
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
    parser = argparse.ArgumentParser(description='Neural network RGB classification experiments.')

    # Required arguments
    parser.add_argument('--input', '-i', required=True, help='Path to the input RGB image')
    parser.add_argument('--shape', '-s', required=True, help='Neural network shape (e.g., "[5]*40")')
    parser.add_argument('--epochs', '-e', type=int, required=True, help='Number of training epochs')

    # Optional arguments
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    parser.add_argument('--points', '-p', type=int, default=5000, help='Number of points to sample (default: 5000)')
    parser.add_argument('--batch-size', '-b', type=int, default=1024, help='Batch size for training (default: 1024)')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--output-dir', '-o', default='results', help='Output directory for results (default: results)')
    parser.add_argument('--snapshot-dir', default='snapshots', help='Directory for epoch snapshots (default: snapshots)')
    parser.add_argument('--optimizer', type=str, default='adam',
                      choices=['adam', 'sgd', 'sgd_momentum', 'rmsprop'],
                      help='Optimizer to use (default: adam)')
    parser.add_argument('--momentum', type=float, default=0.9,
                      help='Momentum factor for SGD with momentum (default: 0.9)')
    parser.add_argument('--final-activation', type=str, default='relu',
                      choices=['relu', 'sigmoid', 'leaky_relu'],
                      help='Activation function for final layer (default: relu)')
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
    parser.add_argument('--checkpoint-dir', type=str, default=None,
                      help='Directory to save checkpoints (default: same as output-dir)')

    # Kolmogorov regularization arguments
    parser.add_argument('--kolmogorov-shape', type=str, default=None,
                      help='Weight predictor network shape (e.g., "[5, 5]"). If not specified, no regularization.')
    parser.add_argument('--kolmogorov-weight', type=float, default=0.0,
                      help='Weight for Kolmogorov regularization loss (default: 0.0, disabled)')
    parser.add_argument('--kolmogorov-loss-type', type=str, default='cross_entropy',
                      choices=['mse', 'cross_entropy', 'gaussian_nll', 'laplacian_nll'],
                      help='Loss function for weight prediction (default: cross_entropy)')
    parser.add_argument('--kolmogorov-bins', type=int, default=256,
                      help='Number of bins for cross_entropy quantization (default: 256)')
    parser.add_argument('--kolmogorov-weight-min', type=float, default=-3.0,
                      help='Minimum weight value for quantization (default: -3.0)')
    parser.add_argument('--kolmogorov-weight-max', type=float, default=3.0,
                      help='Maximum weight value for quantization (default: 3.0)')
    parser.add_argument('--kolmogorov-lr', type=float, default=None,
                      help='Learning rate for weight predictor (default: same as main network)')

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
    params_str = f"{args.shape}_{args.points}_{args.batch_size}_{args.learning_rate}_{args.optimizer}_{args.final_activation}"
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
logger.info("=== Starting polytope-classifier training ===")
logger.info(f"Input image: {args.input}")
logger.info(f"Network shape: {args.shape}")
logger.info(f"Training epochs: {args.epochs}")
logger.info(f"Points: {args.points}")
logger.info(f"Optimizer: {args.optimizer}")
logger.info(f"Learning rate: {args.learning_rate}")
logger.info(f"Final activation: {args.final_activation}")
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
        snapshot_dir=args.snapshot_dir,
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
