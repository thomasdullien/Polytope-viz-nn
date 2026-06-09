import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class WeightPredictorNetwork(nn.Module):
    def __init__(self, layer_sizes: list[int], loss_type: str = 'mse', num_bins: int = 256, debug: bool = False):
        super().__init__()
        self.debug = debug
        self.loss_type = loss_type
        self.num_bins = num_bins
        layers = []
        prev_dim = 3
        for size in layer_sizes:
            linear_layer = nn.Linear(prev_dim, size)
            nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.constant_(linear_layer.bias, 0.1)
            layers.extend([linear_layer, nn.LeakyReLU(negative_slope=0.01)])
            prev_dim = size
        self.hidden_layers = nn.Sequential(*layers)
        if loss_type == 'cross_entropy':
            self.output_layer = nn.Linear(prev_dim, num_bins)
        elif loss_type in ['gaussian_nll', 'laplacian_nll']:
            self.output_layer = nn.Linear(prev_dim, 2)
        else:
            self.output_layer = nn.Linear(prev_dim, 1)
        nn.init.kaiming_normal_(self.output_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, weight_positions):
        output = self.output_layer(self.hidden_layers(weight_positions))
        if self.loss_type == 'cross_entropy':
            return F.log_softmax(output, dim=-1)
        return output


def quantize_weights(weights, num_bins, weight_range):
    min_val, max_val = weight_range
    clipped = torch.clamp(weights, min_val, max_val)
    normalized = (clipped - min_val) / (max_val - min_val)
    return (normalized * (num_bins - 1)).long()


def gaussian_nll_loss(actual, predicted_mean, predicted_log_var):
    var = torch.exp(predicted_log_var)
    loss = 0.5 * (predicted_log_var + ((actual - predicted_mean) ** 2) / var + math.log(2 * math.pi))
    return loss.mean()


def laplacian_nll_loss(actual, predicted_location, predicted_log_scale):
    scale = torch.exp(predicted_log_scale)
    loss = predicted_log_scale + math.log(2) + torch.abs(actual - predicted_location) / scale
    return loss.mean()


def compute_kolmogorov_loss(weight_predictor, main_network, device, weight_range=(-3.0, 3.0)):
    weight_positions, actual_weights = main_network.get_weight_enumeration()
    if weight_positions.device != device:
        weight_positions = weight_positions.to(device)
    if actual_weights.device != device:
        actual_weights = actual_weights.to(device)
    predictions = weight_predictor(weight_positions)
    loss_type = weight_predictor.loss_type
    if loss_type == 'mse':
        return F.mse_loss(predictions.squeeze(), actual_weights)
    if loss_type == 'cross_entropy':
        min_val, max_val = weight_range
        clipped = torch.clamp(actual_weights, min_val, max_val)
        normalized = (clipped - min_val) / (max_val - min_val)
        continuous_bins = normalized * (weight_predictor.num_bins - 1)
        temperature = 0.1
        bin_indices = torch.arange(weight_predictor.num_bins, device=actual_weights.device, dtype=torch.float32)
        distances = -torch.abs(continuous_bins.unsqueeze(1) - bin_indices.unsqueeze(0)) / temperature
        soft_targets = F.softmax(distances, dim=1)
        return -(soft_targets * predictions).sum(dim=1).mean()
    if loss_type == 'gaussian_nll':
        return gaussian_nll_loss(actual_weights, predictions[:, 0], predictions[:, 1])
    if loss_type == 'laplacian_nll':
        return laplacian_nll_loss(actual_weights, predictions[:, 0], predictions[:, 1])
    raise ValueError(f'Unknown loss type: {loss_type}')
