import torch
import torch.nn as nn


class ReLU2(nn.Module):
    def forward(self, x):
        return torch.relu(x).square()


class PolytopeNet(nn.Module):
    def __init__(self, input_dim: int, layer_sizes: list[int], output_dim: int = 1, final_activation: str | None = None, hidden_activation: str = 'leaky_relu', debug: bool = False, logger=None):
        super().__init__()
        layers = []
        self.activation_layers = []
        self.debug = debug
        self.logger = logger
        self.final_activation = final_activation
        self.hidden_activation = hidden_activation

        rng_state = torch.get_rng_state()
        prev_dim = input_dim
        for i, size in enumerate(layer_sizes):
            linear_layer = nn.Linear(prev_dim, size)
            nn.init.kaiming_normal_(linear_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
            nn.init.constant_(linear_layer.bias, 0.1)
            layers.append(linear_layer)

            activation_layer = self._make_hidden_activation(hidden_activation)
            layers.append(activation_layer)
            self.activation_layers.append(activation_layer)
            self.register_buffer(f'hash_coeffs_{i}', torch.randint(1, 2**31 - 1, (size,), dtype=torch.int64))
            prev_dim = size

        output_layer = nn.Linear(prev_dim, output_dim)
        nn.init.kaiming_normal_(output_layer.weight, mode='fan_in', nonlinearity='leaky_relu')
        nn.init.constant_(output_layer.bias, 0.1)
        layers.append(output_layer)

        if final_activation:
            if final_activation == 'relu':
                layers.append(nn.ReLU())
            elif final_activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif final_activation == 'leaky_relu':
                layers.append(nn.LeakyReLU(negative_slope=0.01))
            else:
                raise ValueError(f'Unknown final activation: {final_activation}')

        self.network = nn.Sequential(*layers)
        torch.set_rng_state(rng_state)

    @staticmethod
    def _make_hidden_activation(name: str):
        if name == 'leaky_relu':
            return nn.LeakyReLU(negative_slope=0.01)
        if name == 'relu':
            return nn.ReLU()
        if name == 'relu2':
            return ReLU2()
        raise ValueError(f'Unknown hidden activation: {name}')

    def forward(self, x):
        polytope_hash = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
        current = x
        layer_idx = 0
        for i, layer in enumerate(self.network):
            pre_activation = current
            current = layer(current)
            if layer_idx < len(self.activation_layers) and layer is self.activation_layers[layer_idx]:
                activation_pattern = pre_activation > 0
                hash_coeffs = getattr(self, f'hash_coeffs_{layer_idx}')
                polytope_hash += (activation_pattern * hash_coeffs).sum(dim=1)
                layer_idx += 1
                if self.debug and self.logger:
                    inactive_neurons = (pre_activation <= 0).float().mean().item()
                    self.logger.debug(f"Layer {i//2} {layer.__class__.__name__}: inactive neurons = {inactive_neurons:.2%}, range = [{current.min():.6f}, {current.max():.6f}]")
            elif self.debug and self.logger and isinstance(layer, nn.Linear):
                self.logger.debug(f"Layer {i//2} Linear: range = [{current.min():.6f}, {current.max():.6f}]")
        return current, polytope_hash

    def get_weight_enumeration(self):
        if not hasattr(self, '_weight_position_cache'):
            self._build_weight_position_cache()
        actual_weights = []
        for module in self.network:
            if isinstance(module, nn.Linear):
                actual_weights.append(module.weight.flatten())
                actual_weights.append(module.bias)
        return self._weight_position_cache.clone(), torch.cat(actual_weights)

    def _build_weight_position_cache(self):
        weight_positions = []
        linear_layers = [m for m in self.network if isinstance(m, nn.Linear)]
        num_layers = len(linear_layers)
        layer_counter = 0
        for module in self.network:
            if isinstance(module, nn.Linear):
                layer_id_norm = layer_counter / max(num_layers - 1, 1)
                device = module.weight.device
                out_features, in_features = module.weight.shape
                neuron_indices = torch.arange(out_features, device=device, dtype=torch.float32)
                weight_indices = torch.arange(in_features, device=device, dtype=torch.float32)
                neuron_norm = neuron_indices / max(out_features - 1, 1)
                weight_norm = weight_indices / max(in_features - 1, 1)
                neuron_grid, weight_grid = torch.meshgrid(neuron_norm, weight_norm, indexing='ij')
                layer_grid = torch.full_like(neuron_grid, layer_id_norm)
                weight_positions.append(torch.stack([layer_grid.flatten(), neuron_grid.flatten(), weight_grid.flatten()], dim=1))
                weight_positions.append(torch.stack([
                    torch.full((out_features,), layer_id_norm, device=device),
                    neuron_norm,
                    torch.ones(out_features, device=device),
                ], dim=1))
                layer_counter += 1
        self._weight_position_cache = torch.cat(weight_positions, dim=0)

    def invalidate_weight_position_cache(self):
        if hasattr(self, '_weight_position_cache'):
            del self._weight_position_cache
