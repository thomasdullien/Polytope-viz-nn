import os
import torch
import torch.nn as nn

CONSTANT_OUTPUT_THRESHOLD = 30
constant_activation_map = None
constant_output_counter = 0
network_param_history = {}
loss_history = []
MAX_LOSS_HISTORY = 200


def reset_training_diagnostics():
    global constant_activation_map, constant_output_counter, network_param_history, loss_history
    constant_activation_map = None
    constant_output_counter = 0
    network_param_history = {}
    loss_history = []


def hash_network_params(network):
    param_hash = 0
    for param in network.parameters():
        param_bytes = param.detach().cpu().numpy().tobytes()
        param_hash = hash((param_hash, hash(param_bytes)))
    return param_hash


def check_optimization_loop(network, epoch, logger):
    global network_param_history
    param_hash = hash_network_params(network)
    if param_hash in network_param_history:
        last_seen_epoch = network_param_history[param_hash]
        logger.warning(f"Network parameters at epoch {epoch} match those from epoch {last_seen_epoch}.")
        logger.warning("The optimization is looping. Aborting training.")
        return True
    network_param_history[param_hash] = epoch
    return False


def check_loss_stagnation(current_loss, epoch, network_outputs=None, logger=None):
    global loss_history
    loss_history.append(current_loss)
    if len(loss_history) > MAX_LOSS_HISTORY:
        loss_history = loss_history[-MAX_LOSS_HISTORY:]
    if len(loss_history) < 30:
        return False
    min_recent = min(loss_history[-30:])
    if current_loss <= min_recent:
        return False
    if network_outputs is None:
        return False
    outputs = network_outputs.detach().cpu()
    is_constant = torch.allclose(outputs, outputs[0].expand_as(outputs), atol=1e-6)
    if is_constant and logger:
        logger.warning(f"Loss appears stagnant at epoch {epoch} and outputs are constant.")
    return bool(is_constant)


def dump_network_weights(network, filename):
    with open(filename, 'w') as f:
        linear_layers = [m for m in network.network if isinstance(m, nn.Linear)]
        f.write(f"Network architecture: {[l.in_features for l in linear_layers] + [linear_layers[-1].out_features]}\n")
        f.write(f"Number of layers: {len(linear_layers)}\n\n")
        for i, module in enumerate(network.network):
            if isinstance(module, nn.Linear):
                f.write(f"Layer {i} (Linear):\n")
                f.write(f"  Shape: {module.weight.shape}\n")
                f.write(f"  Weight range: min={module.weight.min():.6f}, max={module.weight.max():.6f}\n")
                f.write(f"  Weight mean: {module.weight.mean():.6f}, std={module.weight.std(unbiased=False):.6f}\n")
                f.write(f"  Bias range: min={module.bias.min():.6f}, max={module.bias.max():.6f}\n")
                f.write(f"  Bias mean: {module.bias.mean():.6f}, std={module.bias.std(unbiased=False):.6f}\n\n")
                weight_matrix = module.weight.detach().cpu().numpy()
                for row in weight_matrix:
                    f.write("    " + " ".join(f"{x:.6f}" for x in row) + "\n")
                bias_vector = module.bias.detach().cpu().numpy()
                f.write("\n  Bias vector:\n")
                f.write("    " + " ".join(f"{x:.6f}" for x in bias_vector) + "\n\n")
            elif isinstance(module, (nn.LeakyReLU, nn.ReLU)) or module.__class__.__name__ == 'ReLU2':
                f.write(f"Layer {i} ({module.__class__.__name__} activation)\n\n")
            elif isinstance(module, nn.Sigmoid):
                f.write(f"Layer {i} (Sigmoid activation)\n\n")
