import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from .kolmogorov import compute_kolmogorov_loss


class CompositeOptimizer:
    """Small wrapper that lets the training loop treat multiple optimizers as one."""

    def __init__(self, optimizers):
        self.optimizers = [optimizer for optimizer in optimizers if optimizer is not None]

    @property
    def param_groups(self):
        groups = []
        for optimizer in self.optimizers:
            groups.extend(optimizer.param_groups)
        return groups

    def zero_grad(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()

    def step(self):
        for optimizer in self.optimizers:
            optimizer.step()

    def state_dict(self):
        return {
            'type': 'CompositeOptimizer',
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
        }

    def load_state_dict(self, state_dict):
        if state_dict.get('type') != 'CompositeOptimizer':
            raise ValueError('Cannot load non-composite optimizer state into CompositeOptimizer')
        optimizer_states = state_dict['optimizers']
        if len(optimizer_states) != len(self.optimizers):
            raise ValueError(
                f"Composite optimizer state has {len(optimizer_states)} optimizers, "
                f"but current optimizer has {len(self.optimizers)}"
            )
        for optimizer, optimizer_state in zip(self.optimizers, optimizer_states):
            optimizer.load_state_dict(optimizer_state)


def _adam_kwargs(args):
    return {
        'eps': getattr(args, 'adam_eps', 1e-8),
        'weight_decay': getattr(args, 'weight_decay', 0.0),
        'amsgrad': getattr(args, 'amsgrad', False),
    }


def _create_muon_optimizer(parameters, args, learning_rate):
    if not hasattr(optim, 'Muon'):
        raise RuntimeError('torch.optim.Muon is not available in this PyTorch installation')
    params = list(parameters)
    muon_params = [param for param in params if param.requires_grad and param.ndim == 2]
    fallback_params = [param for param in params if param.requires_grad and param.ndim != 2]
    optimizers = []
    if muon_params:
        optimizers.append(
            optim.Muon(
                muon_params,
                lr=learning_rate,
                weight_decay=getattr(args, 'weight_decay', 0.0),
                momentum=getattr(args, 'muon_momentum', 0.95),
                nesterov=getattr(args, 'muon_nesterov', True),
                eps=getattr(args, 'muon_eps', 1e-7),
                ns_steps=getattr(args, 'muon_ns_steps', 5),
                adjust_lr_fn=getattr(args, 'muon_adjust_lr_fn', None),
            )
        )
    if fallback_params:
        optimizers.append(optim.AdamW(fallback_params, lr=learning_rate, **_adam_kwargs(args)))
    if not optimizers:
        raise ValueError('No trainable parameters found for Muon optimizer')
    if len(optimizers) == 1:
        return optimizers[0]
    return CompositeOptimizer(optimizers)


def create_optimizer(network, args):
    optimizer_type = args.resume_optimizer if getattr(args, 'resume_optimizer', None) else args.optimizer
    learning_rate = args.resume_lr if getattr(args, 'resume_lr', None) else args.learning_rate
    if optimizer_type == 'adam':
        return optim.Adam(network.parameters(), lr=learning_rate, **_adam_kwargs(args))
    if optimizer_type == 'adamw':
        return optim.AdamW(network.parameters(), lr=learning_rate, **_adam_kwargs(args))
    if optimizer_type == 'muon':
        return _create_muon_optimizer(network.parameters(), args, learning_rate)
    if optimizer_type == 'sgd':
        return optim.SGD(network.parameters(), lr=learning_rate)
    if optimizer_type == 'sgd_momentum':
        return optim.SGD(network.parameters(), lr=learning_rate, momentum=getattr(args, 'momentum', 0.9))
    if optimizer_type == 'rmsprop':
        return optim.RMSprop(network.parameters(), lr=learning_rate)
    raise ValueError(f"Unknown optimizer: {optimizer_type}")


def create_weight_predictor_optimizer(weight_predictor, args, fallback_lr):
    lr = args.kolmogorov_lr if getattr(args, 'kolmogorov_lr', None) else fallback_lr
    optimizer_type = args.resume_optimizer if getattr(args, 'resume_optimizer', None) else args.optimizer
    if optimizer_type == 'adam':
        return optim.Adam(weight_predictor.parameters(), lr=lr, **_adam_kwargs(args))
    if optimizer_type == 'adamw':
        return optim.AdamW(weight_predictor.parameters(), lr=lr, **_adam_kwargs(args))
    if optimizer_type == 'muon':
        return _create_muon_optimizer(weight_predictor.parameters(), args, lr)
    if optimizer_type == 'sgd':
        return optim.SGD(weight_predictor.parameters(), lr=lr)
    if optimizer_type == 'sgd_momentum':
        return optim.SGD(weight_predictor.parameters(), lr=lr, momentum=args.momentum)
    if optimizer_type == 'rmsprop':
        return optim.RMSprop(weight_predictor.parameters(), lr=lr)
    raise ValueError(f"Unknown optimizer: {optimizer_type}")


def _clip_gradients(module, max_norm):
    if max_norm is None:
        return
    if max_norm <= 0:
        raise ValueError('--grad-clip-norm must be positive when specified')
    torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=max_norm)


def train_network(network, optimizer, train_data, val_data, task_config, epochs, batch_size, args=None, weight_predictor=None, weight_predictor_optimizer=None, logger=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    if weight_predictor is not None:
        weight_predictor.to(device)

    train_inputs = torch.tensor(train_data[:, :-1], dtype=torch.float32).to(device)
    val_inputs = torch.tensor(val_data[:, :-1], dtype=torch.float32).to(device)
    if task_config.loss_kind == 'cross_entropy':
        train_targets = torch.tensor(train_data[:, -1], dtype=torch.long).to(device)
        val_targets = torch.tensor(val_data[:, -1], dtype=torch.long).to(device)
        criterion = nn.CrossEntropyLoss()
    else:
        train_targets = torch.tensor(train_data[:, -1], dtype=torch.float32).unsqueeze(1).to(device)
        val_targets = torch.tensor(val_data[:, -1], dtype=torch.float32).unsqueeze(1).to(device)
        criterion = nn.MSELoss()

    train_loader = DataLoader(TensorDataset(train_inputs, train_targets), batch_size=batch_size, shuffle=True)
    kolmogorov_weight = args.kolmogorov_weight if args and hasattr(args, 'kolmogorov_weight') else 0.0
    use_kolmogorov = weight_predictor is not None and kolmogorov_weight > 0.0

    for epoch in range(epochs):
        network.train()
        if weight_predictor is not None:
            weight_predictor.train()
        total_loss = 0.0
        total_task_loss = 0.0
        total_kolmogorov_loss = 0.0
        num_batches = 0
        for batch_inputs, batch_targets in train_loader:
            optimizer.zero_grad()
            if weight_predictor_optimizer is not None:
                weight_predictor_optimizer.zero_grad()
            outputs, _ = network(batch_inputs)
            task_loss = criterion(outputs, batch_targets)
            if use_kolmogorov:
                weight_range = (args.kolmogorov_weight_min, args.kolmogorov_weight_max) if args and hasattr(args, 'kolmogorov_weight_min') else (-3.0, 3.0)
                kolmogorov_loss = compute_kolmogorov_loss(weight_predictor, network._orig_mod if hasattr(network, '_orig_mod') else network, device, weight_range)
                loss = task_loss + kolmogorov_weight * kolmogorov_loss
                total_kolmogorov_loss += kolmogorov_loss.item()
            else:
                loss = task_loss
            loss.backward()
            grad_clip_norm = getattr(args, 'grad_clip_norm', None) if args else None
            _clip_gradients(network, grad_clip_norm)
            if weight_predictor is not None:
                _clip_gradients(weight_predictor, grad_clip_norm)
            optimizer.step()
            if weight_predictor_optimizer is not None:
                weight_predictor_optimizer.step()
            total_loss += loss.item()
            total_task_loss += task_loss.item()
            num_batches += 1

        avg_combined_loss = total_loss / num_batches
        avg_task_loss = total_task_loss / num_batches
        avg_kolmogorov_loss = total_kolmogorov_loss / num_batches if use_kolmogorov else 0.0
        network.eval()
        if weight_predictor is not None:
            weight_predictor.eval()
        with torch.no_grad():
            val_outputs, _ = network(val_inputs)
            val_loss = criterion(val_outputs, val_targets).item()
        if logger and getattr(args, 'debug', False):
            logger.debug(f"Epoch {epoch+1}/{epochs}")
            logger.debug(f"Combined Loss: {avg_combined_loss:.6f}")
            logger.debug(f"Task Loss: {avg_task_loss:.6f}")
            if use_kolmogorov:
                logger.debug(f"Kolmogorov Loss: {avg_kolmogorov_loss:.6f}")
            logger.debug(f"Validation Loss: {val_loss:.6f}")
    return avg_task_loss, val_loss, avg_kolmogorov_loss if use_kolmogorov else 0.0, avg_combined_loss
