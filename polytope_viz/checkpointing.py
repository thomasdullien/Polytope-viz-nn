import os
import random
import numpy as np
import torch


def _unwrap_compiled(network):
    return network._orig_mod if hasattr(network, '_orig_mod') else network


def save_checkpoint(network, optimizer, epoch, output_dir, network_shape_b64, random_seed, args=None, weight_predictor=None, weight_predictor_optimizer=None, final_activation=None, logger=None):
    checkpoint_dir = args.checkpoint_dir if args and getattr(args, 'checkpoint_dir', None) else output_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    seed_str = str(random_seed) if random_seed is not None else 'none'
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{network_shape_b64}_{seed_str}_epoch_{epoch:06d}.pt")
    model = _unwrap_compiled(network)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer_type': args.optimizer if args else 'adam',
        'learning_rate': args.learning_rate if args else 0.001,
        'momentum': getattr(args, 'momentum', None) if args else None,
        'adam_eps': getattr(args, 'adam_eps', 1e-8) if args else 1e-8,
        'weight_decay': getattr(args, 'weight_decay', 0.0) if args else 0.0,
        'amsgrad': getattr(args, 'amsgrad', False) if args else False,
        'muon_eps': getattr(args, 'muon_eps', 1e-7) if args else 1e-7,
        'muon_momentum': getattr(args, 'muon_momentum', 0.95) if args else 0.95,
        'muon_ns_steps': getattr(args, 'muon_ns_steps', 5) if args else 5,
        'muon_nesterov': getattr(args, 'muon_nesterov', True) if args else True,
        'muon_adjust_lr_fn': getattr(args, 'muon_adjust_lr_fn', None) if args else None,
        'grad_clip_norm': getattr(args, 'grad_clip_norm', None) if args else None,
        'pytorch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
        'network_shape_b64': network_shape_b64,
        'random_seed': random_seed,
        'final_activation': final_activation,
        'args': {
            'shape': args.shape if args else None,
            'points': args.points if args else 5000,
            'batch_size': args.batch_size if args else 1024,
            'learning_rate': args.learning_rate if args else 0.001,
            'optimizer': args.optimizer if args else 'adam',
            'momentum': getattr(args, 'momentum', 0.9) if args else 0.9,
            'adam_eps': getattr(args, 'adam_eps', 1e-8) if args else 1e-8,
            'weight_decay': getattr(args, 'weight_decay', 0.0) if args else 0.0,
            'amsgrad': getattr(args, 'amsgrad', False) if args else False,
            'muon_eps': getattr(args, 'muon_eps', 1e-7) if args else 1e-7,
            'muon_momentum': getattr(args, 'muon_momentum', 0.95) if args else 0.95,
            'muon_ns_steps': getattr(args, 'muon_ns_steps', 5) if args else 5,
            'muon_nesterov': getattr(args, 'muon_nesterov', True) if args else True,
            'muon_adjust_lr_fn': getattr(args, 'muon_adjust_lr_fn', None) if args else None,
            'grad_clip_norm': getattr(args, 'grad_clip_norm', None) if args else None,
            'final_activation': final_activation,
        },
    }
    if weight_predictor is not None:
        checkpoint['weight_predictor_state_dict'] = weight_predictor.state_dict()
        checkpoint['weight_predictor_loss_type'] = weight_predictor.loss_type
        checkpoint['weight_predictor_num_bins'] = weight_predictor.num_bins
        if weight_predictor_optimizer is not None:
            checkpoint['weight_predictor_optimizer_state_dict'] = weight_predictor_optimizer.state_dict()
        if args:
            checkpoint['kolmogorov_shape'] = args.kolmogorov_shape
            checkpoint['kolmogorov_weight'] = args.kolmogorov_weight
            checkpoint['kolmogorov_lr'] = getattr(args, 'kolmogorov_lr', None)
    if torch.cuda.is_available():
        checkpoint['cuda_rng_state'] = torch.cuda.get_rng_state()
        checkpoint['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, checkpoint_path)
    if logger:
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def load_checkpoint(checkpoint_path, network, optimizer=None, restore_rng=True, restore_optimizer=True, logger=None, weight_predictor=None, weight_predictor_optimizer=None):
    if not os.path.exists(checkpoint_path):
        raise ValueError(f"Checkpoint file not found: {checkpoint_path}")
    if logger:
        logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model = _unwrap_compiled(network)
    model.load_state_dict(checkpoint['model_state_dict'])
    if logger:
        logger.info(f"Model state loaded from epoch {checkpoint['epoch']}")
    if restore_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if logger:
            logger.info(f"Optimizer state restored ({checkpoint.get('optimizer_type', 'unknown')})")
    if weight_predictor is not None and 'weight_predictor_state_dict' in checkpoint:
        weight_predictor.load_state_dict(checkpoint['weight_predictor_state_dict'])
        if logger:
            logger.info('Weight predictor state restored')
    if restore_optimizer and weight_predictor_optimizer is not None and 'weight_predictor_optimizer_state_dict' in checkpoint:
        weight_predictor_optimizer.load_state_dict(checkpoint['weight_predictor_optimizer_state_dict'])
        if logger:
            logger.info('Weight predictor optimizer state restored')
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
        if logger:
            logger.info('RNG states restored')
    return checkpoint
