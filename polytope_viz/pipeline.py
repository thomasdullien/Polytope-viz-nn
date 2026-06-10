import os
import random
import subprocess
import time
from datetime import datetime

import cv2
import imageio
import numpy as np
import torch
from torchinfo import summary

from .checkpointing import load_checkpoint, save_checkpoint
from .config import CLASSIFIER_TASK, GRAYSCALE_TASK, TaskConfig
from .data import preprocess_classifier_image, preprocess_grayscale_image, preprocess_video, sample_data
from .diagnostics import check_loss_stagnation, check_optimization_loop, dump_network_weights, reset_training_diagnostics
from .kolmogorov import WeightPredictorNetwork
from .models import PolytopeNet
from .shape import parse_layer_sizes
from .training import create_optimizer, create_weight_predictor_optimizer, train_network
from .visualization import save_kernel_smoothed_image, visualize_classifier, visualize_grayscale


def set_random_seed(seed, logger):
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    logger.info('Random seed set for all libraries')


def _current_optimizer_metadata(args):
    optimizer_type = args.resume_optimizer if getattr(args, 'resume_optimizer', None) else args.optimizer
    learning_rate = args.resume_lr if getattr(args, 'resume_lr', None) else args.learning_rate
    momentum = args.momentum if optimizer_type == 'sgd_momentum' else None
    return optimizer_type, learning_rate, momentum


def _compact_optimizer_label(args, optimizer_type):
    if args is None or optimizer_type is None:
        return None
    grad_clip = getattr(args, 'grad_clip_norm', None)
    if optimizer_type in {'adam', 'adamw'}:
        return f"Opt: {optimizer_type};{args.adam_eps:g};{args.weight_decay:g};{grad_clip if grad_clip is not None else 'none'}"
    if optimizer_type == 'muon':
        return f"Opt: muon;{args.muon_eps:g};{args.weight_decay:g};{grad_clip if grad_clip is not None else 'none'}"
    if optimizer_type == 'sgd_momentum':
        return f"Opt: sgd_momentum;{args.momentum:g};{grad_clip if grad_clip is not None else 'none'}"
    return f"Opt: {optimizer_type};{grad_clip if grad_clip is not None else 'none'}"


def _create_weight_predictor(args, learning_rate, debug, device, logger):
    if not (args and getattr(args, 'kolmogorov_shape', None) and args.kolmogorov_weight > 0.0):
        return None, None
    logger.info('Creating weight predictor network for Kolmogorov regularization')
    layer_sizes = parse_layer_sizes(args.kolmogorov_shape)
    loss_type = getattr(args, 'kolmogorov_loss_type', 'mse')
    num_bins = getattr(args, 'kolmogorov_bins', 256)
    weight_predictor = WeightPredictorNetwork(layer_sizes, loss_type=loss_type, num_bins=num_bins, debug=debug).to(device)
    weight_predictor_optimizer = create_weight_predictor_optimizer(weight_predictor, args, learning_rate)
    wp_params = sum(p.numel() for p in weight_predictor.parameters() if p.requires_grad)
    logger.info(f"Weight predictor parameters: {wp_params}")
    logger.info(f"Kolmogorov loss type: {loss_type}")
    logger.info(f"Kolmogorov weight: {args.kolmogorov_weight}")
    if loss_type == 'cross_entropy':
        logger.info(f"Kolmogorov bins: {num_bins}")
    return weight_predictor, weight_predictor_optimizer


def _preprocess_input(input_path, is_video, task_config, logger):
    if task_config.name == 'classifier':
        if is_video:
            raise NotImplementedError('Video processing is not supported for RGB classification')
        return preprocess_classifier_image(input_path, logger), 2
    if is_video:
        return preprocess_video(input_path), 3
    return preprocess_grayscale_image(input_path), 2


def _visualize(task_config, **kwargs):
    if task_config.name == 'classifier':
        return visualize_classifier(**kwargs)
    return visualize_grayscale(**kwargs)


def _snapshot_write_params(args):
    if args is None:
        return 'png', None
    snapshot_format = getattr(args, 'snapshot_format', 'png')
    if snapshot_format == 'jpg':
        return 'jpg', [cv2.IMWRITE_JPEG_QUALITY, getattr(args, 'jpeg_quality', 90)]
    png_compression = getattr(args, 'png_compression', None)
    if png_compression is not None:
        return 'png', [cv2.IMWRITE_PNG_COMPRESSION, png_compression]
    return 'png', None


def full_pipeline(input_path, task_config: TaskConfig, is_video=False, train_size=5000, val_size=None, layer_sizes=None, epochs=10, batch_size=1024, learning_rate=0.001, output_dir='results', snapshot_dir='snapshots', random_seed=None, network_shape_b64=None, network_shape_str=None, debug=False, args=None, logger=None):
    reset_training_diagnostics()
    layer_sizes = layer_sizes or [10] * 8
    training_start_time = time.time()
    logger.info(f"Starting full training pipeline at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(snapshot_dir, exist_ok=True)

    (data, (height, width)), input_dim = _preprocess_input(input_path, is_video, task_config, logger)
    if val_size is None:
        val_size = batch_size
    train_data, val_data = sample_data(data, train_size, val_size)

    seed_str = str(random_seed) if random_seed is not None else 'none'
    if task_config.name == 'grayscale' and not (args and args.resume):
        smoothed_image_path = os.path.join(output_dir, f"{os.path.basename(input_path)}_{network_shape_b64}_{seed_str}-kernel-smoothed.png")
        save_kernel_smoothed_image(train_data, (height, width), smoothed_image_path, input_path, sigma=getattr(args, 'smoothing_sigma', 3.0), logger=logger)

    final_activation = getattr(args, 'final_activation', None) if task_config.name == 'classifier' else None
    hidden_activation = getattr(args, 'activation', 'leaky_relu')
    network = PolytopeNet(input_dim, layer_sizes, output_dim=task_config.output_dim, final_activation=final_activation, hidden_activation=hidden_activation, debug=debug, logger=logger)
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params}")
    summary(network, input_size=(1024, input_dim))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    network.to(device)
    optimizer = create_optimizer(network, args)
    weight_predictor, weight_predictor_optimizer = _create_weight_predictor(args, learning_rate, debug, device, logger)

    if args and args.use_compile and hasattr(torch, 'compile'):
        logger.info('Using torch.compile for network acceleration')
        network = torch.compile(network)
    elif args and args.use_compile:
        logger.warning('torch.compile requested but not available. Requires PyTorch 2.0+')
        logger.info('Continuing without compilation')

    start_epoch = 0
    current_optimizer_type, current_learning_rate, current_momentum = _current_optimizer_metadata(args)
    if args and args.resume:
        logger.info(f"Resuming from checkpoint: {args.resume}")
        restore_optimizer = args.resume_optimizer is None
        checkpoint_data = load_checkpoint(args.resume, network, optimizer, restore_rng=True, restore_optimizer=restore_optimizer, logger=logger, weight_predictor=weight_predictor, weight_predictor_optimizer=weight_predictor_optimizer)
        start_epoch = checkpoint_data['epoch'] + 1
        if args.resume_optimizer:
            logger.info(f"Switching optimizer from {checkpoint_data.get('optimizer_type', 'unknown')} to {args.resume_optimizer}")
            optimizer = create_optimizer(network, args)
        if args.resume_lr:
            logger.info(f"Changing learning rate from {checkpoint_data.get('learning_rate', 'unknown')} to {args.resume_lr}")
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.resume_lr
        current_optimizer_type, current_learning_rate, current_momentum = _current_optimizer_metadata(args)
        logger.info(f"Resuming training from epoch {start_epoch}")

    current_optimizer_label = _compact_optimizer_label(args, current_optimizer_type)

    save_interval = getattr(args, 'save_interval', 1)
    checkpoint_interval = getattr(args, 'checkpoint_interval', None)
    snapshot_ext, image_write_params = _snapshot_write_params(args)
    boundary_frames = []

    for epoch in range(start_epoch, epochs):
        train_loss, val_loss, kolmogorov_loss, combined_loss = train_network(network, optimizer, train_data, val_data, task_config, epochs=1, batch_size=batch_size, args=args, weight_predictor=weight_predictor, weight_predictor_optimizer=weight_predictor_optimizer, logger=logger)
        weighted_k_loss = args.kolmogorov_weight * kolmogorov_loss if (weight_predictor is not None and args.kolmogorov_weight > 0.0) else 0.0
        logger.info(f"Epoch {epoch + 1}/{epochs} - Combined Loss: {combined_loss:.6f} (Task: {train_loss:.6f} + λ·K: {weighted_k_loss:.6f}), Val Loss: {val_loss:.6f}")

        network.eval()
        with torch.no_grad():
            val_inputs = torch.tensor(val_data[:, :-1], dtype=torch.float32).to(device)
            val_outputs, _ = network(val_inputs)
        model_for_dump = network._orig_mod if hasattr(network, '_orig_mod') else network
        if check_optimization_loop(model_for_dump, epoch, logger):
            dump_network_weights(model_for_dump, os.path.join(output_dir, 'weights_optimization_loop.txt'))
            break
        if check_loss_stagnation(train_loss, epoch, val_outputs, logger):
            dump_network_weights(model_for_dump, os.path.join(output_dir, 'weights_stagnation.txt'))
            break

        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            epoch_str = f"{epoch + 1:06d}"
            output_path = os.path.join(snapshot_dir, f"{os.path.basename(input_path)}_{network_shape_b64}_{seed_str}_epoch_{epoch_str}.{snapshot_ext}")
            start_time = time.time()
            result = _visualize(
                task_config,
                network=network,
                data=data,
                train_data=train_data,
                val_data=val_data,
                image_shape=(height, width),
                output_path=output_path,
                target_image_path=input_path,
                logger=logger,
                train_loss=train_loss,
                val_loss=val_loss,
                network_shape_str=network_shape_str,
                random_seed=random_seed,
                epoch=epoch,
                num_points=args.points if args else None,
                learning_rate=current_learning_rate if args else None,
                optimizer=current_optimizer_label,
                momentum=current_momentum,
                chunk_size=getattr(args, 'chunk_size', None),
                final_activation=final_activation,
                image_write_params=image_write_params,
                empty_cache_each_chunk=getattr(args, 'empty_cache_each_viz_chunk', False),
            )
            duration = time.time() - start_time
            if task_config.name == 'classifier':
                misclassified, total = result
                logger.info(f"Visualization @ epoch {epoch + 1}: {duration:.2f}s - Misclassified: {misclassified}/{total}")
            else:
                logger.info(f"Visualization @ epoch {epoch + 1}: {duration:.2f}s")
        if checkpoint_interval and (epoch + 1) % checkpoint_interval == 0:
            save_checkpoint(network, optimizer, epoch, output_dir, network_shape_b64, random_seed, args=args, weight_predictor=weight_predictor, weight_predictor_optimizer=weight_predictor_optimizer, final_activation=final_activation, logger=logger)

    final_checkpoint_path = save_checkpoint(network, optimizer, epochs - 1, output_dir, network_shape_b64, random_seed, args=args, weight_predictor=weight_predictor, weight_predictor_optimizer=weight_predictor_optimizer, final_activation=final_activation, logger=logger)
    logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
    model_for_dump = network._orig_mod if hasattr(network, '_orig_mod') else network
    dump_network_weights(model_for_dump, os.path.join(output_dir, f"weights_{network_shape_b64}_{seed_str}.txt"))

    logger.info('Starting video generation...')
    base_filename = f"{os.path.basename(input_path)}_{network_shape_b64}_{seed_str}"
    glob_pattern = os.path.join(snapshot_dir, f"{base_filename}_epoch_*.{snapshot_ext}")
    video_output_path = os.path.join(output_dir, f"{base_filename}.mp4")
    ffmpeg_command = ['ffmpeg', '-loglevel', 'quiet', '-y', '-framerate', '24', '-pattern_type', 'glob', '-i', glob_pattern, '-c:v', 'libx264', '-crf', '18', '-pix_fmt', 'yuv420p', video_output_path]
    try:
        logger.info(f"Running ffmpeg command: {' '.join(ffmpeg_command)}")
        result = subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        logger.info('ffmpeg stdout:\n' + result.stdout)
        logger.info('ffmpeg stderr:\n' + result.stderr)
        logger.info(f"Video generated successfully: {video_output_path}")
    except FileNotFoundError:
        logger.error("ffmpeg not found. Please ensure ffmpeg is installed and in your system's PATH.")
    except subprocess.CalledProcessError as e:
        logger.error('ffmpeg command failed.')
        logger.error('ffmpeg stdout:\n' + e.stdout)
        logger.error('ffmpeg stderr:\n' + e.stderr)
