import argparse
import base64

from .config import CLASSIFIER_TASK, GRAYSCALE_TASK
from .logging_utils import setup_logger
from .pipeline import full_pipeline, set_random_seed
from .shape import parse_layer_sizes


def _add_common_arguments(parser):
    parser.add_argument('--input', '-i', required=True, help='Path to the input image')
    parser.add_argument('--shape', '-s', required=True, help='Neural network shape (e.g., "[5]*40")')
    parser.add_argument('--epochs', '-e', type=int, required=True, help='Number of training epochs')
    parser.add_argument('--seed', type=int, help='Random seed for reproducibility')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug output')
    parser.add_argument('--points', '-p', type=int, default=5000, help='Number of points to sample (default: 5000)')
    parser.add_argument('--batch-size', '-b', type=int, default=1024, help='Batch size for training (default: 1024)')
    parser.add_argument('--learning-rate', '-l', type=float, default=0.001, help='Learning rate (default: 0.001)')
    parser.add_argument('--output-dir', '-o', default='results', help='Output directory for results (default: results)')
    parser.add_argument('--snapshot-dir', default='snapshots', help='Directory for epoch snapshots (default: snapshots)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adamw', 'muon', 'sgd', 'sgd_momentum', 'rmsprop'], help='Optimizer to use (default: adam)')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum factor for SGD with momentum (default: 0.9)')
    parser.add_argument('--adam-eps', type=float, default=1e-8, help='Epsilon for Adam/AdamW denominator stability (default: 1e-8)')
    parser.add_argument('--weight-decay', type=float, default=0.0, help='Weight decay for Adam/AdamW (default: 0.0)')
    parser.add_argument('--amsgrad', action='store_true', help='Use AMSGrad variant for Adam/AdamW')
    parser.add_argument('--muon-eps', type=float, default=1e-7, help='Epsilon for Muon Newton-Schulz numerical stability (default: 1e-7)')
    parser.add_argument('--muon-momentum', type=float, default=0.95, help='Momentum for Muon (default: 0.95)')
    parser.add_argument('--muon-ns-steps', type=int, default=5, help='Newton-Schulz iteration steps for Muon (default: 5)')
    parser.add_argument('--muon-nesterov', action=argparse.BooleanOptionalAction, default=True, help='Enable Nesterov momentum for Muon (default: enabled)')
    parser.add_argument('--muon-adjust-lr-fn', type=str, default=None, choices=['original', 'match_rms_adamw'], help='Muon LR adjustment function (default: PyTorch default)')
    parser.add_argument('--grad-clip-norm', type=float, default=None, help='Clip gradient norm before optimizer step (default: disabled)')
    parser.add_argument('--use-compile', action='store_true', help='Use torch.compile to accelerate the neural network')
    parser.add_argument('--save-interval', type=int, default=1, help='Save boundary visualization every N epochs (default: 1)')
    parser.add_argument('--snapshot-format', choices=['png', 'jpg'], default='png', help='Snapshot image format (default: png)')
    parser.add_argument('--png-compression', type=int, default=None, choices=range(10), metavar='{0-9}', help='PNG compression level for snapshots (OpenCV default if omitted)')
    parser.add_argument('--jpeg-quality', type=int, default=90, help='JPEG quality for snapshots when --snapshot-format jpg (default: 90)')
    parser.add_argument('--chunk-size', type=int, default=128 * 1024, help='Number of points to process at once during visualization (default: 128K)')
    parser.add_argument('--empty-cache-each-viz-chunk', action='store_true', help='Call torch.cuda.empty_cache() after each visualization chunk to reduce fragmentation at the cost of speed')
    parser.add_argument('--log-file', type=str, default=None, help='Path to the log file (default: log to console only)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint file to resume training from')
    parser.add_argument('--resume-optimizer', type=str, default=None, choices=['adam', 'adamw', 'muon', 'sgd', 'sgd_momentum', 'rmsprop'], help='Override optimizer when resuming')
    parser.add_argument('--resume-lr', type=float, default=None, help='Override learning rate when resuming')
    parser.add_argument('--checkpoint-interval', type=int, default=None, help='Save checkpoint every N epochs')
    parser.add_argument('--checkpoint-dir', type=str, default=None, help='Directory to save checkpoints')
    parser.add_argument('--kolmogorov-shape', type=str, default=None, help='Weight predictor network shape (e.g., "[5, 5]")')
    parser.add_argument('--kolmogorov-weight', type=float, default=0.0, help='Weight for Kolmogorov regularization loss')
    parser.add_argument('--kolmogorov-loss-type', type=str, default='cross_entropy', choices=['mse', 'cross_entropy', 'gaussian_nll', 'laplacian_nll'], help='Loss function for weight prediction')
    parser.add_argument('--kolmogorov-bins', type=int, default=256, help='Number of bins for cross_entropy quantization')
    parser.add_argument('--kolmogorov-weight-min', type=float, default=-3.0, help='Minimum weight value for quantization')
    parser.add_argument('--kolmogorov-weight-max', type=float, default=3.0, help='Maximum weight value for quantization')
    parser.add_argument('--kolmogorov-lr', type=float, default=None, help='Learning rate for weight predictor')


def _network_shape_b64(args, include_final_activation=False):
    params = f"{args.shape}_{args.points}_{args.batch_size}_{args.learning_rate}_{args.optimizer}"
    if include_final_activation:
        params += f"_{args.final_activation}"
    if args.optimizer == 'sgd_momentum':
        params += f"_mom{args.momentum}"
    return base64.b64encode(params.encode()).decode()


def _run(args, task_config, network_shape_b64, logger):
    logger.info(f"=== Starting {'polytope-classifier' if task_config.name == 'classifier' else 'polytope-viz-nn'} training ===")
    logger.info(f"Input image: {args.input}")
    logger.info(f"Network shape: {args.shape}")
    logger.info(f"Training epochs: {args.epochs}")
    logger.info(f"Points: {args.points}")
    logger.info(f"Optimizer: {args.optimizer}")
    logger.info(f"Learning rate: {args.learning_rate}")
    if args.optimizer in {'adam', 'adamw'} or args.resume_optimizer in {'adam', 'adamw'}:
        logger.info(f"Adam eps: {args.adam_eps}")
        logger.info(f"Weight decay: {args.weight_decay}")
        logger.info(f"AMSGrad: {args.amsgrad}")
    if args.optimizer == 'muon' or args.resume_optimizer == 'muon':
        logger.info(f"Muon eps: {args.muon_eps}")
        logger.info(f"Muon momentum: {args.muon_momentum}")
        logger.info(f"Muon Newton-Schulz steps: {args.muon_ns_steps}")
        logger.info(f"Muon Nesterov: {args.muon_nesterov}")
        logger.info(f"Muon LR adjustment: {args.muon_adjust_lr_fn or 'default'}")
        logger.info(f"Muon fallback AdamW eps: {args.adam_eps}")
        logger.info(f"Weight decay: {args.weight_decay}")
    if args.grad_clip_norm is not None:
        logger.info(f"Gradient clip norm: {args.grad_clip_norm}")
    if task_config.name == 'classifier':
        logger.info(f"Final activation: {args.final_activation}")
    logger.info(f"Output directory: {args.output_dir}")
    if args.seed is not None:
        logger.info(f"Random seed: {args.seed}")
    set_random_seed(args.seed, logger)
    try:
        full_pipeline(
            input_path=args.input,
            task_config=task_config,
            is_video=False,
            train_size=args.points,
            layer_sizes=parse_layer_sizes(args.shape),
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            output_dir=args.output_dir,
            snapshot_dir=args.snapshot_dir,
            random_seed=args.seed,
            network_shape_b64=network_shape_b64,
            network_shape_str=args.shape,
            debug=args.debug,
            args=args,
            logger=logger,
        )
        logger.info('=== Training completed successfully ===')
    except Exception as e:
        logger.error(f"Training failed with error: {str(e)}")
        logger.exception('Exception details:')
        raise


def run_grayscale_cli(argv=None):
    parser = argparse.ArgumentParser(description='Neural network experiments for image processing.')
    _add_common_arguments(parser)
    parser.add_argument('--smoothing-sigma', type=float, default=3.0, help='Sigma parameter for Gaussian kernel smoothing (default: 3.0)')
    args = parser.parse_args(argv)
    logger = setup_logger(GRAYSCALE_TASK.logger_name, args.log_file, args.debug)
    _run(args, GRAYSCALE_TASK, _network_shape_b64(args), logger)


def run_classifier_cli(argv=None):
    parser = argparse.ArgumentParser(description='Neural network RGB classification experiments.')
    _add_common_arguments(parser)
    parser.add_argument('--final-activation', type=str, default='relu', choices=['relu', 'sigmoid', 'leaky_relu'], help='Activation function for final layer (default: relu)')
    args = parser.parse_args(argv)
    logger = setup_logger(CLASSIFIER_TASK.logger_name, args.log_file, args.debug)
    _run(args, CLASSIFIER_TASK, _network_shape_b64(args, include_final_activation=True), logger)
