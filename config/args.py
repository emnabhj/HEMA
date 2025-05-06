import argparse
import torch
import os
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description='HEMA: Hybrid Efficient Multimodal Architecture for AVSR'
    )

    # Dataset parameters
    parser.add_argument('--dataset', type=str, choices=['lrs2', 'lrw', 'voxceleb2'], default='lrs2')
    parser.add_argument('--data_root', type=str, default='./datasets/')
    parser.add_argument('--lrs2_directory', type=str, default='./datasets/lrs2')
    parser.add_argument('--lrw_directory', type=str, default='./datasets/lrw')
    parser.add_argument('--voxceleb2_directory', type=str, default='./datasets/voxceleb2')
    parser.add_argument('--noise_directory', type=str, default='./datasets/DEMAND')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mixed_precision', action='store_true')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--log_dir', type=str, default='./logs')
    parser.add_argument('--log_interval', type=int, default=10)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--eval_interval', type=int, default=1)

    # Model parameters
    parser.add_argument('--lp_dim', type=int, default=256)
    parser.add_argument('--capsnet_routing_iterations', type=int, default=3)
    parser.add_argument('--contrastive_loss_weight', type=float, default=0.2)
    parser.add_argument('--lp_regularization', type=float, default=0.01)

    # Audio augmentation
    parser.add_argument('--audio_noise_level', type=float, default=0.1)
    parser.add_argument('--spectral_masking', action='store_true')
    parser.add_argument('--temporal_warping', action='store_true')

    # Modality
    parser.add_argument('--modality', type=str, choices=['audio', 'visual', 'audiovisual'], default='audiovisual')

    # Testing
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--test_noise_levels', nargs='+', type=float, default=[0, -5, -10])
    parser.add_argument('--test_checkpoint', type=str, default=None)
    parser.add_argument('--test_batch_size', type=int, default=16)

    # Multisource training
    parser.add_argument('--multisource', action='store_true')
    parser.add_argument('--pretrain_on_voxceleb2', action='store_true')
    parser.add_argument('--pretrain_epochs', type=int, default=20)

    # Unique run ID
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser.add_argument('--run_id', type=str, default=f'hema_{timestamp}')

    args = parser.parse_args()

    # Create necessary directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    return args
