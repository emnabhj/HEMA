"""
Hyperparameters for the HEMA model based on the research paper.
These values were determined through grid search and ablation studies.
"""

# Visual processing hyperparameters
VISUAL_CONFIG = {
    'frame_dim': (128, 64),  # Input frame dimensions (height, width)
    'lp_dim': 256,  # Linear projection output dimension
    'lp_sparsity': 0.01,  # L1 regularization coefficient for LP layer
    'capsnet_routing_iterations': 3,  # Number of routing iterations in CapsNet
    'augmentation': {
        'rotation_range': (-15, 15),  # Rotation range in degrees
        'gaussian_noise_range': (0.0, 0.2),  # Range for Gaussian noise
    }
}

# Audio processing hyperparameters
AUDIO_CONFIG = {
    'sample_rate': 16000,  # Audio sample rate in Hz
    'n_fft': 400,  # FFT window size
    'hop_length': 160,  # Hop length for STFT (10ms at 16kHz)
    'n_mels': 80,  # Number of mel filterbanks
    'augmentation': {
        'noise_level_range': (0.0, 0.2),  # Range for noise level (eta)
        'time_stretch_range': (0.8, 1.2),  # Range for time stretching (alpha)
        'critical_bands': {
            'fricatives': (1000, 4000),  # Hz range for fricatives
            'plosives': (0, 500),  # Hz range for plosives
        }
    }
}

# Fusion module hyperparameters
FUSION_CONFIG = {
    'attention_dim': 16,  # Dimension of attention projections
    'contrastive_loss_weight': 0.2,  # Weight for contrastive loss (beta)
}

# Training hyperparameters
TRAINING_CONFIG = {
    'batch_size': 32,
    'initial_lr': 1e-4,
    'min_lr': 1e-6,
    'weight_decay': 0.05,
    'max_epochs': 100,
    'pretrain_epochs': 20,  # For VoxCeleb2 pretraining
    'fine_tuning_lr': 5e-5,  # Reduced learning rate for fine-tuning
    'progressive_augmentation': {
        'initial_noise_level': 0.05,
        'final_noise_level': 0.2,
        'warmup_epochs': 10,
    }
}

# Model architecture hyperparameters
MODEL_CONFIG = {
    'visual': {
        'input_channels': 1,  # Grayscale images
        'lp_input_dim': 128 * 64,  # 128x64 pixels
        'primary_capsules': 32,  # Number of primary capsule types
        'primary_capsule_dim': 8,  # Dimension of primary capsule output
        'digit_capsules': 10,  # Number of digit capsule types
        'digit_capsule_dim': 16,  # Dimension of digit capsule output
    },
    'audio': {
        'input_channels': 1,  # Single channel audio spectrogram
        'conformer_dim': 256,  # Dimension of conformer encoders
        'conformer_layers': 6,  # Number of conformer encoder layers
        'conformer_heads': 4,  # Number of attention heads in conformer
    },
    'fusion': {
        'hidden_size': 512,  # Hidden size of fusion module
    }
}

# Dictionary mapping SNR values to noise levels for testing
SNR_TO_NOISE_LEVEL = {
    10: 0.05,
    5: 0.1,
    0: 0.15,
    -5: 0.2,
    -10: 0.25
}


class HParams:
    """
    Hyperparameters for HEMA model
    This class stores all hyperparameters as attributes and allows easy access.
    """

    def __init__(self):
        # Training hyperparameters
        self.batch_size = TRAINING_CONFIG['batch_size']
        self.initial_lr = TRAINING_CONFIG['initial_lr']
        self.min_lr = TRAINING_CONFIG['min_lr']
        self.weight_decay = TRAINING_CONFIG['weight_decay']
        self.max_epochs = TRAINING_CONFIG['max_epochs']
        self.pretrain_epochs = TRAINING_CONFIG['pretrain_epochs']
        self.fine_tuning_lr = TRAINING_CONFIG['fine_tuning_lr']
        self.progressive_augmentation = TRAINING_CONFIG['progressive_augmentation']

        # Visual processing hyperparameters
        self.visual_config = VISUAL_CONFIG

        # Audio processing hyperparameters
        self.audio_config = AUDIO_CONFIG

        # Fusion module hyperparameters
        self.fusion_config = FUSION_CONFIG

        # Model architecture hyperparameters
        self.model_config = MODEL_CONFIG

        # SNR to noise level mapping
        self.snr_to_noise_level = SNR_TO_NOISE_LEVEL

        # Other constants
        self.vocab_size = 5000  # Placeholder
        self.pad_idx = 0  # Padding index
        self.fusion_dim = FUSION_CONFIG['attention_dim']
        self.contrastive_weight = FUSION_CONFIG['contrastive_loss_weight']

    def get_all_params(self):
        """
        Return all hyperparameters in a dictionary format.
        """
        params = {
            "batch_size": self.batch_size,
            "initial_lr": self.initial_lr,
            "min_lr": self.min_lr,
            "weight_decay": self.weight_decay,
            "max_epochs": self.max_epochs,
            "pretrain_epochs": self.pretrain_epochs,
            "fine_tuning_lr": self.fine_tuning_lr,
            "progressive_augmentation": self.progressive_augmentation,
            "visual_config": self.visual_config,
            "audio_config": self.audio_config,
            "fusion_config": self.fusion_config,
            "model_config": self.model_config,
            "snr_to_noise_level": self.snr_to_noise_level,
            "vocab_size": self.vocab_size,
            "pad_idx": self.pad_idx,
            "fusion_dim": self.fusion_dim,
            "contrastive_weight": self.contrastive_weight
        }
        return params
