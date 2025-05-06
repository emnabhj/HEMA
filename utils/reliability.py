# utils/reliability.py

import torch
import torch.nn.functional as F


def estimate_audio_snr(audio_signal):
    """
    Estimate the Signal-to-Noise Ratio (SNR) for the given audio signal.

    Args:
        audio_signal: Input audio signal [batch_size, time_steps]

    Returns:
        snr_normalized: Normalized SNR value between 0 and 1
    """
    # Compute the noise signal (using a simple approach here, assume first part as noise)
    noise_signal = audio_signal[:, :int(0.1 * audio_signal.shape[1])]  # First 10% of the signal as noise estimate
    clean_signal = audio_signal[:, int(0.1 * audio_signal.shape[1]):]  # Rest as clean signal

    # Compute power of signal and noise
    signal_power = torch.mean(clean_signal ** 2, dim=1)
    noise_power = torch.mean(noise_signal ** 2, dim=1)

    # Compute SNR
    eps = 1e-10  # To avoid division by zero
    snr = signal_power / (noise_power + eps)

    # Convert to dB
    snr_db = 10 * torch.log10(snr + eps)

    # Clip SNR to reasonable range (-10dB to 50dB)
    snr_db = torch.clamp(snr_db, min=-10.0, max=50.0)

    # Normalize to [0, 1]
    snr_normalized = (snr_db + 10) / 60.0

    return snr_normalized


def compute_visual_sharpness(visual_frames):
    """
    Compute the sharpness of visual frames based on the variance of Sobel gradients.

    Args:
        visual_frames: Input visual frames [batch_size, channels, height, width]

    Returns:
        sharpness: Sharpness score between 0 and 1
    """
    batch_size = visual_frames.shape[0]

    # Convert to grayscale if necessary (simple weighted sum for RGB to grayscale conversion)
    if visual_frames.shape[1] > 1:  # If RGB
        gray_frames = 0.299 * visual_frames[:, 0] + 0.587 * visual_frames[:, 1] + 0.114 * visual_frames[:, 2]
    else:
        gray_frames = visual_frames[:, 0]  # Already grayscale

    # Initialize Sobel kernels
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(visual_frames.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(visual_frames.device)

    # Reshape for convolution
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    # Apply Sobel operators to detect edges
    gray_frames = gray_frames.unsqueeze(1)  # Add channel dimension
    grad_x = F.conv2d(gray_frames, sobel_x, padding=1)
    grad_y = F.conv2d(gray_frames, sobel_y, padding=1)

    # Compute gradient magnitude
    grad_magnitude = torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-10)

    # Compute variance of gradient magnitude (sharpness measure)
    sharpness = torch.var(grad_magnitude.view(batch_size, -1), dim=1)

    # Normalize sharpness to [0, 1]
    sharpness = torch.clamp(sharpness / 100.0, min=0.0, max=1.0)

    return sharpness.unsqueeze(1)  # Add dimension for consistency
