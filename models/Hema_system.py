#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.audio_model import AudioModel
from models.visual_model import VisualModel
from models.fusion import AttentionFusionModule
from utils.reliability import estimate_audio_snr, compute_visual_sharpness


class HEMASystem(nn.Module):
    """
    Hybrid Efficient Multimodal Architecture (HEMA) for Audiovisual Speech Recognition.

    Combines:
    1. Linear Projection + CapsNet visual pathway
    2. Augmented audio processing
    3. Attention-guided fusion with reliability metrics
    """

    def __init__(self, args, hyperparams):
        super(HEMASystem, self).__init__()
        self.args = args
        self.hyperparams = hyperparams

        # Initialize the audio processing branch
        self.audio_model = AudioModel(
            input_dim=hyperparams["audio_input_dim"],
            hidden_dim=hyperparams["audio_hidden_dim"],
            output_dim=hyperparams["audio_output_dim"],
            dropout=hyperparams["audio_dropout"]
        )

        # Initialize the visual processing branch with LP+CapsNet
        self.visual_model = VisualModel(
            input_dim=(hyperparams["visual_input_dim_h"],
                       hyperparams["visual_input_dim_w"],
                       hyperparams["visual_input_channels"]),
            lp_dim=hyperparams["lp_dim"],
            capsnet_dim=hyperparams["capsnet_dim"],
            capsnet_n_routes=hyperparams["capsnet_n_routes"],
            dropout=hyperparams["visual_dropout"]
        )

        # Initialize the attention fusion module
        self.fusion = AttentionFusionModule(
            audio_dim=hyperparams["audio_output_dim"],
            visual_dim=hyperparams["capsnet_dim"],
            output_dim=hyperparams["fusion_output_dim"]
        )

        # Output classifier for word prediction
        self.classifier = nn.Linear(
            hyperparams["fusion_output_dim"],
            hyperparams["vocab_size"]
        )

        # Loss functions
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrastive_loss_weight = hyperparams["contrastive_loss_weight"]

    def contrastive_loss(self, audio_features, visual_features):
        """
        Contrastive loss to align audio and visual representations
        """
        batch_size = audio_features.size(0)

        # Normalize features for cosine similarity
        audio_features = F.normalize(audio_features, p=2, dim=1)
        visual_features = F.normalize(visual_features, p=2, dim=1)

        # Compute similarity matrix
        similarity = torch.mm(audio_features, visual_features.t())

        # Labels: diagonal elements (i,i) are positive pairs
        labels = torch.arange(batch_size).to(audio_features.device)

        # Compute loss (cross-entropy with temperature scaling)
        temperature = 0.07
        loss = F.cross_entropy(similarity / temperature, labels)

        return loss

    def compute_reliability_metrics(self, audio_input, visual_input):
        """
        Compute real-time reliability metrics for audio and visual inputs
        """
        # Estimate audio SNR
        audio_snr = estimate_audio_snr(audio_input)

        # Compute visual sharpness
        visual_sharpness = compute_visual_sharpness(visual_input)

        return audio_snr, visual_sharpness

    def forward(self, audio_input, visual_input, audio_reliability=None, visual_reliability=None):
        """
        Forward pass through the HEMA system

        Args:
            audio_input: Tensor of audio spectrograms [batch_size, channels, time, freq]
            visual_input: Tensor of lip region images [batch_size, channels, height, width]
            audio_reliability: Optional pre-computed audio reliability scores
            visual_reliability: Optional pre-computed visual reliability scores

        Returns:
            logits: Output class logits
            audio_features: Audio feature representations
            visual_features: Visual feature representations
            fused_features: Fused multimodal representations
        """
        # Process audio input
        audio_features = self.audio_model(audio_input)

        # Process visual input
        visual_features = self.visual_model(visual_input)

        # If reliability metrics are not provided, compute them
        if audio_reliability is None or visual_reliability is None:
            audio_reliability, visual_reliability = self.compute_reliability_metrics(
                audio_input, visual_input
            )

        # Normalize reliability scores to sum to 1
        total_reliability = audio_reliability + visual_reliability + 1e-8  # Avoid division by zero
        audio_weight = audio_reliability / total_reliability
        visual_weight = visual_reliability / total_reliability

        # Fuse features with attention and reliability weights
        fused_features = self.fusion(
            audio_features, visual_features,
            audio_weight, visual_weight
        )

        # Final classification
        logits = self.classifier(fused_features)

        return logits, audio_features, visual_features, fused_features

    def compute_loss(self, logits, targets, audio_features, visual_features):
        """
        Compute the combined loss (CE + contrastive)
        """
        # Cross-entropy loss for classification
        ce_loss = self.ce_loss(logits, targets)

        # Contrastive loss for audio-visual alignment
        cont_loss = self.contrastive_loss(audio_features, visual_features)

        # Combined loss
        total_loss = ce_loss + self.contrastive_loss_weight * cont_loss

        return total_loss, ce_loss, cont_loss

    def infer(self, audio_input, visual_input=None):
        """
        Inference method that handles missing modalities gracefully
        """
        device = next(self.parameters()).device
        batch_size = audio_input.size(0) if audio_input is not None else visual_input.size(0)

        # Handle missing audio
        if audio_input is None:
            # Create zero tensor for audio with proper dimensions
            audio_input = torch.zeros(
                (batch_size, self.hyperparams["audio_input_dim"],
                 self.hyperparams["audio_seq_len"]),
                device=device
            )
            audio_reliability = torch.zeros(batch_size, 1, device=device)
            visual_reliability = torch.ones(batch_size, 1, device=device)

        # Handle missing visual
        elif visual_input is None:
            # Create zero tensor for visual with proper dimensions
            visual_input = torch.zeros(
                (batch_size, self.hyperparams["visual_input_channels"],
                 self.hyperparams["visual_input_dim_h"],
                 self.hyperparams["visual_input_dim_w"]),
                device=device
            )
            audio_reliability = torch.ones(batch_size, 1, device=device)
            visual_reliability = torch.zeros(batch_size, 1, device=device)

        # Both modalities present
        else:
            audio_reliability, visual_reliability = self.compute_reliability_metrics(
                audio_input, visual_input
            )

        # Forward pass
        logits, _, _, _ = self.forward(
            audio_input, visual_input,
            audio_reliability, visual_reliability
        )

        # Get predictions
        predictions = torch.argmax(logits, dim=1)

        return predictions, logits
