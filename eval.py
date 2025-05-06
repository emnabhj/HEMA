#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from config.args import parse_args
from models.audio_model import AudioModel
from models.visual_model import VisualModel
from models.fusion import AttentionFusionModule  # Correction de l'importation
from data.dataset import AVSRDataset
from utils.metrics import compute_wer #
from utils.reliability import estimate_audio_snr, compute_visual_sharpness
from config.hyperparameters import HParams

hyperparams = HParams()


class AVSRSystem(nn.Module):
    def __init__(self, args, hyperparameters):
        super(AVSRSystem, self).__init__()
        self.args = args
        self.hyperparams = hyperparameters

        # Initialize models
        self.audio_model = AudioModel(
            input_dim=hyperparameters["audio_input_dim"],
            output_dim=hyperparameters["audio_output_dim"]
        )

        self.visual_model = VisualModel(
            input_dim=(hyperparameters["visual_input_dim_h"],
                       hyperparameters["visual_input_dim_w"],
                       hyperparameters["visual_input_channels"]),
            lp_dim=hyperparameters["lp_dim"],
            capsnet_dim=hyperparameters["capsnet_dim"],
            capsnet_n_routes=hyperparameters["capsnet_n_routes"]
        )

        # Initialize fusion module
        self.fusion = AttentionFusionModule(  # Correction ici
            audio_dim=hyperparameters["audio_output_dim"],
            visual_dim=hyperparameters["capsnet_dim"],
            output_dim=hyperparameters["fusion_output_dim"]
        )

        # Output layer for word classification
        self.classifier = nn.Linear(
            hyperparameters["fusion_output_dim"],
            hyperparameters["vocab_size"]
        )

    def forward(self, audio_input, visual_input, audio_reliability=None, visual_reliability=None):
        # Process audio input
        audio_features = self.audio_model(audio_input)

        # Process visual input
        visual_features = self.visual_model(visual_input)

        # If reliability metrics are not provided, compute them
        if audio_reliability is None:
            audio_reliability = torch.ones(audio_input.size(0), 1).to(audio_input.device)
        if visual_reliability is None:
            visual_reliability = torch.ones(visual_input.size(0), 1).to(visual_input.device)

        # Fusion
        fused_features = self.fusion(audio_features, visual_features,
                                     audio_reliability, visual_reliability)

        # Classification
        logits = self.classifier(fused_features)

        return logits, audio_features, visual_features, fused_features


def evaluate(model, dataloader, device, args):
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # Unpack batch
            audio_input = batch['audio'].to(device)
            visual_input = batch['video'].to(device)
            targets = batch['text'].to(device)

            # Compute reliability metrics
            audio_snr = batch.get('audio_snr', None)
            if audio_snr is not None:
                audio_snr = audio_snr.to(device)

            visual_sharpness = batch.get('visual_sharpness', None)
            if visual_sharpness is not None:
                visual_sharpness = visual_sharpness.to(device)

            # Forward pass
            logits, _, _, _ = model(audio_input, visual_input, audio_snr, visual_sharpness)

            # Calculate loss
            loss = criterion(logits, targets)
            total_loss += loss.item()

            # Store predictions and targets
            predictions = torch.argmax(logits, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

            if batch_idx % args.log_interval == 0:
                print(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")

    # Calculate metrics
    wer = compute_wer(all_predictions, all_targets, dataloader.dataset.idx_to_word)
    cer = compute_cer(all_predictions, all_targets, dataloader.dataset.idx_to_word)

    avg_loss = total_loss / len(dataloader)

    return {
        'loss': avg_loss,
        'wer': wer,
        'cer': cer
    }


def evaluate_with_noise(model, dataset, device, args):
    """Evaluate model under different noise conditions"""
    noise_levels = [-10, -5, 0, 5, 10, 15, 20]  # SNR in dB
    results = {}

    for snr in noise_levels:
        print(f"\nEvaluating with SNR = {snr} dB")
        # Create dataset with specific noise level
        dataset.set_noise_level(snr)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers
        )

        # Evaluate
        metrics = evaluate(model, dataloader, device, args)
        results[snr] = metrics
        print(f"SNR = {snr} dB: WER = {metrics['wer']:.4f}, CER = {metrics['cer']:.4f}")

    return results


def evaluate_modalities(model, dataset, device, args):
    """Evaluate each modality separately and combined"""
    results = {}

    # Audio-only evaluation
    print("\nEvaluating Audio-only")
    dataset.set_mode('audio')  # Assure-toi que `set_mode` existe dans ton dataset
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    audio_metrics = evaluate(model, dataloader, device, args)
    results['audio'] = audio_metrics

    # Visual-only evaluation
    print("\nEvaluating Visual-only")
    dataset.set_mode('visual')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    visual_metrics = evaluate(model, dataloader, device, args)
    results['visual'] = visual_metrics

    # Combined evaluation
    print("\nEvaluating Audio-Visual")
    dataset.set_mode('audiovisual')
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )
    av_metrics = evaluate(model, dataloader, device, args)
    results['audiovisual'] = av_metrics

    # Print summary
    print("\n=== Evaluation Summary ===")
    print(f"Audio-only WER: {results['audio']['wer']:.4f}")
    print(f"Visual-only WER: {results['visual']['wer']:.4f}")
    print(f"Audio-Visual WER: {results['audiovisual']['wer']:.4f}")

    return results


def main():
    args = parse_args()
    hyperparameters = hyperparams  # Correction de l'accès aux hyperparamètres

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create test dataset
    test_dataset = AVSRDataset(
        root_dir=args.data_dir,
        split='test',
        max_len=args.max_len,
        augment=False
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers
    )

    # Load model
    model = AVSRSystem(args, hyperparameters)

    # Load checkpoint
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        raise ValueError("No checkpoint provided for evaluation")

    model.to(device)

    # Standard evaluation
    if args.eval_mode == 'standard':
        metrics = evaluate(model, test_loader, device, args)
        print(f"Test Loss: {metrics['loss']:.4f}")
        print(f"Test WER: {metrics['wer']:.4f}")
        print(f"Test CER: {metrics['cer']:.4f}")

    # Evaluate with different noise levels
    elif args.eval_mode == 'noise':
        results = evaluate_with_noise(model, test_dataset, device, args)

        # Save results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            np.save(os.path.join(args.output_dir, 'noise_results.npy'), results)

    # Evaluate different modalities
    elif args.eval_mode == 'modality':
        results = evaluate_modalities(model, test_dataset, device, args)

        # Save results
        if args.output_dir:
            os.makedirs(args.output_dir, exist_ok=True)
            np.save(os.path.join(args.output_dir, 'modality_results.npy'), results)

    else:
        raise ValueError(f"Unknown evaluation mode: {args.eval_mode}")


if __name__ == "__main__":
    main()
