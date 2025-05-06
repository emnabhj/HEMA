#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import time
import torch
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm


from config.hyperparameters import HParams
from data.dataset import LRS2Dataset, LRWDataset, VoxCeleb2Dataset
from models.audio_model import AudioModel
from models.visual_model import VisualModel
from models.fusion import AttentionFusionModule
from utils.metrics import compute_wer
from utils.reliability import estimate_audio_snr, compute_visual_sharpness
from config.args import parse_args

args = parse_args()

def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def count_parameters(model):
    """Count the number of trainable parameters in the model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class HEMA(nn.Module):
    """Hybrid Efficient Multimodal Architecture for AVSR"""

    def __init__(self, hparams, args):
        super(HEMA, self).__init__()
        self.visual_model = VisualModel(hparams)
        self.audio_model = AudioModel(hparams)
        self.fusion_module = AttentionFusionModule(hparams)

        # Output projection layer (to vocabulary size)
        self.output_layer = nn.Linear(hparams.fusion_dim, hparams.vocab_size)

        self.contrastive_weight = hparams.contrastive_weight
        self.criterion = nn.CrossEntropyLoss(ignore_index=hparams.pad_idx)

    def forward(self, audio_feats, visual_feats, audio_reliability=None, visual_reliability=None):
        audio_out = self.audio_model(audio_feats)
        visual_out = self.visual_model(visual_feats)

        # If reliability metrics are provided, use them for adaptive fusion
        if audio_reliability is not None and visual_reliability is not None:
            fused_out = self.fusion_module(audio_out, visual_out,
                                           audio_reliability, visual_reliability)
        else:
            fused_out = self.fusion_module(audio_out, visual_out)

        logits = self.output_layer(fused_out)
        return logits, audio_out, visual_out

    def compute_loss(self, logits, targets, audio_out, visual_out):
        # Main cross-entropy loss
        ce_loss = self.criterion(logits.view(-1, logits.size(-1)), targets.view(-1))

        # Optional contrastive loss for alignment between modalities
        if self.contrastive_weight > 0:
            # Normalize embeddings
            audio_emb = F.normalize(audio_out, p=2, dim=-1)
            visual_emb = F.normalize(visual_out, p=2, dim=-1)

            # Compute similarity matrix
            sim_matrix = torch.matmul(audio_emb, visual_emb.transpose(1, 2))

            # Labels are on the diagonal (identity matrix)
            batch_size, seq_len = audio_emb.size(0), audio_emb.size(1)
            labels = torch.eye(seq_len).unsqueeze(0).repeat(batch_size, 1, 1).to(audio_emb.device)

            # Contrastive loss
            contrastive_loss = F.binary_cross_entropy_with_logits(sim_matrix, labels)

            # Total loss
            loss = ce_loss + self.contrastive_weight * contrastive_loss
        else:
            loss = ce_loss

        return loss


def train_epoch(model, dataloader, optimizer, device, epoch, args):
    model.train()
    epoch_loss = 0

    progress_bar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for batch_idx, batch in enumerate(progress_bar):
        # Get the data
        audio_features = batch['audio_features'].to(device)
        visual_features = batch['visual_features'].to(device)
        targets = batch['text_tokens'].to(device)

        # Estimate reliability metrics
        audio_snr = None
        visual_sharpness = None

        if args.use_reliability_metrics:
            # Compute reliability metrics
            audio_snr = estimate_audio_snr(batch['audio_raw']).to(device)
            visual_sharpness = compute_visual_sharpness(batch['visual_raw']).to(device)

        # Forward pass
        logits, audio_out, visual_out = model(audio_features, visual_features,
                                              audio_snr, visual_sharpness)

        # Compute loss
        loss = model.compute_loss(logits, targets, audio_out, visual_out)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        optimizer.step()

        # Update metrics
        epoch_loss += loss.item()

        # Update progress bar
        progress_bar.set_postfix(loss=f'{loss.item():.4f}')

        if batch_idx % args.log_interval == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(audio_features)}/{len(dataloader.dataset)} '
                  f'({100. * batch_idx / len(dataloader):.0f}%)]\tLoss: {loss.item():.6f}')

    avg_loss = epoch_loss / len(dataloader)
    print(f'Train Epoch: {epoch}, Average Loss: {avg_loss:.4f}')

    return avg_loss


def validate(model, dataloader, device):
    model.eval()
    val_loss = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Validation'):
            # Get the data
            audio_features = batch['audio_features'].to(device)
            visual_features = batch['visual_features'].to(device)
            targets = batch['text_tokens'].to(device)

            # Forward pass
            logits, audio_out, visual_out = model(audio_features, visual_features)

            # Compute loss
            loss = model.compute_loss(logits, targets, audio_out, visual_out)
            val_loss += loss.item()

            # Get predictions
            preds = torch.argmax(logits, dim=-1)

            # Store predictions and targets for WER calculation
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    # Calculate WER
    wer = compute_wer(all_preds, all_targets)

    avg_loss = val_loss / len(dataloader)
    print(f'Validation Loss: {avg_loss:.4f}, WER: {wer:.4f}')

    return avg_loss, wer


def main():
    # Parse arguments
    args = Args().parse()

    # Set random seed for reproducibility
    set_seed(args.seed)

    # Get hyperparameters
    hparams = HParams()

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create datasets
    train_dataset = LRS2Dataset(args.lrs2_path, 'train', hparams.max_seq_len)

    # Add additional datasets if specified
    if args.use_multisource:
        lrw_dataset = LRWDataset(args.lrw_path, 'train', hparams.max_seq_len)
        voxceleb_dataset = VoxCeleb2Dataset(args.voxceleb_path, 'train', hparams.max_seq_len)

        # Combine datasets (simple concatenation for now)
        # A more sophisticated approach would involve weighted sampling
        from torch.utils.data import ConcatDataset
        train_dataset = ConcatDataset([train_dataset, lrw_dataset, voxceleb_dataset])

    val_dataset = LRS2Dataset(args.lrs2_path, 'val', hparams.max_seq_len)

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Create model
    model = HEMA(hparams, args).to(device)

    # Print model statistics
    num_params = count_parameters(model)
    print(f"Number of trainable parameters: {num_params}")

    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create learning rate scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=args.min_learning_rate,
    )

    # Training loop
    best_val_wer = float('inf')
    for epoch in range(1, args.epochs + 1):
        # Train for one epoch
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch, args)

        # Validate
        val_loss, val_wer = validate(model, val_loader, device)

        # Update learning rate
        scheduler.step()

        # Save checkpoint if validation WER improved
        if val_wer < best_val_wer:
            best_val_wer = val_wer
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_wer': val_wer,
                'train_loss': train_loss,
                'val_loss': val_loss,
            }
            torch.save(checkpoint, os.path.join(args.checkpoint_dir, f'best_model_epoch_{epoch}_wer_{val_wer:.4f}.pt'))
            print(f'Checkpoint saved at epoch {epoch} with WER {val_wer:.4f}')

        # Always save latest model
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'val_wer': val_wer,
            'train_loss': train_loss,
            'val_loss': val_loss,
        }
        torch.save(checkpoint, os.path.join(args.checkpoint_dir, 'latest_model.pt'))

    print(f'Training completed. Best validation WER: {best_val_wer:.4f}')


if __name__ == '__main__':
    main()