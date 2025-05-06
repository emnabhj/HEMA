import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
import torchaudio
import random
from typing import Optional, Tuple, List, Dict, Union
import pandas as pd

from data.preprocess_audio import AudioPreprocessor
from data.preprocess_visual import LipPreprocessor
from utils.reliability import estimate_audio_snr, compute_visual_sharpness

class AVSRDataset(Dataset):
    """Base class for Audio-Visual Speech Recognition datasets."""

    def __init__(
            self,
            root_dir: str,
            split: str,
            mode: str = "av",  # 'a' for audio-only, 'v' for visual-only, 'av' for audio-visual
            max_seq_len: int = 150,
            augment: bool = False,
            noise_level: float = 0.1,
    ):
        """
        Initialize the dataset.

        Args:
            root_dir: Root directory of the dataset
            split: 'train', 'val', or 'test'
            mode: Modality mode ('a', 'v', or 'av')
            max_seq_len: Maximum sequence length
            augment: Whether to apply data augmentation
            noise_level: Level of noise for audio augmentation
        """
        self.root_dir = root_dir
        self.split = split
        self.mode = mode
        self.max_seq_len = max_seq_len
        self.augment = augment and split == 'train'  # Only augment during training
        self.noise_level = noise_level

        self.audio_processor = AudioPreprocessor()
        self.visual_processor = LipPreprocessor()

        self.audio_reliability_estimator = AudioReliabilityEstimator()
        self.visual_reliability_estimator = VisualReliabilityEstimator()

        self.samples = []
        self._load_data()

    def _load_data(self):
        """Load dataset information. To be implemented by subclasses."""
        raise NotImplementedError

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        sample = self.samples[idx]

        audio_feat = torch.zeros(1)
        visual_feat = torch.zeros(1)

        if self.mode in ['a', 'av']:
            audio_feat = self._get_audio_features(sample)

        if self.mode in ['v', 'av']:
            visual_feat = self._get_visual_features(sample)

        transcript = self._get_transcript(sample)

        audio_reliability = 1.0
        visual_reliability = 1.0

        if self.mode == 'av':
            audio_reliability = self.audio_reliability_estimator.estimate(audio_feat)
            visual_reliability = self.visual_reliability_estimator.estimate(visual_feat)

        return {
            'audio': audio_feat,
            'visual': visual_feat,
            'text': transcript,
            'audio_reliability': audio_reliability,
            'visual_reliability': visual_reliability,
            'sample_id': sample['id']
        }

    def _get_audio_features(self, sample):
        """Extract audio features."""
        audio_path = sample['audio_path']
        waveform, sample_rate = torchaudio.load(audio_path)

        audio_feat = self.audio_processor.process(
            waveform,
            sample_rate,
            augment=self.augment,
            noise_level=self.noise_level
        )

        return audio_feat

    def _get_visual_features(self, sample):
        """Extract visual features from lip region."""
        video_path = sample['video_path']
        visual_feat = self.visual_processor.process_video(
            video_path,
            augment=self.augment
        )

        return visual_feat

    def _get_transcript(self, sample):
        """Get the transcript."""
        return sample['transcript']

    def collate_fn(self, batch):
        """Custom collate function to handle variable length sequences."""
        batch = sorted(batch, key=lambda x: x['audio'].shape[0] if self.mode == 'a' else x['visual'].shape[0],
                       reverse=True)

        if self.mode in ['a', 'av']:
            audio_lengths = [x['audio'].shape[0] for x in batch]
            max_audio_len = min(max(audio_lengths), self.max_seq_len)
            padded_audio = torch.zeros(len(batch), max_audio_len, batch[0]['audio'].shape[1])

            for i, x in enumerate(batch):
                length = min(x['audio'].shape[0], max_audio_len)
                padded_audio[i, :length] = x['audio'][:length]
        else:
            padded_audio = torch.zeros(1)
            audio_lengths = [0]

        if self.mode in ['v', 'av']:
            visual_lengths = [x['visual'].shape[0] for x in batch]
            max_visual_len = min(max(visual_lengths), self.max_seq_len)
            padded_visual = torch.zeros(len(batch), max_visual_len, batch[0]['visual'].shape[1])

            for i, x in enumerate(batch):
                length = min(x['visual'].shape[0], max_visual_len)
                padded_visual[i, :length] = x['visual'][:length]
        else:
            padded_visual = torch.zeros(1)
            visual_lengths = [0]

        transcripts = [x['text'] for x in batch]
        audio_reliability = torch.tensor([x['audio_reliability'] for x in batch])
        visual_reliability = torch.tensor([x['visual_reliability'] for x in batch])
        sample_ids = [x['sample_id'] for x in batch]

        return {
            'audio': padded_audio,
            'audio_lengths': torch.tensor(audio_lengths),
            'visual': padded_visual,
            'visual_lengths': torch.tensor(visual_lengths),
            'text': transcripts,
            'audio_reliability': audio_reliability,
            'visual_reliability': visual_reliability,
            'sample_ids': sample_ids
        }

class LRS2Dataset(AVSRDataset):
    """Dataset class for LRS2."""

    def __init__(
            self,
            root_dir: str,
            split: str,
            mode: str = "av",
            max_seq_len: int = 150,
            augment: bool = False,
            noise_level: float = 0.1,
    ):
        super().__init__(root_dir, split, mode, max_seq_len, augment, noise_level)

    def _load_data(self):
        """Load LRS2 dataset information."""
        split_file = os.path.join(self.root_dir, f"{self.split}.txt")

        with open(split_file, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            parts = line.split(' ', 1)
            if len(parts) != 2:
                continue

            sample_id, transcript = parts

            video_path = os.path.join(self.root_dir, self.split, sample_id, 'video.mp4')
            audio_path = os.path.join(self.root_dir, self.split, sample_id, 'audio.wav')

            if not (os.path.exists(video_path) and os.path.exists(audio_path)):
                continue

            self.samples.append({
                'id': sample_id,
                'video_path': video_path,
                'audio_path': audio_path,
                'transcript': transcript
            })

