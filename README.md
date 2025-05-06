# HEMA: Hybrid Efficient Multimodal Architecture for Robust Audiovisual Speech Recognition

This repository contains the implementation of HEMA, a novel framework for robust Audiovisual Speech Recognition (AVSR) as described in the paper "HEMA: A Hybrid Efficient Multimodal Architecture for Robust Audiovisual Speech Recognition".

HEMA combines a lightweight visual front-end with physics-inspired audio augmentation and adaptive multimodal fusion to achieve state-of-the-art performance on the LRS2 dataset.

## Key Features

- **Hybrid Visual Front-end**: Combining Linear Projection (LP) for dimensionality reduction with Capsule Networks (CapsNet) for preserving hierarchical spatial relationships in lip movements
- **Physics-inspired Audio Augmentation**: Including critical band spectral masking and nonlinear time warping for enhanced robustness to diverse acoustic distortions
- **Confidence-aware Fusion Module**: Dynamically weights audio and visual streams based on real-time reliability estimates
- **Multi-source Training**: Leverages multiple datasets (LRS2, LRW, VoxCeleb2) for improved generalization

## Requirements

### System Requirements
- Python 3.8+
- CUDA 11.3+ (if using NVIDIA GPU)
- ffmpeg

### Python Packages
- torch==1.12.1
- torchvision==0.13.1
- numpy==1.23.3
- opencv-python==4.6.0
- tqdm==4.64.1
- matplotlib==3.6.1
- editdistance==0.6.1
- scipy==1.9.2
- librosa==0.9.2
- Pillow==9.2.0
- tensorboardX==2.5.1

## Project Structure

```
hema_avsr/
├── config/
│   ├── args.py                  # Configuration des arguments
│   └── hyperparameters.py       # Hyperparamètres pour les modèles
├── data/
│   ├── dataset.py               # Classes de dataset pour LRS2, LRW et VoxCeleb2
│   ├── preprocess_audio.py      # Prétraitement audio avec augmentation
│   └── preprocess_visual.py     # Prétraitement visuel avec extraction de la région des lèvres
├── models/
│   ├── audio_model.py           # Modèle audio avec traitement des spectrogrammes
│   ├── capsnet.py               # Implémentation du CapsNet
│   ├── fusion.py                # Module de fusion à base d'attention
│   ├── linear_projection.py     # Module de projection linéaire
│   └── visual_model.py          # Modèle visuel avec LP + CapsNet
├── utils/
│   ├── metrics.py               # Calcul du WER et autres métriques
│   └── reliability.py           # Métriques de fiabilité audio/visuelle en temps réel
├── train.py                     # Script d'entraînement principal
├── eval.py                      # Script d'évaluation
└── README.md                    # Documentation
```

## Results

HEMA achieves state-of-the-art performance on the LRS2 benchmark:

| Model | Modality | WER (%) |
|-------|----------|---------|
| Audio Model with Data Augmentation | Audio Only | 1.1% |
| Hybrid LP + CapsNet Model | Visual Only | 8.9% |
| HEMA (Full Integration) | Audio-Visual | 1.0% |
| HEMA | SNR = 0 dB | 2.5% |
| HEMA | SNR = -5 dB | 4.2% |
| HEMA | SNR = -10 dB | 6.8% |

## How to Use

### Data Preparation

1. Download the LRS2 dataset from the [official source](https://www.robots.ox.ac.uk/~vgg/data/lip_reading/lrs2.html)
2. Organize the dataset according to the expected structure
3. Run preprocessing:
   ```
   python preprocess.py --data_path /path/to/lrs2 --dataset lrs2
   ```

### Training

1. Configure training parameters in `config/args.py`
2. Train the visual-only model:
   ```
   python train.py --modality visual
   ```
3. Train the audio-only model:
   ```
   python train.py --modality audio
   ```
4. Train the full audiovisual model:
   ```
   python train.py --modality audiovisual --visual_pretrained /path/to/visual_model.pth --audio_pretrained /path/to/audio_model.pth
   ```



## Training Details

- We train on multiple datasets (LRS2, LRW, VoxCeleb2) for improved generalization
- Physics-inspired data augmentation is applied to the audio stream during training
- We use a hybrid training paradigm:
  1. First pretrain on VoxCeleb2 to learn speaker-agnostic acoustic features
  2. Fine-tune on LRS2 and LRW with reduced learning rate
  3. During fine-tuning, noise augmentation is progressively intensified
- Training is performed on 4×NVIDIA GEFORCE RTX GPUs with mixed precision



