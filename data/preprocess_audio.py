import os
import numpy as np
import torch
import torchaudio
import random
from scipy import signal
import librosa
import matplotlib.pyplot as plt
from config.hyperparameters import *


class AudioPreprocessor:
    """
    Classe pour le prétraitement audio avec augmentation physique inspirée
    comme décrit dans l'article HEMA
    """

    def __init__(self, sample_rate=16000, n_fft=512, hop_length=160, n_mels=80):
        """
        Initialisation du préprocesseur audio

        Args:
            sample_rate: fréquence d'échantillonnage (16kHz par défaut)
            n_fft: taille de la FFT
            hop_length: saut entre trames
            n_mels: nombre de bandes mel
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels

        # Pour la transformation mel-spectrogram
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        # Charger des échantillons de bruit de la base DEMAND si disponible
        self.noise_samples = []
        noise_dir = "data/DEMAND"
        if os.path.exists(noise_dir):
            for noise_file in os.listdir(noise_dir)[:10]:  # Limiter à 10 fichiers de bruit
                if noise_file.endswith(".wav"):
                    noise_path = os.path.join(noise_dir, noise_file)
                    noise, _ = torchaudio.load(noise_path)
                    noise = noise.mean(dim=0) if noise.shape[0] > 1 else noise[0]  # Convertir en mono
                    self.noise_samples.append(noise)

    def audio_to_mel_spectrogram(self, audio):
        """Convertit un signal audio en mel-spectrogram"""
        if isinstance(audio, np.ndarray):
            audio = torch.from_numpy(audio).float()

        # Normaliser l'amplitude
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()

        # Calculer le mel-spectrogram
        mel_spec = self.mel_transform(audio)

        # Convertir en dB avec un seuil minimum à -100dB
        mel_spec = 20 * torch.log10(torch.clamp(mel_spec, min=1e-5))
        mel_spec = torch.clamp(mel_spec, min=-100.0)

        # Normaliser entre 0 et 1
        mel_spec = (mel_spec + 100) / 100

        return mel_spec

    def add_noise(self, audio, snr_range=(0, 20)):
        """
        Ajoute du bruit au signal audio avec un SNR variable

        Args:
            audio: signal audio torch tensor
            snr_range: plage de SNR en dB (min, max)

        Returns:
            audio avec bruit ajouté
        """
        if len(self.noise_samples) == 0:
            # Générer du bruit blanc si pas d'échantillons DEMAND disponibles
            noise = torch.randn_like(audio)
        else:
            # Sélectionner aléatoirement un échantillon de bruit
            noise_sample = random.choice(self.noise_samples)

            # Si le bruit est plus court que l'audio, le répéter
            if len(noise_sample) < len(audio):
                repeats = int(np.ceil(len(audio) / len(noise_sample)))
                noise = noise_sample.repeat(repeats)[:len(audio)]
            else:
                # Si le bruit est plus long, prendre une section aléatoire
                start = random.randint(0, len(noise_sample) - len(audio))
                noise = noise_sample[start:start + len(audio)]

        # Calculer les puissances
        audio_power = torch.mean(audio ** 2)
        noise_power = torch.mean(noise ** 2)

        # SNR aléatoire dans la plage spécifiée
        snr = random.uniform(snr_range[0], snr_range[1])

        # Calculer le facteur de mise à l'échelle pour le bruit
        if noise_power > 0:
            scale = torch.sqrt(audio_power / (noise_power * 10 ** (snr / 10)))
            scaled_noise = noise * scale
            noisy_audio = audio + scaled_noise
        else:
            noisy_audio = audio

        return noisy_audio

    def spectral_masking(self, mel_spec, n_masks=2, mask_width_range=(5, 20)):
        """
        Applique un masquage spectral au mel-spectrogram

        Args:
            mel_spec: mel-spectrogram (B, F, T) ou (F, T)
            n_masks: nombre de masques à appliquer
            mask_width_range: plage de largeur de masque en bins (min, max)

        Returns:
            mel-spectrogram masqué
        """
        masked_spec = mel_spec.clone()

        # Gestion des dimensions batch ou non
        if len(mel_spec.shape) == 3:
            B, F, T = mel_spec.shape
            for b in range(B):
                for _ in range(n_masks):
                    # Masquer prioritairement les bandes critiques (fricatives: 1-4kHz, plosives: <500Hz)
                    # Pour les fricatives (environ 1/3 supérieur du spectrogramme)
                    if random.random() < 0.5:
                        f_start = random.randint(int(2 * F / 3), F - 1 - mask_width_range[1])
                    # Pour les plosives (environ 1/4 inférieur du spectrogramme)
                    else:
                        f_start = random.randint(0, int(F / 4))

                    f_width = random.randint(mask_width_range[0], mask_width_range[1])
                    f_end = min(f_start + f_width, F)

                    masked_spec[b, f_start:f_end, :] = 0
        else:
            F, T = mel_spec.shape
            for _ in range(n_masks):
                # Même logique que ci-dessus mais pour les données sans batch
                if random.random() < 0.5:
                    f_start = random.randint(int(2 * F / 3), F - 1 - mask_width_range[1])
                else:
                    f_start = random.randint(0, int(F / 4))

                f_width = random.randint(mask_width_range[0], mask_width_range[1])
                f_end = min(f_start + f_width, F)

                masked_spec[f_start:f_end, :] = 0

        return masked_spec

    def time_warping(self, audio, warping_range=(0.8, 1.2)):
        """
        Applique une déformation temporelle non linéaire au signal audio

        Args:
            audio: signal audio torch tensor
            warping_range: plage de facteur d'étirement (min, max)

        Returns:
            audio déformé temporellement
        """
        # Convertir en numpy pour le traitement
        audio_np = audio.numpy() if isinstance(audio, torch.Tensor) else audio

        # Facteur d'étirement aléatoire
        alpha = random.uniform(warping_range[0], warping_range[1])

        # Calculer la nouvelle longueur
        new_length = int(len(audio_np) * alpha)

        # Appliquer la déformation temporelle avec phase vocoder pour préserver le timbre
        audio_warped = librosa.effects.time_stretch(audio_np, rate=1 / alpha)

        # S'assurer que la longueur est correcte
        if len(audio_warped) > len(audio_np):
            audio_warped = audio_warped[:len(audio_np)]
        elif len(audio_warped) < len(audio_np):
            # Padding avec des zéros
            audio_warped = np.pad(audio_warped, (0, len(audio_np) - len(audio_warped)))

        # Reconvertir en torch si nécessaire
        if isinstance(audio, torch.Tensor):
            audio_warped = torch.from_numpy(audio_warped).float()

        return audio_warped

    def estimate_snr(self, noisy_audio):
        """
        Estime le SNR du signal audio bruité en utilisant un filtre de Wiener

        Args:
            noisy_audio: signal audio bruité

        Returns:
            SNR estimé en dB
        """
        # Convertir en numpy pour le traitement
        audio_np = noisy_audio.numpy() if isinstance(noisy_audio, torch.Tensor) else noisy_audio

        # Estimer le bruit en utilisant une fenêtre de 20ms au début (si possible)
        noise_length = min(int(0.02 * self.sample_rate), len(audio_np) // 10)
        if len(audio_np) > noise_length * 2:
            noise_estimate = audio_np[:noise_length]
            noise_power = np.mean(noise_estimate ** 2) + 1e-10

            # Appliquer un filtre de Wiener pour estimer le signal propre
            # Calculer le spectre du signal bruité
            nperseg = min(512, len(audio_np))
            f, t, Zxx = signal.stft(audio_np, fs=self.sample_rate, nperseg=nperseg)

            # Estimer le spectre du bruit
            _, _, Nxx = signal.stft(noise_estimate, fs=self.sample_rate, nperseg=nperseg)
            noise_spec = np.mean(np.abs(Nxx) ** 2, axis=1)

            # Appliquer le filtre de Wiener (version simplifiée)
            noise_spec_full = np.tile(noise_spec[:, np.newaxis], (1, Zxx.shape[1]))
            signal_spec = np.maximum(0, np.abs(Zxx) ** 2 - noise_spec_full)

            # Calculer les puissances
            signal_power = np.mean(signal_spec) + 1e-10

            # Calculer le SNR
            snr_db = 10 * np.log10(signal_power / noise_power)

            return max(0, min(50, snr_db))  # Limiter le SNR entre 0 et 50 dB
        else:
            # Si l'audio est trop court, renvoyer une valeur par défaut
            return 20.0

    def augment_audio(self, audio, augment_level=1.0):
        """
        Applique toutes les augmentations avec un niveau contrôlable

        Args:
            audio: signal audio torch tensor
            augment_level: niveau d'augmentation (0.0 à 1.0)

        Returns:
            audio augmenté
        """
        if augment_level <= 0:
            return audio

        # Appliquer les augmentations en fonction du niveau
        if random.random() < augment_level * 0.8:
            # SNR plus bas pour des niveaux d'augmentation plus élevés
            min_snr = max(0, 20 - augment_level * 20)
            max_snr = max(5, 30 - augment_level * 20)
            audio = self.add_noise(audio, snr_range=(min_snr, max_snr))

        if random.random() < augment_level * 0.7:
            # Déformation temporelle plus prononcée pour des niveaux d'augmentation plus élevés
            min_warp = max(0.9, 1.0 - augment_level * 0.2)
            max_warp = min(1.1, 1.0 + augment_level * 0.2)
            audio = self.time_warping(audio, warping_range=(min_warp, max_warp))

        return audio

    def process_audio_file(self, audio_path, augment_level=1.0, return_snr=False):
        """
        Traite un fichier audio: chargement, augmentation et extraction de caractéristiques

        Args:
            audio_path: chemin vers le fichier audio
            augment_level: niveau d'augmentation (0.0 à 1.0)
            return_snr: si True, renvoie aussi le SNR estimé

        Returns:
            mel_spectrogram et optionnellement le SNR
        """
        # Charger l'audio
        audio, orig_sr = torchaudio.load(audio_path)

        # Convertir en mono si nécessaire
        audio = audio.mean(dim=0) if audio.shape[0] > 1 else audio[0]

        # Rééchantillonner si nécessaire
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            audio = resampler(audio)

        # Normaliser l'amplitude
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()

        # Appliquer les augmentations
        augmented_audio = self.augment_audio(audio, augment_level)

        # Calculer le mel-spectrogram
        mel_spec = self.audio_to_mel_spectrogram(augmented_audio)

        # Appliquer le masquage spectral si augmentation active
        if augment_level > 0 and random.random() < augment_level * 0.5:
            n_masks = int(1 + augment_level * 2)  # 1-3 masques selon le niveau
            mel_spec = self.spectral_masking(mel_spec, n_masks=n_masks)

        if return_snr:
            snr = self.estimate_snr(augmented_audio)
            return mel_spec, snr
        else:
            return mel_spec

    def visualize_augmentation(self, audio_path):
        """
        Visualise l'effet des augmentations sur un fichier audio

        Args:
            audio_path: chemin vers le fichier audio
        """
        # Charger l'audio original
        audio, orig_sr = torchaudio.load(audio_path)
        audio = audio.mean(dim=0) if audio.shape[0] > 1 else audio[0]
        if orig_sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_sr, self.sample_rate)
            audio = resampler(audio)
        if audio.abs().max() > 0:
            audio = audio / audio.abs().max()

        # Calculer le mel-spectrogram original
        mel_spec_orig = self.audio_to_mel_spectrogram(audio)

        # Appliquer les augmentations
        audio_noise = self.add_noise(audio.clone(), snr_range=(0, 0))  # SNR fixe à 0dB
        mel_spec_noise = self.audio_to_mel_spectrogram(audio_noise)

        audio_warp = self.time_warping(audio.clone(), warping_range=(0.8, 0.8))  # Facteur fixe à 0.8
        mel_spec_warp = self.audio_to_mel_spectrogram(audio_warp)

        mel_spec_mask = self.spectral_masking(mel_spec_orig.clone(), n_masks=2)

        # Visualisation
        plt.figure(figsize=(15, 10))

        plt.subplot(2, 2, 1)
        plt.title('Original')
        plt.imshow(mel_spec_orig.numpy(), aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(2, 2, 2)
        plt.title('With Noise (0dB SNR)')
        plt.imshow(mel_spec_noise.numpy(), aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(2, 2, 3)
        plt.title('Time Warped (0.8x)')
        plt.imshow(mel_spec_warp.numpy(), aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')

        plt.subplot(2, 2, 4)
        plt.title('Spectral Masked')
        plt.imshow(mel_spec_mask.numpy(), aspect='auto', origin='lower')
        plt.colorbar(format='%+2.0f dB')

        plt.tight_layout()
        plt.savefig('audio_augmentations.png')
        plt.close()

        print(f"Visualization saved as 'audio_augmentations.png'")


if __name__ == "__main__":
    # Test et démonstration
    processor = AudioPreprocessor()

    # Tester avec un fichier audio si disponible
    test_file = "data/test_audio.wav"
    if os.path.exists(test_file):
        processor.visualize_augmentation(test_file)

        # Tester l'estimation du SNR
        audio, _ = torchaudio.load(test_file)
        audio = audio.mean(dim=0) if audio.shape[0] > 1 else audio[0]
        noisy_audio = processor.add_noise(audio, snr_range=(10, 10))
        estimated_snr = processor.estimate_snr(noisy_audio)
        print(f"Audio augmenté avec SNR=10dB, estimation: {estimated_snr:.2f}dB")
    else:
        print(f"Fichier test {test_file} non trouvé. Créer un fichier audio pour tester.")