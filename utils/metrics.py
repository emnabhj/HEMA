import numpy as np
import torch
import re
import string
import jiwer


class compute_wer:
    """
    Classe pour calculer les métriques d'évaluation pour AVSR
    """

    def __init__(self, char_list=None):
        """
        Initialise la classe de métriques.

        Args:
            char_list: Liste de caractères pour le décodage CTC
        """
        self.char_list = char_list
        self.punctuation_regex = re.compile(f'[{re.escape(string.punctuation)}]')

    def ctc_decode(self, log_probs, blank_id=0):
        """
        Décodage glouton CTC pour convertir les probabilités en texte.

        Args:
            log_probs: Probabilités logarithmiques de sortie du modèle [B, T, C]
            blank_id: ID du token blanc CTC

        Returns:
            Liste de chaînes de caractères décodées
        """
        if not torch.is_tensor(log_probs):
            log_probs = torch.tensor(log_probs)

        # Obtenir les indices les plus probables
        _, max_indices = torch.max(log_probs, dim=-1)  # [B, T]

        batch_size = max_indices.shape[0]
        decoded_outputs = []

        for b in range(batch_size):
            # Obtenir les indices pour cet échantillon du batch
            indices = max_indices[b].cpu().numpy()

            # Décodage glouton CTC
            previous = blank_id
            decoded = []

            for idx in indices:
                # Ignorer les répétitions et les blancs
                if idx != previous and idx != blank_id:
                    decoded.append(idx)
                previous = idx

            # Convertir les indices en caractères
            if self.char_list:
                text = ''.join([self.char_list[i] for i in decoded])
            else:
                text = ' '.join([str(i) for i in decoded])

            decoded_outputs.append(text)

        return decoded_outputs

    def calculate_wer(self, hypotheses, references):
        """
        Calcule le Word Error Rate (WER).

        Args:
            hypotheses: Liste des textes prédits
            references: Liste des textes de référence

        Returns:
            WER (float)
        """
        # Prétraitement des textes
        processed_hyp = []
        processed_ref = []

        for hyp, ref in zip(hypotheses, references):
            # Convertir en minuscules
            hyp = hyp.lower()
            ref = ref.lower()

            # Supprimer la ponctuation
            hyp = self.punctuation_regex.sub(' ', hyp)
            ref = self.punctuation_regex.sub(' ', ref)

            # Normaliser les espaces multiples
            hyp = ' '.join(hyp.split())
            ref = ' '.join(ref.split())

            processed_hyp.append(hyp)
            processed_ref.append(ref)

        # Calculer le WER avec jiwer
        wer = jiwer.wer(processed_ref, processed_hyp)

        return wer

    def calculate_cer(self, hypotheses, references):
        """
        Calcule le Character Error Rate (CER).

        Args:
            hypotheses: Liste des textes prédits
            references: Liste des textes de référence

        Returns:
            CER (float)
        """
        # Prétraitement des textes
        processed_hyp = []
        processed_ref = []

        for hyp, ref in zip(hypotheses, references):
            # Convertir en minuscules
            hyp = hyp.lower()
            ref = ref.lower()

            # Supprimer la ponctuation
            hyp = self.punctuation_regex.sub('', hyp)
            ref = self.punctuation_regex.sub('', ref)

            # Supprimer les espaces
            hyp = ''.join(hyp.split())
            ref = ''.join(ref.split())

            processed_hyp.append(hyp)
            processed_ref.append(ref)

        # Calculer le CER avec jiwer
        transformation = jiwer.Compose([
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.ReduceToSingleChar(),
            jiwer.ReduceToListOfChars()
        ])

        cer = jiwer.wer(processed_ref, processed_hyp, truth_transform=transformation,
                        hypothesis_transform=transformation)

        return cer

    def calculate_metrics(self, log_probs, targets):
        """
        Calcule toutes les métriques de performance.

        Args:
            log_probs: Probabilités logarithmiques de sortie du modèle
            targets: Textes cibles de référence

        Returns:
            Dictionnaire des métriques calculées
        """
        hypotheses = self.ctc_decode(log_probs)
        wer = self.calculate_wer(hypotheses, targets)
        cer = self.calculate_cer(hypotheses, targets)

        return {
            'wer': wer,
            'cer': cer,
            'hypotheses': hypotheses,
            'references': targets
        }

    def output_results(self, metrics, output_file=None):
        """
        Affiche les résultats des métriques et les écrit éventuellement dans un fichier.

        Args:
            metrics: Dictionnaire de métriques calculées
            output_file: Chemin du fichier de sortie (optionnel)
        """
        print(f"Word Error Rate (WER): {metrics['wer']:.4f}")
        print(f"Character Error Rate (CER): {metrics['cer']:.4f}")

        if output_file:
            with open(output_file, 'w') as f:
                f.write(f"WER: {metrics['wer']:.4f}\n")
                f.write(f"CER: {metrics['cer']:.4f}\n\n")

                f.write("Comparaison des hypothèses et références:\n")
                for i, (hyp, ref) in enumerate(zip(metrics['hypotheses'], metrics['references'])):
                    f.write(f"Exemple {i + 1}:\n")
                    f.write(f"  Référence: {ref}\n")
                    f.write(f"  Hypothèse: {hyp}\n\n")


# Fonction pour évaluer les modèles uni-modaux et multimodal
def evaluate_models(audio_only_outputs, visual_only_outputs, av_outputs, targets, metrics_calculator):
    """
    Évalue les performances des modèles audio-seul, visuel-seul et audiovisuel.

    Args:
        audio_only_outputs: Sorties du modèle audio-seul
        visual_only_outputs: Sorties du modèle visuel-seul
        av_outputs: Sorties du modèle audiovisuel
        targets: Textes cibles de référence
        metrics_calculator: Instance de AVSRMetrics

    Returns:
        Dictionnaire des métriques calculées pour chaque modèle
    """
    audio_metrics = metrics_calculator.calculate_metrics(audio_only_outputs, targets)
    visual_metrics = metrics_calculator.calculate_metrics(visual_only_outputs, targets)
    av_metrics = metrics_calculator.calculate_metrics(av_outputs, targets)

    results = {
        'audio_only': {
            'wer': audio_metrics['wer'],
            'cer': audio_metrics['cer']
        },
        'visual_only': {
            'wer': visual_metrics['wer'],
            'cer': visual_metrics['cer']
        },
        'audiovisual': {
            'wer': av_metrics['wer'],
            'cer': av_metrics['cer']
        }
    }

    return results


# Fonction pour évaluer la robustesse sous différentes conditions de bruit
def evaluate_noise_robustness(model, test_loader, snr_levels, metrics_calculator):
    """
    Évalue la robustesse du modèle sous différents niveaux de bruit.

    Args:
        model: Modèle à évaluer
        test_loader: DataLoader pour les données de test
        snr_levels: Liste des niveaux de SNR à tester
        metrics_calculator: Instance de AVSRMetrics

    Returns:
        Dictionnaire des métriques calculées pour chaque niveau de SNR
    """
    results = {}

    for snr in snr_levels:
        print(f"Évaluation avec SNR = {snr} dB")
        all_outputs = []
        all_targets = []

        for batch in test_loader:
            # Ajouter du bruit au niveau SNR spécifié
            audio = batch['audio']
            noise_scale = 10 ** (-snr / 20)
            noise = torch.randn_like(audio) * noise_scale
            noisy_audio = audio + noise

            # Mettre à jour le batch avec l'audio bruité
            batch['audio'] = noisy_audio

            with torch.no_grad():
                outputs = model(batch)

            all_outputs.append(outputs)
            all_targets.extend(batch['text'])

        # Concaténer les sorties de tous les batches
        all_outputs = torch.cat(all_outputs, dim=0)

        # Calculer les métriques
        metrics = metrics_calculator.calculate_metrics(all_outputs, all_targets)
        results[snr] = {
            'wer': metrics['wer'],
            'cer': metrics['cer']
        }

    return results