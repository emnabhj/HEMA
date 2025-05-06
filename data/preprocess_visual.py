import cv2
import dlib
import numpy as np
import os
from tqdm import tqdm


class LipPreprocessor:
    """
    Préprocesseur pour extraire la région des lèvres à partir des images faciales
    basé sur les points de repère faciaux détectés par dlib.
    """

    def __init__(self, face_predictor_path="shape_predictor_68_face_landmarks.dat",
                 target_size=(128, 64),
                 grayscale=True,
                 augmentation=True):
        """
        Initialise le préprocesseur de lèvres.

        Args:
            face_predictor_path: Chemin vers le fichier de prédiction de points de repère
            target_size: Taille cible (width, height) de la région des lèvres
            grayscale: Convertir la sortie en niveaux de gris
            augmentation: Appliquer des augmentations physiques
        """
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(face_predictor_path)
        self.target_size = target_size
        self.grayscale = grayscale
        self.augmentation = augmentation

    def extract_mouth_roi(self, frame):
        """
        Extrait la région des lèvres à partir d'une image.

        Args:
            frame: Image d'entrée (BGR format from OpenCV)

        Returns:
            Region d'intérêt des lèvres redimensionnée
        """
        # Détection du visage
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)

        if len(faces) == 0:
            # Si aucun visage n'est détecté, renvoie une image noire
            return np.zeros((self.target_size[1], self.target_size[0],
                             1 if self.grayscale else 3), dtype=np.uint8)

        # Prend le plus grand visage
        face = max(faces, key=lambda rect: rect.width() * rect.height())

        # Obtenir les points de repère faciaux
        landmarks = self.predictor(gray, face)

        # Points de repère des lèvres (indices 48-68 dans le modèle à 68 points)
        mouth_points = np.array([[landmarks.part(i).x, landmarks.part(i).y]
                                 for i in range(48, 68)])

        # Ajouter une marge autour des lèvres
        x_min = np.min(mouth_points[:, 0]) - 10
        y_min = np.min(mouth_points[:, 1]) - 10
        x_max = np.max(mouth_points[:, 0]) + 10
        y_max = np.max(mouth_points[:, 1]) + 10

        # S'assurer que les coordonnées sont valides
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        # Extraire la région des lèvres
        mouth_roi = frame[int(y_min):int(y_max), int(x_min):int(x_max)]

        # Vérifier si la ROI est valide
        if mouth_roi.size == 0:
            return np.zeros((self.target_size[1], self.target_size[0],
                             1 if self.grayscale else 3), dtype=np.uint8)

        # Redimensionner
        mouth_roi = cv2.resize(mouth_roi, self.target_size)

        # Convertir en niveaux de gris si nécessaire
        if self.grayscale:
            if len(mouth_roi.shape) == 3:
                mouth_roi = cv2.cvtColor(mouth_roi, cv2.COLOR_BGR2GRAY)
                mouth_roi = mouth_roi[:, :, np.newaxis]  # Ajouter une dimension de canal

        return mouth_roi

    def apply_augmentation(self, mouth_roi):
        """
        Applique des augmentations physiques à la région des lèvres.

        Args:
            mouth_roi: Région d'intérêt des lèvres

        Returns:
            Région des lèvres augmentée
        """
        if not self.augmentation:
            return mouth_roi

        # Extraire dimensions
        h, w = mouth_roi.shape[:2]
        center = (w // 2, h // 2)

        # Rotation aléatoire (±15 degrés)
        angle = np.random.uniform(-15, 15)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_roi = cv2.warpAffine(mouth_roi, rotation_matrix, (w, h))

        # Bruit gaussien
        noise_level = np.random.uniform(0, 0.2)
        if len(rotated_roi.shape) == 3:
            noise = np.random.normal(0, noise_level * 255, rotated_roi.shape).astype(np.int16)
            noisy_roi = np.clip(rotated_roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        else:
            noise = np.random.normal(0, noise_level * 255, rotated_roi.shape[:2]).astype(np.int16)
            noisy_roi = np.clip(rotated_roi.astype(np.int16) + noise, 0, 255).astype(np.uint8)
            if len(rotated_roi.shape) == 3:
                noisy_roi = noisy_roi[:, :, np.newaxis]

        return noisy_roi

    def process_frame(self, frame):
        """
        Traite une seule image pour extraire la région des lèvres.

        Args:
            frame: Image d'entrée

        Returns:
            Région des lèvres prétraitée
        """
        mouth_roi = self.extract_mouth_roi(frame)

        if self.augmentation:
            mouth_roi = self.apply_augmentation(mouth_roi)

        return mouth_roi

    def process_video(self, video_path, output_dir=None, max_frames=None):
        """
        Traite une vidéo pour extraire les régions des lèvres de chaque image.

        Args:
            video_path: Chemin vers le fichier vidéo
            output_dir: Répertoire de sortie pour sauvegarder les images (optionnel)
            max_frames: Nombre maximum d'images à traiter (optionnel)

        Returns:
            Liste des régions des lèvres extraites
        """
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        # Créer le répertoire de sortie si nécessaire
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames:
            total_frames = min(total_frames, max_frames)

        for _ in tqdm(range(total_frames), desc=f"Processing {os.path.basename(video_path)}"):
            ret, frame = cap.read()
            if not ret:
                break

            mouth_roi = self.process_frame(frame)
            frames.append(mouth_roi)

            # Sauvegarder l'image si un répertoire de sortie est spécifié
            if output_dir:
                output_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
                if self.grayscale:
                    cv2.imwrite(output_path, mouth_roi.squeeze())
                else:
                    cv2.imwrite(output_path, mouth_roi)

            frame_count += 1
            if max_frames and frame_count >= max_frames:
                break

        cap.release()
        return frames

    def compute_sharpness(self, frame):
        """
        Calcule la netteté de l'image en utilisant la variance des gradients de Sobel.

        Args:
            frame: Image d'entrée

        Returns:
            Valeur de netteté
        """
        if len(frame.shape) == 3 and frame.shape[2] > 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:
            gray = frame.squeeze()
        else:
            gray = frame

        # Appliquer le filtre de Sobel
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        # Calculer le gradient
        gradient_magnitude = np.sqrt(sobelx ** 2 + sobely ** 2)

        # Calculer la variance comme mesure de netteté
        sharpness = np.var(gradient_magnitude)

        return sharpness


def batch_process(video_dir, output_base_dir, file_extension='.mp4'):
    """
    Traite un lot de vidéos pour extraire les régions des lèvres.

    Args:
        video_dir: Répertoire contenant les vidéos
        output_base_dir: Répertoire de base pour la sortie
        file_extension: Extension des fichiers vidéo à traiter
    """
    preprocessor = LipPreprocessor()

    for root, _, files in os.walk(video_dir):
        for file in files:
            if file.endswith(file_extension):
                video_path = os.path.join(root, file)

                # Créer un sous-répertoire correspondant pour la sortie
                rel_path = os.path.relpath(root, video_dir)
                output_dir = os.path.join(output_base_dir, rel_path, os.path.splitext(file)[0])

                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                preprocessor.process_video(video_path, output_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prétraitement de vidéos pour l'extraction de régions des lèvres")
    parser.add_argument("--video_path", type=str, help="Chemin vers une vidéo ou un répertoire de vidéos")
    parser.add_argument("--output_dir", type=str, help="Répertoire de sortie pour les images traitées")
    parser.add_argument("--batch", action="store_true", help="Traiter un lot de vidéos")
    parser.add_argument("--file_ext", type=str, default=".mp4",
                        help="Extension des fichiers vidéo pour le traitement par lot")

    args = parser.parse_args()

    if args.batch:
        batch_process(args.video_path, args.output_dir, args.file_ext)
    else:
        preprocessor = LipPreprocessor()
        preprocessor.process_video(args.video_path, args.output_dir)