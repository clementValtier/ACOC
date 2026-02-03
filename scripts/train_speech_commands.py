#!/usr/bin/env python3
"""
Training ACOC sur Speech Commands (Audio Classification)
========================================================
Classification de commandes vocales.
Utilise des MFCC (Mel-Frequency Cepstral Coefficients) comme features.
"""

import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
import os

from base_trainer import BaseACOCTrainer, create_onehot_collate_fn
from acoc import SystemConfig


class SubsetSC(SPEECHCOMMANDS):
    """Sous-ensemble de Speech Commands."""

    def __init__(self, subset: str = None):
        super().__init__("./data", download=True)

        # Filtrer par subset (train/test)
        def load_list(filename):
            filepath = os.path.join(self._path, filename)
            with open(filepath) as fileobj:
                return [os.path.normpath(os.path.join(self._path, line.strip())) for line in fileobj]

        if subset == "validation":
            self._walker = load_list("validation_list.txt")
        elif subset == "testing":
            self._walker = load_list("testing_list.txt")
        elif subset == "training":
            excludes = set(load_list("validation_list.txt") + load_list("testing_list.txt"))
            self._walker = [w for w in self._walker if w not in excludes]


class SpeechCommandsTrainer(BaseACOCTrainer):
    """Trainer spécifique pour Speech Commands."""

    # 10 commandes les plus communes
    LABELS = [
        'yes', 'no', 'up', 'down', 'left',
        'right', 'on', 'off', 'stop', 'go'
    ]

    LABELS_FR = [
        'Oui', 'Non', 'Haut', 'Bas', 'Gauche',
        'Droite', 'Allumer', 'Éteindre', 'Stop', 'Go'
    ]

    def __init__(self, num_cycles: int = 30, batch_size: int = 64, n_mfcc: int = 40):
        super().__init__(num_cycles, batch_size)
        self.n_mfcc = n_mfcc
        self.label_to_idx = {label: idx for idx, label in enumerate(self.LABELS)}

    def get_config(self) -> SystemConfig:
        # MFCC: n_mfcc coefficients × frames
        # On fixe à n_mfcc × 100 frames = 4000 features (avec padding/troncature)
        return SystemConfig(
            device=self.device,
            input_dim=self.n_mfcc * 100,  # Features audio aplaties
            hidden_dim=512,
            output_dim=len(self.LABELS),
            num_variants=5,
            saturation_threshold=0.8,
            min_cycles_before_expand=10,
            expansion_cooldown=15,
            use_cnn=False  # MLP pour audio (ou CNN 1D possible)
        )

    def _transform_audio(self, waveform, sample_rate):
        """Transforme l'audio en features MFCC."""
        # Resampler à 8kHz si nécessaire
        if sample_rate != 8000:
            resampler = torchaudio.transforms.Resample(sample_rate, 8000)
            waveform = resampler(waveform)

        # MFCC (n_mels doit être >= n_mfcc)
        mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=8000,
            n_mfcc=self.n_mfcc,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': self.n_mfcc}
        )
        mfcc = mfcc_transform(waveform)

        # Padding/troncature à 100 frames
        target_length = 100
        if mfcc.shape[-1] < target_length:
            # Padding
            pad_length = target_length - mfcc.shape[-1]
            mfcc = torch.nn.functional.pad(mfcc, (0, pad_length))
        else:
            # Troncature
            mfcc = mfcc[..., :target_length]

        return mfcc.flatten()  # [n_mfcc, 100] -> [n_mfcc * 100]

    def _collate_fn(self, batch):
        """Collate function custom pour audio."""
        tensors = []
        targets = []

        for waveform, sample_rate, label, *_ in batch:
            # Label to index - filtrer d'abord
            target_idx = self.label_to_idx.get(label, -1)
            if target_idx == -1:
                continue  # Ignorer les labels qu'on ne veut pas

            # Transform audio seulement si le label est valide
            features = self._transform_audio(waveform, sample_rate)
            tensors.append(features)
            targets.append(target_idx)

        if len(tensors) == 0:
            # Retourner des tenseurs vides au lieu de None
            return torch.zeros((0, self.n_mfcc * 100)), torch.zeros((0, len(self.LABELS)))

        # Stack et one-hot
        tensors = torch.stack(tensors)
        targets = torch.tensor(targets)
        targets_onehot = torch.nn.functional.one_hot(
            targets, num_classes=len(self.LABELS)
        ).float()

        return tensors, targets_onehot

    def get_dataloaders(self) -> tuple:
        """Charge Speech Commands avec torchaudio."""
        print("📥 Téléchargement de Speech Commands...")

        train_dataset = SubsetSC(subset="training")
        test_dataset = SubsetSC(subset="testing")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=0
        )

        return train_loader, test_loader

    def get_class_names(self) -> list:
        return self.LABELS_FR

    def get_dataset_name(self) -> str:
        return "speech_commands"

    def get_dataset_info(self) -> dict:
        return {
            "Input": f"{self.n_mfcc * 100} (MFCC {self.n_mfcc}×100 frames)",
            "Hidden": 512,
            "Classes": ", ".join(self.LABELS_FR)
        }


if __name__ == '__main__':
    trainer = SpeechCommandsTrainer(num_cycles=30, batch_size=64, n_mfcc=40)
    trainer.run()
