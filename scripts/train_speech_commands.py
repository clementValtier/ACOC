#!/usr/bin/env python3
"""
Training ACOC on Speech Commands (Audio Classification)
======================================================
Speech command classification.
Uses MFCC (Mel-Frequency Cepstral Coefficients) as features.
"""

import torch
import torchaudio
from torch.utils.data import DataLoader
from torchaudio.datasets import SPEECHCOMMANDS
import os

from base_trainer import BaseACOCTrainer, create_onehot_collate_fn
from acoc import SystemConfig


class SubsetSC(SPEECHCOMMANDS):
    """Subset of Speech Commands."""

    def __init__(self, subset: str | None = None):
        super().__init__("./data", download=True)

        # Filter by subset (train/test)
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

    # 10 most common commands
    LABELS = [
        'yes', 'no', 'up', 'down', 'left',
        'right', 'on', 'off', 'stop', 'go'
    ]

    LABELS_FR = [
        'Yes', 'No', 'Up', 'Down', 'Left',
        'Right', 'On', 'Off', 'Stop', 'Go'
    ]

    def __init__(self, num_cycles: int = 30, batch_size: int = 64, n_mfcc: int = 40):
        super().__init__(num_cycles, batch_size)
        self.n_mfcc = n_mfcc
        self.label_to_idx = {label: idx for idx, label in enumerate(self.LABELS)}

        # Pre-create audio transforms to avoid recreating them for each batch
        self.resampler = torchaudio.transforms.Resample(16000, 8000)
        self.mfcc_transform = torchaudio.transforms.MFCC(
            sample_rate=8000,
            n_mfcc=self.n_mfcc,
            melkwargs={'n_fft': 400, 'hop_length': 160, 'n_mels': self.n_mfcc}
        )

    def get_config(self) -> SystemConfig:
        # MFCC: n_mfcc coefficients × frames
        # Fixed to n_mfcc × 100 frames = 4000 features (with padding/truncation)
        return SystemConfig(
            device=self.device,
            input_dim=self.n_mfcc * 100,  # Flattened audio features
            hidden_dim=512,
            output_dim=len(self.LABELS),
            num_variants=5,
            saturation_threshold=0.8,
            min_cycles_before_expand=10,
            expansion_cooldown=15,
            use_cnn=False  # MLP for audio (or 1D CNN possible)
        )

    def _transform_audio(self, waveform, sample_rate):
        """Transform audio to MFCC features."""
        # Resample to 8kHz if necessary
        if sample_rate != 8000:
            waveform = self.resampler(waveform)

        # Use pre-created MFCC transform
        mfcc = self.mfcc_transform(waveform)

        # Padding/truncation to 100 frames
        target_length = 100
        if mfcc.shape[-1] < target_length:
            # Padding
            pad_length = target_length - mfcc.shape[-1]
            mfcc = torch.nn.functional.pad(mfcc, (0, pad_length))
        else:
            # Truncation
            mfcc = mfcc[..., :target_length]

        return mfcc.flatten()  # [n_mfcc, 100] -> [n_mfcc * 100]

    def _collate_fn(self, batch):
        """Custom collate function for audio."""
        tensors = []
        targets = []

        for waveform, sample_rate, label, *_ in batch:
            # Label to index - filter first
            target_idx = self.label_to_idx.get(label, -1)
            if target_idx == -1:
                continue  # Skip labels we don't want

            # Transform audio only if label is valid
            features = self._transform_audio(waveform, sample_rate)
            tensors.append(features)
            targets.append(target_idx)

        if len(tensors) == 0:
            # Return empty tensors instead of None
            return torch.zeros((0, self.n_mfcc * 100)), torch.zeros((0, len(self.LABELS)))

        # Stack and one-hot
        tensors = torch.stack(tensors)
        targets = torch.tensor(targets)
        targets_onehot = torch.nn.functional.one_hot(
            targets, num_classes=len(self.LABELS)
        ).float()

        return tensors, targets_onehot

    def get_dataloaders(self) -> tuple:
        """Load Speech Commands with torchaudio."""
        print("📥 Downloading Speech Commands...")

        train_dataset = SubsetSC(subset="training")
        test_dataset = SubsetSC(subset="testing")

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=2, persistent_workers=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=2, persistent_workers=True
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
