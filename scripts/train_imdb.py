#!/usr/bin/env python3
"""
Training ACOC on IMDB (Text Classification)
==========================================
Sentiment classification on movie reviews.
Uses TF-IDF for text embeddings.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import numpy as np

from base_trainer import BaseACOCTrainer, create_onehot_collate_fn
from acoc import SystemConfig


class IMDBTrainer(BaseACOCTrainer):
    """Trainer spécifique pour IMDB sentiment analysis."""

    CLASSES = ['Négatif', 'Positif']

    def __init__(self, num_cycles: int = 100, batch_size: int = 64, max_features: int = 5000):
        super().__init__(num_cycles, batch_size)
        self.max_features = max_features
        self.vectorizer = None

    def get_config(self) -> SystemConfig:
        return SystemConfig(
            device=self.device,
            input_dim=self.max_features,
            hidden_dim=512,
            output_dim=2,
            num_variants=5,
            saturation_threshold=0.8,
            min_cycles_before_expand=10,
            expansion_cooldown=15,
            use_cnn=False  # MLP pour texte
        )

    def get_dataloaders(self) -> tuple:
        """Load IMDB with Hugging Face datasets."""
        try:
            from datasets import load_dataset
            print("📥 Downloading IMDB via Hugging Face...")

            # Load dataset
            dataset = load_dataset('imdb')
            train_data = dataset['train']
            test_data = dataset['test']

            # Extract texts and labels
            train_texts = train_data['text']
            train_labels = train_data['label']
            test_texts = test_data['text']
            test_labels = test_data['label']

            # TF-IDF vectorization with L2 normalization
            print(f"  TF-IDF vectorization (max_features={self.max_features})...")
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                stop_words='english',
                max_df=0.7,
                min_df=5,
                norm='l2',  # L2 normalization to stabilize learning
                sublinear_tf=True  # Use log(tf) instead of raw tf
            )

            train_features = self.vectorizer.fit_transform(train_texts).toarray()
            test_features = self.vectorizer.transform(test_texts).toarray()

            # Convert to tensors
            train_X = torch.FloatTensor(train_features)
            train_y = torch.LongTensor(train_labels)
            test_X = torch.FloatTensor(test_features)
            test_y = torch.LongTensor(test_labels)

            # One-hot encoding
            train_y_onehot = torch.nn.functional.one_hot(train_y, num_classes=2).float()
            test_y_onehot = torch.nn.functional.one_hot(test_y, num_classes=2).float()

            # Create datasets
            train_dataset = TensorDataset(train_X, train_y_onehot)
            test_dataset = TensorDataset(test_X, test_y_onehot)

            train_loader = DataLoader(
                train_dataset, batch_size=self.batch_size, shuffle=True,
                num_workers=2, persistent_workers=True
            )
            test_loader = DataLoader(
                test_dataset, batch_size=self.batch_size, shuffle=False,
                num_workers=2, persistent_workers=True
            )

            return train_loader, test_loader

        except ImportError:
            print("❌ Error: 'datasets' not installed.")
            print("   Install with: pip install datasets")
            raise

    def get_class_names(self) -> list:
        return self.CLASSES

    def get_dataset_name(self) -> str:
        return "imdb"

    def get_dataset_info(self) -> dict:
        return {
            "Input": f"{self.max_features} (TF-IDF features)",
            "Hidden": 512,
            "Classes": "Negative, Positive (sentiment analysis)"
        }


if __name__ == '__main__':
    trainer = IMDBTrainer(num_cycles=100, batch_size=64, max_features=5000)
    trainer.run()
