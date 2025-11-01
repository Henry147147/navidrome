#!/usr/bin/env python3
"""
train.py - Train Text-to-Audio Embedding Projection Models

This script trains 3 separate projection models to map text embeddings (Qwen3-Embedding-4B)
into the audio embedding spaces (MuQ, MERT, Music2Latent) using contrastive learning.

Models trained:
1. MuQ Projection: 2560D → 1536D
2. MERT Projection: 2560D → 76,800D
3. Music2Latent Projection: 2560D → 576D
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from transformers import AutoModel, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from tqdm import tqdm


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Data
    data_dir: str = "data/musicbench_embeddings"
    hdf5_filename: str = "embeddings.h5"

    # Model selection (which audio encoder to train for)
    audio_encoder: str = "muq"  # 'muq', 'mert', or 'latent'

    # Architecture
    text_model_name: str = "Qwen/Qwen3-Embedding-4B"
    text_dim: int = 2560
    hidden_dim: int = 1024  # Will be adjusted based on audio encoder
    audio_dim: int = 1536  # Will be set based on audio encoder

    # LoRA
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1

    # Training
    epochs: int = 20
    batch_size: int = 32
    gradient_accumulation_steps: int = 8  # Effective batch = 32 * 8 = 256
    learning_rate: float = 5e-5  # For projection
    text_lr: float = 2e-5  # For text encoder LoRA
    weight_decay: float = 0.1
    warmup_steps: int = 2000
    max_grad_norm: float = 1.0

    # Loss
    temperature: float = 0.07

    # Optimization
    use_amp: bool = True  # Mixed precision
    amp_dtype: str = "bfloat16"  # or "float16"

    # Checkpointing
    checkpoint_dir: str = "checkpoints"
    save_every_n_epochs: int = 5
    early_stopping_patience: int = 5

    # Logging
    log_dir: str = "logs/tensorboard"
    log_every_n_steps: int = 10

    # Device
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42


class MusicBenchEmbeddingDataset(Dataset):
    """Dataset for pre-computed embeddings"""

    def __init__(self, hdf5_path: str, split: str, audio_encoder: str):
        """
        Args:
            hdf5_path: Path to HDF5 file
            split: 'train', 'val', or 'test'
            audio_encoder: 'muq', 'mert', or 'latent'
        """
        self.hdf5_path = hdf5_path
        self.split = split
        self.audio_encoder = audio_encoder

        # Open HDF5 to get metadata
        with h5py.File(hdf5_path, 'r') as f:
            self.length = len(f[split]['captions'])
            self.audio_dim = f[split][f'{audio_encoder}_embeddings'].shape[1]

        logger.info(f"Loaded {split} split: {self.length} samples, audio_dim={self.audio_dim}")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Open HDF5 file for each access (required for multiprocessing)
        with h5py.File(self.hdf5_path, 'r') as f:
            audio_emb = f[self.split][f'{self.audio_encoder}_embeddings'][idx]
            caption = f[self.split]['captions'][idx]
            audio_id = f[self.split]['audio_ids'][idx]

        return {
            'audio_embedding': torch.tensor(audio_emb, dtype=torch.float32),
            'caption': caption,
            'audio_id': audio_id
        }


class TextEncoder(nn.Module):
    """Text encoder with optional LoRA fine-tuning"""

    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config

        logger.info(f"Loading text encoder: {config.text_model_name}")

        # Load base model
        self.model = AutoModel.from_pretrained(
            config.text_model_name,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if config.amp_dtype == "bfloat16" else torch.float32
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config.text_model_name)

        # Apply LoRA if enabled
        if config.use_lora:
            logger.info(f"Applying LoRA (rank={config.lora_rank}, alpha={config.lora_alpha})")
            lora_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                r=config.lora_rank,
                lora_alpha=config.lora_alpha,
                lora_dropout=config.lora_dropout,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                               "gate_proj", "up_proj", "down_proj"],
                bias="none"
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()
        else:
            # Freeze base model if not using LoRA
            for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, texts: List[str]) -> torch.Tensor:
        """
        Encode texts to embeddings

        Args:
            texts: List of text strings

        Returns:
            Embeddings of shape [batch_size, text_dim]
        """
        # Tokenize
        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=512,  # Limit for efficiency
            return_tensors="pt"
        ).to(self.model.device)

        # Get embeddings
        with torch.cuda.amp.autocast(
            enabled=self.config.use_amp,
            dtype=torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16
        ):
            outputs = self.model(**inputs)
            # Use last token embedding (Qwen3 default)
            embeddings = outputs.last_hidden_state[:, -1, :]

        return embeddings


class ProjectionHead(nn.Module):
    """Projection head to map text embeddings to audio embedding space"""

    def __init__(self, text_dim: int, hidden_dim: int, audio_dim: int):
        super().__init__()

        self.text_dim = text_dim
        self.audio_dim = audio_dim

        # Text projection with larger capacity for high-dimensional targets
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, audio_dim),
            nn.BatchNorm1d(audio_dim)
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights with Xavier uniform"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, text_emb: torch.Tensor, audio_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Project embeddings and normalize

        Args:
            text_emb: Text embeddings [batch_size, text_dim]
            audio_emb: Audio embeddings [batch_size, audio_dim]

        Returns:
            Normalized text and audio features
        """
        # Project text to audio space
        text_features = self.text_proj(text_emb)

        # L2 normalize both
        text_features = F.normalize(text_features, p=2, dim=-1)
        audio_features = F.normalize(audio_emb, p=2, dim=-1)

        return text_features, audio_features


class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss (CLIP-style)"""

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, text_features: torch.Tensor, audio_features: torch.Tensor) -> torch.Tensor:
        """
        Compute symmetric InfoNCE loss

        Args:
            text_features: Normalized text embeddings [batch_size, dim]
            audio_features: Normalized audio embeddings [batch_size, dim]

        Returns:
            Loss value
        """
        batch_size = text_features.shape[0]

        # Compute similarity matrix
        logits = text_features @ audio_features.T / self.temperature  # [B, B]

        # Labels: diagonal elements are positive pairs
        labels = torch.arange(batch_size, device=logits.device)

        # Symmetric loss
        loss_text_to_audio = F.cross_entropy(logits, labels)
        loss_audio_to_text = F.cross_entropy(logits.T, labels)

        return (loss_text_to_audio + loss_audio_to_text) / 2


class RetrievalMetrics:
    """Compute retrieval metrics (R@K, MRR, Median Rank)"""

    @staticmethod
    def compute_metrics(text_features: torch.Tensor, audio_features: torch.Tensor) -> Dict[str, float]:
        """
        Compute retrieval metrics

        Args:
            text_features: [N, D]
            audio_features: [N, D]

        Returns:
            Dictionary of metrics
        """
        N = text_features.shape[0]

        # Compute similarity matrix
        sim_matrix = text_features @ audio_features.T  # [N, N]

        # Text-to-audio retrieval
        t2a_ranks = []
        for i in range(N):
            # Get similarities for this text
            sims = sim_matrix[i]  # [N]
            # Sort in descending order
            sorted_indices = torch.argsort(sims, descending=True)
            # Find rank of correct audio (index i)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            t2a_ranks.append(rank)

        # Audio-to-text retrieval
        a2t_ranks = []
        for i in range(N):
            # Get similarities for this audio
            sims = sim_matrix[:, i]  # [N]
            # Sort in descending order
            sorted_indices = torch.argsort(sims, descending=True)
            # Find rank of correct text (index i)
            rank = (sorted_indices == i).nonzero(as_tuple=True)[0].item() + 1
            a2t_ranks.append(rank)

        t2a_ranks = np.array(t2a_ranks)
        a2t_ranks = np.array(a2t_ranks)

        # Compute metrics
        metrics = {
            # Text-to-audio
            't2a_R@1': (t2a_ranks <= 1).mean() * 100,
            't2a_R@5': (t2a_ranks <= 5).mean() * 100,
            't2a_R@10': (t2a_ranks <= 10).mean() * 100,
            't2a_MRR': (1.0 / t2a_ranks).mean(),
            't2a_median_rank': np.median(t2a_ranks),
            # Audio-to-text
            'a2t_R@1': (a2t_ranks <= 1).mean() * 100,
            'a2t_R@5': (a2t_ranks <= 5).mean() * 100,
            'a2t_R@10': (a2t_ranks <= 10).mean() * 100,
            'a2t_MRR': (1.0 / a2t_ranks).mean(),
            'a2t_median_rank': np.median(a2t_ranks),
        }

        return metrics


class Trainer:
    """Training manager"""

    def __init__(self, config: TrainingConfig):
        self.config = config

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        # Create directories
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        # Initialize models
        logger.info("Initializing models...")
        self.text_encoder = TextEncoder(config).to(config.device)
        self.projection = ProjectionHead(
            config.text_dim,
            config.hidden_dim,
            config.audio_dim
        ).to(config.device)

        # Loss function
        self.criterion = InfoNCELoss(temperature=config.temperature)

        # Optimizer
        param_groups = [
            {'params': self.projection.parameters(), 'lr': config.learning_rate},
        ]
        if config.use_lora:
            param_groups.append({
                'params': [p for p in self.text_encoder.parameters() if p.requires_grad],
                'lr': config.text_lr
            })

        self.optimizer = AdamW(param_groups, weight_decay=config.weight_decay)

        # Load datasets
        hdf5_path = os.path.join(config.data_dir, config.hdf5_filename)
        self.train_dataset = MusicBenchEmbeddingDataset(hdf5_path, 'train', config.audio_encoder)
        self.val_dataset = MusicBenchEmbeddingDataset(hdf5_path, 'val', config.audio_encoder)
        self.test_dataset = MusicBenchEmbeddingDataset(hdf5_path, 'test', config.audio_encoder)

        # Update audio_dim from dataset
        config.audio_dim = self.train_dataset.audio_dim

        # DataLoaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

        # Calculate total steps
        self.total_steps = (len(self.train_loader) // config.gradient_accumulation_steps) * config.epochs

        # Learning rate scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.total_steps - config.warmup_steps,
            eta_min=1e-7
        )

        # GradScaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)

        # TensorBoard writer
        log_dir = os.path.join(config.log_dir, config.audio_encoder)
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.global_step = 0
        self.best_val_r1 = 0.0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        logger.info(f"Training configuration:")
        logger.info(f"  Audio encoder: {config.audio_encoder}")
        logger.info(f"  Audio dimension: {config.audio_dim}")
        logger.info(f"  Text dimension: {config.text_dim}")
        logger.info(f"  Hidden dimension: {config.hidden_dim}")
        logger.info(f"  Training samples: {len(self.train_dataset)}")
        logger.info(f"  Validation samples: {len(self.val_dataset)}")
        logger.info(f"  Test samples: {len(self.test_dataset)}")
        logger.info(f"  Effective batch size: {config.batch_size * config.gradient_accumulation_steps}")
        logger.info(f"  Total steps: {self.total_steps}")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch"""
        self.text_encoder.train()
        self.projection.train()

        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{self.config.epochs}")

        for batch_idx, batch in enumerate(pbar):
            # Move to device
            audio_emb = batch['audio_embedding'].to(self.config.device)
            captions = batch['caption']

            # Forward pass
            with torch.cuda.amp.autocast(
                enabled=self.config.use_amp,
                dtype=torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16
            ):
                # Encode text
                text_emb = self.text_encoder(captions)

                # Project to audio space
                text_features, audio_features = self.projection(text_emb, audio_emb)

                # Compute loss
                loss = self.criterion(text_features, audio_features)
                loss = loss / self.config.gradient_accumulation_steps

            # Backward pass
            self.scaler.scale(loss).backward()

            # Update weights
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(self.projection.parameters()) + list(self.text_encoder.parameters()),
                    self.config.max_grad_norm
                )

                # Optimizer step
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # Warmup or scheduler
                if self.global_step < self.config.warmup_steps:
                    lr_scale = (self.global_step + 1) / self.config.warmup_steps
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] * lr_scale / max(lr_scale - 1 + 1, 1e-8)
                else:
                    self.scheduler.step()

                self.global_step += 1

                # Log
                if self.global_step % self.config.log_every_n_steps == 0:
                    self.writer.add_scalar('train/loss', loss.item() * self.config.gradient_accumulation_steps, self.global_step)
                    self.writer.add_scalar('train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            pbar.set_postfix({'loss': total_loss / num_batches})

        return {'loss': total_loss / num_batches}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, split: str) -> Dict[str, float]:
        """Evaluate on a dataset split"""
        self.text_encoder.eval()
        self.projection.eval()

        all_text_features = []
        all_audio_features = []
        total_loss = 0.0

        for batch in tqdm(loader, desc=f"Evaluating {split}"):
            audio_emb = batch['audio_embedding'].to(self.config.device)
            captions = batch['caption']

            with torch.cuda.amp.autocast(
                enabled=self.config.use_amp,
                dtype=torch.bfloat16 if self.config.amp_dtype == "bfloat16" else torch.float16
            ):
                # Encode text
                text_emb = self.text_encoder(captions)

                # Project
                text_features, audio_features = self.projection(text_emb, audio_emb)

                # Loss
                loss = self.criterion(text_features, audio_features)

            total_loss += loss.item()
            all_text_features.append(text_features.cpu())
            all_audio_features.append(audio_features.cpu())

        # Concatenate all features
        all_text_features = torch.cat(all_text_features, dim=0)
        all_audio_features = torch.cat(all_audio_features, dim=0)

        # Compute metrics
        metrics = RetrievalMetrics.compute_metrics(all_text_features, all_audio_features)
        metrics['loss'] = total_loss / len(loader)

        return metrics

    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], prefix: str = ""):
        """Save checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'global_step': self.global_step,
            'projection_state_dict': self.projection.state_dict(),
            'text_encoder_state_dict': self.text_encoder.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'config': asdict(self.config),
            'metrics': metrics,
            'best_val_r1': self.best_val_r1,
            'best_val_loss': self.best_val_loss
        }

        filename = f"{self.config.audio_encoder}_{prefix}.pt" if prefix else f"{self.config.audio_encoder}_epoch{epoch}.pt"
        path = os.path.join(self.config.checkpoint_dir, filename)
        torch.save(checkpoint, path)
        logger.info(f"Checkpoint saved to {path}")

    def train(self):
        """Main training loop"""
        logger.info("Starting training...")

        for epoch in range(self.config.epochs):
            # Train
            train_metrics = self.train_epoch(epoch)
            logger.info(f"Epoch {epoch+1} - Train loss: {train_metrics['loss']:.4f}")

            # Validate
            val_metrics = self.evaluate(self.val_loader, 'val')
            logger.info(f"Epoch {epoch+1} - Val loss: {val_metrics['loss']:.4f}")
            logger.info(f"  Val t2a R@1: {val_metrics['t2a_R@1']:.2f}%")
            logger.info(f"  Val t2a R@5: {val_metrics['t2a_R@5']:.2f}%")
            logger.info(f"  Val t2a R@10: {val_metrics['t2a_R@10']:.2f}%")
            logger.info(f"  Val t2a MRR: {val_metrics['t2a_MRR']:.4f}")

            # Log to TensorBoard
            for key, value in val_metrics.items():
                self.writer.add_scalar(f'val/{key}', value, epoch)

            # Save best checkpoints
            if val_metrics['t2a_R@1'] > self.best_val_r1:
                self.best_val_r1 = val_metrics['t2a_R@1']
                self.save_checkpoint(epoch, val_metrics, prefix="best_r1")
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

            if val_metrics['loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['loss']
                self.save_checkpoint(epoch, val_metrics, prefix="best_loss")

            # Periodic checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, val_metrics)

            # Save last
            self.save_checkpoint(epoch, val_metrics, prefix="last")

            # Early stopping
            if self.epochs_without_improvement >= self.config.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

        # Final evaluation on test set
        logger.info("\nEvaluating on test set...")
        test_metrics = self.evaluate(self.test_loader, 'test')
        logger.info(f"Test loss: {test_metrics['loss']:.4f}")
        logger.info(f"Test t2a R@1: {test_metrics['t2a_R@1']:.2f}%")
        logger.info(f"Test t2a R@5: {test_metrics['t2a_R@5']:.2f}%")
        logger.info(f"Test t2a R@10: {test_metrics['t2a_R@10']:.2f}%")
        logger.info(f"Test t2a MRR: {test_metrics['t2a_MRR']:.4f}")

        # Save test metrics
        test_metrics_path = os.path.join(
            self.config.checkpoint_dir,
            f"{self.config.audio_encoder}_test_metrics.json"
        )
        with open(test_metrics_path, 'w') as f:
            json.dump(test_metrics, f, indent=2)

        logger.info(f"\nTraining complete!")
        logger.info(f"Best val R@1: {self.best_val_r1:.2f}%")
        logger.info(f"Best val loss: {self.best_val_loss:.4f}")

        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train text-to-audio projection models")
    parser.add_argument('--audio-encoder', type=str, required=True,
                       choices=['muq', 'mert', 'latent'],
                       help='Which audio encoder to train for')
    parser.add_argument('--data-dir', type=str, default='data/musicbench_embeddings',
                       help='Directory containing embeddings HDF5 file')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='logs/tensorboard',
                       help='Directory for TensorBoard logs')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size per GPU')
    parser.add_argument('--lr', type=float, default=5e-5,
                       help='Learning rate for projection')
    parser.add_argument('--text-lr', type=float, default=2e-5,
                       help='Learning rate for text encoder LoRA')
    parser.add_argument('--hidden-dim', type=int, default=None,
                       help='Hidden dimension (default: auto based on audio encoder)')
    parser.add_argument('--no-lora', action='store_true',
                       help='Disable LoRA fine-tuning of text encoder')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # Set hidden dim based on audio encoder if not specified
    if args.hidden_dim is None:
        hidden_dims = {'muq': 1024, 'mert': 2048, 'latent': 1024}
        args.hidden_dim = hidden_dims[args.audio_encoder]

    # Create config
    config = TrainingConfig(
        audio_encoder=args.audio_encoder,
        data_dir=args.data_dir,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        text_lr=args.text_lr,
        hidden_dim=args.hidden_dim,
        use_lora=not args.no_lora,
        seed=args.seed
    )

    # Train
    trainer = Trainer(config)
    trainer.train()


if __name__ == '__main__':
    main()
