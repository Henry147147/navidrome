"""Main training script for text embedding models."""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from tqdm import tqdm
import wandb

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.text_embedders.muq_text_embedder import MuQTextEmbedder
from models.text_embedders.mert_text_embedder import MERTTextEmbedder
from models.text_embedders.music2latent_text_embedder import Music2LatentTextEmbedder
from training.losses import InfoNCELoss, CombinedLoss
from utils.data_loader import create_dataloader


class Trainer:
    """Trainer for text embedding models."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        loss_fn: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler,
        device: str = "cuda",
        checkpoint_dir: Path = Path("checkpoints"),
        log_interval: int = 100,
        use_wandb: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        """
        Initialize trainer.

        Args:
            model: Text embedding model
            train_loader: Training data loader
            val_loader: Validation data loader
            loss_fn: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            device: Device to train on
            checkpoint_dir: Directory to save checkpoints
            log_interval: Steps between logging
            use_wandb: Whether to use Weights & Biases
            logger: Logger instance
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.logger = logger or logging.getLogger("Trainer")

        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_infonce = 0.0
        total_mse = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (input_ids, attention_mask, audio_embeddings) in enumerate(pbar):
            # Move to device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            audio_embeddings = audio_embeddings.to(self.device)

            # Forward pass
            text_embeddings = self.model(input_ids, attention_mask)

            # Compute loss
            if isinstance(self.loss_fn, CombinedLoss):
                loss, infonce_loss, mse_loss = self.loss_fn(text_embeddings, audio_embeddings)
                total_infonce += infonce_loss.item()
                total_mse += mse_loss.item()
            else:
                loss = self.loss_fn(text_embeddings, audio_embeddings)
                infonce_loss = loss
                mse_loss = torch.tensor(0.0)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Log to wandb
            if self.use_wandb and self.global_step % self.log_interval == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/infonce_loss': infonce_loss.item() if isinstance(infonce_loss, torch.Tensor) else infonce_loss,
                    'train/mse_loss': mse_loss.item() if isinstance(mse_loss, torch.Tensor) else mse_loss,
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'train/epoch': self.current_epoch,
                }, step=self.global_step)

        # Update learning rate
        if self.scheduler is not None:
            self.scheduler.step()

        avg_loss = total_loss / num_batches
        avg_infonce = total_infonce / num_batches if total_infonce > 0 else 0
        avg_mse = total_mse / num_batches if total_mse > 0 else 0

        return {
            'loss': avg_loss,
            'infonce_loss': avg_infonce,
            'mse_loss': avg_mse,
        }

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        total_infonce = 0.0
        total_mse = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc="Validation")

        for input_ids, attention_mask, audio_embeddings in pbar:
            # Move to device
            input_ids = input_ids.to(self.device)
            attention_mask = attention_mask.to(self.device)
            audio_embeddings = audio_embeddings.to(self.device)

            # Forward pass
            text_embeddings = self.model(input_ids, attention_mask)

            # Compute loss
            if isinstance(self.loss_fn, CombinedLoss):
                loss, infonce_loss, mse_loss = self.loss_fn(text_embeddings, audio_embeddings)
                total_infonce += infonce_loss.item()
                total_mse += mse_loss.item()
            else:
                loss = self.loss_fn(text_embeddings, audio_embeddings)

            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_infonce = total_infonce / num_batches if total_infonce > 0 else 0
        avg_mse = total_mse / num_batches if total_mse > 0 else 0

        return {
            'loss': avg_loss,
            'infonce_loss': avg_infonce,
            'mse_loss': avg_mse,
        }

    def save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_val_loss': self.best_val_loss,
        }

        # Save latest checkpoint
        checkpoint_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, checkpoint_path)
        self.logger.info(f"Saved checkpoint to {checkpoint_path}")

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Saved best checkpoint to {best_path}")

            # Also save the model in HuggingFace format
            model_save_dir = self.checkpoint_dir / "best_model"
            self.model.save_pretrained(str(model_save_dir))
            self.logger.info(f"Saved model to {model_save_dir}")

    def load_checkpoint(self, checkpoint_path: Path):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        self.logger.info(f"Loaded checkpoint from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")

    def train(self, num_epochs: int):
        """Train for multiple epochs."""
        self.logger.info(f"Starting training for {num_epochs} epochs")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch()
            self.logger.info(f"Epoch {epoch} - Train Loss: {train_metrics['loss']:.4f}")

            # Validate
            val_metrics = self.validate()
            self.logger.info(f"Epoch {epoch} - Val Loss: {val_metrics['loss']:.4f}")

            # Log to wandb
            if self.use_wandb:
                wandb.log({
                    'val/loss': val_metrics['loss'],
                    'val/infonce_loss': val_metrics['infonce_loss'],
                    'val/mse_loss': val_metrics['mse_loss'],
                    'epoch': epoch,
                }, step=self.global_step)

            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            self.save_checkpoint(is_best=is_best)

        self.logger.info("Training complete!")
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train text embedding model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb logging')
    args = parser.parse_args()

    # Load config
    with open(args.config, 'r') as f:
        config = json.load(f)

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger("train")

    # Initialize wandb
    use_wandb = not args.no_wandb
    if use_wandb:
        wandb.init(
            project=config['wandb']['project'],
            name=config['wandb']['name'],
            config=config,
        )

    # Create model
    model_type = config['model']['type']
    if model_type == 'muq':
        model = MuQTextEmbedder(**config['model']['params'])
    elif model_type == 'mert':
        model = MERTTextEmbedder(**config['model']['params'])
    elif model_type == 'music2latent':
        model = Music2LatentTextEmbedder(**config['model']['params'])
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    logger.info(f"Created {model_type} model with {sum(p.numel() for p in model.parameters())} parameters")

    # Create data loaders
    train_loader = create_dataloader(
        embeddings_file=Path(config['data']['train_embeddings']),
        metadata_file=Path(config['data']['train_metadata']),
        tokenizer=model.tokenizer,
        **config['data']['loader_params']
    )

    val_loader = create_dataloader(
        embeddings_file=Path(config['data']['val_embeddings']),
        metadata_file=Path(config['data']['val_metadata']),
        tokenizer=model.tokenizer,
        batch_size=config['data']['loader_params']['batch_size'],
        shuffle=False,
        num_workers=config['data']['loader_params']['num_workers'],
    )

    # Create loss function
    loss_type = config['training']['loss']['type']
    if loss_type == 'infonce':
        loss_fn = InfoNCELoss(**config['training']['loss']['params'])
    elif loss_type == 'combined':
        loss_fn = CombinedLoss(**config['training']['loss']['params'])
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        **config['training']['optimizer']
    )

    # Create scheduler
    num_training_steps = len(train_loader) * config['training']['num_epochs']
    warmup_steps = int(num_training_steps * config['training']['scheduler']['warmup_ratio'])

    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=num_training_steps - warmup_steps,
        eta_min=config['training']['scheduler']['min_lr']
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_steps]
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config['training']['device'],
        checkpoint_dir=Path(config['training']['checkpoint_dir']),
        log_interval=config['training']['log_interval'],
        use_wandb=use_wandb,
        logger=logger,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))

    # Train
    trainer.train(config['training']['num_epochs'])

    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
