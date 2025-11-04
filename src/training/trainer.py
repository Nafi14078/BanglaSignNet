import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time
import os
from pathlib import Path
from tqdm import tqdm


class BanglaSignTrainer:
    def __init__(self, model, train_loader, val_loader, vocabulary, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.config = config

        # Training setup
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocabulary['word_to_idx']['<PAD>'])
        self.optimizer = optim.Adam(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=1e-4
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.8)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        # Create output directory
        self.output_dir = Path(config['paths']['models'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            src = batch['src'].transpose(0, 1).to(self.device)  # (seq_len, batch_size, features)
            tgt_input = batch['tgt_input'].transpose(0, 1).to(self.device)  # (tgt_len, batch_size)
            tgt_output = batch['tgt_output'].transpose(0, 1).to(self.device)  # (tgt_len, batch_size)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input)  # (tgt_len, batch_size, vocab_size)

            # Calculate loss
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item() * src.size(1)  # Multiply by batch size
            total_samples += src.size(1)

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / total_samples:.4f}'
            })

        avg_loss = total_loss / total_samples
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        total_correct = 0
        total_tokens = 0

        with torch.no_grad():
            for batch in self.val_loader:
                src = batch['src'].transpose(0, 1).to(self.device)
                tgt_input = batch['tgt_input'].transpose(0, 1).to(self.device)
                tgt_output = batch['tgt_output'].transpose(0, 1).to(self.device)

                # Forward pass
                output = self.model(src, tgt_input)

                # Calculate loss
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                # Calculate accuracy
                predictions = output.argmax(dim=-1)
                correct = (predictions == tgt_output) & (tgt_output != self.vocabulary['word_to_idx']['<PAD>'])
                total_correct += correct.sum().item()
                total_tokens += (tgt_output != self.vocabulary['word_to_idx']['<PAD>']).sum().item()

                total_loss += loss.item() * src.size(1)
                total_samples += src.size(1)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        self.val_losses.append(avg_loss)

        return avg_loss, accuracy

    def train(self):
        """Complete training loop"""
        print("Starting training...")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_accuracy = self.validate()

            # Update learning rate
            self.scheduler.step()

            # Print epoch summary
            print(f"Epoch {epoch + 1:03d}/{self.config['training']['epochs']}: "
                  f"Train Loss: {train_loss:.4f}, "
                  f"Val Loss: {val_loss:.4f}, "
                  f"Val Acc: {val_accuracy:.4f}, "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.6f}")

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint(best=True)
                print(f"New best model saved with val_loss: {val_loss:.4f}")

            # Save regular checkpoint
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint()

            # Early stopping check
            if epoch > 10 and val_loss > self.best_val_loss * 1.5:
                print("Early stopping triggered!")
                break

        print("Training completed!")
        self.plot_training_curve()

    def save_checkpoint(self, best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'vocabulary': self.vocabulary,
            'config': self.config
        }

        if best:
            filename = self.output_dir / "best_model.pth"
        else:
            filename = self.output_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth"

        torch.save(checkpoint, filename)
        print(f"Checkpoint saved: {filename}")

    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.best_val_loss = checkpoint['best_val_loss']

        print(f"Checkpoint loaded: {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch + 1}")

    def plot_training_curve(self):
        """Plot training and validation loss curves"""
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(10, 6))
            plt.plot(self.train_losses, label='Training Loss')
            plt.plot(self.val_losses, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.grid(True)

            # Save plot
            plot_path = self.output_dir / "training_curve.png"
            plt.savefig(plot_path)
            plt.close()

            print(f"Training curve saved: {plot_path}")

        except ImportError:
            print("Matplotlib not available, skipping plot generation")

    def evaluate_sample(self, dataset, sample_idx=0):
        """Evaluate a single sample"""
        self.model.eval()

        sample = dataset[sample_idx]
        src = sample['src'].unsqueeze(1).to(self.device)  # Add batch dimension
        tgt_input = sample['tgt_input'].unsqueeze(1).to(self.device)

        with torch.no_grad():
            # Greedy decoding
            sos_token = self.vocabulary['word_to_idx']['<SOS>']
            eos_token = self.vocabulary['word_to_idx']['<EOS>']

            prediction = self.model.predict(src.squeeze(1), sos_token, eos_token)
            predicted_sentence = dataset.indices_to_sentence(prediction.tolist())

        print(f"\n=== Sample Evaluation ===")
        print(f"Original Sentence: {sample['sentence']}")
        print(f"Predicted Sentence: {predicted_sentence}")
        print(f"Gloss: {sample['gloss']}")
        print(f"Video ID: {sample['video_id']}")


def create_data_loaders(dataset, config, train_ratio=0.8):
    """Create train and validation data loaders"""
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader


def main():
    """Main training function"""
    from src.modeling.transformer_model import BanglaSignTransformer
    from src.data_processing.sign_language_dataset import BanglaSignLanguageDataset
    from src.utils.config import load_config

    # Load configuration
    config = load_config()

    # Load vocabulary
    with open('data/annotations/vocabulary.json', 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    # Create dataset
    dataset = BanglaSignLanguageDataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        pose_dir="data/processed/pose_sequences",
        vocabulary_file="data/annotations/vocabulary.json"
    )

    if len(dataset) == 0:
        print("No data available for training!")
        return

    # Create data loaders
    train_loader, val_loader = create_data_loaders(dataset, config, train_ratio=0.8)

    # Create model
    model = BanglaSignTransformer(
        input_dim=375,  # From pose extraction
        vocab_size=len(vocabulary['word_to_idx']),
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['encoder_layers'],
        num_decoder_layers=config['model']['decoder_layers'],
        dropout=config['model']['dropout'],
        max_seq_length=20
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    trainer = BanglaSignTrainer(model, train_loader, val_loader, vocabulary, config)

    # Start training
    trainer.train()

    # Evaluate a sample
    trainer.evaluate_sample(dataset)


if __name__ == "__main__":
    main()