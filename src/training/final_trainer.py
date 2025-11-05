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


class FinalBanglaSignTrainer:
    def __init__(self, model, train_loader, val_loader, vocabulary, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.config = config

        # Training setup
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocabulary['word_to_idx']['<PAD>'])
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, verbose=True
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_exact_match = 0
        self.train_losses = []
        self.val_losses = []
        self.val_accuracies = []
        self.exact_matches = []

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = Path(config['paths']['models']) / "final"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} Training")

        for batch_idx, batch in enumerate(progress_bar):
            # Move data to device
            src = batch['src'].transpose(0, 1).to(self.device)
            tgt_input = batch['tgt_input'].transpose(0, 1).to(self.device)
            tgt_output = batch['tgt_output'].transpose(0, 1).to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input)

            # Calculate loss
            loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update statistics
            total_loss += loss.item() * src.size(1)
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
        """Validate with detailed metrics"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        total_correct = 0
        total_tokens = 0
        exact_matches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
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

                # Check exact matches
                for i in range(predictions.size(1)):
                    pred_seq = predictions[:, i]
                    target_seq = tgt_output[:, i]
                    # Remove padding for comparison
                    pred_seq = pred_seq[pred_seq != self.vocabulary['word_to_idx']['<PAD>']]
                    target_seq = target_seq[target_seq != self.vocabulary['word_to_idx']['<PAD>']]
                    if torch.equal(pred_seq, target_seq):
                        exact_matches += 1

                total_loss += loss.item() * src.size(1)
                total_samples += src.size(1)

        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        exact_match_rate = exact_matches / total_samples if total_samples > 0 else 0

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(accuracy)
        self.exact_matches.append(exact_match_rate)

        return avg_loss, accuracy, exact_match_rate

    def train(self):
        """Complete training loop"""
        print("🚀 Starting FINAL training with 500 samples!")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        print(f"🔤 Vocabulary size: {len(self.vocabulary['word_to_idx'])}")
        print(f"⚙️  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_accuracy, exact_match_rate = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)

            # Print epoch summary
            print(f"📈 Epoch {epoch + 1:03d}: "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Val Acc: {val_accuracy:.4f} | "
                  f"Exact Match: {exact_match_rate:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Save best model based on exact match rate
            if exact_match_rate > self.best_exact_match:
                self.best_exact_match = exact_match_rate
                self.best_val_loss = val_loss
                self.save_checkpoint(best=True)
                print(f"🎯 NEW BEST! Exact Match: {exact_match_rate:.4f}")

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint()

            # Early stopping
            if epoch > 20 and exact_match_rate > 0.8:
                print(f"🎉 Target achieved! Exact Match > 80%")
                break
            elif epoch > 30 and val_loss > self.best_val_loss * 1.5:
                print("🛑 Early stopping triggered!")
                break

        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time:.2f} seconds!")
        print(f"🏆 Best Exact Match: {self.best_exact_match:.4f}")

        self.plot_training_curve()
        self.evaluate_multiple_samples()

    def evaluate_multiple_samples(self, num_samples=10):
        """Evaluate multiple samples"""
        from src.data_processing.large_sign_language_dataset import LargeBanglaSignLanguageDataset

        dataset = LargeBanglaSignLanguageDataset(
            annotations_file="data/annotations/filtered_annotations.csv",
            pose_dir="data/processed/pose_sequences_full",
            vocabulary_file="data/annotations/vocabulary.json"
        )

        self.model.eval()

        print(f"\n{'=' * 60}")
        print("FINAL MODEL PREDICTIONS")
        print(f"{'=' * 60}")

        correct_predictions = 0

        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            src = sample['src'].unsqueeze(1).to(self.device)

            with torch.no_grad():
                sos_token = self.vocabulary['word_to_idx']['<SOS>']
                eos_token = self.vocabulary['word_to_idx']['<EOS>']

                prediction = self.model.predict(src.squeeze(1), sos_token, eos_token)
                predicted_sentence = dataset.indices_to_sentence(prediction.tolist())

            is_correct = sample['sentence'] == predicted_sentence
            if is_correct:
                correct_predictions += 1

            print(f"\n📝 Sample {i + 1}:")
            print(f"   Original: {sample['sentence']}")
            print(f"   Predicted: {predicted_sentence}")
            print(f"   Gloss: {sample['gloss']}")
            print(f"   Status: {'✅ CORRECT' if is_correct else '❌ INCORRECT'}")

        accuracy = correct_predictions / num_samples
        print(f"\n🎯 Prediction Accuracy: {accuracy:.2%} ({correct_predictions}/{num_samples})")

    def save_checkpoint(self, best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'exact_matches': self.exact_matches,
            'best_val_loss': self.best_val_loss,
            'best_exact_match': self.best_exact_match,
            'vocabulary': self.vocabulary,
            'config': self.config
        }

        if best:
            filename = self.output_dir / "best_model.pth"
        else:
            filename = self.output_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth"

        torch.save(checkpoint, filename)

    def plot_training_curve(self):
        """Plot training curves"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Plot losses
            ax1.plot(self.train_losses, label='Training Loss', linewidth=2)
            ax1.plot(self.val_losses, label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss')
            ax1.legend()
            ax1.grid(True)

            # Plot accuracies
            ax2.plot(self.val_accuracies, label='Token Accuracy', linewidth=2)
            ax2.plot(self.exact_matches, label='Exact Match Rate', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Validation Metrics')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plot_path = self.output_dir / "training_curves.png"
            plt.savefig(plot_path)
            plt.close()

            print(f"📊 Training curves saved: {plot_path}")

        except ImportError:
            print("Matplotlib not available, skipping plots")


def create_data_loaders_large(dataset, config, train_ratio=0.8):
    """Create train and validation data loaders for large dataset"""
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
    """Main training function for 500 samples"""
    from src.modeling.transformer_model import BanglaSignTransformer
    from src.data_processing.large_sign_language_dataset import LargeBanglaSignLanguageDataset
    from src.utils.config import load_config

    # Load configuration
    config = load_config()

    # Update config for larger dataset
    config['training']['batch_size'] = 16
    config['training']['learning_rate'] = 0.0001
    config['training']['epochs'] = 100

    # Load vocabulary
    with open('data/annotations/vocabulary.json', 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    # Create large dataset
    print("Loading large dataset...")
    dataset = LargeBanglaSignLanguageDataset(
        annotations_file="data/annotations/filtered_annotations.csv",
        pose_dir="data/processed/pose_sequences_full",
        vocabulary_file="data/annotations/vocabulary.json"
    )

    if len(dataset) == 0:
        print("No data available for training!")
        return

    print(f"✅ Loaded {len(dataset)} samples")

    # Create data loaders
    train_loader, val_loader = create_data_loaders_large(dataset, config, train_ratio=0.8)

    # Create model (using original architecture since we have more data)
    model = BanglaSignTransformer(
        input_dim=375,
        vocab_size=len(vocabulary['word_to_idx']),
        d_model=256,  # Balanced size
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dropout=0.1,
        max_seq_length=20
    )

    print(f"🔧 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    trainer = FinalBanglaSignTrainer(model, train_loader, val_loader, vocabulary, config)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()