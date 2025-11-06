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


class Final1922Trainer:
    def __init__(self, model, train_loader, val_loader, vocabulary, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.config = config

        # Optimized training setup for 1,922 samples
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocabulary['word_to_idx']['<PAD>'])
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=0.01
        )

        # Learning rate scheduler optimized for this dataset size
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # First restart after 10 epochs
            T_mult=2,  # Double the cycle each time
            eta_min=1e-7  # Minimum learning rate
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

        # Create final output directory
        self.output_dir = Path(config['paths']['models']) / "final_1922"
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

            # Update learning rate
            self.scheduler.step(self.current_epoch + batch_idx / len(self.train_loader))

            # Update statistics
            total_loss += loss.item() * src.size(1)
            total_samples += src.size(1)

            # Update progress bar
            current_lr = self.optimizer.param_groups[0]['lr']
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / total_samples:.4f}',
                'lr': f'{current_lr:.2e}'
            })

        avg_loss = total_loss / total_samples
        self.train_losses.append(avg_loss)

        return avg_loss

    def validate(self):
        """Comprehensive validation"""
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

                # Calculate metrics
                predictions = output.argmax(dim=-1)
                correct = (predictions == tgt_output) & (tgt_output != self.vocabulary['word_to_idx']['<PAD>'])
                total_correct += correct.sum().item()
                total_tokens += (tgt_output != self.vocabulary['word_to_idx']['<PAD>']).sum().item()

                # Exact matches
                for i in range(predictions.size(1)):
                    pred_seq = predictions[:, i]
                    target_seq = tgt_output[:, i]
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
        """Final training loop for 1,922 samples"""
        print("🎯 STARTING FINAL TRAINING WITH 1,922 SAMPLES!")
        print(f"📊 Training samples: {len(self.train_loader.dataset):,}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset):,}")
        print(f"🔤 Vocabulary size: {len(self.vocabulary['word_to_idx'])}")
        print(f"⚙️  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, val_accuracy, exact_match_rate = self.validate()

            epoch_time = time.time() - epoch_start

            # Print epoch summary
            print(f"📈 Epoch {epoch + 1:03d} | "
                  f"Time: {epoch_time:.1f}s | "
                  f"Train: {train_loss:.4f} | "
                  f"Val: {val_loss:.4f} | "
                  f"Acc: {val_accuracy:.4f} | "
                  f"Exact: {exact_match_rate:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Save best model
            if exact_match_rate > self.best_exact_match:
                self.best_exact_match = exact_match_rate
                self.best_val_loss = val_loss
                self.save_checkpoint(best=True)
                print(f"🎯 NEW BEST! Exact Match: {exact_match_rate:.4f}")

            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint()

            # Early stopping conditions
            if exact_match_rate > 0.90:  # 90% accuracy
                print(f"🎉 EXCELLENT! >90% Exact Match Achieved!")
                break
            elif exact_match_rate > 0.80 and epoch > 30:  # Good performance
                print(f"🎉 VERY GOOD! >80% Exact Match")
                break
            elif epoch > 50 and val_loss > self.best_val_loss * 1.8:
                print("🛑 Early stopping - validation loss increasing")
                break

        total_time = time.time() - start_time
        minutes = total_time // 60
        seconds = total_time % 60

        print(f"\n✅ FINAL TRAINING COMPLETED!")
        print(f"⏱️  Total time: {minutes:.0f}m {seconds:.0f}s")
        print(f"🏆 Best Exact Match: {self.best_exact_match:.4f}")

        self.plot_training_curves()
        self.final_evaluation()

    def final_evaluation(self, num_samples=15):
        """Final comprehensive evaluation"""
        from src.data_processing.large_sign_language_dataset import LargeBanglaSignLanguageDataset

        dataset = LargeBanglaSignLanguageDataset(
            annotations_file="data/annotations/dataset_annotations.csv",
            pose_dir="data/processed/pose_sequences_full",
            vocabulary_file="data/annotations/vocabulary.json"
        )

        self.model.eval()

        print(f"\n{'=' * 60}")
        print("🎯 FINAL MODEL EVALUATION - 1,922 SAMPLES")
        print(f"{'=' * 60}")

        correct_predictions = 0
        samples_evaluated = min(num_samples, len(dataset))

        for i in range(samples_evaluated):
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
                status = "✅ CORRECT"
            else:
                status = "❌ INCORRECT"

            print(f"\n📝 Sample {i + 1}:")
            print(f"   Original: {sample['sentence']}")
            print(f"   Predicted: {predicted_sentence}")
            print(f"   Gloss: {sample['gloss']}")
            print(f"   Status: {status}")

        accuracy = correct_predictions / samples_evaluated

        print(f"\n🎯 FINAL ACCURACY: {accuracy:.2%} ({correct_predictions}/{samples_evaluated})")

        # Performance assessment
        if accuracy >= 0.80:
            print("🏆 OUTSTANDING PERFORMANCE! >80% accuracy")
        elif accuracy >= 0.70:
            print("👍 EXCELLENT PERFORMANCE! >70% accuracy")
        elif accuracy >= 0.60:
            print("📈 GOOD PERFORMANCE! >60% accuracy")
        else:
            print("🔧 NEEDS IMPROVEMENT - Consider model adjustments")

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
            'config': self.config,
            'dataset_size': 1922
        }

        if best:
            filename = self.output_dir / "best_model.pth"
        else:
            filename = self.output_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth"

        torch.save(checkpoint, filename)
        print(f"💾 Checkpoint saved: {filename}")

    def plot_training_curves(self):
        """Plot training curves"""
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            # Loss curves
            ax1.plot(self.train_losses, label='Training Loss', linewidth=2)
            ax1.plot(self.val_losses, label='Validation Loss', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.set_title('Training and Validation Loss - 1,922 Samples')
            ax1.legend()
            ax1.grid(True)

            # Accuracy curves
            ax2.plot(self.val_accuracies, label='Token Accuracy', linewidth=2, color='green')
            ax2.plot(self.exact_matches, label='Exact Match Rate', linewidth=2, color='red')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Accuracy')
            ax2.set_title('Validation Accuracy - 1,922 Samples')
            ax2.legend()
            ax2.grid(True)

            plt.tight_layout()
            plot_path = self.output_dir / "final_training_curves.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()

            print(f"📊 Training curves saved: {plot_path}")

        except ImportError:
            print("Matplotlib not available, skipping plots")


def main():
    """Final training with complete 1,922 samples"""
    from src.modeling.transformer_model import BanglaSignTransformer
    from src.data_processing.large_sign_language_dataset import LargeBanglaSignLanguageDataset
    from src.utils.config import load_config

    # Load configuration
    config = load_config()

    # Optimized configuration for 1,922 samples
    config['training']['batch_size'] = 16
    config['training']['learning_rate'] = 0.0003
    config['training']['epochs'] = 80

    # Balanced model architecture
    config['model']['d_model'] = 256
    config['model']['encoder_layers'] = 4
    config['model']['decoder_layers'] = 4

    # Load vocabulary
    with open('data/annotations/vocabulary.json', 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    # Create final dataset
    print("🚀 Loading final dataset with 1,922 samples...")
    dataset = LargeBanglaSignLanguageDataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        pose_dir="data/processed/pose_sequences_full",
        vocabulary_file="data/annotations/vocabulary.json"
    )

    print(f"✅ Loaded {len(dataset)} samples!")

    # Create data loaders (85/15 split)
    from src.training.trainer import create_data_loaders
    train_loader, val_loader = create_data_loaders(dataset, config, train_ratio=0.85)

    print(f"📊 Training: {len(train_loader.dataset):,} samples")
    print(f"📊 Validation: {len(val_loader.dataset):,} samples")

    # Create optimized model
    model = BanglaSignTransformer(
        input_dim=375,
        vocab_size=len(vocabulary['word_to_idx']),
        d_model=config['model']['d_model'],
        nhead=config['model']['nhead'],
        num_encoder_layers=config['model']['encoder_layers'],
        num_decoder_layers=config['model']['decoder_layers'],
        dropout=0.1,
        max_seq_length=20
    )

    print(f"🔧 Final model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create final trainer
    trainer = Final1922Trainer(model, train_loader, val_loader, vocabulary, config)

    # Start final training
    trainer.train()


if __name__ == "__main__":
    main()