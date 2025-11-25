import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm


class FullWordLevelBanglaSignTrainer:
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
            weight_decay=0.05
        )

        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, verbose=True
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_gloss_accuracy = 0
        self.train_losses = []
        self.val_losses = []
        self.gloss_accuracies = []

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = Path(config['paths']['models']) / "full_word_level_old_updated_1"
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
        """Validate with gloss-level accuracy"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        total_correct = 0
        total_tokens = 0
        exact_gloss_matches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                src = batch['src'].transpose(0, 1).to(self.device)
                tgt_input = batch['tgt_input'].transpose(0, 1).to(self.device)
                tgt_output = batch['tgt_output'].transpose(0, 1).to(self.device)

                # Forward pass
                output = self.model(src, tgt_input)

                # Calculate loss
                loss = self.criterion(output.reshape(-1, output.size(-1)), tgt_output.reshape(-1))

                # Calculate token accuracy
                predictions = output.argmax(dim=-1)
                correct = (predictions == tgt_output) & (tgt_output != self.vocabulary['word_to_idx']['<PAD>'])
                total_correct += correct.sum().item()
                total_tokens += (tgt_output != self.vocabulary['word_to_idx']['<PAD>']).sum().item()

                # Check exact gloss sequence matches
                for i in range(predictions.size(1)):
                    pred_seq = predictions[:, i]
                    target_seq = tgt_output[:, i]
                    # Remove padding for comparison
                    pred_seq = pred_seq[pred_seq != self.vocabulary['word_to_idx']['<PAD>']]
                    target_seq = target_seq[target_seq != self.vocabulary['word_to_idx']['<PAD>']]
                    if torch.equal(pred_seq, target_seq):
                        exact_gloss_matches += 1

                total_loss += loss.item() * src.size(1)
                total_samples += src.size(1)

        avg_loss = total_loss / total_samples
        token_accuracy = total_correct / total_tokens if total_tokens > 0 else 0
        gloss_accuracy = exact_gloss_matches / total_samples if total_samples > 0 else 0

        self.val_losses.append(avg_loss)
        self.gloss_accuracies.append(gloss_accuracy)

        return avg_loss, token_accuracy, gloss_accuracy

    def train(self):
        """Full dataset training loop"""
        print("🎯 Starting FULL DATASET Word-Level Bangla Sign Language Recognition!")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        print(f"🔤 Gloss vocabulary size: {len(self.vocabulary['word_to_idx'])}")
        print(f"⚙️  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate
            val_loss, token_accuracy, gloss_accuracy = self.validate()

            # Update learning rate
            self.scheduler.step(val_loss)

            # Print epoch summary
            print(f"📈 Epoch {epoch + 1:03d}: "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Token Acc: {token_accuracy:.4f} | "
                  f"Gloss Acc: {gloss_accuracy:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Save best model based on gloss accuracy
            if gloss_accuracy > self.best_gloss_accuracy:
                self.best_gloss_accuracy = gloss_accuracy
                self.best_val_loss = val_loss
                self.save_checkpoint(best=True)
                print(f"🎯 NEW BEST! Gloss Accuracy: {gloss_accuracy:.4f}")

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint()

            # Early stopping conditions
            if epoch > 20 and gloss_accuracy > 0.90:
                print(f"🎉 Excellent! Gloss Accuracy > 90%")
                break
            elif epoch > 30 and gloss_accuracy > 0.85:
                print(f"🎉 Very Good! Gloss Accuracy > 85%")
                break
            elif epoch > 40 and val_loss > self.best_val_loss * 1.3:
                print("🛑 Early stopping triggered!")
                break

        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time / 60:.2f} minutes!")
        print(f"🏆 Best Gloss Accuracy: {self.best_gloss_accuracy:.4f}")

        self.evaluate_gloss_predictions()

    def evaluate_gloss_predictions(self, num_samples=15):
        """Evaluate gloss sequence predictions"""
        from src.data_processing.gloss_dataset import BanglaSignGlossDataset

        dataset = BanglaSignGlossDataset(
            annotations_file="data/annotations/dataset_annotations.csv",  # Use full dataset
            pose_dir="data/processed/pose_sequences_full",
            vocabulary_file="data/annotations/gloss_vocabulary.json"
        )

        self.model.eval()

        print(f"\n{'=' * 70}")
        print("FINAL WORD-LEVEL GLOSS PREDICTION RESULTS")
        print(f"{'=' * 70}")

        correct_predictions = 0
        samples_by_length = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}
        correct_by_length = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0}

        for i in range(min(num_samples, len(dataset))):
            sample = dataset[i]
            src = sample['src'].unsqueeze(1).to(self.device)

            with torch.no_grad():
                sos_token = self.vocabulary['word_to_idx']['<SOS>']
                eos_token = self.vocabulary['word_to_idx']['<EOS>']

                prediction = self.model.predict(src.squeeze(1), sos_token, eos_token)
                predicted_gloss = dataset.indices_to_gloss(prediction.tolist())

            is_correct = sample['gloss_sequence'] == predicted_gloss
            gloss_length = len(sample['gloss_sequence'].split())

            samples_by_length[gloss_length] += 1
            if is_correct:
                correct_predictions += 1
                correct_by_length[gloss_length] += 1

            status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
            print(f"\n📝 Sample {i + 1:2d} ({gloss_length} words): {status}")
            print(f"   Natural: {sample['natural_sentence']}")
            print(f"   Target:  {sample['gloss_sequence']}")
            print(f"   Predicted: {predicted_gloss}")

        # Calculate overall and per-length accuracy
        overall_accuracy = correct_predictions / num_samples

        print(f"\n{'=' * 70}")
        print("ACCURACY BREAKDOWN")
        print(f"{'=' * 70}")
        print(f"Overall Accuracy: {overall_accuracy:.2%} ({correct_predictions}/{num_samples})")

        for length in sorted(samples_by_length.keys()):
            if samples_by_length[length] > 0:
                length_accuracy = correct_by_length[length] / samples_by_length[length]
                print(
                    f"  {length}-word sequences: {length_accuracy:.2%} ({correct_by_length[length]}/{samples_by_length[length]})")

    def save_checkpoint(self, best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'gloss_accuracies': self.gloss_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_gloss_accuracy': self.best_gloss_accuracy,
            'vocabulary': self.vocabulary,
            'config': self.config,
            'dataset_info': {
                'total_samples': 1922,
                'vocabulary_size': 39,
                'average_gloss_length': 5.18
            }
        }

        if best:
            filename = self.output_dir / "best_full_gloss_model_old_updated_1.pth"
        else:
            filename = self.output_dir / f"full_checkpoint_epoch_{self.current_epoch + 1}.pth"

        torch.save(checkpoint, filename)
        print(f"💾 Checkpoint saved: {filename}")


def create_data_loaders_full(dataset, config, train_ratio=0.85):
    """Create train and validation data loaders for full dataset"""
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
    """Main training function for full dataset"""
    from src.modeling.transformer_model import BanglaSignTransformer
    from src.data_processing.gloss_dataset import BanglaSignGlossDataset
    from src.utils.config import load_config

    # Load configuration
    config = load_config()

    # Update config for full dataset training
    config['training']['batch_size'] = 32  # Larger batch size for more data
    config['training']['learning_rate'] = 0.00008
    config['training']['epochs'] = 80

    # Load GLOSS vocabulary
    with open('data/annotations/gloss_vocabulary.json', 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    # Create gloss dataset with FULL data
    print("Loading FULL word-level gloss dataset...")
    dataset = BanglaSignGlossDataset(
        annotations_file="data/annotations/dataset_annotations.csv",  # Full dataset
        pose_dir="data/processed/pose_sequences_full",
        vocabulary_file="data/annotations/gloss_vocabulary.json",
        max_gloss_length=8
    )

    if len(dataset) == 0:
        print("No data available for training!")
        return

    print(f"✅ Loaded {len(dataset)} word-level samples")

    # Create data loaders with 85/15 split (more training data)
    train_loader, val_loader = create_data_loaders_full(dataset, config, train_ratio=0.85)

    # Create model optimized for word-level recognition
    model = BanglaSignTransformer(
        input_dim=375,
        vocab_size=len(vocabulary['word_to_idx']),
        d_model=256,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dropout=0.3,
        max_seq_length=8  # Matches our gloss sequence length
    )

    print(f"🔧 Word-level model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create full dataset trainer
    trainer = FullWordLevelBanglaSignTrainer(model, train_loader, val_loader, vocabulary, config)

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()