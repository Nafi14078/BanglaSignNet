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
    def __init__(self, model, train_loader, val_loader, vocabulary, config, dataset, val_indices):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.config = config
        self.dataset = dataset  # Store full dataset for autoregressive validation
        self.val_indices = val_indices  # Store validation indices

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
        self.best_autoregressive_acc = 0  # Track best autoregressive accuracy
        self.train_losses = []
        self.val_losses = []
        self.gloss_accuracies = []
        self.autoregressive_accs = []  # Track autoregressive accuracy per epoch

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Create output directory
        self.output_dir = Path(config['paths']['models']) / "full_word_level_fixed_seed"
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
        """Validate with gloss-level accuracy (teacher forcing)"""
        self.model.eval()
        total_loss = 0
        total_samples = 0
        total_correct = 0
        total_tokens = 0
        exact_gloss_matches = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating (Teacher Forcing)"):
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

    def validate_autoregressive(self, num_samples=50):
        """
        CRITICAL: Validate using autoregressive prediction (actual inference)
        This shows the TRUE accuracy you'll see at test time
        """
        self.model.eval()

        sos_token = self.vocabulary['word_to_idx']['<SOS>']
        eos_token = self.vocabulary['word_to_idx']['<EOS>']

        correct = 0
        total = 0

        # Sample from validation indices
        sample_indices = self.val_indices[:num_samples]

        with torch.no_grad():
            for idx in tqdm(sample_indices, desc="Validating (Autoregressive)", leave=False):
                sample = self.dataset[idx]
                src = sample['src'].to(self.device)
                target_gloss = sample['gloss_sequence']

                # Autoregressive prediction (what happens at test time)
                prediction = self.model.predict(src, sos_token, eos_token)
                predicted_gloss = self.dataset.indices_to_gloss(prediction.tolist())

                if predicted_gloss == target_gloss:
                    correct += 1
                total += 1

        accuracy = correct / total if total > 0 else 0
        return accuracy

    def train(self):
        """Full dataset training loop"""
        print("🎯 Starting FIXED SEED Word-Level Bangla Sign Language Recognition!")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        print(f"🔤 Gloss vocabulary size: {len(self.vocabulary['word_to_idx'])}")
        print(f"⚙️  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"✅ Using fixed seed for reproducible splits")

        start_time = time.time()

        for epoch in range(self.current_epoch, self.config['training']['epochs']):
            self.current_epoch = epoch

            # Train
            train_loss = self.train_epoch()

            # Validate (teacher forcing - fast but optimistic)
            val_loss, token_accuracy, gloss_accuracy_tf = self.validate()

            # CRITICAL: Validate autoregressive (every 5 epochs - shows TRUE accuracy)
            if (epoch + 1) % 5 == 0 or epoch == 0:
                autoregressive_acc = self.validate_autoregressive(num_samples=50)
                self.autoregressive_accs.append(autoregressive_acc)
                print(f"\n   🎯 Autoregressive Accuracy: {autoregressive_acc:.4f} (TRUE inference accuracy)")
            else:
                autoregressive_acc = None

            # Update learning rate
            self.scheduler.step(val_loss)

            # Print epoch summary
            print(f"📈 Epoch {epoch + 1:03d}: "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | "
                  f"Token Acc (TF): {token_accuracy:.4f} | "
                  f"Gloss Acc (TF): {gloss_accuracy_tf:.4f} | "
                  f"LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Save best model based on autoregressive accuracy (when available)
            if autoregressive_acc is not None:
                if autoregressive_acc > self.best_autoregressive_acc:
                    self.best_autoregressive_acc = autoregressive_acc
                    self.best_val_loss = val_loss
                    self.save_checkpoint(best=True)
                    print(f"   🎯 NEW BEST! Autoregressive: {autoregressive_acc:.4f}")
            else:
                # Between autoregressive checks, use teacher forcing as proxy
                if gloss_accuracy_tf > self.best_gloss_accuracy:
                    self.best_gloss_accuracy = gloss_accuracy_tf
                    self.save_checkpoint(best=True)

            # Save regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint()

            # Early stopping conditions
            if epoch > 20 and self.best_autoregressive_acc > 0.90:
                print(f"🎉 Excellent! Autoregressive Accuracy > 90%")
                break
            elif epoch > 30 and self.best_autoregressive_acc > 0.85:
                print(f"🎉 Very Good! Autoregressive Accuracy > 85%")
                break
            elif epoch > 40 and val_loss > self.best_val_loss * 1.3:
                print("🛑 Early stopping triggered!")
                break

        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time / 60:.2f} minutes!")
        print(f"🏆 Best Autoregressive Accuracy: {self.best_autoregressive_acc:.4f}")
        print(f"📊 Best Teacher Forcing Accuracy: {self.best_gloss_accuracy:.4f}")

        # Final comprehensive evaluation
        self.evaluate_gloss_predictions()

    def evaluate_gloss_predictions(self, num_samples=100):
        """Final evaluation on validation set"""
        self.model.eval()

        print(f"\n{'=' * 70}")
        print(f"FINAL WORD-LEVEL GLOSS PREDICTION RESULTS ({num_samples} val samples)")
        print(f"{'=' * 70}")

        sos_token = self.vocabulary['word_to_idx']['<SOS>']
        eos_token = self.vocabulary['word_to_idx']['<EOS>']

        correct_predictions = 0
        samples_by_length = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}
        correct_by_length = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0}

        # Position-wise accuracy
        position_stats = {i: {'correct': 0, 'total': 0} for i in range(8)}

        # Use validation indices
        eval_indices = self.val_indices[:num_samples]

        for i, idx in enumerate(eval_indices):
            sample = self.dataset[idx]
            src = sample['src'].to(self.device)
            target_gloss = sample['gloss_sequence']

            with torch.no_grad():
                prediction = self.model.predict(src, sos_token, eos_token)
                predicted_gloss = self.dataset.indices_to_gloss(prediction.tolist())

            is_correct = target_gloss == predicted_gloss
            gloss_length = len(target_gloss.split())

            samples_by_length[gloss_length] += 1
            if is_correct:
                correct_predictions += 1
                correct_by_length[gloss_length] += 1

            # Position-wise accuracy
            pred_words = predicted_gloss.split()
            target_words = target_gloss.split()

            for pos in range(min(len(pred_words), len(target_words))):
                position_stats[pos]['total'] += 1
                if pred_words[pos] == target_words[pos]:
                    position_stats[pos]['correct'] += 1

            # Show first 15
            if i < 15:
                status = "✅ CORRECT" if is_correct else "❌ INCORRECT"
                print(f"\n📝 Sample {i + 1:2d} ({gloss_length} words): {status}")
                print(f"   Natural: {sample['natural_sentence']}")
                print(f"   Target:  {target_gloss}")
                print(f"   Predicted: {predicted_gloss}")

        # Calculate overall accuracy
        overall_accuracy = correct_predictions / len(eval_indices)

        print(f"\n{'=' * 70}")
        print("ACCURACY BREAKDOWN")
        print(f"{'=' * 70}")
        print(f"Overall Accuracy: {overall_accuracy:.2%} ({correct_predictions}/{len(eval_indices)})")

        # Per-length accuracy
        print(f"\n📏 Accuracy by Sequence Length:")
        for length in sorted(samples_by_length.keys()):
            if samples_by_length[length] > 0:
                length_accuracy = correct_by_length[length] / samples_by_length[length]
                print(
                    f"   {length}-word sequences: {length_accuracy:.2%} ({correct_by_length[length]}/{samples_by_length[length]})")

        # Position-wise accuracy
        print(f"\n📍 Accuracy by Position:")
        for pos in sorted(position_stats.keys()):
            if position_stats[pos]['total'] > 0:
                pos_acc = position_stats[pos]['correct'] / position_stats[pos]['total']
                print(
                    f"   Position {pos}: {pos_acc:.2%} ({position_stats[pos]['correct']}/{position_stats[pos]['total']})")

        # Context vs Verb analysis
        context_positions = [0, 1, 2]
        context_correct = sum(position_stats[p]['correct'] for p in context_positions if position_stats[p]['total'] > 0)
        context_total = sum(position_stats[p]['total'] for p in context_positions if position_stats[p]['total'] > 0)
        context_acc = context_correct / context_total if context_total > 0 else 0

        max_pos = max([p for p in position_stats if position_stats[p]['total'] > 0], default=3)
        verb_acc = position_stats[max_pos]['correct'] / position_stats[max_pos]['total'] if position_stats[max_pos][
                                                                                                'total'] > 0 else 0

        print(f"\n🎯 Context vs Verb Analysis:")
        print(f"   Context (positions 0-2): {context_acc:.2%}")
        print(f"   Verb (position {max_pos}): {verb_acc:.2%}")
        print(f"   Context-Verb Gap: {(context_acc - verb_acc):.2%}")

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
            'autoregressive_accs': self.autoregressive_accs,
            'best_val_loss': self.best_val_loss,
            'best_gloss_accuracy': self.best_gloss_accuracy,
            'best_autoregressive_acc': self.best_autoregressive_acc,
            'vocabulary': self.vocabulary,
            'config': self.config,
            'val_indices': self.val_indices,  # Save for reproducibility
            'dataset_info': {
                'total_samples': 1922,
                'vocabulary_size': 39,
                'average_gloss_length': 5.18,
                'train_size': len(self.train_loader.dataset),
                'val_size': len(self.val_loader.dataset)
            }
        }

        if best:
            filename = self.output_dir / "best_fixed_seed_model.pth"
        else:
            filename = self.output_dir / f"checkpoint_epoch_{self.current_epoch + 1}.pth"

        torch.save(checkpoint, filename)
        print(f"   💾 Checkpoint saved: {filename.name}")


def create_data_loaders_full(dataset, config, train_ratio=0.85, seed=42):
    """Create train and validation data loaders with FIXED SEED"""
    dataset_size = len(dataset)
    train_size = int(train_ratio * dataset_size)
    val_size = dataset_size - train_size

    # CRITICAL FIX: Use fixed seed for reproducibility
    print(f"\n🎲 Creating data split with seed={seed} (reproducible)")
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )

    print(f"   ✅ Training: {len(train_dataset)} samples")
    print(f"   ✅ Validation: {len(val_dataset)} samples")
    print(f"   📊 Split ratio: {train_ratio:.0%} / {1 - train_ratio:.0%}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,  # Don't shuffle validation
        num_workers=0
    )

    # Return validation indices for later evaluation
    val_indices = val_dataset.indices

    return train_loader, val_loader, val_indices


def main():
    """Main training function for full dataset"""
    from src.modeling.transformer_model import BanglaSignTransformer
    from src.data_processing.gloss_dataset import BanglaSignGlossDataset
    from src.utils.config import load_config

    # Load configuration
    config = load_config()

    # Update config for full dataset training
    config['training']['batch_size'] = 32
    config['training']['learning_rate'] = 0.00008
    config['training']['epochs'] = 50

    # Load GLOSS vocabulary
    with open('data/annotations/gloss_vocabulary.json', 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    # Create gloss dataset with FULL data
    print("=" * 70)
    print("FIXED SEED BANGLA SIGN LANGUAGE TRAINER")
    print("=" * 70)
    print("\n📂 Loading dataset...")
    dataset = BanglaSignGlossDataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        pose_dir="data/processed/pose_sequences_full",
        vocabulary_file="data/annotations/gloss_vocabulary.json",
        max_gloss_length=8
    )

    if len(dataset) == 0:
        print("❌ No data available for training!")
        return

    print(f"✅ Loaded {len(dataset)} word-level samples")

    # Create data loaders with FIXED SEED (reproducible)
    train_loader, val_loader, val_indices = create_data_loaders_full(
        dataset, config, train_ratio=0.85, seed=42
    )

    # Create model optimized for word-level recognition
    print("\n🔧 Creating model...")
    model = BanglaSignTransformer(
        input_dim=375,
        vocab_size=len(vocabulary['word_to_idx']),
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.2,
        max_seq_length=8
    )

    print(f"✅ Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer with fixed seed
    trainer = FullWordLevelBanglaSignTrainer(
        model, train_loader, val_loader, vocabulary, config, dataset, val_indices
    )

    # Start training
    print("\n" + "=" * 70)
    trainer.train()


if __name__ == "__main__":
    main()