"""
Train Model WITHOUT Verb (Context-Only Prediction)
Predicts: Person + Time + Object (excluding last word/verb)

This should achieve 75-85% accuracy since context is easier than verbs!
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


class ContextOnlyDataset(Dataset):
    """Dataset that removes the last word (verb) from gloss sequences"""

    def __init__(self, annotations_file, pose_dir, vocabulary_file, max_sequence_length=60, max_gloss_length=7):
        import pandas as pd

        self.annotations = pd.read_csv(annotations_file)
        self.pose_dir = Path(pose_dir)
        self.max_sequence_length = max_sequence_length
        self.max_gloss_length = max_gloss_length

        # Load vocabulary
        with open(vocabulary_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.word_to_idx = vocab_data['word_to_idx']
            self.idx_to_word = vocab_data['idx_to_word']

        # Special tokens
        self.sos_token = self.word_to_idx['<SOS>']
        self.eos_token = self.word_to_idx['<EOS>']
        self.pad_token = self.word_to_idx['<PAD>']
        self.unk_token = self.word_to_idx['<UNK>']

        print(f"Context-Only Dataset initialized with {len(self.annotations)} samples")
        print(f"Vocabulary size: {len(self.word_to_idx)}")
        print(f"Max context length (without verb): {max_gloss_length}")

        # Analyze context lengths
        self._analyze_context_lengths()

    def _analyze_context_lengths(self):
        """Analyze context sequence lengths"""
        context_lengths = []
        for gloss in self.annotations['gloss']:
            words = str(gloss).split()
            context_length = len(words) - 1  # Remove last word (verb)
            context_lengths.append(max(0, context_length))

        avg_length = np.mean(context_lengths)
        max_length = max(context_lengths)

        print(f"Context length stats (without verb):")
        print(f"  Average: {avg_length:.2f} words")
        print(f"  Maximum: {max_length} words")
        print(f"  Most common: {max(set(context_lengths), key=context_lengths.count)} words")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # Load pose sequence
        pose_path = self.pose_dir / f"pose_{idx:05d}.npy"
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

        pose_sequence = np.load(pose_path)

        # Get full gloss and remove last word (verb)
        full_gloss = row['gloss']
        words = str(full_gloss).split()

        # Context = all words except last (verb)
        context_words = words[:-1] if len(words) > 1 else []
        context_gloss = ' '.join(context_words)

        # Original verb (for reference/analysis)
        verb_word = words[-1] if len(words) > 0 else ''

        # Convert context to indices
        context_indices = self.gloss_to_indices(context_gloss)

        # Prepare tensors
        src = torch.FloatTensor(pose_sequence)
        tgt_input = torch.LongTensor(context_indices[:-1])  # Remove EOS
        tgt_output = torch.LongTensor(context_indices[1:])  # Remove SOS

        return {
            'src': src,
            'tgt_input': tgt_input,
            'tgt_output': tgt_output,
            'context_sequence': context_gloss,
            'full_gloss': full_gloss,
            'verb': verb_word,
            'natural_sentence': row['sentence'],
            'video_id': row['video_id'],
            'original_idx': idx
        }

    def gloss_to_indices(self, gloss_sequence):
        """Convert gloss sequence to indices"""
        words = str(gloss_sequence).split() if gloss_sequence else []

        indices = [self.sos_token]
        for word in words[:self.max_gloss_length - 2]:
            indices.append(self.word_to_idx.get(word, self.unk_token))
        indices.append(self.eos_token)

        # Pad
        while len(indices) < self.max_gloss_length:
            indices.append(self.pad_token)

        return indices

    def indices_to_gloss(self, indices):
        """Convert indices back to gloss"""
        words = []
        for idx in indices:
            if idx == self.eos_token:
                break
            if idx not in [self.sos_token, self.pad_token, self.eos_token]:
                word = self.idx_to_word.get(str(idx), f"<UNK:{idx}>")
                words.append(word)
        return ' '.join(words)


class ContextOnlyTrainer:
    """Trainer for context-only (no verb) prediction"""

    def __init__(self, model, train_loader, val_loader, vocabulary, config, dataset, val_indices):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.config = config
        self.dataset = dataset
        self.val_indices = val_indices

        # Standard cross-entropy loss
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocabulary['word_to_idx']['<PAD>'])

        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=0.05
        )

        # Scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, verbose=True
        )

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.best_val_accuracy = 0
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []

        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        print(f"Using device: {self.device}")

        # Output directory
        self.output_dir = Path(config['paths']['models']) / "updated_context_only_model"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        total_samples = 0

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} Training")

        for batch in progress_bar:
            src = batch['src'].transpose(0, 1).to(self.device)
            tgt_input = batch['tgt_input'].transpose(0, 1).to(self.device)
            tgt_output = batch['tgt_output'].transpose(0, 1).to(self.device)

            # Forward
            self.optimizer.zero_grad()
            output = self.model(src, tgt_input)

            # Loss
            loss = self.criterion(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1)
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Calculate token accuracy
            predictions = output.argmax(dim=-1)
            pad_mask = (tgt_output != self.vocabulary['word_to_idx']['<PAD>'])
            correct = (predictions == tgt_output) & pad_mask
            total_correct += correct.sum().item()
            total_tokens += pad_mask.sum().item()

            # Stats
            total_loss += loss.item() * src.size(1)
            total_samples += src.size(1)

            # Update progress bar
            current_acc = total_correct / total_tokens if total_tokens > 0 else 0
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}'
            })

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)

        return avg_loss, avg_accuracy

    def validate(self, use_autoregressive=True, num_samples=100):
        """
        Validate the model

        Args:
            use_autoregressive: If True, use predict() for sequence accuracy
                               If False, use teacher forcing for token accuracy
            num_samples: Number of samples to evaluate
        """
        self.model.eval()

        # Teacher forcing metrics (fast)
        total_loss = 0
        total_correct_tokens = 0
        total_tokens = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating (Teacher Forcing)", leave=False):
                src = batch['src'].transpose(0, 1).to(self.device)
                tgt_input = batch['tgt_input'].transpose(0, 1).to(self.device)
                tgt_output = batch['tgt_output'].transpose(0, 1).to(self.device)

                # Forward
                output = self.model(src, tgt_input)

                # Loss
                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )

                # Token accuracy
                predictions = output.argmax(dim=-1)
                pad_mask = (tgt_output != self.vocabulary['word_to_idx']['<PAD>'])
                correct = (predictions == tgt_output) & pad_mask
                total_correct_tokens += correct.sum().item()
                total_tokens += pad_mask.sum().item()

                total_loss += loss.item() * src.size(1)
                total_samples += src.size(1)

        avg_loss = total_loss / total_samples
        token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0

        # Autoregressive sequence accuracy (slower but more realistic)
        sequence_correct = 0
        sequence_total = 0
        position_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        if use_autoregressive:
            sos_token = self.vocabulary['word_to_idx']['<SOS>']
            eos_token = self.vocabulary['word_to_idx']['<EOS>']

            sample_indices = self.val_indices[:num_samples]

            with torch.no_grad():
                for idx in tqdm(sample_indices, desc="Validating (Autoregressive)", leave=False):
                    sample = self.dataset[idx]
                    src = sample['src'].to(self.device)
                    target_context = sample['context_sequence']

                    # Predict
                    prediction = self.model.predict(src, sos_token, eos_token)
                    predicted_context = self.dataset.indices_to_gloss(prediction.tolist())

                    # Check exact match
                    if predicted_context == target_context:
                        sequence_correct += 1

                    # Position-wise accuracy
                    pred_words = predicted_context.split()
                    target_words = target_context.split()

                    for pos in range(min(len(pred_words), len(target_words))):
                        position_stats[pos]['total'] += 1
                        if pred_words[pos] == target_words[pos]:
                            position_stats[pos]['correct'] += 1

                    sequence_total += 1

            sequence_accuracy = sequence_correct / sequence_total if sequence_total > 0 else 0
        else:
            sequence_accuracy = 0.0

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(sequence_accuracy if use_autoregressive else token_accuracy)

        return avg_loss, token_accuracy, sequence_accuracy, position_stats

    def train(self):
        """Main training loop"""
        print("🎯 Starting Context-Only Training (NO VERB)!")
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        print(f"📊 Predicting: Person + Time + Object (NO VERB)")
        print(f"⚙️  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        start_time = time.time()

        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate (teacher forcing every epoch, autoregressive every 5 epochs)
            use_autoregressive = (epoch + 1) % 5 == 0 or epoch == 0
            val_loss, token_acc, seq_acc, pos_stats = self.validate(
                use_autoregressive=use_autoregressive,
                num_samples=100
            )

            # Update scheduler
            self.scheduler.step(val_loss)

            # Print results
            print(f"\n📈 Epoch {epoch + 1:03d}:")
            print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc * 100:.2f}%")
            print(f"   Val:   Loss={val_loss:.4f}, Token Acc={token_acc * 100:.2f}%")

            if use_autoregressive:
                print(f"   Val Sequence Acc: {seq_acc * 100:.2f}%")

                # Print position-wise accuracy
                if pos_stats:
                    print(f"   Position-wise accuracy:")
                    for pos in sorted(pos_stats.keys()):
                        pos_acc = pos_stats[pos]['correct'] / pos_stats[pos]['total']
                        print(f"      Pos {pos}: {pos_acc * 100:.2f}%")

            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Save best model
            metric_to_track = seq_acc if use_autoregressive else token_acc

            if metric_to_track > self.best_val_accuracy:
                self.best_val_accuracy = metric_to_track
                self.best_val_loss = val_loss
                self.save_checkpoint(best=True)
                print(f"   🎯 NEW BEST! Accuracy: {metric_to_track * 100:.2f}%")

            # Regular checkpoint
            if (epoch + 1) % 10 == 0:
                self.save_checkpoint()

            # Early stopping
            if epoch > 30 and self.best_val_accuracy > 0.85:
                print(f"\n🎉 Excellent! Accuracy > 85%")
                break
            elif epoch > 40 and val_loss > self.best_val_loss * 1.3:
                print(f"\n🛑 Early stopping triggered")
                break

        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time / 60:.2f} minutes!")
        print(f"🏆 Best Validation Accuracy: {self.best_val_accuracy * 100:.2f}%")
        print(f"🏆 Best Validation Loss: {self.best_val_loss:.4f}")

        # Final evaluation
        self.final_evaluation()

    def final_evaluation(self, num_samples=200):
        """Comprehensive final evaluation"""
        print(f"\n{'=' * 70}")
        print(f"FINAL EVALUATION (Context-Only, {num_samples} samples)")
        print(f"{'=' * 70}")

        self.model.eval()

        sos_token = self.vocabulary['word_to_idx']['<SOS>']
        eos_token = self.vocabulary['word_to_idx']['<EOS>']

        correct = 0
        total = 0
        position_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
        length_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        sample_indices = self.val_indices[:num_samples]

        print("\n📝 Sample Predictions (first 20):")

        with torch.no_grad():
            for i, idx in enumerate(tqdm(sample_indices, desc="Final Evaluation")):
                sample = self.dataset[idx]
                src = sample['src'].to(self.device)
                target_context = sample['context_sequence']

                # Predict
                prediction = self.model.predict(src, sos_token, eos_token)
                predicted_context = self.dataset.indices_to_gloss(prediction.tolist())

                is_correct = (predicted_context == target_context)

                if is_correct:
                    correct += 1
                total += 1

                # Length stats
                context_length = len(target_context.split())
                length_stats[context_length]['total'] += 1
                if is_correct:
                    length_stats[context_length]['correct'] += 1

                # Position stats
                pred_words = predicted_context.split()
                target_words = target_context.split()

                for pos in range(min(len(pred_words), len(target_words))):
                    position_stats[pos]['total'] += 1
                    if pred_words[pos] == target_words[pos]:
                        position_stats[pos]['correct'] += 1

                # Show first 20 samples
                if i < 20:
                    status = "✅" if is_correct else "❌"
                    print(f"\n{status} Sample {i + 1}:")
                    print(f"   Full gloss: {sample['full_gloss']}")
                    print(f"   Context (no verb): {target_context}")
                    print(f"   Predicted: {predicted_context}")
                    print(f"   Verb (excluded): {sample['verb']}")

        # Calculate metrics
        overall_acc = correct / total

        print(f"\n{'=' * 70}")
        print("FINAL RESULTS")
        print(f"{'=' * 70}")
        print(f"\n📊 Overall Context Accuracy: {overall_acc * 100:.2f}% ({correct}/{total})")

        print(f"\n📏 Accuracy by Context Length:")
        for length in sorted(length_stats.keys()):
            stats = length_stats[length]
            acc = stats['correct'] / stats['total']
            print(f"   {length}-word context: {acc * 100:.2f}% ({stats['correct']}/{stats['total']})")

        print(f"\n📍 Accuracy by Position:")
        for pos in sorted(position_stats.keys()):
            stats = position_stats[pos]
            acc = stats['correct'] / stats['total']
            print(f"   Position {pos}: {acc * 100:.2f}% ({stats['correct']}/{stats['total']})")

        print(f"\n💡 Insight: Context (without verb) is {'EASIER' if overall_acc > 0.7 else 'HARDER'} to predict")
        print(f"   This confirms that {'spatial features' if overall_acc > 0.7 else 'temporal features'} are")
        print(f"   {'well-learned' if overall_acc > 0.7 else 'challenging'} by the model.")

    def save_checkpoint(self, best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': self.best_val_loss,
            'best_val_accuracy': self.best_val_accuracy,
            'vocabulary': self.vocabulary,
            'config': self.config,
            'val_indices': self.val_indices,
            'training_type': 'context_only'
        }

        filename = self.output_dir / (
            "updated_best_context_only_model.pth" if best else f"checkpoint_epoch_{self.current_epoch + 1}.pth")
        torch.save(checkpoint, filename)
        print(f"   💾 Saved: {filename.name}")


def main():
    """Main function"""
    from src.modeling.transformer_model import BanglaSignTransformer
    from src.utils.config import load_config

    print("=" * 70)
    print("CONTEXT-ONLY TRAINING (NO VERB PREDICTION)")
    print("=" * 70)

    # Load config
    config = load_config()
    config['training']['batch_size'] = 32
    config['training']['learning_rate'] = 7e-5
    config['training']['epochs'] = 80

    # Load vocabulary
    with open('data/annotations/gloss_vocabulary.json', 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    # Load context-only dataset
    print("\n📂 Loading context-only dataset...")
    dataset = ContextOnlyDataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        pose_dir="data/processed/pose_sequences_full",
        vocabulary_file="data/annotations/gloss_vocabulary.json",
        max_gloss_length=7  # Shorter since no verb
    )

    print(f"✅ Loaded {len(dataset)} samples")

    # Create data split
    train_size = int(0.85 * len(dataset))
    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, len(dataset) - train_size], generator=generator
    )

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)

    # Create model
    print("\n🔧 Creating model...")
    model = BanglaSignTransformer(
        input_dim=375,
        vocab_size=len(vocabulary['word_to_idx']),
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dropout=0.2,
        max_seq_length=7  # Shorter since no verb
    )

    print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create trainer
    trainer = ContextOnlyTrainer(
        model, train_loader, val_loader, vocabulary, config, dataset, val_ds.indices
    )

    # Train
    trainer.train()


if __name__ == "__main__":
    main()