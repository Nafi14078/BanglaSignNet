"""
🚀 ENHANCED TRAINER with:
1. ✅ Scheduled Sampling (reduces teacher forcing gradually)
2. ✅ Beam Search Decoding (better predictions)
3. ✅ Data Augmentation (via dataset)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import json
import time
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict


class EnhancedContextOnlyTrainer:
    """Enhanced trainer with scheduled sampling and beam search"""

    def __init__(self, model, train_loader, val_loader, vocabulary, config, dataset, val_indices):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.vocabulary = vocabulary
        self.config = config
        self.dataset = dataset
        self.val_indices = val_indices

        # Loss and optimizer
        self.criterion = nn.CrossEntropyLoss(ignore_index=vocabulary['word_to_idx']['<PAD>'])
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=0.05
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8, verbose=True
        )

        # 🆕 SCHEDULED SAMPLING settings
        self.use_scheduled_sampling = config['training'].get('scheduled_sampling', True)
        self.ss_start_epoch = config['training'].get('ss_start_epoch', 5)
        self.ss_decay_rate = config['training'].get('ss_decay_rate', 0.02)

        # 🆕 BEAM SEARCH settings
        self.use_beam_search = config['training'].get('use_beam_search', True)
        self.beam_width = config['training'].get('beam_width', 5)

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

        print(f"🎯 Enhanced Trainer Initialized")
        print(f"   Device: {self.device}")
        print(f"   Scheduled Sampling: {'✅ Enabled' if self.use_scheduled_sampling else '❌ Disabled'}")
        print(
            f"   Beam Search: {'✅ Enabled (width=' + str(self.beam_width) + ')' if self.use_beam_search else '❌ Disabled'}")

        # Output directory
        self.output_dir = Path(config['paths']['models']) / "enhanced_context_model"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_teacher_forcing_ratio(self):
        """
        🆕 SCHEDULED SAMPLING: Calculate teacher forcing ratio

        Starts at 1.0 (100% teacher forcing) and gradually decreases
        Formula: max(0.5, 1.0 - (epoch - start_epoch) * decay_rate)
        """
        if not self.use_scheduled_sampling:
            return 1.0  # Always use teacher forcing

        if self.current_epoch < self.ss_start_epoch:
            return 1.0  # Full teacher forcing for first few epochs

        # Gradually decrease
        ratio = 1.0 - (self.current_epoch - self.ss_start_epoch) * self.ss_decay_rate
        ratio = max(0.5, ratio)  # Never go below 50%

        return ratio

    def train_epoch_with_scheduled_sampling(self):
        """
        🆕 Training with SCHEDULED SAMPLING

        During training, we gradually shift from teacher forcing to model predictions
        This helps the model learn to recover from its own errors
        """
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_tokens = 0
        total_samples = 0

        teacher_forcing_ratio = self.get_teacher_forcing_ratio()

        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch + 1} Training")
        progress_bar.set_postfix({'tf_ratio': f'{teacher_forcing_ratio:.2f}'})

        for batch in progress_bar:
            src = batch['src'].transpose(0, 1).to(self.device)  # (seq_len, batch, features)
            tgt_input = batch['tgt_input'].transpose(0, 1).to(self.device)  # (tgt_len, batch)
            tgt_output = batch['tgt_output'].transpose(0, 1).to(self.device)

            batch_size = src.size(1)
            tgt_len = tgt_input.size(0)

            self.optimizer.zero_grad()

            # Encode once
            memory = self.model.encoder(src)

            # 🆕 SCHEDULED SAMPLING: Mix teacher forcing and model predictions
            decoder_input = tgt_input[0:1, :]  # Start with SOS token

            outputs = []

            for t in range(tgt_len):
                # Generate mask
                tgt_mask = self.model.generate_square_subsequent_mask(decoder_input.size(0)).to(self.device)

                # Decode
                output = self.model.decoder(decoder_input, memory, tgt_mask=tgt_mask)
                outputs.append(output[-1:, :, :])  # Keep last output

                # Decide next input: teacher forcing or model prediction
                if t < tgt_len - 1:
                    use_teacher_forcing = np.random.rand() < teacher_forcing_ratio

                    if use_teacher_forcing:
                        # Use ground truth
                        next_token = tgt_input[t + 1:t + 2, :]
                    else:
                        # Use model prediction
                        next_token = output[-1, :, :].argmax(dim=-1, keepdim=True).transpose(0, 1)

                    decoder_input = torch.cat([decoder_input, next_token], dim=0)

            # Stack outputs
            output = torch.cat(outputs, dim=0)

            # Calculate loss
            loss = self.criterion(
                output.reshape(-1, output.size(-1)),
                tgt_output.reshape(-1)
            )

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Calculate accuracy
            predictions = output.argmax(dim=-1)
            pad_mask = (tgt_output != self.vocabulary['word_to_idx']['<PAD>'])
            correct = (predictions == tgt_output) & pad_mask
            total_correct += correct.sum().item()
            total_tokens += pad_mask.sum().item()

            # Stats
            total_loss += loss.item() * batch_size
            total_samples += batch_size

            # Update progress
            current_acc = total_correct / total_tokens if total_tokens > 0 else 0
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{current_acc:.4f}',
                'tf': f'{teacher_forcing_ratio:.2f}'
            })

        avg_loss = total_loss / total_samples
        avg_accuracy = total_correct / total_tokens if total_tokens > 0 else 0

        self.train_losses.append(avg_loss)
        self.train_accuracies.append(avg_accuracy)

        return avg_loss, avg_accuracy

    def validate_with_beam_search(self, num_samples=100):
        """
        🆕 Validation with BEAM SEARCH

        Uses beam search for more accurate predictions during validation
        """
        self.model.eval()

        # Token-level metrics (fast, with teacher forcing)
        total_loss = 0
        total_correct_tokens = 0
        total_tokens = 0
        total_samples = 0

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating (Teacher Forcing)", leave=False):
                src = batch['src'].transpose(0, 1).to(self.device)
                tgt_input = batch['tgt_input'].transpose(0, 1).to(self.device)
                tgt_output = batch['tgt_output'].transpose(0, 1).to(self.device)

                output = self.model(src, tgt_input)

                loss = self.criterion(
                    output.reshape(-1, output.size(-1)),
                    tgt_output.reshape(-1)
                )

                predictions = output.argmax(dim=-1)
                pad_mask = (tgt_output != self.vocabulary['word_to_idx']['<PAD>'])
                correct = (predictions == tgt_output) & pad_mask
                total_correct_tokens += correct.sum().item()
                total_tokens += pad_mask.sum().item()

                total_loss += loss.item() * src.size(1)
                total_samples += src.size(1)

        avg_loss = total_loss / total_samples
        token_accuracy = total_correct_tokens / total_tokens if total_tokens > 0 else 0

        # 🆕 Sequence-level accuracy with BEAM SEARCH
        sequence_correct = 0
        sequence_total = 0
        position_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

        sos_token = self.vocabulary['word_to_idx']['<SOS>']
        eos_token = self.vocabulary['word_to_idx']['<EOS>']

        sample_indices = self.val_indices[:num_samples]

        with torch.no_grad():
            for idx in tqdm(sample_indices, desc=f"Validating ({'Beam' if self.use_beam_search else 'Greedy'})",
                            leave=False):
                sample = self.dataset[idx]
                src = sample['src'].to(self.device)
                target_context = sample['context_sequence']

                # 🆕 Use beam search or greedy
                if self.use_beam_search:
                    prediction = self.model.beam_search(
                        src, sos_token, eos_token,
                        beam_width=self.beam_width
                    )
                else:
                    prediction = self.model.predict(src, sos_token, eos_token)

                predicted_context = self.dataset.indices_to_gloss(prediction.tolist())

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

        self.val_losses.append(avg_loss)
        self.val_accuracies.append(sequence_accuracy)

        return avg_loss, token_accuracy, sequence_accuracy, position_stats

    def train(self):
        """Main training loop with all enhancements"""
        print("=" * 70)
        print("🚀 ENHANCED CONTEXT-ONLY TRAINING")
        print("=" * 70)
        print(f"📊 Training samples: {len(self.train_loader.dataset)}")
        print(f"📊 Validation samples: {len(self.val_loader.dataset)}")
        print(f"🎯 Predicting: Person + Time + Object (NO VERB)")
        print(f"⚙️  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"✨ Enhancements:")
        print(f"   - Scheduled Sampling: {self.use_scheduled_sampling}")
        print(f"   - Beam Search (width={self.beam_width}): {self.use_beam_search}")
        print(f"   - Data Augmentation: Check dataset")
        print("=" * 70)

        start_time = time.time()

        for epoch in range(self.config['training']['epochs']):
            self.current_epoch = epoch

            # Train with scheduled sampling
            train_loss, train_acc = self.train_epoch_with_scheduled_sampling()

            # Validate with beam search (every 5 epochs for speed)
            use_full_eval = (epoch + 1) % 5 == 0 or epoch == 0
            val_loss, token_acc, seq_acc, pos_stats = self.validate_with_beam_search(
                num_samples=100 if use_full_eval else 50
            )

            # Update scheduler
            self.scheduler.step(val_loss)

            # Print results
            tf_ratio = self.get_teacher_forcing_ratio()
            print(f"\n📈 Epoch {epoch + 1:03d}:")
            print(f"   Train: Loss={train_loss:.4f}, Acc={train_acc * 100:.2f}% (TF={tf_ratio:.2f})")
            print(f"   Val:   Loss={val_loss:.4f}, Token Acc={token_acc * 100:.2f}%")

            if use_full_eval:
                print(f"   Val Sequence Acc: {seq_acc * 100:.2f}%")

                if pos_stats:
                    print(f"   Position-wise accuracy:")
                    for pos in sorted(pos_stats.keys())[:5]:
                        pos_acc = pos_stats[pos]['correct'] / pos_stats[pos]['total']
                        print(f"      Pos {pos}: {pos_acc * 100:.2f}%")

            print(f"   LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Save best model
            metric_to_track = seq_acc if use_full_eval else token_acc

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
            elif epoch > 50 and val_loss > self.best_val_loss * 1.3:
                print(f"\n🛑 Early stopping triggered")
                break

        training_time = time.time() - start_time
        print(f"\n✅ Training completed in {training_time / 60:.2f} minutes!")
        print(f"🏆 Best Validation Accuracy: {self.best_val_accuracy * 100:.2f}%")

    def save_checkpoint(self, best=False):
        """Save checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
            'vocabulary': self.vocabulary,
            'config': self.config,
            'enhancements': {
                'scheduled_sampling': self.use_scheduled_sampling,
                'beam_search': self.use_beam_search,
                'beam_width': self.beam_width
            }
        }

        filename = self.output_dir / (
            "best_enhanced_model.pth" if best else f"checkpoint_epoch_{self.current_epoch + 1}.pth"
        )
        torch.save(checkpoint, filename)
        print(f"   💾 Saved: {filename.name}")


def main():
    """Main function with all enhancements"""
    import sys
    sys.path.append('src')

    from src.modeling.transformer_model import BanglaSignTransformer
    from src.utils.config import load_config

    print("=" * 70)
    print("🚀 ENHANCED TRAINING PIPELINE")
    print("=" * 70)

    # Load config with enhancements
    config = load_config()
    config['training']['batch_size'] = 32
    config['training']['learning_rate'] = 7e-5
    config['training']['epochs'] = 80

    # 🆕 Enhancement settings
    config['training']['scheduled_sampling'] = True
    config['training']['ss_start_epoch'] = 10  # Start scheduled sampling after 10 epochs
    config['training']['ss_decay_rate'] = 0.02  # Decay rate per epoch
    config['training']['use_beam_search'] = True
    config['training']['beam_width'] = 5

    # Load vocabulary
    with open('data/annotations/gloss_vocabulary.json', 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    # 🆕 Load AUGMENTED dataset
    print("\n📂 Loading dataset with augmentation...")
    from src.data_processing.augmented_dataset import AugmentedBanglaSignDataset

    train_dataset = AugmentedBanglaSignDataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        pose_dir="data/processed/pose_sequences_full",
        vocabulary_file="data/annotations/gloss_vocabulary.json",
        max_gloss_length=7,
        augment=True,  # 🆕 Enable augmentation for training
        augment_prob=0.6
    )

    val_dataset = AugmentedBanglaSignDataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        pose_dir="data/processed/pose_sequences_full",
        vocabulary_file="data/annotations/gloss_vocabulary.json",
        max_gloss_length=7,
        augment=False  # No augmentation for validation
    )

    # Split data
    train_size = int(0.85 * len(train_dataset))
    generator = torch.Generator().manual_seed(42)
    indices = torch.randperm(len(train_dataset), generator=generator).tolist()

    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False, num_workers=0)

    print(f"✅ Train: {len(train_subset)} samples (with augmentation)")
    print(f"✅ Val: {len(val_subset)} samples (no augmentation)")

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
        max_seq_length=7
    )

    print(f"✅ Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Create enhanced trainer
    trainer = EnhancedContextOnlyTrainer(
        model, train_loader, val_loader, vocabulary, config,
        val_dataset, val_indices
    )

    # Train with all enhancements!
    trainer.train()


if __name__ == "__main__":
    main()