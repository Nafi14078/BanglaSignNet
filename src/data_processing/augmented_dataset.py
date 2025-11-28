import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
from pathlib import Path


class AugmentedBanglaSignDataset(Dataset):
    """
    🆕 Dataset withPOSE DATA AUGMENTATION

    Augmentation techniques:
    1. Temporal: Speed variation (0.8x - 1.2x)
    2. Spatial: Pose noise injection (Gaussian noise)
    3. Sequence: Random frame dropping
    """

    def __init__(self, annotations_file, pose_dir, vocabulary_file,
                 max_sequence_length=60, max_gloss_length=7,
                 augment=True, augment_prob=0.5):
        """
        Args:
            augment: Whether to apply augmentation
            augment_prob: Probability of applying each augmentation
        """
        self.annotations = pd.read_csv(annotations_file)
        self.pose_dir = Path(pose_dir)
        self.max_sequence_length = max_sequence_length
        self.max_gloss_length = max_gloss_length

        # 🆕 Augmentation settings
        self.augment = augment
        self.augment_prob = augment_prob

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

        aug_status = "WITH AUGMENTATION" if augment else "WITHOUT AUGMENTATION"
        print(f"Dataset initialized {aug_status}")
        print(f"  Samples: {len(self.annotations)}")
        print(f"  Vocabulary: {len(self.word_to_idx)} words")
        print(f"  Augmentation probability: {augment_prob}")

    def temporal_augmentation(self, pose_sequence):
        """
        🆕 TEMPORAL AUGMENTATION: Speed variation
        Randomly speeds up or slows down the sequence
        """
        if not self.augment or np.random.rand() > self.augment_prob:
            return pose_sequence

        # Random speed factor (0.8x to 1.2x)
        speed_factor = np.random.uniform(0.8, 1.2)

        original_length = len(pose_sequence)
        new_length = int(original_length * speed_factor)
        new_length = min(max(new_length, 10), self.max_sequence_length)  # Clamp

        # Interpolate to new length
        indices = np.linspace(0, original_length - 1, new_length)
        augmented = np.zeros((new_length, pose_sequence.shape[1]))

        for i, idx in enumerate(indices):
            lower = int(np.floor(idx))
            upper = min(int(np.ceil(idx)), original_length - 1)
            weight = idx - lower

            augmented[i] = (1 - weight) * pose_sequence[lower] + weight * pose_sequence[upper]

        # Pad to max_sequence_length
        if len(augmented) < self.max_sequence_length:
            padding = np.zeros((self.max_sequence_length - len(augmented), pose_sequence.shape[1]))
            augmented = np.vstack([augmented, padding])

        return augmented

    def spatial_augmentation(self, pose_sequence):
        """
        🆕 SPATIAL AUGMENTATION: Add Gaussian noise to pose landmarks
        Simulates small variations in pose detection
        """
        if not self.augment or np.random.rand() > self.augment_prob:
            return pose_sequence

        # Add small Gaussian noise (σ = 0.01 to 0.02)
        noise_std = np.random.uniform(0.01, 0.02)
        noise = np.random.normal(0, noise_std, pose_sequence.shape)

        augmented = pose_sequence + noise

        # Clip to valid range [0, 1] for normalized coordinates
        augmented = np.clip(augmented, 0, 1)

        return augmented

    def sequence_dropout(self, pose_sequence):
        """
        🆕 SEQUENCE DROPOUT: Randomly drop frames
        Simulates missing frames or occlusions
        """
        if not self.augment or np.random.rand() > self.augment_prob:
            return pose_sequence

        # Drop 5-15% of frames randomly
        dropout_rate = np.random.uniform(0.05, 0.15)

        # Find non-zero frames (actual data, not padding)
        non_zero_mask = np.any(pose_sequence != 0, axis=1)
        non_zero_indices = np.where(non_zero_mask)[0]

        if len(non_zero_indices) == 0:
            return pose_sequence

        # Keep random subset
        num_keep = max(int(len(non_zero_indices) * (1 - dropout_rate)), 1)
        keep_indices = np.sort(np.random.choice(non_zero_indices, num_keep, replace=False))

        # Create new sequence with dropped frames
        augmented = np.zeros_like(pose_sequence)
        augmented[:len(keep_indices)] = pose_sequence[keep_indices]

        return augmented

    def apply_augmentation(self, pose_sequence):
        """Apply all augmentation techniques"""
        if not self.augment:
            return pose_sequence

        # Apply augmentations in sequence
        pose_sequence = self.temporal_augmentation(pose_sequence)
        pose_sequence = self.spatial_augmentation(pose_sequence)
        pose_sequence = self.sequence_dropout(pose_sequence)

        return pose_sequence

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # Load pose sequence
        pose_path = self.pose_dir / f"pose_{idx:05d}.npy"
        if not pose_path.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

        pose_sequence = np.load(pose_path)

        # 🆕 Apply augmentation during training
        pose_sequence = self.apply_augmentation(pose_sequence)

        # Get context (all words except last/verb)
        full_gloss = row['gloss']
        words = str(full_gloss).split()
        context_words = words[:-1] if len(words) > 1 else []
        context_gloss = ' '.join(context_words)
        verb_word = words[-1] if len(words) > 0 else ''

        # Convert to indices
        context_indices = self.gloss_to_indices(context_gloss)

        # Prepare tensors
        src = torch.FloatTensor(pose_sequence)
        tgt_input = torch.LongTensor(context_indices[:-1])
        tgt_output = torch.LongTensor(context_indices[1:])

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


def test_augmentation():
    """Test augmentation techniques"""
    print("Testing Data Augmentation...")

    # Create dataset with augmentation
    dataset = AugmentedBanglaSignDataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        pose_dir="data/processed/pose_sequences_full",
        vocabulary_file="data/annotations/gloss_vocabulary.json",
        augment=True,
        augment_prob=0.8
    )

    # Get one sample multiple times to see augmentation
    print("\nTesting augmentation on same sample:")
    for i in range(3):
        sample = dataset[0]
        pose = sample['src'].numpy()
        non_zero = np.sum(np.any(pose != 0, axis=1))
        print(f"  Augmentation {i + 1}: {non_zero} non-zero frames, shape={pose.shape}")

    print("✅ Augmentation working!")


if __name__ == "__main__":
    test_augmentation()