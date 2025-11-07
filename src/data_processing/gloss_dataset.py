import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
from pathlib import Path


class BanglaSignGlossDataset(Dataset):
    def __init__(self, annotations_file, pose_dir, vocabulary_file, max_sequence_length=60, max_gloss_length=8):
        self.annotations = pd.read_csv(annotations_file)

        # Load GLOSS vocabulary (not the natural sentence vocabulary)
        with open(vocabulary_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.word_to_idx = vocab_data['word_to_idx']
            self.idx_to_word = vocab_data['idx_to_word']

        self.pose_dir = Path(pose_dir)
        self.max_sequence_length = max_sequence_length
        self.max_gloss_length = max_gloss_length

        # Special tokens
        self.sos_token = self.word_to_idx['<SOS>']
        self.eos_token = self.word_to_idx['<EOS>']
        self.pad_token = self.word_to_idx['<PAD>']
        self.unk_token = self.word_to_idx['<UNK>']

        print(f"Gloss dataset initialized with {len(self.annotations)} samples")
        print(f"Gloss vocabulary size: {len(self.word_to_idx)}")
        print(f"Max gloss length: {max_gloss_length}")

        # Verify vocabulary matches gloss data
        self.verify_vocabulary()

    def verify_vocabulary(self):
        """Verify that vocabulary covers the gloss data"""
        missing_words = set()
        total_words = 0
        covered_words = 0

        for gloss in self.annotations['gloss']:
            words = str(gloss).split()
            total_words += len(words)
            for word in words:
                if word in self.word_to_idx:
                    covered_words += 1
                else:
                    missing_words.add(word)

        coverage = covered_words / total_words if total_words > 0 else 0
        print(f"Vocabulary coverage: {coverage:.2%} ({covered_words}/{total_words} words)")

        if missing_words:
            print(f"Missing words in vocabulary: {list(missing_words)[:10]}{'...' if len(missing_words) > 10 else ''}")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        row = self.annotations.iloc[idx]

        # Load pose sequence
        pose_path = self.pose_dir / f"pose_{idx:05d}.npy"

        if not pose_path.exists():
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

        pose_sequence = np.load(pose_path)  # Shape: (seq_len, features)

        # Use GLOSS instead of natural sentence for word-level recognition
        gloss_sequence = row['gloss']  # This is the word-level gloss sequence

        # Convert gloss sequence to indices
        gloss_indices = self.gloss_to_indices(gloss_sequence)

        # Prepare source (pose) and target (gloss) sequences
        src = torch.FloatTensor(pose_sequence)  # (seq_len, features)

        # For teacher forcing: input is SOS + gloss, target is gloss + EOS
        tgt_input = torch.LongTensor(gloss_indices[:-1])  # Remove EOS for input
        tgt_output = torch.LongTensor(gloss_indices[1:])  # Remove SOS for output

        return {
            'src': src,  # Pose sequence: (seq_len, features)
            'tgt_input': tgt_input,  # Decoder input: (gloss_len-1,)
            'tgt_output': tgt_output,  # Decoder target: (gloss_len-1,)
            'natural_sentence': row['sentence'],  # For reference
            'gloss_sequence': gloss_sequence,  # Target word sequence
            'video_id': row['video_id'],
            'original_idx': idx
        }

    def gloss_to_indices(self, gloss_sequence):
        """Convert gloss sequence to token indices with special tokens"""
        # Split gloss into individual words
        words = str(gloss_sequence).split()

        # Add SOS and EOS tokens
        indices = [self.sos_token]
        for word in words[:self.max_gloss_length - 2]:  # -2 for SOS and EOS
            indices.append(self.word_to_idx.get(word, self.unk_token))
        indices.append(self.eos_token)

        # Pad to max_gloss_length
        while len(indices) < self.max_gloss_length:
            indices.append(self.pad_token)

        return indices

    def indices_to_gloss(self, indices):
        """Convert token indices back to gloss sequence"""
        words = []
        for idx in indices:
            if idx == self.eos_token:
                break
            if idx not in [self.sos_token, self.pad_token, self.eos_token]:
                word = self.idx_to_word.get(str(idx), f"<UNK:{idx}>")
                words.append(word)
        return ' '.join(words)


def test_gloss_dataset():
    """Test the gloss dataset class"""
    print("Testing BanglaSignGlossDataset...")

    dataset = BanglaSignGlossDataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        pose_dir="data/processed/pose_sequences_full",
        vocabulary_file="data/annotations/gloss_vocabulary.json",  # Use gloss vocabulary!
        max_gloss_length=8  # Gloss sequences are shorter
    )

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Source shape: {sample['src'].shape}")
        print(f"Target input shape: {sample['tgt_input'].shape}")
        print(f"Natural Sentence: {sample['natural_sentence']}")
        print(f"Gloss Sequence: {sample['gloss_sequence']}")
        print(f"Target indices: {sample['tgt_input'].tolist()}")
        print(f"Reconstructed gloss: {dataset.indices_to_gloss(sample['tgt_input'].tolist())}")

        # Test a few more samples to verify vocabulary coverage
        print(f"\nTesting vocabulary coverage on more samples:")
        for i in range(1, min(5, len(dataset))):
            sample = dataset[i]
            reconstructed = dataset.indices_to_gloss(sample['tgt_input'].tolist())
            print(f"Sample {i}: {sample['gloss_sequence']} -> {reconstructed}")

        return dataset
    else:
        print("No samples found in dataset!")
        return None


if __name__ == "__main__":
    dataset = test_gloss_dataset()