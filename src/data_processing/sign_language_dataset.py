import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import json
from pathlib import Path


class BanglaSignLanguageDataset(Dataset):
    def __init__(self, annotations_file, pose_dir, vocabulary_file, max_sequence_length=60, max_sentence_length=20):
        self.annotations = pd.read_csv(annotations_file)

        # Load vocabulary
        with open(vocabulary_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
            self.word_to_idx = vocab_data['word_to_idx']
            self.idx_to_word = vocab_data['idx_to_word']

        self.pose_dir = Path(pose_dir)
        self.max_sequence_length = max_sequence_length
        self.max_sentence_length = max_sentence_length

        # Special tokens
        self.sos_token = self.word_to_idx['<SOS>']
        self.eos_token = self.word_to_idx['<EOS>']
        self.pad_token = self.word_to_idx['<PAD>']

        # Filter annotations to only include processed pose data
        self.valid_indices = self._filter_valid_samples()

        print(f"Dataset initialized with {len(self.valid_indices)} samples")
        print(f"Vocabulary size: {len(self.word_to_idx)}")

    def _filter_valid_samples(self):
        """Filter samples that have corresponding pose data"""
        valid_indices = []

        for idx, row in self.annotations.iterrows():
            pose_path = self.pose_dir / f"pose_{idx:05d}.npy"
            if pose_path.exists():
                valid_indices.append(idx)

        return valid_indices

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        original_idx = self.valid_indices[idx]
        row = self.annotations.iloc[original_idx]

        # Load pose sequence
        pose_path = self.pose_dir / f"pose_{original_idx:05d}.npy"
        pose_sequence = np.load(pose_path)  # Shape: (seq_len, features)

        # Convert sentence to indices
        sentence_indices = self.sentence_to_indices(row['sentence'])

        # Prepare source (pose) and target (sentence) sequences
        src = torch.FloatTensor(pose_sequence)  # (seq_len, features)

        # For teacher forcing: input is SOS + sentence, target is sentence + EOS
        tgt_input = torch.LongTensor(sentence_indices[:-1])  # Remove EOS for input
        tgt_output = torch.LongTensor(sentence_indices[1:])  # Remove SOS for output

        return {
            'src': src,  # Pose sequence: (seq_len, features)
            'tgt_input': tgt_input,  # Decoder input: (tgt_len-1,)
            'tgt_output': tgt_output,  # Decoder target: (tgt_len-1,)
            'sentence': row['sentence'],
            'gloss': row['gloss'],
            'video_id': row['video_id'],
            'original_idx': original_idx
        }

    def sentence_to_indices(self, sentence):
        """Convert Bangla sentence to token indices with special tokens"""
        words = sentence.split()

        # Add SOS and EOS tokens
        indices = [self.sos_token]
        for word in words[:self.max_sentence_length - 2]:  # -2 for SOS and EOS
            indices.append(self.word_to_idx.get(word, self.word_to_idx['<UNK>']))
        indices.append(self.eos_token)

        # Pad to max_sentence_length
        while len(indices) < self.max_sentence_length:
            indices.append(self.pad_token)

        return indices

    def indices_to_sentence(self, indices):
        """Convert token indices back to sentence"""
        words = []
        for idx in indices:
            if idx == self.eos_token:
                break
            if idx not in [self.sos_token, self.pad_token, self.eos_token]:
                words.append(self.idx_to_word[str(idx)])
        return ' '.join(words)


def test_dataset():
    """Test the dataset class"""
    print("Testing BanglaSignLanguageDataset...")

    dataset = BanglaSignLanguageDataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        pose_dir="data/processed/pose_sequences",
        vocabulary_file="data/annotations/vocabulary.json"
    )

    if len(dataset) > 0:
        sample = dataset[0]
        print(f"Source shape: {sample['src'].shape}")
        print(f"Target input shape: {sample['tgt_input'].shape}")
        print(f"Target output shape: {sample['tgt_output'].shape}")
        print(f"Sentence: {sample['sentence']}")
        print(f"Target indices: {sample['tgt_input'].tolist()}")
        print(f"Reconstructed: {dataset.indices_to_sentence(sample['tgt_input'].tolist())}")

        return dataset
    else:
        print("No samples found in dataset!")
        return None


if __name__ == "__main__":
    dataset = test_dataset()