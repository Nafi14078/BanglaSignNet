import pandas as pd
import json
from pathlib import Path


def rebuild_gloss_vocabulary():
    """Rebuild vocabulary using gloss sequences instead of natural sentences"""

    # Load annotations
    annotations_df = pd.read_csv("data/annotations/dataset_annotations.csv")

    # Extract words from GLOSS sequences (not natural sentences)
    all_words = []
    for gloss in annotations_df['gloss']:
        words = str(gloss).split()
        all_words.extend(words)

    # Create vocabulary from gloss words only
    vocabulary = sorted(set(all_words))

    print(f"Gloss vocabulary: {len(vocabulary)} unique words")
    print(f"Sample gloss words: {vocabulary[:20]}")

    # Add special tokens
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}
    word_to_idx['<SOS>'] = len(vocabulary)
    word_to_idx['<EOS>'] = len(vocabulary) + 1
    word_to_idx['<PAD>'] = len(vocabulary) + 2
    word_to_idx['<UNK>'] = len(vocabulary) + 3

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    # Save gloss vocabulary
    vocabulary_info = {
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': len(word_to_idx),
        'special_tokens': {
            '<SOS>': len(vocabulary),
            '<EOS>': len(vocabulary) + 1,
            '<PAD>': len(vocabulary) + 2,
            '<UNK>': len(vocabulary) + 3
        }
    }

    # Save to new file
    with open('data/annotations/gloss_vocabulary.json', 'w', encoding='utf-8') as f:
        json.dump(vocabulary_info, f, ensure_ascii=False, indent=2)

    print(f"\n=== Gloss Vocabulary Summary ===")
    print(f"Total words: {len(vocabulary)}")
    print(f"Vocabulary size (with special tokens): {len(word_to_idx)}")
    print(f"Most common gloss words: {vocabulary[:10]}")

    # Analyze gloss sequence lengths
    gloss_lengths = [len(str(gloss).split()) for gloss in annotations_df['gloss']]
    print(f"Average gloss length: {sum(gloss_lengths) / len(gloss_lengths):.2f} words")
    print(f"Max gloss length: {max(gloss_lengths)} words")
    print(f"Min gloss length: {min(gloss_lengths)} words")

    return vocabulary_info


if __name__ == "__main__":
    rebuild_gloss_vocabulary()