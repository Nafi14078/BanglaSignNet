import pandas as pd
import json
from collections import Counter


def analyze_gloss_vocabulary():
    """Analyze the gloss vocabulary and data"""

    # Load annotations
    annotations_df = pd.read_csv("data/annotations/dataset_annotations.csv")

    # Load gloss vocabulary
    with open('data/annotations/gloss_vocabulary.json', 'r', encoding='utf-8') as f:
        gloss_vocab = json.load(f)

    # Load original vocabulary for comparison
    with open('data/annotations/vocabulary.json', 'r', encoding='utf-8') as f:
        original_vocab = json.load(f)

    print("=== Vocabulary Analysis ===")
    print(f"Original vocabulary size: {len(original_vocab['word_to_idx'])}")
    print(f"Gloss vocabulary size: {len(gloss_vocab['word_to_idx'])}")

    # Analyze gloss word frequencies
    all_gloss_words = []
    for gloss in annotations_df['gloss']:
        words = str(gloss).split()
        all_gloss_words.extend(words)

    word_freq = Counter(all_gloss_words)

    print(f"\nMost common gloss words:")
    for word, freq in word_freq.most_common(15):
        print(f"  {word}: {freq} occurrences")

    # Check for missing words in gloss vocabulary
    missing_in_gloss = [word for word in word_freq.keys() if word not in gloss_vocab['word_to_idx']]

    print(f"\nWords in data but missing from gloss vocabulary: {len(missing_in_gloss)}")
    if missing_in_gloss:
        print(f"Missing words: {missing_in_gloss}")

    # Check gloss sequence patterns
    gloss_lengths = [len(str(gloss).split()) for gloss in annotations_df['gloss']]

    print(f"\nGloss Sequence Length Analysis:")
    print(f"  Average length: {sum(gloss_lengths) / len(gloss_lengths):.2f} words")
    print(f"  Max length: {max(gloss_lengths)} words")
    print(f"  Min length: {min(gloss_lengths)} words")
    print(f"  Most common length: {Counter(gloss_lengths).most_common(1)[0][0]} words")

    # Show sample gloss sequences
    print(f"\nSample Gloss Sequences:")
    for i in range(min(10, len(annotations_df))):
        row = annotations_df.iloc[i]
        print(f"  {i + 1:2d}. Natural: {row['sentence']}")
        print(f"      Gloss: {row['gloss']}")
        print(f"      Length: {len(str(row['gloss']).split())} words")


if __name__ == "__main__":
    analyze_gloss_vocabulary()