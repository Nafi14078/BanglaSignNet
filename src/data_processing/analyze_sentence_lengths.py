import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


def analyze_sentence_lengths():
    """Analyze the distribution of sentence lengths in the dataset"""

    # Load annotations
    annotations_df = pd.read_csv("data/annotations/dataset_annotations.csv")

    print("=== SENTENCE LENGTH ANALYSIS ===")
    print(f"Total samples: {len(annotations_df)}")

    # Analyze natural sentences
    natural_lengths = [len(str(sentence).split()) for sentence in annotations_df['sentence']]

    # Analyze gloss sequences
    gloss_lengths = [len(str(gloss).split()) for gloss in annotations_df['gloss']]

    # Create detailed analysis
    analyze_length_distribution("Natural Sentences", natural_lengths)
    analyze_length_distribution("Gloss Sequences", gloss_lengths)

    # Compare both
    compare_distributions(natural_lengths, gloss_lengths)

    # Save detailed breakdown to file
    save_length_breakdown(natural_lengths, gloss_lengths)


def analyze_length_distribution(name, lengths):
    """Analyze and print distribution for one type of sentences"""
    print(f"\n--- {name} ---")

    length_counter = Counter(lengths)
    total = len(lengths)

    print(f"Total: {total}")
    print(f"Average length: {sum(lengths) / total:.2f} words")
    print(f"Min length: {min(lengths)} words")
    print(f"Max length: {max(lengths)} words")
    print(f"Most common length: {length_counter.most_common(1)[0][0]} words")

    print(f"\nLength distribution:")
    for length in sorted(length_counter.keys()):
        count = length_counter[length]
        percentage = (count / total) * 100
        print(f"  {length:2d} words: {count:4d} samples ({percentage:6.2f}%)")


def compare_distributions(natural_lengths, gloss_lengths):
    """Compare natural sentences vs gloss sequences"""
    print(f"\n--- COMPARISON ---")

    natural_counter = Counter(natural_lengths)
    gloss_counter = Counter(gloss_lengths)

    print(f"{'Length':6} {'Natural':8} {'Gloss':8} {'Diff':8}")
    print("-" * 40)

    all_lengths = set(natural_counter.keys()) | set(gloss_counter.keys())

    for length in sorted(all_lengths):
        natural_count = natural_counter.get(length, 0)
        gloss_count = gloss_counter.get(length, 0)
        diff = natural_count - gloss_count

        print(f"{length:6d} {natural_count:8d} {gloss_count:8d} {diff:8d}")


def save_length_breakdown(natural_lengths, gloss_lengths):
    """Save detailed breakdown to JSON file"""
    natural_counter = Counter(natural_lengths)
    gloss_counter = Counter(gloss_lengths)

    breakdown = {
        'natural_sentences': {
            'total': len(natural_lengths),
            'average_length': sum(natural_lengths) / len(natural_lengths),
            'min_length': min(natural_lengths),
            'max_length': max(natural_lengths),
            'length_distribution': dict(natural_counter)
        },
        'gloss_sequences': {
            'total': len(gloss_lengths),
            'average_length': sum(gloss_lengths) / len(gloss_lengths),
            'min_length': min(gloss_lengths),
            'max_length': max(gloss_lengths),
            'length_distribution': dict(gloss_counter)
        }
    }

    # Save to file
    with open('data/annotations/sentence_length_analysis.json', 'w', encoding='utf-8') as f:
        json.dump(breakdown, f, indent=2, ensure_ascii=False)

    print(f"\n💾 Detailed analysis saved to: data/annotations/sentence_length_analysis.json")


def plot_length_distributions():
    """Create visualization of sentence length distributions"""
    try:
        # Load data
        annotations_df = pd.read_csv("data/annotations/dataset_annotations.csv")

        natural_lengths = [len(str(sentence).split()) for sentence in annotations_df['sentence']]
        gloss_lengths = [len(str(gloss).split()) for gloss in annotations_df['gloss']]

        # Create subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Plot natural sentences
        sns.histplot(natural_lengths, bins=range(1, max(natural_lengths) + 2), ax=ax1, kde=True)
        ax1.set_title('Natural Sentences Length Distribution')
        ax1.set_xlabel('Number of Words')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # Add counts on bars
        natural_counts = Counter(natural_lengths)
        for length, count in natural_counts.items():
            ax1.text(length, count + 5, str(count), ha='center', va='bottom', fontsize=9)

        # Plot gloss sequences
        sns.histplot(gloss_lengths, bins=range(1, max(gloss_lengths) + 2), ax=ax2, kde=True, color='orange')
        ax2.set_title('Gloss Sequences Length Distribution')
        ax2.set_xlabel('Number of Words')
        ax2.set_ylabel('Frequency')
        ax2.grid(True, alpha=0.3)

        # Add counts on bars
        gloss_counts = Counter(gloss_lengths)
        for length, count in gloss_counts.items():
            ax2.text(length, count + 5, str(count), ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig('data/annotations/sentence_length_distribution.png', dpi=300, bbox_inches='tight')
        plt.show()

        print(f"📊 Visualization saved to: data/annotations/sentence_length_distribution.png")

    except ImportError:
        print("Matplotlib/Seaborn not available. Install with: pip install matplotlib seaborn")


if __name__ == "__main__":
    analyze_sentence_lengths()
    plot_length_distributions()