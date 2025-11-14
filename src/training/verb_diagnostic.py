import torch
import json
from src.data_processing.gloss_dataset import BanglaSignGlossDataset


def diagnose_verb_issue():
    """Diagnose why verb accuracy is 0%"""

    # Load vocabulary
    with open('data/annotations/gloss_vocabulary.json', 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    print("=== Vocabulary Analysis ===")
    print(f"Vocabulary size: {len(vocabulary['word_to_idx'])}")

    # Check the problematic verbs
    problem_verbs = ['না', 'হবে', 'নেই', 'হয়েছে', 'খাওয়া।', 'খাওয়ায়া']
    print("\n=== Problem Verb Check ===")
    for verb in problem_verbs:
        if verb in vocabulary['word_to_idx']:
            idx = vocabulary['word_to_idx'][verb]
            print(f"  '{verb}' -> index {idx}")
        else:
            print(f"  '{verb}' -> NOT FOUND in vocabulary!")

    # Check what's actually in the data
    dataset = BanglaSignGlossDataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        pose_dir="data/processed/pose_sequences_full",
        vocabulary_file="data/annotations/gloss_vocabulary.json"
    )

    print(f"\n=== Sample Analysis ===")
    verb_distribution = {}

    for i in range(min(50, len(dataset))):
        sample = dataset[i]
        gloss = sample['gloss_sequence']
        words = gloss.split()
        last_word = words[-1] if words else ""

        verb_distribution[last_word] = verb_distribution.get(last_word, 0) + 1

        if i < 10:
            print(f"  '{gloss}' -> last word: '{last_word}'")

    print(f"\n=== Final Word Distribution ===")
    for word, count in sorted(verb_distribution.items(), key=lambda x: x[1], reverse=True):
        print(f"  '{word}': {count} occurrences")


if __name__ == "__main__":
    diagnose_verb_issue()