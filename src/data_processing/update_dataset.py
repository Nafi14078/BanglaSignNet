import pandas as pd
import json
from pathlib import Path


def update_dataset_with_all_samples():
    """Update the dataset to use all 500 processed samples"""

    # Load original annotations
    annotations_df = pd.read_csv("data/annotations/dataset_annotations.csv")

    # Check how many pose files we have
    pose_dir = Path("data/processed/pose_sequences_full")
    pose_files = list(pose_dir.glob("pose_*.npy"))

    print(f"Total annotations: {len(annotations_df)}")
    print(f"Pose files available: {len(pose_files)}")

    # Create a new annotations file with only the samples that have pose data
    valid_indices = []
    for pose_file in pose_files:
        try:
            # Extract index from filename "pose_00000.npy"
            idx = int(pose_file.stem.split('_')[1])
            if idx < len(annotations_df):
                valid_indices.append(idx)
        except:
            continue

    # Create filtered dataset
    filtered_annotations = annotations_df.iloc[valid_indices].copy()

    print(f"Samples with pose data: {len(filtered_annotations)}")

    # Save filtered annotations
    filtered_annotations.to_csv("data/annotations/filtered_annotations.csv", index=False)

    # Update dataset summary
    with open('data/annotations/vocabulary.json', 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)

    summary = {
        'total_samples': len(filtered_annotations),
        'total_videos': len(filtered_annotations['video_path'].unique()),
        'vocabulary_size': vocabulary['vocab_size'],
        'average_sentence_length': filtered_annotations['sentence_length'].mean(),
        'max_sentence_length': filtered_annotations['sentence_length'].max(),
        'dataset_structure': {
            'columns': list(filtered_annotations.columns),
            'pose_files_directory': "pose_sequences_full"
        }
    }

    with open('data/annotations/filtered_dataset_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n=== Updated Dataset Summary ===")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Vocabulary size: {summary['vocabulary_size']}")
    print(f"Average sentence length: {summary['average_sentence_length']:.2f} words")
    print(f"Max sentence length: {summary['max_sentence_length']} words")

    return filtered_annotations


if __name__ == "__main__":
    update_dataset_with_all_samples()