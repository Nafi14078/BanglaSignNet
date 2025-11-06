import pandas as pd
from pathlib import Path
from pose_extraction_pipeline import PoseExtractionPipeline
import time


def process_remaining_samples():
    """Process the remaining 1,422 samples to complete the dataset"""
    pipeline = PoseExtractionPipeline()

    # Load all annotations
    annotations_df = pd.read_csv("data/annotations/dataset_annotations.csv")
    total_samples = len(annotations_df)

    # Check how many we already have
    existing_pose_dir = Path("data/processed/pose_sequences_full")
    existing_files = list(existing_pose_dir.glob("pose_*.npy"))

    print(f"📊 Dataset Statistics:")
    print(f"   Total samples: {total_samples}")
    print(f"   Already processed: {len(existing_files)}")
    print(f"   Remaining to process: {total_samples - len(existing_files)}")

    # Process only the remaining samples
    start_time = time.time()

    pose_data, summary = pipeline.process_dataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        output_dir="data/processed/pose_sequences_full",
        sample_size=None  # Process all, but it will skip existing ones
    )

    processing_time = time.time() - start_time
    minutes = processing_time // 60
    seconds = processing_time % 60

    print(f"\n✅ DATASET PROCESSING COMPLETED!")
    print(f"📊 Final count: {summary['successful']} samples")
    print(f"⏱️  Time taken: {minutes:.0f}m {seconds:.0f}s")
    print(f"💾 Output directory: data/processed/pose_sequences_full")

    # Verify final count
    final_files = list(existing_pose_dir.glob("pose_*.npy"))
    print(f"🔍 Verification: {len(final_files)} pose files found")

    return summary


if __name__ == "__main__":
    process_remaining_samples()