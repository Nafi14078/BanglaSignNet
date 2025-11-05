import pandas as pd
from pathlib import Path
from pose_extraction_pipeline import PoseExtractionPipeline


def process_larger_dataset():
    """Process a larger subset of the dataset"""
    pipeline = PoseExtractionPipeline()

    # Process 500 samples instead of 20
    sample_size = 500

    print(f"Processing {sample_size} samples for better training...")

    pose_data, summary = pipeline.process_dataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        output_dir="data/processed/pose_sequences_full",
        sample_size=sample_size
    )

    print(f"Successfully processed {summary['successful']} samples")
    return summary


if __name__ == "__main__":
    process_larger_dataset()