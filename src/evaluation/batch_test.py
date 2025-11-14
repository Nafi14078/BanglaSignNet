import torch
import json
from pathlib import Path
from test_model import ModelTester


def batch_test_different_sizes():
    """Test model on different sample sizes"""

    MODEL_PATH = "models/full_word_level_updated_transformer/full_checkpoint_epoch_80.pth"
    ANNOTATIONS_FILE = "data/annotations/dataset_annotations.csv"
    POSE_DIR = "data/processed/pose_sequences_full"
    VOCABULARY_FILE = "data/annotations/gloss_vocabulary.json"

    # Different test sizes to try
    test_sizes = [50, 100, 200, 300, 500, 1000]

    print("🔬 Batch Testing on Different Sample Sizes")
    print("=" * 60)

    results_summary = {}

    for size in test_sizes:
        print(f"\n📊 Testing on {size} samples...")

        tester = ModelTester(
            model_path=MODEL_PATH,
            annotations_file=ANNOTATIONS_FILE,
            pose_dir=POSE_DIR,
            vocabulary_file=VOCABULARY_FILE,
            test_size=size
        )

        results = tester.test_on_samples()
        accuracy = results['correct'] / (results['correct'] + results['incorrect'])

        results_summary[size] = {
            'accuracy': accuracy,
            'correct': results['correct'],
            'total': results['correct'] + results['incorrect'],
            'by_length': results['by_length']
        }

        print(f"   Accuracy: {accuracy:.2%}")

    # Print summary
    print(f"\n{'=' * 60}")
    print("📈 BATCH TESTING SUMMARY")
    print(f"{'=' * 60}")

    for size, result in results_summary.items():
        print(f"Sample Size {size:4d}: {result['accuracy']:.2%} ({result['correct']}/{result['total']})")

    # Save batch results
    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / "batch_testing_summary.json", 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n💾 Batch results saved to: {output_dir / 'batch_testing_summary.json'}")


if __name__ == "__main__":
    batch_test_different_sizes()