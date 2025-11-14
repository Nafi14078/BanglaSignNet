import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.modeling.transformer_model import BanglaSignTransformer
from src.data_processing.gloss_dataset import BanglaSignGlossDataset


class ModelTester:
    def __init__(self, model_path, annotations_file, pose_dir, vocabulary_file, test_size=300):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load vocabulary
        with open(vocabulary_file, 'r', encoding='utf-8') as f:
            self.vocabulary = json.load(f)

        # Load model
        self.model = self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()

        # Load test dataset
        self.dataset = BanglaSignGlossDataset(
            annotations_file=annotations_file,
            pose_dir=pose_dir,
            vocabulary_file=vocabulary_file,
            max_gloss_length=8
        )

        # Select test samples (first test_size samples)
        self.test_indices = list(range(min(test_size, len(self.dataset))))
        print(f"Testing on {len(self.test_indices)} samples")

    def load_model(self, model_path):
        """Load trained model from checkpoint"""
        checkpoint = torch.load(model_path, map_location=self.device)

        # Create model with same architecture
        model = BanglaSignTransformer(
            input_dim=375,
            vocab_size=len(self.vocabulary['word_to_idx']),
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.2,
            max_seq_length=8
        )

        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✅ Model loaded from: {model_path}")
        print(f"📊 Trained for {checkpoint['epoch']} epochs")
        print(f"🏆 Best gloss accuracy: {checkpoint.get('best_gloss_accuracy', 'N/A')}")

        return model

    def test_on_samples(self, num_samples=None):
        """Test model on specified number of samples"""
        if num_samples is None:
            num_samples = len(self.test_indices)

        test_indices = self.test_indices[:num_samples]

        print(f"\n🧪 Testing on {len(test_indices)} samples...")
        print("=" * 80)

        results = {
            'correct': 0,
            'incorrect': 0,
            'by_length': {},
            'detailed_results': []
        }

        for idx in tqdm(test_indices, desc="Testing"):
            sample = self.dataset[idx]
            src = sample['src'].unsqueeze(1).to(self.device)

            with torch.no_grad():
                sos_token = self.vocabulary['word_to_idx']['<SOS>']
                eos_token = self.vocabulary['word_to_idx']['<EOS>']

                prediction = self.model.predict(src.squeeze(1), sos_token, eos_token)
                predicted_gloss = self.dataset.indices_to_gloss(prediction.tolist())

            target_gloss = sample['gloss_sequence']
            is_correct = target_gloss == predicted_gloss
            gloss_length = len(target_gloss.split())

            # Update results
            if gloss_length not in results['by_length']:
                results['by_length'][gloss_length] = {'correct': 0, 'total': 0}

            results['by_length'][gloss_length]['total'] += 1
            if is_correct:
                results['correct'] += 1
                results['by_length'][gloss_length]['correct'] += 1
            else:
                results['incorrect'] += 1

            # Store detailed results
            results['detailed_results'].append({
                'index': idx,
                'natural_sentence': sample['natural_sentence'],
                'target_gloss': target_gloss,
                'predicted_gloss': predicted_gloss,
                'is_correct': is_correct,
                'length': gloss_length,
                'video_id': sample['video_id']
            })

        return results

    def print_detailed_results(self, results, num_examples=20):
        """Print detailed test results"""
        total_samples = results['correct'] + results['incorrect']
        accuracy = results['correct'] / total_samples

        print(f"\n{'=' * 80}")
        print(f"📊 COMPREHENSIVE TEST RESULTS ({total_samples} samples)")
        print(f"{'=' * 80}")
        print(f"🎯 Overall Accuracy: {accuracy:.2%} ({results['correct']}/{total_samples})")

        # Accuracy by sequence length
        print(f"\n📈 Accuracy by Sequence Length:")
        for length in sorted(results['by_length'].keys()):
            length_data = results['by_length'][length]
            if length_data['total'] > 0:
                length_accuracy = length_data['correct'] / length_data['total']
                print(
                    f"   {length}-word sequences: {length_accuracy:.2%} ({length_data['correct']}/{length_data['total']})")

        # Show some examples
        print(f"\n🔍 Sample Predictions (first {num_examples}):")
        print("-" * 80)

        correct_examples = [r for r in results['detailed_results'] if r['is_correct']]
        incorrect_examples = [r for r in results['detailed_results'] if not r['is_correct']]

        print("✅ CORRECT PREDICTIONS:")
        for i, example in enumerate(correct_examples[:min(10, len(correct_examples))]):
            print(f"   {i + 1:2d}. Natural: {example['natural_sentence']}")
            print(f"       Target:  {example['target_gloss']}")
            print(f"       Predicted: {example['predicted_gloss']}")
            print()

        print("❌ INCORRECT PREDICTIONS:")
        for i, example in enumerate(incorrect_examples[:min(10, len(incorrect_examples))]):
            print(f"   {i + 1:2d}. Natural: {example['natural_sentence']}")
            print(f"       Target:  {example['target_gloss']}")
            print(f"       Predicted: {example['predicted_gloss']}")
            print()

    def analyze_error_patterns(self, results):
        """Analyze common error patterns"""
        incorrect_results = [r for r in results['detailed_results'] if not r['is_correct']]

        if not incorrect_results:
            print("🎉 No errors to analyze!")
            return

        print(f"\n🔍 ERROR ANALYSIS ({len(incorrect_results)} errors):")
        print("=" * 60)

        # Analyze by error type
        error_types = {
            'wrong_verb': 0,
            'wrong_time': 0,
            'wrong_person': 0,
            'wrong_negation': 0,
            'completely_different': 0
        }

        for error in incorrect_results:
            target_words = set(error['target_gloss'].split())
            predicted_words = set(error['predicted_gloss'].split())

            # Analyze error patterns
            if 'খাওয়ায়া' in target_words or 'খাওয়ায়া' in predicted_words:
                error_types['wrong_verb'] += 1
            elif any(time_word in target_words.union(predicted_words)
                     for time_word in ['সকাল', 'দুপুর', 'বিকাল', 'সন্ধ্যা', 'রাত']):
                error_types['wrong_time'] += 1
            elif any(person_word in target_words.union(predicted_words)
                     for person_word in ['বাবা', 'মা', 'দাদা', 'দাদি', 'চাচা', 'খালা']):
                error_types['wrong_person'] += 1
            elif 'না' in target_words.union(predicted_words) or 'নেই' in target_words.union(predicted_words):
                error_types['wrong_negation'] += 1
            else:
                error_types['completely_different'] += 1

        print("Error Types:")
        for error_type, count in error_types.items():
            if count > 0:
                percentage = count / len(incorrect_results)
                print(f"  {error_type}: {percentage:.1%} ({count} errors)")

    def save_results(self, results, output_file):
        """Save test results to file"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame for easy analysis
        df_data = []
        for result in results['detailed_results']:
            df_data.append({
                'index': result['index'],
                'natural_sentence': result['natural_sentence'],
                'target_gloss': result['target_gloss'],
                'predicted_gloss': result['predicted_gloss'],
                'is_correct': result['is_correct'],
                'length': result['length'],
                'video_id': result['video_id']
            })

        df = pd.DataFrame(df_data)
        df.to_csv(output_file, index=False, encoding='utf-8')

        # Save summary
        summary = {
            'total_samples': len(results['detailed_results']),
            'accuracy': results['correct'] / len(results['detailed_results']),
            'correct_count': results['correct'],
            'incorrect_count': results['incorrect'],
            'accuracy_by_length': {
                str(length): data['correct'] / data['total']
                for length, data in results['by_length'].items()
            }
        }

        summary_file = output_path.with_suffix('.summary.json')
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

        print(f"💾 Results saved to: {output_file}")
        print(f"💾 Summary saved to: {summary_file}")


def main():
    """Test the trained model on 300+ samples"""

    # Configuration
    MODEL_PATH = "models/full_word_level_updated_transformer/full_checkpoint_epoch_80.pth"  # Update path if different
    ANNOTATIONS_FILE = "data/annotations/dataset_annotations.csv"
    POSE_DIR = "data/processed/pose_sequences_full"
    VOCABULARY_FILE = "data/annotations/gloss_vocabulary.json"
    TEST_SIZE = 300  # Test on 300 samples

    print("🧪 Starting Comprehensive Model Testing")
    print("=" * 60)

    # Initialize tester
    tester = ModelTester(
        model_path=MODEL_PATH,
        annotations_file=ANNOTATIONS_FILE,
        pose_dir=POSE_DIR,
        vocabulary_file=VOCABULARY_FILE,
        test_size=TEST_SIZE
    )

    # Run tests
    results = tester.test_on_samples()

    # Print results
    tester.print_detailed_results(results, num_examples=25)

    # Analyze errors
    tester.analyze_error_patterns(results)

    # Save results
    tester.save_results(results, "results/comprehensive_test_results.csv")

    print(f"\n✅ Testing completed on {TEST_SIZE} samples!")


if __name__ == "__main__":
    main()