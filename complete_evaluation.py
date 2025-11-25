"""
Comprehensive Evaluation Script
Shows: All accuracies, ROC curves, Confusion matrix, Precision/Recall/F1
For: Context-Only Model (or any model)

Usage:
    python comprehensive_evaluation.py
"""

import torch
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict, Counter
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, roc_curve, auc,
    accuracy_score, balanced_accuracy_score
)
from sklearn.preprocessing import label_binarize
import warnings

warnings.filterwarnings('ignore')

from src.modeling.transformer_model import BanglaSignTransformer

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Create results directory
Path('results/comprehensive').mkdir(parents=True, exist_ok=True)


class ComprehensiveEvaluator:
    """Complete evaluation with all metrics and visualizations"""

    def __init__(self, model_path, dataset, vocabulary, device='cpu'):
        self.model_path = model_path
        self.dataset = dataset
        self.vocabulary = vocabulary
        self.device = torch.device(device)

        # Load model
        print(f"\n📂 Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Determine model type
        self.is_context_only = checkpoint.get('training_type') == 'context_only'

        # Create model
        self.model = BanglaSignTransformer(
            input_dim=375,
            vocab_size=len(vocabulary['word_to_idx']),
            d_model=128,
            nhead=4,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dropout=0.2,
            max_seq_length=7 if self.is_context_only else 8
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Tokens
        self.sos_token = vocabulary['word_to_idx']['<SOS>']
        self.eos_token = vocabulary['word_to_idx']['<EOS>']
        self.pad_token = vocabulary['word_to_idx']['<PAD>']

        # Get validation indices
        self.val_indices = checkpoint.get('val_indices', list(range(300)))

        print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
        print(f"   Model type: {'Context-Only' if self.is_context_only else 'Full'}")
        print(f"   Validation samples: {len(self.val_indices)}")

        # Storage for results
        self.results = {
            'predictions': [],
            'targets': [],
            'correct': [],
            'all_pred_tokens': [],
            'all_target_tokens': [],
            'position_stats': defaultdict(lambda: {'correct': 0, 'total': 0}),
            'length_stats': defaultdict(lambda: {'correct': 0, 'total': 0}),
        }

    def evaluate(self, num_samples=300):
        """Evaluate model and collect all data"""

        eval_indices = self.val_indices[:num_samples]

        print(f"\n{'=' * 70}")
        print(f"EVALUATING ON {len(eval_indices)} SAMPLES")
        print(f"{'=' * 70}")

        for idx in tqdm(eval_indices, desc="Evaluating"):
            sample = self.dataset[idx]
            src = sample['src'].to(self.device)

            # Get target based on model type
            if self.is_context_only:
                target = sample['context_sequence']
            else:
                target = sample['gloss_sequence'] if 'gloss_sequence' in sample else sample['context_sequence']

            # Predict
            with torch.no_grad():
                prediction = self.model.predict(src, self.sos_token, self.eos_token)
                predicted = self.dataset.indices_to_gloss(prediction.tolist())

            is_correct = (predicted == target)

            # Store sequence-level results
            self.results['predictions'].append(predicted)
            self.results['targets'].append(target)
            self.results['correct'].append(is_correct)

            # Token-level results
            pred_words = predicted.split()
            target_words = target.split()

            for pred_word, target_word in zip(pred_words, target_words):
                if pred_word in self.vocabulary['word_to_idx'] and target_word in self.vocabulary['word_to_idx']:
                    self.results['all_pred_tokens'].append(self.vocabulary['word_to_idx'][pred_word])
                    self.results['all_target_tokens'].append(self.vocabulary['word_to_idx'][target_word])

            # Length stats
            seq_len = len(target_words)
            self.results['length_stats'][seq_len]['total'] += 1
            if is_correct:
                self.results['length_stats'][seq_len]['correct'] += 1

            # Position stats
            for pos in range(min(len(pred_words), len(target_words))):
                self.results['position_stats'][pos]['total'] += 1
                if pred_words[pos] == target_words[pos]:
                    self.results['position_stats'][pos]['correct'] += 1

        return self.results

    def calculate_all_metrics(self):
        """Calculate all possible metrics"""

        print(f"\n{'=' * 70}")
        print("CALCULATING ALL METRICS")
        print(f"{'=' * 70}")

        metrics = {}

        # 1. Sequence-level accuracy
        total = len(self.results['correct'])
        correct = sum(self.results['correct'])
        metrics['sequence_accuracy'] = correct / total if total > 0 else 0

        print(f"\n📊 Sequence-Level Metrics:")
        print(f"   Exact Match Accuracy: {metrics['sequence_accuracy'] * 100:.2f}% ({correct}/{total})")

        # 2. Token-level metrics
        if len(self.results['all_pred_tokens']) > 0:
            metrics['token_accuracy'] = accuracy_score(
                self.results['all_target_tokens'],
                self.results['all_pred_tokens']
            )

            metrics['balanced_accuracy'] = balanced_accuracy_score(
                self.results['all_target_tokens'],
                self.results['all_pred_tokens']
            )

            # Precision, Recall, F1
            precision, recall, f1, support = precision_recall_fscore_support(
                self.results['all_target_tokens'],
                self.results['all_pred_tokens'],
                average='weighted',
                zero_division=0
            )

            metrics['precision_weighted'] = precision
            metrics['recall_weighted'] = recall
            metrics['f1_weighted'] = f1

            # Macro averages
            precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
                self.results['all_target_tokens'],
                self.results['all_pred_tokens'],
                average='macro',
                zero_division=0
            )

            metrics['precision_macro'] = precision_macro
            metrics['recall_macro'] = recall_macro
            metrics['f1_macro'] = f1_macro

            print(f"\n📊 Token-Level Metrics:")
            print(f"   Token Accuracy: {metrics['token_accuracy'] * 100:.2f}%")
            print(f"   Balanced Accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")
            print(f"\n   Weighted Average:")
            print(f"      Precision: {precision * 100:.2f}%")
            print(f"      Recall:    {recall * 100:.2f}%")
            print(f"      F1-Score:  {f1 * 100:.2f}%")
            print(f"\n   Macro Average:")
            print(f"      Precision: {precision_macro * 100:.2f}%")
            print(f"      Recall:    {recall_macro * 100:.2f}%")
            print(f"      F1-Score:  {f1_macro * 100:.2f}%")

        # 3. Position-wise accuracy
        metrics['position_accuracies'] = {}
        print(f"\n📍 Position-wise Accuracy:")
        for pos in sorted(self.results['position_stats'].keys()):
            stats = self.results['position_stats'][pos]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            metrics['position_accuracies'][pos] = acc
            print(f"   Position {pos}: {acc * 100:.2f}% ({stats['correct']}/{stats['total']})")

        # 4. Length-wise accuracy
        metrics['length_accuracies'] = {}
        print(f"\n📏 Accuracy by Sequence Length:")
        for length in sorted(self.results['length_stats'].keys()):
            stats = self.results['length_stats'][length]
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            metrics['length_accuracies'][length] = acc
            print(f"   {length}-word: {acc * 100:.2f}% ({stats['correct']}/{stats['total']})")

        return metrics

    def plot_confusion_matrix(self, top_n=20):
        """Plot confusion matrix for top N tokens"""

        print(f"\n📊 Generating confusion matrix (top {top_n} tokens)...")

        # Get top N most common tokens
        token_counts = Counter(self.results['all_target_tokens'])
        top_tokens = [t for t, _ in token_counts.most_common(top_n)]

        # Filter to only top tokens
        filtered_preds = []
        filtered_targets = []
        for p, t in zip(self.results['all_pred_tokens'], self.results['all_target_tokens']):
            if t in top_tokens and p in top_tokens:
                filtered_preds.append(p)
                filtered_targets.append(t)

        if len(filtered_targets) == 0:
            print("   ⚠️  Not enough data for confusion matrix")
            return

        # Create confusion matrix
        cm = confusion_matrix(filtered_targets, filtered_preds, labels=top_tokens)

        # Get labels
        idx_to_word = {int(k): v for k, v in self.vocabulary['idx_to_word'].items()}
        labels = [idx_to_word.get(t, f"ID_{t}") for t in top_tokens]

        # Normalize by row (true labels)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_normalized = np.nan_to_num(cm_normalized)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))

        # Raw counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[0], cbar_kws={'label': 'Count'})
        axes[0].set_title(f'Confusion Matrix - Raw Counts (Top {top_n} Tokens)', fontsize=14, pad=20)
        axes[0].set_ylabel('True Label', fontsize=12)
        axes[0].set_xlabel('Predicted Label', fontsize=12)

        # Normalized
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='RdYlGn',
                    xticklabels=labels, yticklabels=labels,
                    ax=axes[1], cbar_kws={'label': 'Proportion'}, vmin=0, vmax=1)
        axes[1].set_title(f'Confusion Matrix - Normalized (Top {top_n} Tokens)', fontsize=14, pad=20)
        axes[1].set_ylabel('True Label', fontsize=12)
        axes[1].set_xlabel('Predicted Label', fontsize=12)

        plt.tight_layout()
        plt.savefig('results/comprehensive/confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("   💾 Saved: results/comprehensive/confusion_matrix.png")
        plt.close()

    def plot_roc_curves(self, top_n=10):
        """Plot ROC curves for top N tokens"""

        print(f"\n📊 Generating ROC curves (top {top_n} tokens)...")

        # Get all unique tokens
        all_tokens = sorted(set(self.results['all_target_tokens'] + self.results['all_pred_tokens']))
        n_classes = len(all_tokens)

        if n_classes < 2:
            print("   ⚠️  Not enough classes for ROC curves")
            return

        # Binarize labels
        y_true = label_binarize(self.results['all_target_tokens'], classes=all_tokens)
        y_pred = label_binarize(self.results['all_pred_tokens'], classes=all_tokens)

        # Calculate micro-average ROC
        fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_pred.ravel())
        roc_auc_micro = auc(fpr_micro, tpr_micro)

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Micro-average
        axes[0].plot(fpr_micro, tpr_micro, linewidth=3,
                     label=f'Micro-average (AUC = {roc_auc_micro:.3f})', color='navy')
        axes[0].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')
        axes[0].set_xlabel('False Positive Rate', fontsize=12)
        axes[0].set_ylabel('True Positive Rate', fontsize=12)
        axes[0].set_title('ROC Curve - Micro Average', fontsize=14)
        axes[0].legend(loc='lower right', fontsize=10)
        axes[0].grid(True, alpha=0.3)

        # Top N classes
        token_counts = Counter(self.results['all_target_tokens'])
        top_tokens = [t for t, _ in token_counts.most_common(top_n)]

        idx_to_word = {int(k): v for k, v in self.vocabulary['idx_to_word'].items()}

        colors = plt.cm.tab10(np.linspace(0, 1, top_n))

        for i, token_idx in enumerate(top_tokens):
            if token_idx in all_tokens:
                token_pos = all_tokens.index(token_idx)
                if token_pos < y_true.shape[1]:
                    fpr, tpr, _ = roc_curve(y_true[:, token_pos], y_pred[:, token_pos])
                    roc_auc = auc(fpr, tpr)
                    label_name = idx_to_word.get(token_idx, f"ID_{token_idx}")
                    axes[1].plot(fpr, tpr, linewidth=2, color=colors[i],
                                 label=f'{label_name} (AUC={roc_auc:.2f})')

        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random')
        axes[1].set_xlabel('False Positive Rate', fontsize=12)
        axes[1].set_ylabel('True Positive Rate', fontsize=12)
        axes[1].set_title(f'ROC Curves - Top {top_n} Most Common Tokens', fontsize=14)
        axes[1].legend(loc='lower right', fontsize=9)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('results/comprehensive/roc_curves.png', dpi=300, bbox_inches='tight')
        print("   💾 Saved: results/comprehensive/roc_curves.png")
        plt.close()

        return roc_auc_micro

    def plot_accuracy_breakdown(self, metrics):
        """Plot comprehensive accuracy breakdown"""

        print(f"\n📊 Generating accuracy breakdown plots...")

        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # 1. Overall metrics bar chart
        ax1 = fig.add_subplot(gs[0, 0])
        overall_metrics = {
            'Sequence\nAccuracy': metrics['sequence_accuracy'] * 100,
            'Token\nAccuracy': metrics.get('token_accuracy', 0) * 100,
            'Balanced\nAccuracy': metrics.get('balanced_accuracy', 0) * 100,
        }
        bars = ax1.bar(overall_metrics.keys(), overall_metrics.values(),
                       color=['#2ecc71', '#3498db', '#9b59b6'], alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Accuracy (%)', fontsize=11)
        ax1.set_title('Overall Accuracy Metrics', fontsize=12, fontweight='bold')
        ax1.set_ylim([0, 100])
        ax1.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width() / 2., height + 2,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 2. Precision, Recall, F1
        ax2 = fig.add_subplot(gs[0, 1])
        prf_metrics = {
            'Precision': metrics.get('precision_weighted', 0) * 100,
            'Recall': metrics.get('recall_weighted', 0) * 100,
            'F1-Score': metrics.get('f1_weighted', 0) * 100,
        }
        bars = ax2.bar(prf_metrics.keys(), prf_metrics.values(),
                       color=['#e74c3c', '#f39c12', '#1abc9c'], alpha=0.8, edgecolor='black')
        ax2.set_ylabel('Score (%)', fontsize=11)
        ax2.set_title('Weighted Precision, Recall, F1', fontsize=12, fontweight='bold')
        ax2.set_ylim([0, 100])
        ax2.grid(True, alpha=0.3, axis='y')
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + 2,
                     f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

        # 3. Macro vs Weighted
        ax3 = fig.add_subplot(gs[0, 2])
        x = np.arange(3)
        width = 0.35
        weighted = [metrics.get('precision_weighted', 0) * 100,
                    metrics.get('recall_weighted', 0) * 100,
                    metrics.get('f1_weighted', 0) * 100]
        macro = [metrics.get('precision_macro', 0) * 100,
                 metrics.get('recall_macro', 0) * 100,
                 metrics.get('f1_macro', 0) * 100]
        ax3.bar(x - width / 2, weighted, width, label='Weighted', alpha=0.8, color='steelblue', edgecolor='black')
        ax3.bar(x + width / 2, macro, width, label='Macro', alpha=0.8, color='coral', edgecolor='black')
        ax3.set_ylabel('Score (%)', fontsize=11)
        ax3.set_title('Weighted vs Macro Averaging', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(['Precision', 'Recall', 'F1'])
        ax3.legend()
        ax3.set_ylim([0, 100])
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Position-wise accuracy
        ax4 = fig.add_subplot(gs[1, :2])
        positions = sorted(metrics['position_accuracies'].keys())
        pos_accs = [metrics['position_accuracies'][p] * 100 for p in positions]
        bars = ax4.bar(positions, pos_accs, alpha=0.8, color='purple', edgecolor='black')
        ax4.set_xlabel('Position in Sequence', fontsize=11)
        ax4.set_ylabel('Accuracy (%)', fontsize=11)
        ax4.set_title('Position-wise Accuracy', fontsize=12, fontweight='bold')
        ax4.set_ylim([0, 100])
        ax4.grid(True, alpha=0.3, axis='y')
        for i, (pos, acc) in enumerate(zip(positions, pos_accs)):
            ax4.text(pos, acc + 2, f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 5. Length-wise accuracy
        ax5 = fig.add_subplot(gs[1, 2])
        lengths = sorted(metrics['length_accuracies'].keys())
        length_accs = [metrics['length_accuracies'][l] * 100 for l in lengths]
        bars = ax5.bar(lengths, length_accs, alpha=0.8, color='teal', edgecolor='black')
        ax5.set_xlabel('Sequence Length (words)', fontsize=11)
        ax5.set_ylabel('Accuracy (%)', fontsize=11)
        ax5.set_title('Accuracy by Sequence Length', fontsize=12, fontweight='bold')
        ax5.set_ylim([0, 100])
        ax5.grid(True, alpha=0.3, axis='y')
        for i, (length, acc) in enumerate(zip(lengths, length_accs)):
            ax5.text(length, acc + 2, f'{acc:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 6. Length distribution
        ax6 = fig.add_subplot(gs[2, :])
        length_counts = [self.results['length_stats'][l]['total'] for l in lengths]
        x = np.arange(len(lengths))
        width = 0.35

        ax6_twin = ax6.twinx()
        bars = ax6.bar(x - width / 2, length_counts, width, label='Count',
                       alpha=0.7, color='lightblue', edgecolor='black')
        line = ax6_twin.plot(x, length_accs, 'ro-', linewidth=2, markersize=8, label='Accuracy')

        ax6.set_xlabel('Sequence Length (words)', fontsize=11)
        ax6.set_ylabel('Count', fontsize=11, color='blue')
        ax6_twin.set_ylabel('Accuracy (%)', fontsize=11, color='red')
        ax6.set_title('Sequence Length Distribution vs Accuracy', fontsize=12, fontweight='bold')
        ax6.set_xticks(x)
        ax6.set_xticklabels([f'{l}-word' for l in lengths])
        ax6.grid(True, alpha=0.3, axis='y')

        # Legends
        lines1, labels1 = ax6.get_legend_handles_labels()
        lines2, labels2 = ax6_twin.get_legend_handles_labels()
        ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        plt.suptitle('Comprehensive Accuracy Breakdown', fontsize=16, fontweight='bold', y=0.995)
        plt.savefig('results/comprehensive/accuracy_breakdown.png', dpi=300, bbox_inches='tight')
        print("   💾 Saved: results/comprehensive/accuracy_breakdown.png")
        plt.close()

    def generate_report(self, metrics, roc_auc_micro):
        """Generate comprehensive text report"""

        print(f"\n📊 Generating comprehensive report...")

        report = []
        report.append("=" * 70)
        report.append("COMPREHENSIVE EVALUATION REPORT")
        report.append("=" * 70)
        report.append(f"\nModel: {self.model_path}")
        report.append(f"Type: {'Context-Only' if self.is_context_only else 'Full Model'}")
        report.append(f"Evaluation samples: {len(self.results['predictions'])}")

        report.append(f"\n{'=' * 70}")
        report.append("ACCURACY METRICS")
        report.append(f"{'=' * 70}")
        report.append(f"\nSequence-Level:")
        report.append(f"  Exact Match Accuracy: {metrics['sequence_accuracy'] * 100:.2f}%")

        if 'token_accuracy' in metrics:
            report.append(f"\nToken-Level:")
            report.append(f"  Token Accuracy: {metrics['token_accuracy'] * 100:.2f}%")
            report.append(f"  Balanced Accuracy: {metrics['balanced_accuracy'] * 100:.2f}%")

        report.append(f"\n{'=' * 70}")
        report.append("PRECISION, RECALL, F1-SCORE")
        report.append(f"{'=' * 70}")
        report.append(f"\nWeighted Average:")
        report.append(f"  Precision: {metrics.get('precision_weighted', 0) * 100:.2f}%")
        report.append(f"  Recall:    {metrics.get('recall_weighted', 0) * 100:.2f}%")
        report.append(f"  F1-Score:  {metrics.get('f1_weighted', 0) * 100:.2f}%")

        report.append(f"\nMacro Average:")
        report.append(f"  Precision: {metrics.get('precision_macro', 0) * 100:.2f}%")
        report.append(f"  Recall:    {metrics.get('recall_macro', 0) * 100:.2f}%")
        report.append(f"  F1-Score:  {metrics.get('f1_macro', 0) * 100:.2f}%")

        report.append(f"\n{'=' * 70}")
        report.append("ROC ANALYSIS")
        report.append(f"{'=' * 70}")
        report.append(f"\nMicro-average AUC: {roc_auc_micro:.4f}")

        report.append(f"\n{'=' * 70}")
        report.append("POSITION-WISE ACCURACY")
        report.append(f"{'=' * 70}")
        for pos in sorted(metrics['position_accuracies'].keys()):
            acc = metrics['position_accuracies'][pos]
            report.append(f"  Position {pos}: {acc * 100:.2f}%")

        report.append(f"\n{'=' * 70}")
        report.append("LENGTH-WISE ACCURACY")
        report.append(f"{'=' * 70}")
        for length in sorted(metrics['length_accuracies'].keys()):
            acc = metrics['length_accuracies'][length]
            stats = self.results['length_stats'][length]
            report.append(f"  {length}-word: {acc * 100:.2f}% ({stats['correct']}/{stats['total']})")

        report.append(f"\n{'=' * 70}")
        report.append("SUMMARY FOR PAPER")
        report.append(f"{'=' * 70}")
        report.append(f"\nOverall Accuracy: {metrics['sequence_accuracy'] * 100:.2f}%")
        report.append(f"Token Accuracy: {metrics.get('token_accuracy', 0) * 100:.2f}%")
        report.append(f"F1-Score (Weighted): {metrics.get('f1_weighted', 0) * 100:.2f}%")
        report.append(f"ROC AUC (Micro): {roc_auc_micro:.4f}")

        # Save report
        report_text = '\n'.join(report)
        with open('results/comprehensive/evaluation_report.txt', 'w', encoding='utf-8') as f:
            f.write(report_text)

        print(report_text)
        print("\n💾 Saved: results/comprehensive/evaluation_report.txt")


def main():
    """Main evaluation function"""

    print("=" * 70)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 70)

    # Configuration
    MODEL_PATH = 'models/updated_context_only_model/checkpoint_epoch_80.pth'
    VOCAB_PATH = 'data/annotations/gloss_vocabulary.json'
    ANNOTATIONS_PATH = 'data/annotations/dataset_annotations.csv'
    POSE_DIR = 'data/processed/pose_sequences_full'

    # Load vocabulary
    print("\n📂 Loading vocabulary...")
    with open(VOCAB_PATH, 'r', encoding='utf-8') as f:
        vocabulary = json.load(f)
    print(f"✅ Vocabulary: {len(vocabulary['word_to_idx'])} words")

    # Load dataset
    print("\n📂 Loading dataset...")

    # Try to load context-only dataset first
    try:
        from src.training.train_context_only import ContextOnlyDataset
        dataset = ContextOnlyDataset(
            annotations_file=ANNOTATIONS_PATH,
            pose_dir=POSE_DIR,
            vocabulary_file=VOCAB_PATH
        )
        print(f"✅ Context-only dataset: {len(dataset)} samples")
    except:
        # Fallback to regular dataset
        from src.data_processing.gloss_dataset import BanglaSignGlossDataset
        dataset = BanglaSignGlossDataset(
            annotations_file=ANNOTATIONS_PATH,
            pose_dir=POSE_DIR,
            vocabulary_file=VOCAB_PATH
        )
        print(f"✅ Regular dataset: {len(dataset)} samples")

    # Create evaluator
    evaluator = ComprehensiveEvaluator(MODEL_PATH, dataset, vocabulary)

    # Run evaluation
    evaluator.evaluate(num_samples=300)

    # Calculate metrics
    metrics = evaluator.calculate_all_metrics()

    # Generate visualizations
    evaluator.plot_confusion_matrix(top_n=20)
    roc_auc = evaluator.plot_roc_curves(top_n=10)
    evaluator.plot_accuracy_breakdown(metrics)

    # Generate report
    evaluator.generate_report(metrics, roc_auc)

    print(f"\n{'=' * 70}")
    print("✅ EVALUATION COMPLETE!")
    print(f"{'=' * 70}")
    print("\n📁 All results saved in: results/comprehensive/")
    print("   - confusion_matrix.png")
    print("   - roc_curves.png")
    print("   - accuracy_breakdown.png")
    print("   - evaluation_report.txt")


if __name__ == "__main__":
    main()