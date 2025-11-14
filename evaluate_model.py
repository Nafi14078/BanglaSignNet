"""
Detailed evaluation script for the trained model
Shows breakdown: Overall, Context-Only, First-3-Words, and Verb-Only accuracy
"""
from src.modeling.transformer_model import BanglaSignTransformer
from src.data_processing.gloss_dataset import BanglaSignGlossDataset
import torch
import json

print("Loading vocabulary and model...")

# Load vocabulary
with open('data/annotations/gloss_vocabulary.json', 'r', encoding='utf-8') as f:
    vocab = json.load(f)

# Create model
model = BanglaSignTransformer(
    input_dim=375,
    vocab_size=len(vocab['word_to_idx']),
    d_model=128,
    nhead=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    dropout=0.2,
    max_seq_length=8
)

# Load best checkpoint from your 20% accuracy model
print("Loading trained model from models/full_word_level_updated_transformer/full_checkpoint_epoch_80.pth...")
checkpoint = torch.load('models/full_word_level_updated_transformer/full_checkpoint_epoch_80.pth', map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print(f"✅ Model loaded from epoch {checkpoint['epoch']}")
print(f"✅ Best training gloss accuracy: {checkpoint.get('best_gloss_accuracy', 'N/A'):.4f}")

# Load dataset
print("\nLoading dataset...")
dataset = BanglaSignGlossDataset(
    annotations_file='data/annotations/dataset_annotations.csv',
    pose_dir='data/processed/pose_sequences_full',
    vocabulary_file='data/annotations/gloss_vocabulary.json'
)

# Setup
device = torch.device('cpu')
model.to(device)
sos_idx = vocab['word_to_idx']['<SOS>']
eos_idx = vocab['word_to_idx']['<EOS>']

print("\n" + "=" * 70)
print("DETAILED EVALUATION RESULTS (First 30 samples)")
print("=" * 70)

correct = 0
last_token_correct = 0
first_3_correct = 0
without_verb_correct = 0
by_length = {}

for i in range(min(30, len(dataset))):
    sample = dataset[i]
    src = sample['src'].to(device)

    with torch.no_grad():
        pred = model.predict(src, sos_idx, eos_idx)
        predicted_gloss = dataset.indices_to_gloss(pred.tolist())

    target_words = sample['gloss_sequence'].split()
    pred_words = predicted_gloss.split()

    target_len = len(target_words)
    is_correct = sample['gloss_sequence'] == predicted_gloss

    # ✅ Check first 3 words (without verb)
    first_3_match = len(target_words) >= 3 and len(pred_words) >= 3 and \
                    target_words[:3] == pred_words[:3]

    # ✅ Check last word (verb) only
    last_match = len(target_words) > 0 and len(pred_words) > 0 and \
                 target_words[-1] == pred_words[-1]

    # ✅ Check everything except last word
    without_verb_match = False
    if len(target_words) > 1 and len(pred_words) > 1:
        without_verb_match = target_words[:-1] == pred_words[:-1]

    # Update counters
    by_length.setdefault(target_len, {'total': 0, 'correct': 0})
    by_length[target_len]['total'] += 1

    if is_correct:
        correct += 1
        by_length[target_len]['correct'] += 1
    if last_match:
        last_token_correct += 1
    if first_3_match:
        first_3_correct += 1
    if without_verb_match:
        without_verb_correct += 1

    # Status indicators
    status = '✅' if is_correct else ('🟡' if without_verb_match else '❌')
    last_status = '✅' if last_match else '❌'

    print(f"\n📝 Sample {i + 1:02d} ({target_len} words): {status} [Verb: {last_status}]")
    print(f"   Natural:   {sample['natural_sentence']}")
    print(f"   Target:    {sample['gloss_sequence']}")
    print(f"   Predicted: {predicted_gloss}")

    if not is_correct and len(target_words) > 0 and len(pred_words) > 0:
        if without_verb_match:
            print(f"   ✓ Context correct, only verb wrong: '{target_words[-1]}' → '{pred_words[-1]}'")
        elif last_match:
            print(f"   ✓ Verb correct, context wrong")
        else:
            print(f"   ✗ Both context and verb wrong")

print(f"\n{'=' * 70}")
print("DETAILED ACCURACY BREAKDOWN")
print(f"{'=' * 70}")
print(f"Overall Accuracy:           {correct / 30:6.1%} ({correct}/30)")
print(f"Context-Only Accuracy:      {without_verb_correct / 30:6.1%} ({without_verb_correct}/30) [without verb]")
print(f"First-3-Words Accuracy:     {first_3_correct / 30:6.1%} ({first_3_correct}/30)")
print(f"Verb-Only Accuracy:         {last_token_correct / 30:6.1%} ({last_token_correct}/30) [last token]")

print(f"\n{'=' * 70}")
print("ACCURACY BY SEQUENCE LENGTH")
print(f"{'=' * 70}")
for length in sorted(by_length.keys()):
    stats = by_length[length]
    acc = stats['correct'] / stats['total']
    print(f"  {length}-word sequences: {acc:.2%} ({stats['correct']}/{stats['total']})")
print(f"{'=' * 70}")

# ✅ Additional Analysis
print(f"\n{'=' * 70}")
print("KEY INSIGHTS")
print(f"{'=' * 70}")

context_vs_verb_gap = (without_verb_correct / 30) - (last_token_correct / 30)
print(f"📊 Context vs Verb Gap: {context_vs_verb_gap:+.1%}")
if context_vs_verb_gap > 0.3:
    print(f"   → Model is MUCH better at context than verbs")
    print(f"   → Focus improvement on verb/tense recognition")
elif context_vs_verb_gap < -0.1:
    print(f"   → Model is better at verbs than context (unusual!)")
else:
    print(f"   → Balanced performance between context and verbs")

if without_verb_correct > correct * 2:
    print(f"\n💡 RECOMMENDATION: The model gets context right {without_verb_correct} times")
    print(f"   but only {correct} complete sequences. Focus on:")
    print(f"   1. More training data with verb variations")
    print(f"   2. Better temporal features for tense distinction")
    print(f"   3. Ensemble models or post-processing for verbs")

print(f"{'=' * 70}")