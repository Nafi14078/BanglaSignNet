import pandas as pd
import json
import os
from pathlib import Path


def load_simple_config():
    """Simple configuration without external dependencies"""
    return {
        'paths': {
            'data_raw': 'data/raw/Ban-Sign-Sent-9K-V1',
            'data_processed': 'data/processed'
        }
    }


def process_excel_metadata():
    """Process the Excel file to create annotations for our dataset"""
    config = load_simple_config()
    data_dir = Path(config['paths']['data_raw'])
    excel_file = data_dir / "Bangla_Sign_Sentence.xlsx"

    print("=== Processing Excel Metadata ===")

    if not excel_file.exists():
        print(f"Error: Excel file not found at {excel_file}")
        return None

    try:
        # Read Excel file
        df = pd.read_excel(excel_file)

        print(f"Excel file loaded successfully!")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        # Display sample data
        print("\n=== First 5 rows ===")
        print(df.head())

        print("\n=== Column Info ===")
        print(df.info())

        # Analyze the structure and map columns
        column_mapping = analyze_columns(df)

        # Create annotations
        annotations = create_annotations(df, column_mapping, data_dir)

        # Build vocabulary
        vocabulary = build_vocabulary(annotations)

        # Save everything
        save_annotations(annotations, vocabulary)

        print("\n=== Processing Complete ===")
        print(f"Total annotations: {len(annotations)}")
        print(f"Vocabulary size: {len(vocabulary['word_to_idx'])}")

        return annotations

    except Exception as e:
        print(f"Error processing Excel file: {e}")
        import traceback
        traceback.print_exc()
        return None


def analyze_columns(df):
    """Analyze Excel columns and map to our expected format"""
    print("\n=== Analyzing Column Structure ===")

    column_mapping = {}

    for col in df.columns:
        col_lower = str(col).lower()
        print(f"Column: '{col}' -> '{col_lower}'")

        # Map based on common patterns
        if any(keyword in col_lower for keyword in ['video', 'file', 'path', 'name', 'id']):
            column_mapping['video_id'] = col
            print(f"  → Mapped as: video_id")

        elif any(keyword in col_lower for keyword in ['gloss', 'sign', 'word', 'token']):
            column_mapping['gloss'] = col
            print(f"  → Mapped as: gloss")

        elif any(keyword in col_lower for keyword in ['sentence', 'text', 'bangla', 'translation', 'bn', 'meaning']):
            column_mapping['sentence'] = col
            print(f"  → Mapped as: sentence")

        elif any(keyword in col_lower for keyword in ['english', 'en', 'translation']):
            column_mapping['english'] = col
            print(f"  → Mapped as: english")

    # If no sentence column found, use gloss as sentence
    if 'sentence' not in column_mapping and 'gloss' in column_mapping:
        column_mapping['sentence'] = column_mapping['gloss']
        print("  → Using 'gloss' as 'sentence'")

    print(f"\nFinal column mapping: {column_mapping}")
    return column_mapping


def create_annotations(df, column_mapping, data_dir):
    """Create annotations by matching Excel data with video files"""
    print("\n=== Creating Annotations ===")

    annotations = []
    video_files = list((data_dir / "Sign_Videos").rglob("*.mp4"))
    print(f"Found {len(video_files)} video files in Sign_Videos folder")

    # Show first few video files
    print("Sample video files:")
    for video in video_files[:5]:
        print(f"  - {video.name}")

    # Create a mapping from video ID to video path
    video_map = {}
    for video_path in video_files:
        video_name = video_path.stem  # Remove extension
        video_map[video_name] = str(video_path)

    # Process each row in Excel
    for idx, row in df.iterrows():
        try:
            # Get video ID from Excel
            if 'video_id' in column_mapping:
                video_id = str(row[column_mapping['video_id']]).strip()
            else:
                # If no video ID column, use row index
                video_id = f"video_{idx + 1:04d}"

            # Find corresponding video file
            video_path = None
            if video_id in video_map:
                video_path = video_map[video_id]
            else:
                # Try partial matching
                for v_name, v_path in video_map.items():
                    if video_id in v_name or v_name in video_id:
                        video_path = v_path
                        break

            # Get sentence text
            if 'sentence' in column_mapping:
                sentence = str(row[column_mapping['sentence']]).strip()
            else:
                sentence = ""

            # Get gloss if available
            gloss = ""
            if 'gloss' in column_mapping:
                gloss = str(row[column_mapping['gloss']]).strip()

            # Only add if we have both video and sentence
            if video_path and sentence:
                annotation = {
                    'id': idx,
                    'video_id': video_id,
                    'video_path': video_path,
                    'sentence': sentence,
                    'gloss': gloss,
                    'sentence_length': len(sentence.split()),
                    'words': sentence.split()
                }

                # Add English translation if available
                if 'english' in column_mapping:
                    annotation['english'] = str(row[column_mapping['english']]).strip()

                annotations.append(annotation)

        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    print(f"Successfully created {len(annotations)} annotations")
    return annotations


def build_vocabulary(annotations):
    """Build vocabulary from all sentences"""
    print("\n=== Building Vocabulary ===")

    all_words = []
    for ann in annotations:
        all_words.extend(ann['words'])

    vocabulary = sorted(set(all_words))
    print(f"Found {len(vocabulary)} unique words")

    # Show sample words
    print(f"Sample words: {vocabulary[:10]}")

    # Create word to index mapping
    word_to_idx = {word: idx for idx, word in enumerate(vocabulary)}

    # Add special tokens
    special_tokens = {
        '<SOS>': len(vocabulary),
        '<EOS>': len(vocabulary) + 1,
        '<PAD>': len(vocabulary) + 2,
        '<UNK>': len(vocabulary) + 3
    }
    word_to_idx.update(special_tokens)

    idx_to_word = {idx: word for word, idx in word_to_idx.items()}

    vocabulary_info = {
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word,
        'vocab_size': len(word_to_idx),
        'special_tokens': special_tokens
    }

    print(f"Final vocabulary size (with special tokens): {len(word_to_idx)}")
    print(f"Special tokens: {special_tokens}")

    return vocabulary_info


def save_annotations(annotations, vocabulary):
    """Save annotations and vocabulary to files"""
    print("\n=== Saving Files ===")

    # Create annotations directory
    annotations_dir = Path("data/annotations")
    annotations_dir.mkdir(parents=True, exist_ok=True)

    # Save annotations as CSV
    annotations_df = pd.DataFrame(annotations)
    annotations_csv_path = annotations_dir / "dataset_annotations.csv"
    annotations_df.to_csv(annotations_csv_path, index=False, encoding='utf-8')
    print(f"Saved annotations to: {annotations_csv_path}")

    # Save vocabulary
    vocabulary_path = annotations_dir / "vocabulary.json"
    with open(vocabulary_path, 'w', encoding='utf-8') as f:
        json.dump(vocabulary, f, ensure_ascii=False, indent=2)
    print(f"Saved vocabulary to: {vocabulary_path}")

    # Save dataset summary
    summary = {
        'total_samples': len(annotations),
        'total_videos': len(set(ann['video_path'] for ann in annotations)),
        'vocabulary_size': vocabulary['vocab_size'],
        'average_sentence_length': sum(ann['sentence_length'] for ann in annotations) / len(annotations),
        'max_sentence_length': max(ann['sentence_length'] for ann in annotations),
        'dataset_structure': {
            'columns_in_excel': list(annotations_df.columns),
            'video_files_directory': "Sign_Videos"
        }
    }

    summary_path = annotations_dir / "dataset_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"Saved dataset summary to: {summary_path}")

    # Print summary
    print(f"\n=== Dataset Summary ===")
    print(f"Total samples: {summary['total_samples']}")
    print(f"Total videos: {summary['total_videos']}")
    print(f"Vocabulary size: {summary['vocabulary_size']}")
    print(f"Average sentence length: {summary['average_sentence_length']:.2f} words")
    print(f"Max sentence length: {summary['max_sentence_length']} words")

    # Show sample annotations
    print(f"\n=== Sample Annotations ===")
    for i in range(min(3, len(annotations))):
        ann = annotations[i]
        print(f"Sample {i + 1}:")
        print(f"  Video: {Path(ann['video_path']).name}")
        print(f"  Sentence: {ann['sentence']}")
        print(f"  Words: {ann['words']}")
        print()


if __name__ == "__main__":
    process_excel_metadata()