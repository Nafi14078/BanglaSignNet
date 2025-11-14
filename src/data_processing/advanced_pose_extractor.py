import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import os
from pathlib import Path
import albumentations as A
from albumentations.core.composition import OneOf
import sys

# Add the src directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class AdvancedPoseExtractor:
    def __init__(self, augment_data=True):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Data augmentation pipeline
        self.augment_data = augment_data
        if augment_data:
            self.augmentation = A.Compose([
                OneOf([
                    A.MotionBlur(blur_limit=3, p=0.3),
                    A.MedianBlur(blur_limit=3, p=0.3),
                    A.GaussianBlur(blur_limit=3, p=0.3),
                ], p=0.5),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.5),
                A.RandomGamma(gamma_limit=(80, 120), p=0.3),
            ])

    def extract_pose_from_video(self, video_path, max_frames=60, augment=False):
        """Extract pose landmarks with optional data augmentation"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Apply data augmentation if enabled
            if augment and self.augment_data:
                try:
                    frame = self.augmentation(image=frame)['image']
                except Exception as e:
                    print(f"Augmentation error: {e}")
                    # Continue without augmentation if it fails

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)

            # Extract pose landmarks with confidence scores
            pose_data = self.extract_landmarks_with_confidence(results)
            frames.append(pose_data)

            frame_count += 1

        cap.release()

        # Pad sequence if needed
        if len(frames) < max_frames:
            padding = [np.zeros_like(frames[0]) for _ in range(max_frames - len(frames))]
            frames.extend(padding)

        sequence = np.array(frames)
        return sequence

    def extract_landmarks_with_confidence(self, results):
        """Extract landmarks with confidence scores and better normalization"""
        landmarks = []
        confidence_scores = []

        # Pose landmarks (33 points)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
                confidence_scores.append(landmark.visibility)
        else:
            landmarks.extend([0.0] * 33 * 3)
            confidence_scores.extend([0.0] * 33)

        # Left hand (21 points)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
                confidence_scores.append(1.0)  # Assume high confidence for detected hands
        else:
            landmarks.extend([0.0] * 21 * 3)
            confidence_scores.extend([0.0] * 21)

        # Right hand (21 points)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
                confidence_scores.append(1.0)
        else:
            landmarks.extend([0.0] * 21 * 3)
            confidence_scores.extend([0.0] * 21)

        # Face landmarks (50 points)
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                if i < 50:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                    confidence_scores.append(1.0)
                else:
                    break
        else:
            landmarks.extend([0.0] * 50 * 3)
            confidence_scores.extend([0.0] * 50)

        # Combine landmarks and confidence scores
        features = np.array(landmarks + confidence_scores)
        return features

    def get_feature_dimensions(self):
        """Get total feature dimensions (landmarks + confidence scores)"""
        # 125 points * 3 coordinates + 125 confidence scores = 500 features
        return 500

    def process_dataset_with_augmentation(self, annotations_file, output_dir, num_augmentations=2, sample_size=None):
        """Process dataset with data augmentation"""
        annotations_df = pd.read_csv(annotations_file)

        if sample_size:
            annotations_df = annotations_df.head(sample_size)
            print(f"Processing sample of {sample_size} videos...")
        else:
            print(f"Processing all {len(annotations_df)} videos...")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pose_data = []
        original_count = 0
        augmented_count = 0
        failed_count = 0

        print(f"Processing {len(annotations_df)} videos with {num_augmentations} augmentations each...")

        for idx, row in tqdm(annotations_df.iterrows(), total=len(annotations_df)):
            video_path = row['video_path']

            # Check if video file exists
            if not os.path.exists(video_path):
                print(f"Video file not found: {video_path}")
                failed_count += 1
                continue

            try:
                # Original pose data
                pose_sequence = self.extract_pose_from_video(video_path, augment=False)
                pose_filename = f"pose_orig_{idx:05d}.npy"
                pose_filepath = output_path / pose_filename
                np.save(pose_filepath, pose_sequence)

                pose_data.append({
                    'pose_path': str(pose_filepath),
                    'video_path': video_path,
                    'sentence': row['sentence'],
                    'gloss': row['gloss'],
                    'original_id': idx,
                    'is_augmented': False
                })
                original_count += 1

                # Augmented versions
                for aug_idx in range(num_augmentations):
                    aug_pose_sequence = self.extract_pose_from_video(video_path, augment=True)
                    aug_pose_filename = f"pose_aug_{idx:05d}_{aug_idx:02d}.npy"
                    aug_pose_filepath = output_path / aug_pose_filename
                    np.save(aug_pose_filepath, aug_pose_sequence)

                    pose_data.append({
                        'pose_path': str(aug_pose_filepath),
                        'video_path': video_path,
                        'sentence': row['sentence'],
                        'gloss': row['gloss'],
                        'original_id': idx,
                        'is_augmented': True,
                        'augmentation_id': aug_idx
                    })
                    augmented_count += 1

            except Exception as e:
                print(f"Error processing {video_path}: {e}")
                failed_count += 1
                continue

        # Save metadata
        metadata_path = output_path / "augmented_pose_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(pose_data, f, indent=2, ensure_ascii=False)

        summary = {
            'total_original': original_count,
            'total_augmented': augmented_count,
            'total_samples': len(pose_data),
            'failed_count': failed_count,
            'augmentation_factor': num_augmentations + 1
        }

        # Save summary
        summary_path = output_path / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== Augmentation Complete ===")
        print(f"✅ Original samples: {original_count}")
        print(f"✅ Augmented samples: {augmented_count}")
        print(f"❌ Failed: {failed_count}")
        print(f"📊 Total samples: {len(pose_data)}")
        print(
            f"📈 Data increased by: {((augmented_count + original_count) / original_count - 1) * 100:.1f}%" if original_count > 0 else "N/A")
        print(f"💾 Output directory: {output_dir}")

        return pose_data, summary


def main():
    """Main function to run pose extraction with augmentation"""
    print("🚀 Starting Advanced Pose Extraction with Data Augmentation")
    print("=" * 60)

    # Initialize extractor
    extractor = AdvancedPoseExtractor(augment_data=True)

    # Process dataset with augmentation
    pose_data, summary = extractor.process_dataset_with_augmentation(
        annotations_file="data/annotations/dataset_annotations.csv",
        output_dir="data/processed/pose_sequences_augmented",
        num_augmentations=2,  # Create 2 augmented versions per video
        sample_size= None # Process first 100 samples for testing
    )

    print(f"\n🎉 Pose extraction completed successfully!")
    print(f"📁 Check the output in: data/processed/pose_sequences_augmented/")


def test_single_video():
    """Test pose extraction on a single video"""
    print("🧪 Testing pose extraction on single video...")

    extractor = AdvancedPoseExtractor(augment_data=False)

    # Find a test video
    annotations_df = pd.read_csv("data/annotations/dataset_annotations.csv")
    if len(annotations_df) > 0:
        test_video = annotations_df.iloc[0]['video_path']

        if os.path.exists(test_video):
            print(f"Testing with: {test_video}")

            # Extract pose without augmentation
            pose_sequence = extractor.extract_pose_from_video(test_video, augment=False)
            print(f"✅ Success! Extracted pose sequence shape: {pose_sequence.shape}")
            print(f"📊 Feature dimensions: {extractor.get_feature_dimensions()}")

            # Test with augmentation
            aug_pose_sequence = extractor.extract_pose_from_video(test_video, augment=True)
            print(f"✅ Augmented pose sequence shape: {aug_pose_sequence.shape}")

            return True
        else:
            print(f"❌ Test video not found: {test_video}")
            return False
    else:
        print("❌ No annotations found!")
        return False


if __name__ == "__main__":
    # First test with a single video
    if test_single_video():
        print("\n" + "=" * 60)
        print("✅ Single video test passed! Starting full processing...")
        print("=" * 60)

        # Then process the full dataset (or sample)
        main()
    else:
        print("❌ Single video test failed! Please check your data setup.")