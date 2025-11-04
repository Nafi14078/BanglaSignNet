import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import json
from tqdm import tqdm
import os
from pathlib import Path


class PoseExtractionPipeline:
    def __init__(self):
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_pose_from_video(self, video_path, max_frames=60):
        """Extract pose landmarks from a video file"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0

        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(frame_rgb)

            # Extract pose landmarks
            pose_data = self.extract_landmarks(results)
            frames.append(pose_data)

            frame_count += 1

        cap.release()

        # Pad sequence if needed
        if len(frames) < max_frames:
            padding = [np.zeros_like(frames[0]) for _ in range(max_frames - len(frames))]
            frames.extend(padding)

        sequence = np.array(frames)
        print(f"  Extracted {len(frames)} frames from {total_frames} total frames")

        return sequence

    def extract_landmarks(self, results):
        """Extract all landmarks from MediaPipe results"""
        landmarks = []

        # Pose landmarks (33 points, 3 coordinates each)
        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            landmarks.extend([0.0] * 33 * 3)  # 33 points × 3 coordinates

        # Left hand (21 points, 3 coordinates each)
        if results.left_hand_landmarks:
            for landmark in results.left_hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            landmarks.extend([0.0] * 21 * 3)

        # Right hand (21 points, 3 coordinates each)
        if results.right_hand_landmarks:
            for landmark in results.right_hand_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
        else:
            landmarks.extend([0.0] * 21 * 3)

        # Face landmarks (limited to 50 points for efficiency)
        if results.face_landmarks:
            for i, landmark in enumerate(results.face_landmarks.landmark):
                if i < 50:  # Use first 50 face landmarks
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                else:
                    break
        else:
            landmarks.extend([0.0] * 50 * 3)

        return np.array(landmarks)

    def get_landmark_dimensions(self):
        """Calculate total dimensions of our pose features"""
        # 33 pose + 21 left hand + 21 right hand + 50 face = 125 points
        # Each point has 3 coordinates (x, y, z)
        return 125 * 3  # 375 dimensions per frame

    def process_dataset(self, annotations_file, output_dir, sample_size=50):
        """Process the dataset - start with small sample for testing"""
        # Load annotations
        annotations_df = pd.read_csv(annotations_file)

        if sample_size:
            annotations_df = annotations_df.head(sample_size)
            print(f"Processing sample of {sample_size} videos...")
        else:
            print(f"Processing all {len(annotations_df)} videos...")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        pose_data = []
        successful = 0
        failed = 0

        print("Starting pose extraction...")
        for idx, row in tqdm(annotations_df.iterrows(), total=len(annotations_df)):
            video_path = row['video_path']

            try:
                print(f"\nProcessing: {Path(video_path).name}")

                # Extract pose sequence
                pose_sequence = self.extract_pose_from_video(video_path)

                # Save pose data
                pose_filename = f"pose_{idx:05d}.npy"
                pose_filepath = output_path / pose_filename
                np.save(pose_filepath, pose_sequence)

                pose_data.append({
                    'pose_path': str(pose_filepath),
                    'video_path': video_path,
                    'sentence': row['sentence'],
                    'gloss': row['gloss'],
                    'original_id': idx,
                    'video_id': row['video_id']
                })

                successful += 1
                print(f"  ✓ Success - saved to {pose_filename}")

            except Exception as e:
                print(f"  ✗ Error processing {Path(video_path).name}: {e}")
                failed += 1
                continue

        # Save metadata
        metadata_path = output_path / "pose_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(pose_data, f, indent=2, ensure_ascii=False)

        # Save processing summary
        summary = {
            'total_processed': len(annotations_df),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(annotations_df) * 100,
            'feature_dimensions': self.get_landmark_dimensions(),
            'max_frames': 60
        }

        summary_path = output_path / "processing_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n=== Pose Extraction Complete ===")
        print(f"Successful: {successful}")
        print(f"Failed: {failed}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Feature dimensions per frame: {summary['feature_dimensions']}")
        print(f"Pose data saved to: {output_dir}")

        return pose_data, summary


def analyze_pose_data(pose_dir):
    """Analyze the extracted pose data"""
    pose_path = Path(pose_dir)
    metadata_file = pose_path / "pose_metadata.json"

    if not metadata_file.exists():
        print("No pose metadata found!")
        return

    with open(metadata_file, 'r') as f:
        metadata = json.load(f)

    print(f"\n=== Pose Data Analysis ===")
    print(f"Total pose sequences: {len(metadata)}")

    # Analyze first sequence
    if metadata:
        first_pose = metadata[0]
        pose_data = np.load(first_pose['pose_path'])
        print(f"Sequence shape: {pose_data.shape}")  # (frames, features)
        print(f"Frames per sequence: {pose_data.shape[0]}")
        print(f"Features per frame: {pose_data.shape[1]}")

        # Show sample data
        print(f"\nSample sequence info:")
        print(f"Video: {Path(first_pose['video_path']).name}")
        print(f"Sentence: {first_pose['sentence']}")
        print(f"Gloss: {first_pose['gloss']}")


def main():
    """Main function to run pose extraction"""
    pipeline = PoseExtractionPipeline()

    print("Starting Bangla Sign Language Pose Extraction Pipeline")
    print("=" * 60)

    # Process a small sample first (for testing)
    sample_size = 20  # Start with 20 videos for quick testing

    pose_data, summary = pipeline.process_dataset(
        annotations_file="data/annotations/dataset_annotations.csv",
        output_dir="data/processed/pose_sequences",
        sample_size=sample_size
    )

    # Analyze the results
    analyze_pose_data("data/processed/pose_sequences")

    print(f"\n✅ Phase 1 Complete! Ready for model development.")
    print(f"Next: We'll build the Transformer model with {sample_size} training samples")


if __name__ == "__main__":
    main()