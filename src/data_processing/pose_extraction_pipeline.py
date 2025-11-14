import os
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
import pandas as pd
from src.utils.config import load_config


class PoseExtractionPipeline:
    def __init__(self):
        self.config = load_config()

    def get_landmark_dimensions(self):
        """
        Returns the expected landmark dimension size for pose sequence.
        Adjust if you use a different keypoint model.
        """
        # Example: 25 landmarks × 3 (x,y,confidence)
        return 75

    def extract_pose_from_video(self, video_path, max_frames=60):
        """
        Extracts pose sequence from a video.
        Returns a NumPy array of shape (frames, features).
        """
        if not os.path.exists(video_path):
            print(f"⚠️  Skipping: video not found -> {video_path}")
            feat_dim = self.get_landmark_dimensions()
            return np.zeros((max_frames, feat_dim), dtype=np.float32)

        cap = cv2.VideoCapture(str(video_path))
        frames = []
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            # ✨ Dummy placeholder: Replace with actual keypoint extraction
            keypoints = self._extract_keypoints(frame)
            frames.append(keypoints)

            if len(frames) >= max_frames:
                break

        cap.release()

        # 🧩 Handle case where no frames are extracted
        if len(frames) == 0:
            feat_dim = self.get_landmark_dimensions()
            return np.zeros((max_frames, feat_dim), dtype=np.float32)

        frames = np.array(frames)

        # 🧩 Pad or trim sequence to fixed length
        if frames.shape[0] < max_frames:
            pad_len = max_frames - frames.shape[0]
            pad = np.zeros((pad_len, frames.shape[1]))
            frames = np.vstack((frames, pad))
        else:
            frames = frames[:max_frames]

        return frames.astype(np.float32)

    def _extract_keypoints(self, frame):
        """
        Dummy keypoint extractor placeholder.
        Replace this with your Mediapipe/OpenPose keypoint extractor.
        """
        # Here we just flatten the frame mean as dummy features.
        # Replace this with your real pose/keypoint data.
        feat_dim = self.get_landmark_dimensions()
        return np.random.rand(feat_dim).astype(np.float32)

    def process_dataset(self, annotations_csv, output_path):
        """
        Loops through dataset annotations and saves pose sequences.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(annotations_csv)

        print(f"🧩 Found {len(df)} annotated samples.")
        print(f"💾 Saving processed poses to: {output_path}")

        for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting poses"):
            video_path = Path(row["video_path"])
            pose_filename = f"pose_vid{row['video_id']}.npy"
            pose_filepath = output_path / pose_filename

            pose_sequence = self.extract_pose_from_video(video_path)
            np.save(pose_filepath, pose_sequence)

        print("✅ Pose extraction complete!")


if __name__ == "__main__":
    print("🚀 Starting BanglaSignNet Pose Extraction Pipeline...")

    # Load config
    config = load_config()

    raw_data_path = Path(config["paths"]["data_raw"])
    processed_path = Path(config["paths"]["data_processed"]) / "pose_sequences_full"
    processed_path.mkdir(parents=True, exist_ok=True)

    annotations_csv = Path("data/annotations/dataset_annotations.csv")
    if not annotations_csv.exists():
        raise FileNotFoundError("❌ dataset_annotations.csv not found in data/annotations/")

    # Initialize and run
    pipeline = PoseExtractionPipeline()
    pipeline.process_dataset(annotations_csv, processed_path)

    print("✅ All videos processed successfully!")
