"""
Module for extracting thermal features from soil videos.

Functions:
- extract_features_from_video: Extract features per frame and save frames from a video.
"""

import cv2
import numpy as np
import os

def extract_features_from_video(video_path, label_name, label_value, frame_skip=30, frames_save_root='saved_frames'):
    """
    Extract thermal features from frames sampled in a video.

    Args:
        video_path (str): Path to the video file.
        label_name (str): Label category name (e.g., "dry" or "wet").
        label_value (int): Numeric label for classification.
        frame_skip (int): Number of frames to skip between samples.
        frames_save_root (str): Root folder to save extracted frames.

    Returns:
        avg_features (np.ndarray): Average [mean, std, max, min] temperature features across sampled frames.
        csv_rows (list of dict): Per-frame features and metadata for all sampled frames.
    """
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    video_frames_dir = os.path.join(frames_save_root, label_name, video_id)
    os.makedirs(video_frames_dir, exist_ok=True)

    csv_rows = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_skip == 0:
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            temp_map = gray.astype(np.float32)
            mean_temp = np.mean(temp_map)
            std_temp = np.std(temp_map)
            max_temp = np.max(temp_map)
            min_temp = np.min(temp_map)

            frame_filename = os.path.join(video_frames_dir, f"frame_{frame_count}.png")
            cv2.imwrite(frame_filename, gray)

            csv_rows.append({
                'video_file': video_id,
                'frame_num': frame_count,
                'mean_temp': mean_temp,
                'std_temp': std_temp,
                'max_temp': max_temp,
                'min_temp': min_temp,
                'label_name': label_name,
                'label': label_value
            })
        frame_count += 1
    cap.release()

    features_arr = np.array([[row['mean_temp'], row['std_temp'], row['max_temp'], row['min_temp']] for row in csv_rows])
    avg_features = features_arr.mean(axis=0) if features_arr.size > 0 else np.zeros(4)

    return avg_features, csv_rows
