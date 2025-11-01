"""
Random Forest Video Dominance Prediction with Side-by-Side Video Stitching
Predicts if videos are dry-dominated or wet-dominated
"""

import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class RandomForestVideoDominancePredictor:
    """Predict video dominance using Random Forest and stitch comparison videos."""
    
    def __init__(self, csv_path, frames_dir, output_dir='rf_output_videos'):
        """
        Initialize predictor.
        
        Args:
            csv_path (str): Path to engineered features CSV
            frames_dir (str): Base frames directory
            output_dir (str): Output directory for videos
        """
        self.df = pd.read_csv(csv_path)
        self.frames_dir = frames_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Top 5 features for RF (best performing)
        self.top_features = [
            'temp_rolling_mean',
            'avg_within_segment_std',
            'segment_intensity_range',
            'segment_intensity_cv',
            'mean_segment_intensity'
        ]
        
        # Color maps
        self.thermal_cmap = cm.get_cmap('inferno')
        self.class_colors = {
            0: (0, 0, 255),    # Dry = Red (BGR)
            1: (255, 0, 0)     # Wet = Blue (BGR)
        }
        self.class_names = {0: 'DRY', 1: 'WET'}
        
        print(f"✓ Loaded {len(self.df)} samples from {csv_path}")
        
    def train_random_forest(self):
        """Train Random Forest on frame-level features."""
        print("\n" + "="*70)
        print("TRAINING RANDOM FOREST")
        print("="*70)
        
        # Prepare data
        X = self.df[self.top_features].values
        y = self.df['label'].values
        
        # Standardize
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train RF
        self.rf_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        )
        self.rf_model.fit(X_train, y_train)
        
        # Evaluate
        train_acc = self.rf_model.score(X_train, y_train)
        test_acc = self.rf_model.score(X_test, y_test)
        
        print(f"\n✓ Model trained")
        print(f"  Train accuracy: {train_acc:.3f}")
        print(f"  Test accuracy:  {test_acc:.3f}")
        
    def predict_frame_labels(self):
        """Predict frame-level dry/wet labels using RF."""
        print("\n" + "="*70)
        print("PREDICTING FRAME-LEVEL LABELS WITH RANDOM FOREST")
        print("="*70)
        
        X = self.df[self.top_features].values
        X_scaled = self.scaler.transform(X)
        
        self.df['rf_prediction'] = self.rf_model.predict(X_scaled)
        
        print(f"✓ Frame predictions complete")
        print(f"  Dry frames (0):  {(self.df['rf_prediction'] == 0).sum()}")
        print(f"  Wet frames (1):  {(self.df['rf_prediction'] == 1).sum()}")
        
    def get_video_dominance(self):
        """Aggregate predictions to video level dominance."""
        print("\n" + "="*70)
        print("VIDEO-LEVEL DOMINANCE PREDICTION")
        print("="*70)
        
        # Group by video
        video_groups = self.df.groupby('video_file')['rf_prediction'].value_counts().unstack(fill_value=0)
        
        # Ensure columns exist (handle case where one class is missing)
        if 0 not in video_groups.columns:
            video_groups[0] = 0
        if 1 not in video_groups.columns:
            video_groups[1] = 0
        
        # Reorder columns
        video_groups = video_groups[[0, 1]]
        
        # Determine dominance - FIXED LINE
        video_groups['total'] = video_groups[0] + video_groups[1]
        video_groups['dominant_class'] = video_groups[[0, 1]].idxmax(axis=1)
        video_groups['dominant_class_name'] = video_groups['dominant_class'].map(self.class_names)
        video_groups['dominance_percent'] = video_groups[[0, 1]].max(axis=1) / video_groups['total'] * 100
        
        print(f"\n✓ Video dominance predictions complete")
        print(f"\n{video_groups}")
        
        # Save results
        video_groups.to_csv(os.path.join(self.output_dir, 'rf_video_dominance.csv'))
        print(f"\n✓ Saved to: rf_video_dominance.csv")
        
        return video_groups

    
    def load_frame(self, video_file, frame_num):
        """Load frame from disk."""
        for label_folder in ['dry', 'wet']:
            frame_path = os.path.join(
                self.frames_dir, label_folder, video_file, f'frame_{frame_num}.png'
            )
            if os.path.exists(frame_path):
                img = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    return img
        return None
    
    def apply_thermal_colormap(self, frame_gray):
        """Apply thermal colormap to grayscale frame."""
        if frame_gray is None:
            return None
        
        frame_min = frame_gray.min()
        frame_max = frame_gray.max()
        if frame_max == frame_min:
            frame_norm = np.zeros_like(frame_gray, dtype=np.float32)
        else:
            frame_norm = (frame_gray - frame_min) / (frame_max - frame_min)
        
        frame_colored = (self.thermal_cmap(frame_norm)[:, :, :3] * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_colored, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
    
    def create_rf_overlay(self, frame, rf_prediction):
        """Create overlay with RF prediction."""
        if frame is None:
            return None
        
        overlay = frame.copy()
        color = self.class_colors[rf_prediction]
        label_text = self.class_names[rf_prediction]
        
        # Semi-transparent overlay - FIXED RECTANGLE COORDINATES
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
        frame_overlay = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
        
        # Add label
        label_color = (255, 255, 255)
        cv2.putText(frame_overlay, label_text, (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 2, label_color, 3)
        
        return frame_overlay

    
    def create_side_by_side_frame(self, original, rf_processed):
        """Create side-by-side comparison."""
        if original is None or rf_processed is None:
            return None
        
        h = max(original.shape[0], rf_processed.shape[0])
        w = original.shape[1]  # This is correct
        
        if original.shape[0] != h:
            original = cv2.resize(original, (w, h))
        if rf_processed.shape[0] != h:
            rf_processed = cv2.resize(rf_processed, (w, h))
        
        side_by_side = np.hstack([original, rf_processed])
        
        # FIXED: Convert w to int if needed
        w_int = int(w)
        
        cv2.putText(side_by_side, "ORIGINAL", (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "RF PREDICTION", (w_int + 20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return side_by_side

    
    def create_conclusion_frame_rf(self, video_name, dry_count, wet_count, 
                               frame_width, frame_height):
        """Create conclusion frame for RF predictions."""
        # FIXED: Convert to int to ensure they are integers, not tuples

        
        conclusion_frame = np.ones((int(frame_height), int(frame_width), 3), dtype=np.uint8) * 240
        
        total = dry_count + wet_count
        dry_pct = (dry_count / total) * 100 if total > 0 else 0
        wet_pct = (wet_count / total) * 100 if total > 0 else 0
        
        # Determine dominant
        if dry_pct > wet_pct:
            dominant = "DRY DOMINANT"
            dominant_color = (0, 0, 255)  # Red
            bg_color = (150, 100, 100)
        else:
            dominant = "WET DOMINANT"
            dominant_color = (255, 0, 0)  # Blue
            bg_color = (100, 100, 150)
        
        # Background overlay
        overlay = conclusion_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), bg_color, -1)
        conclusion_frame = cv2.addWeighted(conclusion_frame, 0.6, overlay, 0.4, 0)
        
        # Title
        cv2.putText(conclusion_frame, "RF CONCLUSION", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        
        # Main conclusion
        cv2.putText(conclusion_frame, dominant, (80, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, dominant_color, 4)
        
        # Statistics
        stats_y = 280
        line_spacing = 70
        
        cv2.putText(conclusion_frame, f"Video: {video_name}", (80, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        stats_y += line_spacing
        cv2.putText(conclusion_frame, f"Total Frames: {total}", (80, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        stats_y += line_spacing
        dry_text = f"Dry Frames (RF): {dry_count} ({dry_pct:.1f}%)"
        cv2.putText(conclusion_frame, dry_text, (80, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)
        
        stats_y += line_spacing
        wet_text = f"Wet Frames (RF): {wet_count} ({wet_pct:.1f}%)"
        cv2.putText(conclusion_frame, wet_text, (80, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
        
        # Progress bar
        bar_y = stats_y + 100
        bar_height = 40
        bar_width = 500
        bar_x = 80
        
        cv2.rectangle(conclusion_frame, (bar_x, bar_y), 
                    (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
        
        dry_portion = int((dry_pct / 100) * bar_width)
        cv2.rectangle(conclusion_frame, (bar_x, bar_y), 
                    (bar_x + dry_portion, bar_y + bar_height), (0, 0, 255), -1)
        
        wet_portion = int((wet_pct / 100) * bar_width)
        cv2.rectangle(conclusion_frame, (bar_x + dry_portion, bar_y), 
                    (bar_x + dry_portion + wet_portion, bar_y + bar_height), (255, 0, 0), -1)
        
        cv2.putText(conclusion_frame, "Dry", (bar_x + 10, bar_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(conclusion_frame, "Wet", (bar_x + bar_width - 60, bar_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return conclusion_frame

    
    def stitch_video_with_rf_predictions(self, video_name, fps=10):
        """Stitch video with RF predictions side-by-side."""
        video_data = self.df[self.df['video_file'] == video_name].sort_values('frame_num')
        
        if len(video_data) == 0:
            print(f"  ⚠ No data for {video_name}")
            return
        
        print(f"\n📹 Processing {video_name} with RF predictions ({len(video_data)} frames)...")
        
        writer = None
        output_path = os.path.join(self.output_dir, f"{video_name}_rf_comparison.mp4")
        
        successful_frames = 0
        skipped_frames = 0
        dry_count = 0
        wet_count = 0
        frame_height = None
        frame_width = None
        
        for idx, row in tqdm(video_data.iterrows(), total=len(video_data), 
                            desc=f"  {video_name}"):
            frame_num = int(row['frame_num'])
            rf_pred = int(row['rf_prediction'])
            
            # Count predictions
            if rf_pred == 0:
                dry_count += 1
            else:
                wet_count += 1
            
            # Load frame
            original = self.load_frame(video_name, frame_num)
            if original is None:
                skipped_frames += 1
                continue
            
            # FIXED: Extract as int immediately
            if frame_height is None:
                frame_height = int(original.shape[0])
                frame_width = int(original.shape[1])
            
            # Apply colormap
            original_colored = self.apply_thermal_colormap(original)
            
            # Create RF overlay
            rf_overlay = self.create_rf_overlay(original_colored, rf_pred)
            
            # Side-by-side
            side_by_side = self.create_side_by_side_frame(original_colored, rf_overlay)
            
            if side_by_side is None:
                skipped_frames += 1
                continue
            
            # Initialize writer
            if writer is None:
                h, w = side_by_side.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                print(f"  ✓ Creating: {output_path} ({w}x{h}@{fps}fps)")
            
            writer.write(side_by_side)
            successful_frames += 1
        
        # Add conclusion frame
        if writer is not None and frame_width is not None:
            conclusion_frame = self.create_conclusion_frame_rf(
                video_name, dry_count, wet_count,
                frame_width * 2, frame_height
            )
            
            num_conclusion_frames = fps * 3  # 3 seconds
            for _ in range(num_conclusion_frames):
                writer.write(conclusion_frame)
            
            print(f"  ✓ Added RF conclusion frame ({dry_count} dry, {wet_count} wet)")
            writer.release()
            print(f"  ✓ Saved: {output_path}")

        
    def stitch_all_videos_with_rf(self, fps=10):
        """Stitch all videos with RF predictions."""
        unique_videos = self.df['video_file'].unique()
        print(f"\n{'='*70}")
        print(f"Stitching {len(unique_videos)} videos with RF predictions")
        print(f"{'='*70}")
        
        for video_name in unique_videos:
            self.stitch_video_with_rf_predictions(video_name, fps=fps)
        
        print(f"\n✓ All videos stitched!")
        print(f"  Output: {self.output_dir}/")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("RANDOM FOREST VIDEO DOMINANCE PREDICTION WITH STITCHING")
    print("="*70)
    
    # Configuration
    csv_path = 'final_clustered_results.csv'
    frames_dir = 'saved_frames'
    output_dir = 'rf_output_videos'
    
    # Initialize predictor
    predictor = RandomForestVideoDominancePredictor(csv_path, frames_dir, output_dir)
    
    # Train RF
    predictor.train_random_forest()
    
    # Predict frame labels
    predictor.predict_frame_labels()
    
    # Get video-level dominance
    video_dominance = predictor.get_video_dominance()
    
    # Stitch videos with RF predictions
    print("\n" + "="*70)
    print("STITCHING VIDEOS WITH RF PREDICTIONS")
    print("="*70)
    predictor.stitch_all_videos_with_rf(fps=10)
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nGenerated videos in: {output_dir}/")
    print("Files:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")
