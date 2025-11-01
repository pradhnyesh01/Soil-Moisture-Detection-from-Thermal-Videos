"""
Stitch thermal video frames into comparison videos.
Directory structure: saved_frames/dry/MOV_12742/frame_0.png, frame_30.png, etc.
Cluster column: cluster_kmeans_name (or cluster_kmeans with numeric values)
"""

import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.cm as cm

frames_directory = '/Users/pradhnyesh/Documents/PJT-1/saved_frames'  # Folder with frame images
results_csv = 'final_clustered_results.csv'          # Your clustering results
output_dir = 'output_videos'                         # Output folder

class ThermalVideoStitcher:
    """Stitch frames with side-by-side cluster overlay comparison."""
    
    def __init__(self, base_frames_dir, csv_results, output_dir='output_videos'):
        """
        Initialize video stitcher.
        
        Args:
            base_frames_dir (str): Base directory (saved_frames)
            csv_results (str): CSV file with clustering results
            output_dir (str): Directory to save output videos
        """
        self.base_frames_dir = base_frames_dir
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Load results
        self.df = pd.read_csv(csv_results)
        print(f"✓ Loaded {len(self.df)} frame predictions")
        print(f"  Columns: {self.df.columns.tolist()[:10]}...")
        
        # Setup color maps
        self.cluster_colors = {
            'dry': (0, 0, 255),      # Red (BGR)
            'wet': (255, 0, 0),      # Blue (BGR)
            0: (0, 0, 255),          # Red
            1: (255, 0, 0)           # Blue
        }
        self.thermal_cmap = cm.get_cmap('inferno')
        
    def load_frame(self, video_name, frame_num):
        """
        Load frame from directory structure.
        
        Args:
            video_name (str): Video filename (e.g., 'MOV_12742')
            frame_num (int): Frame number
        
        Returns:
            np.ndarray: Grayscale thermal frame or None
        """
        # Try dry folder first, then wet folder
        for label_folder in ['dry', 'wet']:
            frame_path = os.path.join(
                self.base_frames_dir, 
                label_folder, 
                video_name, 
                f'frame_{frame_num}.png'
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
        
        # Normalize to 0-1
        frame_min = frame_gray.min()
        frame_max = frame_gray.max()
        if frame_max == frame_min:
            frame_norm = np.zeros_like(frame_gray, dtype=np.float32)
        else:
            frame_norm = (frame_gray - frame_min) / (frame_max - frame_min)
        
        # Apply colormap
        frame_colored = (self.thermal_cmap(frame_norm)[:, :, :3] * 255).astype(np.uint8)
        frame_bgr = cv2.cvtColor(frame_colored, cv2.COLOR_RGB2BGR)
        
        return frame_bgr
    
    def create_overlay(self, frame, cluster_label):
        """
        Create frame with cluster prediction overlay.
        
        Args:
            frame (np.ndarray): Original frame (BGR)
            cluster_label (str or int): Cluster label ('dry', 'wet', 0, or 1)
        
        Returns:
            np.ndarray: Frame with overlay and label
        """
        if frame is None:
            return None
        
        overlay = frame.copy()
        
        # Get color based on cluster
        if isinstance(cluster_label, str):
            color = self.cluster_colors.get(cluster_label.lower(), (128, 128, 128))
            label_text = cluster_label.upper()
        else:
            color = self.cluster_colors.get(cluster_label, (128, 128, 128))
            label_text = "DRY" if cluster_label == 0 else "WET"
        
        # Apply semi-transparent overlay
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), color, -1)
        frame_overlay = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
        
        # Add text label
        label_color = (255, 255, 255) if label_text == "DRY" else (255, 255, 255)
        cv2.putText(frame_overlay, label_text, (20, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, label_color, 3)
        
        return frame_overlay
    
    def create_side_by_side_frame(self, original, processed):
        """Create side-by-side comparison."""
        if original is None or processed is None:
            return None
        
        # Ensure same height
        h = max(original.shape[0], processed.shape[0])
        w = original.shape[1]
        
        if original.shape[0] != h:
            original = cv2.resize(original, (w, h))
        if processed.shape[0] != h:
            processed = cv2.resize(processed, (w, h))
        
        # Concatenate
        side_by_side = np.hstack([original, processed])
        
        # Add labels
        cv2.putText(side_by_side, "ORIGINAL", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(side_by_side, "CLUSTER PREDICTION", (w + 20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        return side_by_side
    
    def stitch_video_for_video(self, video_name, fps=10):
        """
        Create video for a single thermal video.
        
        Args:
            video_name (str): Video name (e.g., 'MOV_12742')
            fps (int): Output frames per second
        """
        # Filter data for this video
        video_data = self.df[self.df['video_file'] == video_name].sort_values('frame_num')
        
        if len(video_data) == 0:
            print(f"  ⚠ No data for {video_name}")
            return
        
        print(f"\n📹 Processing {video_name} ({len(video_data)} frames)...")
        
        writer = None
        output_path = os.path.join(self.output_dir, f"{video_name}_comparison.mp4")
        
        successful_frames = 0
        skipped_frames = 0
        
        for idx, row in tqdm(video_data.iterrows(), total=len(video_data), 
                            desc=f"  {video_name}"):
            frame_num = int(row['frame_num'])
            
            # Get cluster label (use cluster_kmeans_name if available, else cluster_kmeans)
            if 'cluster_kmeans_name' in row:
                cluster = row['cluster_kmeans_name']
            else:
                cluster = int(row['cluster_kmeans'])
            
            # Load original frame
            original = self.load_frame(video_name, frame_num)
            if original is None:
                skipped_frames += 1
                continue
            
            # Apply colormap
            original_colored = self.apply_thermal_colormap(original)
            
            # Create overlay
            processed = self.create_overlay(original_colored, cluster)
            
            # Create side-by-side
            side_by_side = self.create_side_by_side_frame(original_colored, processed)
            
            if side_by_side is None:
                skipped_frames += 1
                continue
            
            # Initialize writer on first successful frame
            if writer is None:
                h, w = side_by_side.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                print(f"  ✓ Creating: {output_path} ({w}x{h}@{fps}fps)")
            
            # Write frame
            writer.write(side_by_side)
            successful_frames += 1
        
        if writer is not None:
            writer.release()
            print(f"  ✓ Saved: {output_path}")
            print(f"    {successful_frames} frames written, {skipped_frames} skipped")
        else:
            print(f"  ✗ No frames written for {video_name}")
    
    def stitch_all_videos(self, fps=10):
        """Create videos for all unique videos in dataset."""
        unique_videos = self.df['video_file'].unique()
        print(f"\n{'='*70}")
        print(f"Found {len(unique_videos)} unique videos")
        print(f"{'='*70}")
        
        for video_name in unique_videos:
            self.stitch_video_for_video(video_name, fps=fps)
        
        print(f"\n✓ All videos processed!")
        print(f"  Output directory: {self.output_dir}/")
    
    def create_summary_report(self):
        """Create summary statistics visualization."""
        print("\n📊 Generating summary statistics...")
        
        # Group by video
        video_stats = self.df.groupby('video_file').agg({
            'cluster_kmeans': ['sum', 'count', 'mean']
        }).reset_index()
        video_stats.columns = ['video_file', 'wet_count', 'total_frames', 'wet_proportion']
        video_stats['dry_count'] = video_stats['total_frames'] - video_stats['wet_count']
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Thermal Soil Moisture Clustering Summary', fontsize=16, fontweight='bold')
        
        # 1. Bar chart
        ax = axes[0, 0]
        x = np.arange(len(video_stats))
        width = 0.35
        ax.bar(x - width/2, video_stats['dry_count'], width, label='Dry', color='red', alpha=0.7)
        ax.bar(x + width/2, video_stats['wet_count'], width, label='Wet', color='blue', alpha=0.7)
        ax.set_xlabel('Video')
        ax.set_ylabel('Frame Count')
        ax.set_title('Dry vs Wet Frame Distribution per Video')
        ax.set_xticks(x)
        ax.set_xticklabels(video_stats['video_file'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # 2. Pie chart
        ax = axes[0, 1]
        total_dry = video_stats['dry_count'].sum()
        total_wet = video_stats['wet_count'].sum()
        ax.pie([total_dry, total_wet], labels=['Dry', 'Wet'], autopct='%1.1f%%',
              colors=['red', 'blue'], startangle=90, textprops={'fontsize': 11})
        ax.set_title('Overall Distribution')
        
        # 3. Line chart
        ax = axes[1, 0]
        ax.plot(video_stats['video_file'], video_stats['dry_count'], marker='o', 
               label='Dry', color='red', linewidth=2, markersize=8)
        ax.plot(video_stats['video_file'], video_stats['wet_count'], marker='s', 
               label='Wet', color='blue', linewidth=2, markersize=8)
        ax.set_xlabel('Video')
        ax.set_ylabel('Frame Count')
        ax.set_title('Frame Counts Trend')
        ax.set_xticklabels(video_stats['video_file'], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Summary table
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
        SUMMARY STATISTICS
        {"─"*35}
        Total Videos:        {len(video_stats)}
        Total Frames:        {video_stats['total_frames'].sum():,}
        
        Overall Dry:         {total_dry:,} ({total_dry/(total_dry+total_wet)*100:.1f}%)
        Overall Wet:         {total_wet:,} ({total_wet/(total_dry+total_wet)*100:.1f}%)
        
        Avg Frames/Video:    {video_stats['total_frames'].mean():.0f}
        Avg Dry/Video:       {video_stats['dry_count'].mean():.0f}
        Avg Wet/Video:       {video_stats['wet_count'].mean():.0f}
        
        Dry Std Dev:         {video_stats['dry_count'].std():.1f}
        Wet Std Dev:         {video_stats['wet_count'].std():.1f}
        """
        ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
               verticalalignment='center', 
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
        
        plt.tight_layout()
        
        # Save
        summary_path = os.path.join(self.output_dir, 'summary_statistics.png')
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        print(f"  ✓ Summary saved: {summary_path}")
        plt.close()
        
        # Also save stats as CSV
        stats_csv = os.path.join(self.output_dir, 'summary_statistics.csv')
        video_stats.to_csv(stats_csv, index=False)
        print(f"  ✓ Stats CSV saved: {stats_csv}")

    def create_conclusion_frame(self, video_name, total_frames, dry_count, wet_count, 
                           frame_width, frame_height):
        """
        Create a final conclusion frame showing overall video classification.
        
        Args:
            video_name (str): Video name
            total_frames (int): Total frames analyzed
            dry_count (int): Number of dry frames
            wet_count (int): Number of wet frames
            frame_width (int): Width of frame
            frame_height (int): Height of frame
        
        Returns:
            np.ndarray: Conclusion frame
        """
        # Create blank frame
        conclusion_frame = np.ones((frame_height, frame_width, 3), dtype=np.uint8) * 240
        
        # Calculate statistics
        dry_percentage = (dry_count / total_frames) * 100 if total_frames > 0 else 0
        wet_percentage = (wet_count / total_frames) * 100 if total_frames > 0 else 0
        
        # Determine dominant state
        if dry_percentage > wet_percentage:
            dominant = "DRY DOMINANT"
            dominant_color = (0, 0, 255)  # Red
            bg_color = (200, 100, 100)     # Red-ish background
        else:
            dominant = "WET DOMINANT"
            dominant_color = (255, 0, 0)   # Blue
            bg_color = (200, 100, 100)     # Blue-ish background
        
        # Add colored background overlay
        overlay = conclusion_frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), bg_color, -1)
        conclusion_frame = cv2.addWeighted(conclusion_frame, 0.6, overlay, 0.4, 0)
        
        # Add title
        cv2.putText(conclusion_frame, "CONCLUSION", (50, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 3)
        
        # Add main conclusion
        cv2.putText(conclusion_frame, dominant, (80, 180),
                cv2.FONT_HERSHEY_SIMPLEX, 2.5, dominant_color, 4)
        
        # Add statistics
        stats_y = 280
        line_spacing = 70
        
        # Video name
        cv2.putText(conclusion_frame, f"Video: {video_name}", (80, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # Total frames
        stats_y += line_spacing
        cv2.putText(conclusion_frame, f"Total Frames: {total_frames}", (80, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)
        
        # Dry count and percentage
        stats_y += line_spacing
        dry_text = f"Dry Frames: {dry_count} ({dry_percentage:.1f}%)"
        cv2.putText(conclusion_frame, dry_text, (80, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 2)  # Red text
        
        # Wet count and percentage
        stats_y += line_spacing
        wet_text = f"Wet Frames: {wet_count} ({wet_percentage:.1f}%)"
        cv2.putText(conclusion_frame, wet_text, (80, stats_y),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)  # Blue text
        
        # Add confidence bar (visual representation)
        bar_y = stats_y + 100
        bar_height = 40
        bar_width = 500
        bar_x = 80
        
        # Draw background bar
        cv2.rectangle(conclusion_frame, (bar_x, bar_y), 
                    (bar_x + bar_width, bar_y + bar_height), (200, 200, 200), -1)
        
        # Draw dry portion
        dry_portion = int((dry_percentage / 100) * bar_width)
        cv2.rectangle(conclusion_frame, (bar_x, bar_y), 
                    (bar_x + dry_portion, bar_y + bar_height), (0, 0, 255), -1)
        
        # Draw wet portion
        wet_portion = int((wet_percentage / 100) * bar_width)
        cv2.rectangle(conclusion_frame, (bar_x + dry_portion, bar_y), 
                    (bar_x + dry_portion + wet_portion, bar_y + bar_height), (255, 0, 0), -1)
        
        # Add bar labels
        cv2.putText(conclusion_frame, "Dry", (bar_x + 10, bar_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(conclusion_frame, "Wet", (bar_x + bar_width - 60, bar_y + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        return conclusion_frame


    def stitch_video_for_video(self, video_name, fps=10):
        """
        Create video for a single thermal video WITH CONCLUSION FRAME.
        
        Args:
            video_name (str): Video name (e.g., 'MOV_12742')
            fps (int): Output frames per second
        """
        # Filter data for this video
        video_data = self.df[self.df['video_file'] == video_name].sort_values('frame_num')
        
        if len(video_data) == 0:
            print(f"  ⚠ No data for {video_name}")
            return
        
        print(f"\n📹 Processing {video_name} ({len(video_data)} frames)...")
        
        writer = None
        output_path = os.path.join(self.output_dir, f"{video_name}_comparison.mp4")
        
        successful_frames = 0
        skipped_frames = 0
        dry_count = 0
        wet_count = 0
        frame_height = None
        frame_width = None
        
        for idx, row in tqdm(video_data.iterrows(), total=len(video_data), 
                            desc=f"  {video_name}"):
            frame_num = int(row['frame_num'])
            
            # Get cluster label
            if 'cluster_kmeans_name' in row:
                cluster = row['cluster_kmeans_name']
            else:
                cluster = int(row['cluster_kmeans'])
            
            # Count frames
            if cluster == 'dry' or cluster == 0:
                dry_count += 1
            else:
                wet_count += 1
            
            # Load original frame
            original = self.load_frame(video_name, frame_num)
            if original is None:
                skipped_frames += 1
                continue
            
            # Store dimensions for conclusion frame
            if frame_height is None:
                frame_height = original.shape[0]
                frame_width = original.shape[1]
            
            # Apply colormap
            original_colored = self.apply_thermal_colormap(original)
            
            # Create overlay
            processed = self.create_overlay(original_colored, cluster)
            
            # Create side-by-side
            side_by_side = self.create_side_by_side_frame(original_colored, processed)
            
            if side_by_side is None:
                skipped_frames += 1
                continue
            
            # Initialize writer on first successful frame
            if writer is None:
                h, w = side_by_side.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
                print(f"  ✓ Creating: {output_path} ({w}x{h}@{fps}fps)")
            
            # Write frame
            writer.write(side_by_side)
            successful_frames += 1
        
        # Create and write conclusion frame (3 seconds at chosen fps)
        if writer is not None and frame_width is not None:
            conclusion_frame = self.create_conclusion_frame(
                video_name, 
                successful_frames, 
                dry_count, 
                wet_count,
                frame_width * 2,  # Double width for side-by-side
                frame_height
            )
            
            # Write conclusion frame multiple times (to show for ~3 seconds)
            num_conclusion_frames = fps * 3  # 3 seconds
            for _ in range(num_conclusion_frames):
                writer.write(conclusion_frame)
            
            print(f"  ✓ Added conclusion frame ({dry_count} dry, {wet_count} wet)")
        
        if writer is not None:
            writer.release()
            print(f"  ✓ Saved: {output_path}")
            print(f"    {successful_frames} frames + conclusion, {skipped_frames} skipped")
        else:
            print(f"  ✗ No frames written for {video_name}")



# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*70)
    print("THERMAL VIDEO STITCHER WITH CLUSTER OVERLAY")
    print("="*70)
    
    # Configuration - UPDATE THESE PATHS
    base_frames_directory = 'saved_frames'  # Your base directory
    results_csv = 'final_clustered_results.csv'  # Your results CSV
    output_dir = 'output_videos'
    
    # Verify paths exist
    if not os.path.exists(base_frames_directory):
        print(f"✗ Error: {base_frames_directory} not found!")
        exit(1)
    
    if not os.path.exists(results_csv):
        print(f"✗ Error: {results_csv} not found!")
        exit(1)
    
    # Initialize and run
    stitcher = ThermalVideoStitcher(base_frames_directory, results_csv, output_dir)
    
    # Create comparison videos
    stitcher.stitch_all_videos(fps=5)
    
    # Generate summary statistics
    stitcher.create_summary_report()
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"Output: {output_dir}/")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        print(f"  - {file}")
