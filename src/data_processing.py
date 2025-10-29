import os
import pandas as pd
from src.feature_extraction import extract_features_from_video, extract_features_with_segmentation

all_frame_features = []

def process_dataset_with_segmentation(base_folder, 
                                      frames_save_root='saved_frames_segmented',
                                      out_csv_path='all_frame_features_segmented.csv',
                                      segmentation_method='kmeans',
                                      n_segments=3):
    """
    Process dataset folder WITH unsupervised segmentation.
    """
    labels_map = {'dry': 0, 'wet': 1}
    data = []
    all_frame_features = []
    all_segmentation_results = []
    
    for label_name, label_value in labels_map.items():
        folder_path = os.path.join(base_folder, label_name)
        if not os.path.exists(folder_path):
            print(f"Warning: Folder {folder_path} does not exist. Skipping...")
            continue
            
        for video_file in os.listdir(folder_path):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(folder_path, video_file)
                print(f"Processing video: {video_path} with label: {label_name}")
                print(f"  Segmentation method: {segmentation_method}, n_segments: {n_segments}")
                
                avg_features, frame_rows, seg_results = extract_features_with_segmentation(
                    video_path, label_name, label_value,
                    frame_skip=30,
                    frames_save_root=frames_save_root,
                    segmentation_method=segmentation_method,
                    n_segments=n_segments
                )
                
                all_frame_features.extend(frame_rows)
                all_segmentation_results.extend(seg_results)
                data.append([*avg_features, label_value])
                
    # Save CSV
    df_frames = pd.DataFrame(all_frame_features)
    df_frames.to_csv(out_csv_path, index=False)
    print(f"\nAll per-frame features with segmentation saved to {out_csv_path}")
    print(f"Preview:")
    print(df_frames.head())

    # Create video-level dataset
    columns = ['mean_temp', 'std_temp', 'max_temp', 'min_temp', 'label']
    df = pd.DataFrame(data, columns=columns)
    print("\nFinished processing dataset folder. Video-level features preview:")
    print(df.head())
    
    return df, all_segmentation_results

def process_dataset_folder(base_folder, frames_save_root='saved_frames', out_csv_path='all_frame_features.csv'):
    labels_map = {'dry': 0, 'wet': 1}
    data = []
    global all_frame_features
    for label_name, label_value in labels_map.items():
        folder_path = os.path.join(base_folder, label_name)
        for video_file in os.listdir(folder_path):
            if video_file.endswith(('.mp4', '.avi', '.mov')):
                video_path = os.path.join(folder_path, video_file)
                print(f"Processing video: {video_path} with label: {label_name}")
                avg_features, frame_rows = extract_features_from_video(
                    video_path, label_name, label_value,
                    frame_skip=30,
                    frames_save_root=frames_save_root
                )
                all_frame_features.extend(frame_rows)
                data.append([*avg_features, label_value])
                
    # Save all per-frame features for all videos into one CSV file
    df_frames = pd.DataFrame(all_frame_features)
    df_frames.to_csv(out_csv_path, index=False)
    print(f"\nAll per-frame features saved to {out_csv_path}. Showing preview:")
    print(df_frames.head())

    # Create video-level dataset for ML training
    columns = ['mean_temp', 'std_temp', 'max_temp', 'min_temp', 'label']
    df = pd.DataFrame(data, columns=columns)
    print("\nFinished processing dataset folder. Video-level features preview:")
    print(df.head())
    return df