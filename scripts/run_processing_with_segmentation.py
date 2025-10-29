import os
from src.data_processing import process_dataset_with_segmentation

if __name__ == "__main__":
    # ============ CONFIGURATION ============
    base_folder = '/Users/pradhnyesh/Documents/dataset'
    frames_save_root = 'saved_frames_segmented'
    output_csv = 'all_frame_features_segmented.csv'
    
    # Choose segmentation method: 'otsu', 'kmeans', 'gmm', 'multilevel'
    segmentation_method = 'kmeans'
    
    # Number of segments (only for kmeans, gmm, multilevel)
    n_segments = 4
    # ======================================
    
    print("="*60)
    print("UNSUPERVISED SEGMENTATION PIPELINE")
    print("="*60)
    print(f"Dataset folder: {base_folder}")
    print(f"Segmentation method: {segmentation_method}")
    print(f"Number of segments: {n_segments}")
    print("="*60)
    
    # Process dataset with segmentation
    dataset_df, segmentation_results = process_dataset_with_segmentation(
        base_folder=base_folder,
        frames_save_root=frames_save_root,
        out_csv_path=output_csv,
        segmentation_method=segmentation_method,
        n_segments=n_segments
    )
    
    print("\n" + "="*60)
    print("PROCESSING COMPLETE!")
    print("="*60)
    print(f"✓ Frames saved to: {frames_save_root}/{{label}}/{{video}}/original/")
    print(f"✓ Segmented frames saved to: {frames_save_root}/{{label}}/{{video}}/segmented/")
    print(f"✓ Per-frame features saved to: {output_csv}")
    print(f"✓ Total frames processed: {len(segmentation_results)}")
    print("="*60)