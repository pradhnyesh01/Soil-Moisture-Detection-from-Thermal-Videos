import os
from src.data_processing import process_dataset_folder

if __name__ == "__main__":
    base_folder = '/Users/pradhnyesh/Documents/dataset'
    frames_save_root = 'saved_frames'
    output_csv = 'all_frame_features.csv'

    dataset_df = process_dataset_folder(base_folder, frames_save_root, output_csv)
