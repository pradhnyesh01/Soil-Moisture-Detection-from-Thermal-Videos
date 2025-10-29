"""
Test segmentation on a single thermal frame.
"""

import cv2
import matplotlib.pyplot as plt
import numpy as np
from src.segmentation import (
    otsu_segmentation, 
    kmeans_segmentation, 
    gmm_segmentation,
    segment_and_analyze_frame
)


def test_single_frame(frame_path):
    """Test all segmentation methods on one frame."""
    
    # Load frame
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    if frame is None:
        print(f"Error: Could not load frame from {frame_path}")
        print("Please update the frame_path in this script.")
        return
    
    print(f"Loaded frame: {frame.shape}")
    print(f"Intensity range: {frame.min()} to {frame.max()}")
    print("\n" + "="*60)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Thermal Frame Segmentation Methods Comparison', fontsize=16)
    
    # Original
    axes[0, 0].imshow(frame, cmap='gray')
    axes[0, 0].set_title('Original Frame')
    axes[0, 0].axis('off')
    
    # 1. Otsu
    print("\n1. Testing Otsu Thresholding...")
    thresh_val, otsu_result = otsu_segmentation(frame)
    axes[0, 1].imshow(otsu_result, cmap='gray')
    axes[0, 1].set_title(f'Otsu (thresh={thresh_val:.1f})')
    axes[0, 1].axis('off')
    print(f"   ✓ Threshold: {thresh_val:.2f}")
    
    # 2. K-Means k=2
    print("\n2. Testing K-Means (k=2)...")
    kmeans2, centers2, _ = kmeans_segmentation(frame, n_clusters=2)
    axes[0, 2].imshow(kmeans2, cmap='viridis')
    axes[0, 2].set_title(f'K-Means (k=2)')
    axes[0, 2].axis('off')
    print(f"   ✓ Centers: {centers2.flatten()}")
    
    # 3. K-Means k=3
    print("\n3. Testing K-Means (k=3)...")
    kmeans3, centers3, _ = kmeans_segmentation(frame, n_clusters=3)
    axes[1, 0].imshow(kmeans3, cmap='viridis')
    axes[1, 0].set_title(f'K-Means (k=3)')
    axes[1, 0].axis('off')
    print(f"   ✓ Centers: {centers3.flatten()}")
    
    # 4. GMM n=3
    print("\n4. Testing GMM (n=3)...")
    gmm3, _, gmm_model = gmm_segmentation(frame, n_components=3)
    axes[1, 1].imshow(gmm3, cmap='plasma')
    axes[1, 1].set_title(f'GMM (n=3)')
    axes[1, 1].axis('off')
    print(f"   ✓ Means: {gmm_model.means_.flatten()}")
    
    # 5. Detailed analysis with K-Means k=3
    print("\n5. Detailed Analysis (K-Means k=3)...")
    results = segment_and_analyze_frame(frame, method='kmeans', n_segments=3)
    
    # Show labeled segments
    labels_colored = (results['labels'] * (255 // 2)).astype(np.uint8)
    axes[1, 2].imshow(labels_colored, cmap='tab10')
    axes[1, 2].set_title('Segment Labels')
    axes[1, 2].axis('off')
    
    print("\n   Segment Statistics:")
    for stat in results['segment_statistics']:
        print(f"     Segment {stat['label']}:")
        print(f"       - Coverage: {stat['percentage']:.1f}%")
        print(f"       - Mean intensity: {stat['mean_intensity']:.1f}")
        print(f"       - Std intensity: {stat['std_intensity']:.1f}")
    
    plt.tight_layout()
    plt.savefig('segmentation_test_results.png', dpi=150, bbox_inches='tight')
    print("\n" + "="*60)
    print("✓ Results saved to: segmentation_test_results.png")
    print("="*60)
    plt.show()


if __name__ == "__main__":
    # UPDATE THIS PATH to point to one of your extracted frames
    frame_path = 'saved_frames1/dry/MOV_12742/frame_0.png'
    
    # If you haven't extracted frames yet, run your original pipeline first:
    # python run_processing.py
    
    test_single_frame(frame_path)