"""
Module for unsupervised segmentation of thermal video frames.

Contains multiple segmentation techniques:
1. Otsu thresholding
2. K-Means clustering  
3. Gaussian Mixture Model (GMM)
4. Multi-level thresholding
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import os


def preprocess_frame(frame, enhance=True):
    """Preprocess thermal frame for better segmentation."""
    if enhance:
        enhanced = cv2.equalizeHist(frame)
        return enhanced
    return frame


def otsu_segmentation(frame, preprocess=True):
    """Apply Otsu's automatic thresholding for binary segmentation."""
    if preprocess:
        frame = preprocess_frame(frame, enhance=True)
    
    thresh_val, binary_img = cv2.threshold(
        frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )
    return thresh_val, binary_img


def kmeans_segmentation(frame, n_clusters=3, preprocess=True):
    """Apply K-Means clustering for multi-class segmentation."""
    if preprocess:
        frame = preprocess_frame(frame, enhance=True)
    
    h, w = frame.shape
    pixel_values = frame.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(
        pixel_values, n_clusters, None, criteria, 10, cv2.KMEANS_PP_CENTERS
    )
    
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(frame.shape)
    
    return segmented_image, centers, labels.reshape(h, w)


def gmm_segmentation(frame, n_components=3, preprocess=True):
    """Apply Gaussian Mixture Model for probabilistic segmentation."""
    if preprocess:
        frame = preprocess_frame(frame, enhance=True)
    
    h, w = frame.shape
    pixel_values = frame.reshape((-1, 1))
    pixel_values = np.float32(pixel_values)
    
    gmm = GaussianMixture(n_components=n_components, covariance_type='tied', random_state=42)
    gmm.fit(pixel_values)
    labels = gmm.predict(pixel_values)
    
    means = gmm.means_.flatten()
    sorted_indices = np.argsort(means)
    label_mapping = {sorted_indices[i]: i for i in range(n_components)}
    
    remapped_labels = np.array([label_mapping[label] for label in labels])
    segmented_image = (remapped_labels * (255 // (n_components - 1))).astype(np.uint8)
    segmented_image = segmented_image.reshape(frame.shape)
    
    return segmented_image, remapped_labels.reshape(h, w), gmm


def multilevel_threshold_segmentation(frame, levels=3, preprocess=True):
    """Apply multi-level thresholding for segmentation."""
    if preprocess:
        frame = preprocess_frame(frame, enhance=True)
    
    thresholds = []
    for i in range(1, levels):
        percentile = (i / levels) * 100
        thresh = np.percentile(frame, percentile)
        thresholds.append(thresh)
    
    segmented = np.zeros_like(frame)
    for i, thresh in enumerate(thresholds):
        if i == 0:
            segmented[frame <= thresh] = i
        else:
            segmented[(frame > thresholds[i-1]) & (frame <= thresh)] = i
    
    if len(thresholds) > 0:
        segmented[frame > thresholds[-1]] = len(thresholds)
    
    segmented_vis = (segmented * (255 // levels)).astype(np.uint8)
    return segmented_vis, thresholds


def segment_and_analyze_frame(frame, method='kmeans', n_segments=3, save_path=None):
    """Comprehensive segmentation and analysis of a single frame."""
    results = {
        'method': method,
        'n_segments': n_segments,
        'original_shape': frame.shape
    }
    
    if method == 'otsu':
        thresh_val, segmented = otsu_segmentation(frame)
        results['threshold'] = thresh_val
        results['segmented_image'] = segmented
        results['labels'] = (segmented > 0).astype(int)
        
    elif method == 'kmeans':
        segmented, centers, labels = kmeans_segmentation(frame, n_clusters=n_segments)
        results['segmented_image'] = segmented
        results['cluster_centers'] = centers.flatten()
        results['labels'] = labels
        
    elif method == 'gmm':
        segmented, labels, gmm = gmm_segmentation(frame, n_components=n_segments)
        results['segmented_image'] = segmented
        results['labels'] = labels
        results['gmm_means'] = gmm.means_.flatten()
        results['gmm_covariances'] = gmm.covariances_
        
    elif method == 'multilevel':
        segmented, thresholds = multilevel_threshold_segmentation(frame, levels=n_segments)
        results['segmented_image'] = segmented
        results['thresholds'] = thresholds
        results['labels'] = segmented // (255 // n_segments)
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Calculate statistics for each segment
    unique_labels = np.unique(results['labels'])
    segment_stats = []
    
    for label in unique_labels:
        mask = (results['labels'] == label)
        segment_pixels = frame[mask]
        
        stats = {
            'label': int(label),
            'pixel_count': int(np.sum(mask)),
            'percentage': float(np.sum(mask) / frame.size * 100),
            'mean_intensity': float(np.mean(segment_pixels)),
            'std_intensity': float(np.std(segment_pixels)),
            'min_intensity': float(np.min(segment_pixels)),
            'max_intensity': float(np.max(segment_pixels))
        }
        segment_stats.append(stats)
    
    results['segment_statistics'] = segment_stats
    
    if save_path:
        cv2.imwrite(save_path, results['segmented_image'])
    
    return results