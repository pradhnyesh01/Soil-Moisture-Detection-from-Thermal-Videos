"""
t-SNE visualization for thermal video features.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def load_and_prepare_data(csv_path):
    """Load CSV and prepare features for t-SNE."""
    print("Loading data...")
    df = pd.read_csv(csv_path)
    
    # Select feature columns (exclude metadata and labels)
    exclude_cols = ['label', 'label_name', 'video_file', 'frame_num', 
                    'seg_method', 'cluster']
    # In tsne_visualization.py, modify load_and_prepare_data():
    best_features = ['std_temp_sqrt', 'std_temp', 'temp_kurtosis_proxy',
                 'segment_intensity_cv', 'temp_rolling_mean']
    features = df[best_features].values
    
    features = np.nan_to_num(features, nan=0.0)
    
    labels = df['label_name'].values if 'label_name' in df.columns else None
    
    print(f"Loaded {features.shape[0]} samples with {features.shape[1]} features")
    return features, labels, best_features, df


def create_tsne_plot(features, labels=None, perplexity=30, max_iter=1000):
    """Create t-SNE visualization."""
    print(f"\nApplying t-SNE (perplexity={perplexity})...")
    
    # Standardize
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # t-SNE - FIXED: use max_iter instead of n_iter
    tsne = TSNE(n_components=2, perplexity=perplexity, max_iter=max_iter, 
                random_state=42, verbose=1)
    features_tsne = tsne.fit_transform(features_scaled)
    
    # Plot
    plt.figure(figsize=(12, 9))
    
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                       c=[colors[i]], label=label, alpha=0.6, 
                       edgecolors='k', linewidth=0.5, s=50)
        
        plt.legend(title='Class', fontsize=12)
    else:
        plt.scatter(features_tsne[:, 0], features_tsne[:, 1],
                   c='steelblue', alpha=0.6, s=50)
    
    plt.xlabel('t-SNE Component 1', fontsize=13)
    plt.ylabel('t-SNE Component 2', fontsize=13)
    plt.title('t-SNE: Thermal Video Features', fontsize=15, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return features_tsne


def create_combined_plot(features, labels=None):
    """Create side-by-side PCA and t-SNE plots."""
    print("\nCreating PCA + t-SNE comparison...")
    
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # PCA
    pca = PCA(n_components=2, random_state=42)
    features_pca = pca.fit_transform(features_scaled)
    
    # t-SNE - FIXED: use max_iter instead of n_iter
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42)
    features_tsne = tsne.fit_transform(features_scaled)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    
    # PCA subplot
    ax = axes[0]
    if labels is not None:
        unique_labels = np.unique(labels)
        colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(features_pca[mask, 0], features_pca[mask, 1],
                      c=[colors[i]], label=label, alpha=0.6,
                      edgecolors='k', linewidth=0.5, s=50)
        ax.legend(title='Class', fontsize=11)
    else:
        ax.scatter(features_pca[:, 0], features_pca[:, 1], c='steelblue', alpha=0.6, s=50)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%})', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%})', fontsize=12)
    ax.set_title('PCA Visualization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # t-SNE subplot
    ax = axes[1]
    if labels is not None:
        for i, label in enumerate(unique_labels):
            mask = labels == label
            ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1],
                      c=[colors[i]], label=label, alpha=0.6,
                      edgecolors='k', linewidth=0.5, s=50)
        ax.legend(title='Class', fontsize=11)
    else:
        ax.scatter(features_tsne[:, 0], features_tsne[:, 1], c='steelblue', alpha=0.6, s=50)
    
    ax.set_xlabel('t-SNE Component 1', fontsize=12)
    ax.set_ylabel('t-SNE Component 2', fontsize=12)
    ax.set_title('t-SNE Visualization', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # UPDATE THIS PATH
    csv_path = 'engineered_features.csv'
    
    print("="*70)
    print("t-SNE VISUALIZATION - Thermal Video Dataset")
    print("="*70)
    
    # Load data
    features, labels, feature_names, df = load_and_prepare_data(csv_path)
    
    # Create t-SNE plot
    features_tsne = create_tsne_plot(features, labels, perplexity=30)
    plt.savefig('tsne_plot.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: tsne_plot.png")
    plt.show()
    
    # Create combined PCA + t-SNE
    fig = create_combined_plot(features, labels)
    plt.savefig('pca_tsne_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: pca_tsne_comparison.png")
    plt.show()
    
    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE!")
    print("="*70)
