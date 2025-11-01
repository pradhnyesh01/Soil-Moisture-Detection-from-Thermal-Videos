import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv('/Users/pradhnyesh/Documents/PJT-1/engineered_features.csv')

# Select top features
top_features = [
    'std_temp_sqrt', 'std_temp', 'temp_kurtosis_proxy',
    'segment_intensity_cv', 'temp_rolling_mean',
    'segment_intensity_range', 'avg_within_segment_std',
    'temp_range_squared', 'min_temp', 'temp_range',
    'temp_cv', 'mean_temp_squared', 'mean_temp_seg_entropy',
    'total_weighted_intensity', 'mean_segment_intensity'
]

X = df[top_features].values
y = df['label'].values

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-Means clustering
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

# Auto-align function
def auto_align_clusters(clusters, true_labels):
    """Automatically align cluster labels to maximize accuracy."""
    acc_original = accuracy_score(true_labels, clusters)
    clusters_flipped = 1 - clusters
    acc_flipped = accuracy_score(true_labels, clusters_flipped)
    
    if acc_flipped > acc_original:
        print(f"✓ Labels flipped: {acc_original:.1%} → {acc_flipped:.1%}")
        return clusters_flipped, acc_flipped
    else:
        print(f"✓ Labels kept: {acc_original:.1%}")
        return clusters, acc_original

# Align clusters
clusters_aligned, accuracy = auto_align_clusters(clusters, y)

from sklearn.metrics import silhouette_score

score = silhouette_score(X_scaled, clusters_aligned)
print(f"Silhouette Score (K-Means, n=2): {score:.3f}")

# Detailed evaluation
print("\n" + "="*60)
print("K-MEANS CLUSTERING RESULTS (n=2)")
print("="*60)
print(f"\nOverall Accuracy: {accuracy:.1%}")

# Confusion matrix
cm = confusion_matrix(y, clusters_aligned)
print("\nConfusion Matrix:")
print("              Predicted")
print("            Dry    Wet")
print(f"True Dry:  {cm[0,0]:4d}  {cm[0,1]:4d}   (Total: {cm[0,0]+cm[0,1]})")
print(f"True Wet:  {cm[1,0]:4d}  {cm[1,1]:4d}   (Total: {cm[1,0]+cm[1,1]})")

# Per-class metrics
dry_recall = cm[0,0] / (cm[0,0] + cm[0,1])
wet_recall = cm[1,1] / (cm[1,0] + cm[1,1])
dry_precision = cm[0,0] / (cm[0,0] + cm[1,0])
wet_precision = cm[1,1] / (cm[0,1] + cm[1,1])

print("\nPer-Class Performance:")
print(f"  Dry - Recall: {dry_recall:.1%}  Precision: {dry_precision:.1%}")
print(f"  Wet - Recall: {wet_recall:.1%}  Precision: {wet_precision:.1%}")

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Dry', 'Wet'], 
            yticklabels=['Dry', 'Wet'])
plt.xlabel('Predicted Label', fontsize=12)
plt.ylabel('True Label', fontsize=12)
plt.title('K-Means Confusion Matrix (n=2, Aligned)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('kmeans_confusion_matrix_aligned.png', dpi=150)
print("\n✓ Confusion matrix saved: kmeans_confusion_matrix_aligned.png")
plt.show()

# Save results
df['cluster_kmeans'] = clusters_aligned
df['cluster_kmeans_name'] = df['cluster_kmeans'].map({0: 'dry', 1: 'wet'})
df.to_csv('final_clustered_results.csv', index=False)
print("✓ Results saved: final_clustered_results.csv")

print("\n" + "="*60)
