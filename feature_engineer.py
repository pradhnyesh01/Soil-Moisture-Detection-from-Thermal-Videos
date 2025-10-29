"""
Advanced feature engineering for thermal video soil moisture classification.
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')


class ThermalFeatureEngineer:
    """Engineer advanced features from thermal video data."""
    
    def __init__(self, csv_path):
        """
        Initialize with CSV containing segmented features.
        
        Args:
            csv_path (str): Path to CSV file
        """
        print(f"Loading data from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        print(f"Loaded {len(self.df)} samples")
        
    def create_all_features(self):
        """Create all feature engineering transformations."""
        print("\n" + "="*70)
        print("FEATURE ENGINEERING PIPELINE")
        print("="*70)
        
        # 1. Temperature-based features
        print("\n1. Creating temperature-based features...")
        self._create_temperature_features()
        
        # 2. Segment-based features
        print("2. Creating segment-based features...")
        self._create_segment_features()
        
        # 3. Statistical features
        print("3. Creating statistical features...")
        self._create_statistical_features()
        
        # 4. Ratio and interaction features
        print("4. Creating ratio and interaction features...")
        self._create_ratio_features()
        
        # 5. Temporal features (if applicable)
        print("5. Creating temporal features...")
        self._create_temporal_features()
        
        # 6. Polynomial features (selected)
        print("6. Creating polynomial features...")
        self._create_polynomial_features()
        
        # 7. Segment distribution features
        print("7. Creating segment distribution features...")
        self._create_distribution_features()
        
        print("\n" + "="*70)
        print(f"Feature engineering complete!")
        print(f"Total features: {len(self.df.columns)}")
        print("="*70)
        
        return self.df
    
    def _create_temperature_features(self):
        """Create temperature-based features."""
        # Temperature range
        self.df['temp_range'] = self.df['max_temp'] - self.df['min_temp']
        
        # Temperature coefficient of variation
        self.df['temp_cv'] = self.df['std_temp'] / (self.df['mean_temp'] + 1e-8)
        
        # Relative temperature metrics
        self.df['temp_above_mean'] = self.df['max_temp'] - self.df['mean_temp']
        self.df['temp_below_mean'] = self.df['mean_temp'] - self.df['min_temp']
        
        # Temperature asymmetry
        self.df['temp_asymmetry'] = (self.df['temp_above_mean'] - self.df['temp_below_mean']) / (self.df['temp_range'] + 1e-8)
        
        # Normalized temperature
        self.df['mean_temp_normalized'] = (self.df['mean_temp'] - self.df['min_temp']) / (self.df['temp_range'] + 1e-8)
        
        print(f"   Added 6 temperature features")
    
    def _create_segment_features(self):
        """Create segment-based features."""
        # Identify segment columns
        seg_percentages = [col for col in self.df.columns if 'seg' in col and 'percentage' in col]
        seg_intensities = [col for col in self.df.columns if 'seg' in col and 'mean_intensity' in col]
        seg_stds = [col for col in self.df.columns if 'seg' in col and 'std_intensity' in col]
        
        n_features = 0
        
        # Segment coverage features
        if len(seg_percentages) >= 2:
            # Dominant segment
            self.df['dominant_segment'] = self.df[seg_percentages].idxmax(axis=1).str.extract('(\d+)').astype(float)
            
            # Segment entropy (diversity)
            seg_data = self.df[seg_percentages].values + 1e-8
            seg_probs = seg_data / seg_data.sum(axis=1, keepdims=True)
            self.df['segment_entropy'] = -(seg_probs * np.log(seg_probs)).sum(axis=1)
            
            # Max segment coverage
            self.df['max_segment_coverage'] = self.df[seg_percentages].max(axis=1)
            self.df['min_segment_coverage'] = self.df[seg_percentages].min(axis=1)
            
            # Coverage imbalance
            self.df['segment_imbalance'] = (self.df['max_segment_coverage'] - self.df['min_segment_coverage']) / 100.0
            
            n_features += 5
        
        # Segment intensity features
        if len(seg_intensities) >= 2:
            # Intensity range across segments
            self.df['segment_intensity_range'] = self.df[seg_intensities].max(axis=1) - self.df[seg_intensities].min(axis=1)
            
            # Intensity gradient (difference between consecutive segments)
            for i in range(len(seg_intensities) - 1):
                self.df[f'intensity_gradient_{i}_{i+1}'] = self.df[seg_intensities[i+1]] - self.df[seg_intensities[i]]
            
            # Mean segment intensity
            self.df['mean_segment_intensity'] = self.df[seg_intensities].mean(axis=1)
            
            # Intensity coefficient of variation
            self.df['segment_intensity_cv'] = self.df[seg_intensities].std(axis=1) / (self.df['mean_segment_intensity'] + 1e-8)
            
            n_features += len(seg_intensities) + 2
        
        # Segment standard deviation features
        if len(seg_stds) >= 2:
            # Within-segment variability
            self.df['avg_within_segment_std'] = self.df[seg_stds].mean(axis=1)
            self.df['max_within_segment_std'] = self.df[seg_stds].max(axis=1)
            
            n_features += 2
        
        print(f"   Added {n_features} segment features")
    
    def _create_statistical_features(self):
        """Create statistical distribution features."""
        # Skewness proxy (from available statistics)
        self.df['temp_skewness_proxy'] = (self.df['mean_temp'] - (self.df['min_temp'] + self.df['max_temp']) / 2) / (self.df['std_temp'] + 1e-8)
        
        # Kurtosis proxy
        self.df['temp_kurtosis_proxy'] = self.df['temp_range'] / (self.df['std_temp'] + 1e-8)
        
        print(f"   Added 2 statistical features")
    
    def _create_ratio_features(self):
        """Create ratio and interaction features."""
        seg_percentages = [col for col in self.df.columns if 'seg' in col and 'percentage' in col]
        seg_intensities = [col for col in self.df.columns if 'seg' in col and 'mean_intensity' in col]
        
        n_features = 0
        
        # Pairwise segment ratios
        if len(seg_percentages) >= 2:
            # FIXED: Ratio of first to last segment (use single column indexing)
            self.df['seg_first_last_ratio'] = self.df[seg_percentages[0]] / (self.df[seg_percentages[-1]] + 1e-8)
            n_features += 1
            
            # Create selected pairwise ratios
            for i in range(len(seg_percentages) - 1):
                self.df[f'seg_ratio_{i}_{i+1}'] = self.df[seg_percentages[i]] / (self.df[seg_percentages[i+1]] + 1e-8)
                n_features += 1
        
        # Weighted features (coverage * intensity)
        if len(seg_percentages) >= 2 and len(seg_intensities) >= 2:
            for i in range(min(len(seg_percentages), len(seg_intensities))):
                self.df[f'weighted_intensity_{i}'] = self.df[seg_percentages[i]] * self.df[seg_intensities[i]]
                n_features += 1
            
            # Total weighted intensity
            weighted_cols = [col for col in self.df.columns if 'weighted_intensity_' in col]
            if weighted_cols:
                self.df['total_weighted_intensity'] = self.df[weighted_cols].sum(axis=1)
                n_features += 1
        
        # Temperature * segment interactions
        if len(seg_percentages) > 0:
            if 'segment_imbalance' in self.df.columns:
                self.df['temp_range_seg_imbalance'] = self.df['temp_range'] * self.df['segment_imbalance']
                n_features += 1
            if 'segment_entropy' in self.df.columns:
                self.df['mean_temp_seg_entropy'] = self.df['mean_temp'] * self.df['segment_entropy']
                n_features += 1
        
        print(f"   Added {n_features} ratio features")

    
    def _create_temporal_features(self):
        """Create temporal features based on frame sequences."""
        n_features = 0
        
        if 'video_file' in self.df.columns and 'frame_num' in self.df.columns:
            # Sort by video and frame
            self.df = self.df.sort_values(['video_file', 'frame_num'])
            
            # Temperature change from previous frame
            self.df['temp_change'] = self.df.groupby('video_file')['mean_temp'].diff()
            
            # Cumulative temperature change
            self.df['temp_cumulative_change'] = self.df.groupby('video_file')['temp_change'].cumsum()
            
            # Temperature change rate
            self.df['temp_change_rate'] = self.df['temp_change'].abs()
            
            # Rolling statistics (window=3 frames)
            self.df['temp_rolling_mean'] = self.df.groupby('video_file')['mean_temp'].transform(
                lambda x: x.rolling(window=3, min_periods=1).mean()
            )
            self.df['temp_rolling_std'] = self.df.groupby('video_file')['mean_temp'].transform(
                lambda x: x.rolling(window=3, min_periods=1).std()
            )
            
            # Frame position in video
            self.df['frame_position'] = self.df.groupby('video_file').cumcount()
            
            # Fill NaN from diff operations
            self.df['temp_change'].fillna(0, inplace=True)
            self.df['temp_cumulative_change'].fillna(0, inplace=True)
            self.df['temp_rolling_std'].fillna(0, inplace=True)
            
            n_features = 6
        
        print(f"   Added {n_features} temporal features")
    
    def _create_polynomial_features(self):
        """Create selected polynomial features."""
        # Square of important features
        self.df['mean_temp_squared'] = self.df['mean_temp'] ** 2
        self.df['std_temp_squared'] = self.df['std_temp'] ** 2
        self.df['temp_range_squared'] = self.df['temp_range'] ** 2
        
        # Sqrt features
        self.df['mean_temp_sqrt'] = np.sqrt(np.abs(self.df['mean_temp']))
        self.df['std_temp_sqrt'] = np.sqrt(np.abs(self.df['std_temp']))
        
        print(f"   Added 5 polynomial features")
    
    def _create_distribution_features(self):
        """Create features describing segment distributions."""
        seg_percentages = [col for col in self.df.columns if 'seg' in col and 'percentage' in col]
        
        n_features = 0
        
        if len(seg_percentages) >= 3:
            # Gini coefficient (inequality measure)
            seg_values = self.df[seg_percentages].values
            sorted_seg = np.sort(seg_values, axis=1)
            n_segs = sorted_seg.shape[1]  # FIXED: renamed from 'n' to 'n_segs'
            index = np.arange(1, n_segs + 1)  # FIXED: use n_segs instead of n
            gini = ((2 * index - n_segs - 1) * sorted_seg).sum(axis=1) / (n_segs * sorted_seg.sum(axis=1) + 1e-8)
            self.df['segment_gini'] = gini
            
            # Herfindahl index (concentration)
            seg_normalized = seg_values / 100.0
            self.df['segment_herfindahl'] = (seg_normalized ** 2).sum(axis=1)
            
            n_features = 2
        
        print(f"   Added {n_features} distribution features")

    
    def get_feature_importance_ready_df(self):
        """
        Return dataframe with only numeric features, ready for ML.
        
        Returns:
            pd.DataFrame: Cleaned dataframe
        """
        # Drop non-feature columns
        exclude_cols = ['video_file', 'frame_num', 'label_name', 'seg_method']
        
        # Select numeric columns only
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Create feature matrix
        X = self.df[feature_cols].copy()
        
        # Handle any remaining NaN
        X.fillna(0, inplace=True)
        
        # Handle infinite values
        X.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Add labels if available
        if 'label' in self.df.columns:
            X['label'] = self.df['label']
        if 'label_name' in self.df.columns:
            X['label_name'] = self.df['label_name']
        
        print(f"\nFeature matrix ready: {X.shape} samples × {X.shape} features")
        
        return X
    
    def save_engineered_features(self, output_path='engineered_features.csv'):
        """Save engineered features to CSV."""
        self.df.to_csv(output_path, index=False)
        print(f"\n✓ Engineered features saved to: {output_path}")


def analyze_feature_correlation(df, label_col='label'):
    """
    Analyze correlation of features with target label.
    
    Args:
        df (pd.DataFrame): Dataframe with features and labels
        label_col (str): Name of label column
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    print("\n" + "="*70)
    print("FEATURE CORRELATION ANALYSIS")
    print("="*70)
    
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['label', 'frame_num']]
    
    # Calculate correlation with label
    if label_col in df.columns:
        correlations = df[feature_cols + [label_col]].corr()[label_col].drop(label_col)
        correlations = correlations.abs().sort_values(ascending=False)
        
        print(f"\nTop 20 features correlated with {label_col}:")
        print(correlations.head(20))
        
        # Plot top features
        plt.figure(figsize=(12, 8))
        correlations.head(20).plot(kind='barh')
        plt.xlabel('Absolute Correlation')
        plt.title('Top 20 Features by Correlation with Target')
        plt.tight_layout()
        plt.savefig('feature_correlations.png', dpi=150)
        print("\n✓ Saved: feature_correlations.png")
        plt.show()
    
    return correlations


def compare_feature_distributions(df, label_col='label_name'):
    """
    Compare feature distributions between classes.
    
    Args:
        df (pd.DataFrame): Dataframe with features and labels
        label_col (str): Name of label column
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import ttest_ind
    
    print("\n" + "="*70)
    print("FEATURE DISTRIBUTION ANALYSIS")
    print("="*70)
    
    if label_col not in df.columns:
        print(f"Label column '{label_col}' not found.")
        return
    
    # Select numeric features
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col not in ['label', 'frame_num']]
    
    # Get class labels
    classes = df[label_col].unique()
    
    if len(classes) != 2:
        print("This analysis is designed for binary classification.")
        return
    
    # Perform t-tests
    significant_features = []
    
    class_0_data = df[df[label_col] == classes[0]]
    class_1_data = df[df[label_col] == classes[1]]

    
    for feature in feature_cols:
        stat, p_value = ttest_ind(class_0_data[feature].dropna(), 
                                   class_1_data[feature].dropna(), 
                                   equal_var=False)
        
        if p_value < 0.05:
            significant_features.append((feature, p_value, abs(stat)))
    
    # Sort by t-statistic
    significant_features.sort(key=lambda x: x, reverse=True)
    
    print(f"\nFound {len(significant_features)} statistically significant features (p < 0.05)")
    print("\nTop 10 most discriminative features:")
    for i, (feature, p_val, t_stat) in enumerate(significant_features[:10], 1):
        print(f"{i:2d}. {feature:40s} (p={p_val:.2e}, |t|={t_stat:.2f})")
    
    # Visualize top features
    top_features = [f for f in significant_features[:6]]
    
    if top_features:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
        
        for idx, feature in enumerate(top_features):
            sns.violinplot(x=label_col, y=feature, data=df, ax=axes[idx])
            axes[idx].set_title(f'{feature}', fontsize=10)
            axes[idx].set_xlabel('')
        
        plt.tight_layout()
        plt.savefig('top_features_distributions.png', dpi=150)
        print("\n✓ Saved: top_features_distributions.png")
        plt.show()
    
    return significant_features


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Configuration
    input_csv = 'all_frame_features_segmented.csv'
    output_csv = 'engineered_features.csv'
    
    print("="*70)
    print(" "*15 + "FEATURE ENGINEERING PIPELINE")
    print(" "*20 + "Thermal Video Dataset")
    print("="*70)
    
    # Initialize feature engineer
    engineer = ThermalFeatureEngineer(input_csv)
    
    # Create all features
    df_engineered = engineer.create_all_features()
    
    # Get ML-ready dataframe
    df_ml = engineer.get_feature_importance_ready_df()
    
    # Save
    engineer.save_engineered_features(output_csv)
    
    # Analysis
    print("\n" + "="*70)
    print("FEATURE ANALYSIS")
    print("="*70)
    
    # Correlation analysis
    correlations = analyze_feature_correlation(df_ml, label_col='label')
    
    # Distribution comparison
    significant_features = compare_feature_distributions(df_ml, label_col='label_name')
    
    print("\n" + "="*70)
    print("FEATURE ENGINEERING COMPLETE!")
    print("="*70)
    print(f"\nOriginal features: 17")
    print(f"Total features after engineering: {df_ml.shape - 2}")  # Exclude label columns
    print(f"\nNext steps:")
    print(f"  1. Use '{output_csv}' for model training")
    print(f"  2. Review 'feature_correlations.png'")
    print(f"  3. Review 'top_features_distributions.png'")
    print(f"  4. Run t-SNE again with engineered features")
    print("="*70)
