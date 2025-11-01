"""
Comprehensive Supervised Learning Comparison
Including: Logistic Regression, Random Forest, SVM, XGBoost, and Gradient Boosting
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, confusion_matrix, classification_report, 
                            roc_auc_score, roc_curve)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# LOAD AND PREPARE DATA
# ============================================================================

print("="*80)
print("SUPERVISED LEARNING COMPARISON FOR THERMAL SOIL MOISTURE CLASSIFICATION")
print("="*80)

# Load data
df = pd.read_csv('engineered_features.csv')
print(f"\n✓ Loaded {len(df)} samples")

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

print(f"✓ Using {len(top_features)} features")
print(f"  Class distribution: {np.bincount(y)}")

# Standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n✓ Train set: {len(X_train)} samples")
print(f"✓ Test set: {len(X_test)} samples")

# ============================================================================
# DEFINE MODELS
# ============================================================================

models = {
    'Logistic Regression': LogisticRegression(
        random_state=42, max_iter=1000, class_weight='balanced'
    ),
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    ),
    'SVM (RBF)': SVC(
        kernel='rbf', C=1.0, random_state=42, probability=True
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42
    ),
    'XGBoost': xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1, 
        random_state=42, eval_metric='logloss', use_label_encoder=False
    )
}

# ============================================================================
# TRAIN AND EVALUATE MODELS
# ============================================================================

results = {}
predictions = {}

print("\n" + "="*80)
print("TRAINING MODELS")
print("="*80)

for model_name, model in models.items():
    print(f"\n[{model_name}]", end=" ")
    
    # Cross-validation
    print("(CV)", end=" ")
    cv_results = cross_validate(
        model, X_train, y_train, cv=5, 
        scoring=['accuracy', 'precision', 'recall', 'f1', 'roc_auc'],
        n_jobs=-1
    )
    
    # Train on full training set
    print("(Train)", end=" ")
    model.fit(X_train, y_train)
    
    # Test predictions
    print("(Test)", end=" ")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    
    results[model_name] = {
        'cv_accuracy_mean': cv_results['test_accuracy'].mean(),
        'cv_accuracy_std': cv_results['test_accuracy'].std(),
        'cv_precision_mean': cv_results['test_precision'].mean(),
        'cv_recall_mean': cv_results['test_recall'].mean(),
        'cv_f1_mean': cv_results['test_f1'].mean(),
        'cv_auc_mean': cv_results['test_roc_auc'].mean(),
        'test_accuracy': acc,
        'test_precision': prec,
        'test_recall': rec,
        'test_f1': f1,
        'test_auc': auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'model': model
    }
    
    predictions[model_name] = y_pred
    
    print(f"✓ Done")

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

print("\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

# Create results dataframe
results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'CV Accuracy': [f"{results[m]['cv_accuracy_mean']:.3f} ± {results[m]['cv_accuracy_std']:.3f}" 
                    for m in results.keys()],
    'Test Accuracy': [f"{results[m]['test_accuracy']:.3f}" for m in results.keys()],
    'Test Precision': [f"{results[m]['test_precision']:.3f}" for m in results.keys()],
    'Test Recall': [f"{results[m]['test_recall']:.3f}" for m in results.keys()],
    'Test F1': [f"{results[m]['test_f1']:.3f}" for m in results.keys()],
    'Test AUC': [f"{results[m]['test_auc']:.3f}" if results[m]['test_auc'] is not None else "N/A" 
                 for m in results.keys()]
})

print("\n")
print(results_df.to_string(index=False))

# ============================================================================
# DETAILED ANALYSIS FOR EACH MODEL
# ============================================================================

print("\n" + "="*80)
print("DETAILED METRICS PER MODEL")
print("="*80)

for model_name in models.keys():
    print(f"\n[{model_name}]")
    print(f"  Cross-Val Accuracy: {results[model_name]['cv_accuracy_mean']:.3f} ± {results[model_name]['cv_accuracy_std']:.3f}")
    print(f"  Test Accuracy:      {results[model_name]['test_accuracy']:.3f}")
    print(f"  Test Precision:     {results[model_name]['test_precision']:.3f}")
    print(f"  Test Recall:        {results[model_name]['test_recall']:.3f}")
    print(f"  Test F1-Score:      {results[model_name]['test_f1']:.3f}")
    if results[model_name]['test_auc'] is not None:
        print(f"  Test AUC-ROC:       {results[model_name]['test_auc']:.3f}")
    
    cm = results[model_name]['confusion_matrix']
    print(f"  Confusion Matrix:")
    print(f"    True Neg:  {cm[0,0]:4d}  |  False Pos: {cm[0,1]:4d}")
    print(f"    False Neg: {cm[1,0]:4d}  |  True Pos:  {cm[1,1]:4d}")

# ============================================================================
# FEATURE IMPORTANCE (for tree-based models)
# ============================================================================

print("\n" + "="*80)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*80)

tree_models = {
    'Random Forest': results['Random Forest']['model'],
    'Gradient Boosting': results['Gradient Boosting']['model'],
    'XGBoost': results['XGBoost']['model']
}

for model_name, model in tree_models.items():
    print(f"\n[{model_name}] Top 10 Important Features:")
    importances = model.feature_importances_
    feature_importance = sorted(zip(top_features, importances),
                               key=lambda x: x[1], reverse=True)
    
    for idx, (feature, importance) in enumerate(feature_importance[:10], 1):
        bar_length = int(importance * 50)
        bar = '█' * bar_length
        print(f"  {idx:2d}. {feature:30s} {importance:6.3f} {bar}")

# ============================================================================
# COMPARISON WITH UNSUPERVISED
# ============================================================================

print("\n" + "="*80)
print("COMPARISON: UNSUPERVISED vs SUPERVISED")
print("="*80)

comparison_data = {
    'Method': ['K-Means (Unsupervised)', 'Logistic Regression', 'Random Forest', 
               'SVM (RBF)', 'Gradient Boosting', 'XGBoost'],
    'Accuracy': [
        0.704,
        results['Logistic Regression']['test_accuracy'],
        results['Random Forest']['test_accuracy'],
        results['SVM (RBF)']['test_auc'] if results['SVM (RBF)']['test_auc'] else results['SVM (RBF)']['test_accuracy'],
        results['Gradient Boosting']['test_accuracy'],
        results['XGBoost']['test_accuracy']
    ]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n")
print(comparison_df.to_string(index=False))

# ============================================================================
# VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Supervised Learning Models Comparison', fontsize=16, fontweight='bold')

# 1. Test Accuracy Comparison
ax = axes[0, 0]
test_accs = [results[m]['test_accuracy'] for m in results.keys()]
colors = ['red' if 'K-Means' in str(results.keys()) else 'steelblue' for _ in results.keys()]
ax.barh(list(results.keys()), test_accs, color='steelblue', alpha=0.7)
ax.axvline(0.704, color='red', linestyle='--', linewidth=2, label='K-Means (70.4%)')
ax.set_xlabel('Test Accuracy')
ax.set_title('Test Accuracy per Model')
ax.legend()
ax.grid(True, alpha=0.3, axis='x')

# 2. CV Accuracy with Error Bars
ax = axes[0, 1]
models_list = list(results.keys())
cv_means = [results[m]['cv_accuracy_mean'] for m in models_list]
cv_stds = [results[m]['cv_accuracy_std'] for m in models_list]
ax.barh(models_list, cv_means, xerr=cv_stds, color='steelblue', alpha=0.7, capsize=5)
ax.set_xlabel('Cross-Validation Accuracy')
ax.set_title('5-Fold CV Accuracy (with Std Dev)')
ax.grid(True, alpha=0.3, axis='x')

# 3. Precision vs Recall
ax = axes[0, 2]
precision = [results[m]['test_precision'] for m in results.keys()]
recall = [results[m]['test_recall'] for m in results.keys()]
ax.scatter(recall, precision, s=200, alpha=0.6, c=range(len(results)), cmap='viridis')
for i, model_name in enumerate(results.keys()):
    ax.annotate(model_name.split('(')[0].strip(), 
               (recall[i], precision[i]), fontsize=9)
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision vs Recall Trade-off')
ax.grid(True, alpha=0.3)
ax.set_xlim([0.4, 1.0])
ax.set_ylim([0.4, 1.0])

# 4. F1-Score Comparison
ax = axes[1, 0]
f1_scores = [results[m]['test_f1'] for m in results.keys()]
ax.bar(range(len(results)), f1_scores, color='steelblue', alpha=0.7)
ax.set_xticks(range(len(results)))
ax.set_xticklabels([m.split('(')[0].strip() for m in results.keys()], rotation=45, ha='right')
ax.set_ylabel('F1-Score')
ax.set_title('F1-Score Comparison')
ax.set_ylim([0, 1])
ax.grid(True, alpha=0.3, axis='y')

# 5. Confusion Matrices (Best Model - XGBoost)
ax = axes[1, 1]
cm_best = results['XGBoost']['confusion_matrix']
sns.heatmap(cm_best, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_xlabel('Predicted')
ax.set_ylabel('True')
ax.set_title('Confusion Matrix (XGBoost - Best Model)')
ax.set_xticklabels(['Dry', 'Wet'])
ax.set_yticklabels(['Dry', 'Wet'])

# 6. Model Rankings
ax = axes[1, 2]
rankings = sorted([(m, results[m]['test_accuracy']) for m in results.keys()],
                 key=lambda x: x[1], reverse=True)
models_ranked = [r[0].split('(')[0].strip() for r in rankings]
accuracies_ranked = [r[1] for r in rankings]
colors_ranked = ['#FFD700' if i == 0 else '#C0C0C0' if i == 1 else '#CD7F32' if i == 2 else 'steelblue'
                for i in range(len(rankings))]
ax.barh(models_ranked, accuracies_ranked, color=colors_ranked, alpha=0.7)
ax.set_xlabel('Test Accuracy')
ax.set_title('Model Rankings')
ax.set_xlim([0.6, 1.0])
for i, v in enumerate(accuracies_ranked):
    ax.text(v + 0.01, i, f'{v:.3f}', va='center')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('supervised_learning_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: supervised_learning_comparison.png")
plt.show()

# ============================================================================
# ROC CURVES
# ============================================================================

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('ROC Curves - Supervised Models', fontsize=16, fontweight='bold')

axes = axes.flatten()

for idx, (model_name, model) in enumerate(results.items()):
    ax = axes[idx]
    
    y_pred_proba = model['model'].predict_proba(X_test)[:, 1] if hasattr(model['model'], 'predict_proba') else None
    
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        auc = model['test_auc']
        
        ax.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc:.3f}')
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'{model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14)
        ax.set_title(f'{model_name}')

plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=150, bbox_inches='tight')
print("✓ Saved: roc_curves_comparison.png")
plt.show()

# ============================================================================
# SAVE RESULTS
# ============================================================================

# Save detailed results to CSV
results_export = []
for model_name in results.keys():
    results_export.append({
        'Model': model_name,
        'CV_Accuracy_Mean': results[model_name]['cv_accuracy_mean'],
        'CV_Accuracy_Std': results[model_name]['cv_accuracy_std'],
        'CV_Precision_Mean': results[model_name]['cv_precision_mean'],
        'CV_Recall_Mean': results[model_name]['cv_recall_mean'],
        'CV_F1_Mean': results[model_name]['cv_f1_mean'],
        'CV_AUC_Mean': results[model_name]['cv_auc_mean'],
        'Test_Accuracy': results[model_name]['test_accuracy'],
        'Test_Precision': results[model_name]['test_precision'],
        'Test_Recall': results[model_name]['test_recall'],
        'Test_F1': results[model_name]['test_f1'],
        'Test_AUC': results[model_name]['test_auc'] if results[model_name]['test_auc'] is not None else 'N/A'
    })

results_export_df = pd.DataFrame(results_export)
results_export_df.to_csv('supervised_learning_results.csv', index=False)
print("✓ Saved: supervised_learning_results.csv")

# ============================================================================
# SUMMARY REPORT
# ============================================================================

print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

best_model = max(results.items(), key=lambda x: x[1]['test_accuracy'])
print(f"\n🏆 Best Performing Model: {best_model[0]}")
print(f"   Test Accuracy: {best_model[1]['test_accuracy']:.3f}")
print(f"   Test F1-Score: {best_model[1]['test_f1']:.3f}")
print(f"   Test AUC: {best_model[1]['test_auc']:.3f}" if best_model[1]['test_auc'] is not None else "")

improvement = (best_model[1]['test_accuracy'] - 0.704) / 0.704 * 100
print(f"\n📈 Improvement over Unsupervised (K-Means):")
print(f"   {best_model[1]['test_accuracy']:.3f} vs 0.704 = +{improvement:.1f}%")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETE!")
print("="*80)
