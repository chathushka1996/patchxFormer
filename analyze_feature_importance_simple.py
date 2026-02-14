"""
Simple Feature Importance Analysis for Weather Parameters
This script analyzes the importance of each weather parameter using:
1. Correlation Analysis
2. Mutual Information
3. Feature Ranking

No trained model required - can be run immediately!
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr
import warnings
warnings.filterwarnings('ignore')

def analyze_feature_importance(root_path, target='Solar Power Output'):
    """
    Analyze feature importance for weather parameters
    
    Args:
        root_path: Path to dataset directory containing train.csv
        target: Name of target column
    """
    
    # Load training data
    train_path = os.path.join(root_path, 'train.csv')
    if not os.path.exists(train_path):
        print(f"Error: {train_path} not found!")
        return
    
    print("Loading data...")
    train_df = pd.read_csv(train_path)
    print(f"Loaded {len(train_df)} samples with {len(train_df.columns)} columns")
    
    # Identify weather parameters (exclude date, dayofyear, timeofday, and target)
    exclude_cols = ['date', 'dayofyear', 'timeofday', target]
    weather_params = [col for col in train_df.columns if col not in exclude_cols]
    
    print(f"\nFound {len(weather_params)} weather parameters:")
    for i, param in enumerate(weather_params, 1):
        print(f"  {i}. {param}")
    
    # Prepare data
    X = train_df[weather_params].values
    y = train_df[target].values
    
    # Remove any NaN values
    mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[mask]
    y = y[mask]
    
    print(f"\nUsing {len(y)} samples after removing NaN values")
    
    # ============================================
    # 1. CORRELATION ANALYSIS
    # ============================================
    print("\n" + "="*70)
    print("1. CORRELATION ANALYSIS")
    print("="*70)
    
    correlations = {}
    correlations_p = {}  # p-values
    
    for i, param in enumerate(weather_params):
        corr, p_value = pearsonr(X[:, i], y)
        correlations[param] = abs(corr)  # Use absolute value
        correlations_p[param] = p_value
    
    # Sort by correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
    
    print("\nCorrelation with Solar Power Output (absolute values):")
    print("-" * 70)
    print(f"{'Parameter':<20} {'Correlation':<15} {'P-value':<15} {'Significant':<15}")
    print("-" * 70)
    
    for param, corr_val in sorted_corr:
        p_val = correlations_p[param]
        significant = "Yes" if p_val < 0.05 else "No"
        print(f"{param:<20} {corr_val:>13.4f}   {p_val:>13.4e}   {significant:<15}")
    
    # Visualize correlations
    plt.figure(figsize=(12, 7))
    params = [x[0] for x in sorted_corr]
    values = [x[1] for x in sorted_corr]
    colors = ['red' if correlations_p[p] < 0.05 else 'gray' for p in params]
    plt.barh(params, values, color=colors)
    plt.xlabel('Absolute Correlation Coefficient', fontsize=12)
    plt.title('Feature Importance: Correlation Analysis\n(Red = Statistically Significant, p < 0.05)', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_importance_correlation.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved plot: feature_importance_correlation.png")
    
    # ============================================
    # 2. MUTUAL INFORMATION ANALYSIS
    # ============================================
    print("\n" + "="*70)
    print("2. MUTUAL INFORMATION ANALYSIS")
    print("="*70)
    print("Calculating mutual information (this may take a moment)...")
    
    # Calculate mutual information
    mi_scores = mutual_info_regression(X, y, random_state=42, n_neighbors=3)
    
    mi_dict = dict(zip(weather_params, mi_scores))
    sorted_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
    
    print("\nMutual Information with Solar Power Output:")
    print("-" * 70)
    print(f"{'Parameter':<20} {'MI Score':<15}")
    print("-" * 70)
    
    for param, mi_val in sorted_mi:
        print(f"{param:<20} {mi_val:>13.4f}")
    
    # Visualize mutual information
    plt.figure(figsize=(12, 7))
    params = [x[0] for x in sorted_mi]
    values = [x[1] for x in sorted_mi]
    plt.barh(params, values, color='green', alpha=0.7)
    plt.xlabel('Mutual Information Score', fontsize=12)
    plt.title('Feature Importance: Mutual Information', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_importance_mutual_info.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved plot: feature_importance_mutual_info.png")
    
    # ============================================
    # 3. COMBINED RANKING
    # ============================================
    print("\n" + "="*70)
    print("3. COMBINED FEATURE IMPORTANCE RANKING")
    print("="*70)
    
    # Normalize scores to 0-1 range
    max_corr = max(correlations.values()) if correlations.values() else 1
    max_mi = max(mi_dict.values()) if mi_dict.values() else 1
    
    results = {}
    for param in weather_params:
        norm_corr = correlations[param] / max_corr if max_corr > 0 else 0
        norm_mi = mi_dict[param] / max_mi if max_mi > 0 else 0
        avg_score = (norm_corr + norm_mi) / 2
        
        results[param] = {
            'correlation': correlations[param],
            'correlation_norm': norm_corr,
            'mutual_info': mi_dict[param],
            'mutual_info_norm': norm_mi,
            'average_score': avg_score,
            'p_value': correlations_p[param]
        }
    
    # Sort by average score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['average_score'], reverse=True)
    
    print("\nOverall Feature Importance Ranking:")
    print("-" * 90)
    print(f"{'Rank':<6} {'Parameter':<20} {'Correlation':<15} {'Mutual Info':<15} {'Avg Score':<15}")
    print("-" * 90)
    
    for rank, (param, scores) in enumerate(sorted_results, 1):
        print(f"{rank:<6} {param:<20} {scores['correlation']:>13.4f}   {scores['mutual_info']:>13.4f}   {scores['average_score']:>13.4f}")
    
    # Save to CSV
    summary_data = []
    for rank, (param, scores) in enumerate(sorted_results, 1):
        summary_data.append({
            'rank': rank,
            'parameter': param,
            'correlation': scores['correlation'],
            'correlation_normalized': scores['correlation_norm'],
            'mutual_info': scores['mutual_info'],
            'mutual_info_normalized': scores['mutual_info_norm'],
            'average_score': scores['average_score'],
            'p_value': scores['p_value'],
            'statistically_significant': 'Yes' if scores['p_value'] < 0.05 else 'No'
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv('feature_importance_summary.csv', index=False)
    print("\n✓ Saved summary to: feature_importance_summary.csv")
    
    # Visualize combined ranking
    plt.figure(figsize=(14, 8))
    params = [x[0] for x in sorted_results]
    corr_norm = [results[p]['correlation_norm'] for p in params]
    mi_norm = [results[p]['mutual_info_norm'] for p in params]
    avg_scores = [results[p]['average_score'] for p in params]
    
    x = np.arange(len(params))
    width = 0.25
    
    plt.barh(x - width, corr_norm, width, label='Correlation (normalized)', alpha=0.8)
    plt.barh(x, mi_norm, width, label='Mutual Info (normalized)', alpha=0.8)
    plt.barh(x + width, avg_scores, width, label='Average Score', alpha=0.8)
    
    plt.yticks(x, params)
    plt.xlabel('Normalized Importance Score', fontsize=12)
    plt.title('Feature Importance: Combined Analysis', fontsize=14)
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_importance_combined.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot: feature_importance_combined.png")
    
    # ============================================
    # 4. CORRELATION MATRIX
    # ============================================
    print("\n" + "="*70)
    print("4. CORRELATION MATRIX")
    print("="*70)
    
    # Create correlation matrix including target
    all_features = weather_params + [target]
    corr_matrix = train_df[all_features].corr()
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8})
    plt.title('Correlation Matrix: Weather Parameters and Solar Power Output', fontsize=14)
    plt.tight_layout()
    plt.savefig('feature_importance_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✓ Saved plot: feature_importance_correlation_matrix.png")
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70)
    print("\nGenerated files:")
    print("  1. feature_importance_correlation.png")
    print("  2. feature_importance_mutual_info.png")
    print("  3. feature_importance_combined.png")
    print("  4. feature_importance_correlation_matrix.png")
    print("  5. feature_importance_summary.csv")
    print("\nTop 3 Most Important Weather Parameters:")
    for i, (param, scores) in enumerate(sorted_results[:3], 1):
        print(f"  {i}. {param} (Score: {scores['average_score']:.4f})")
    
    return results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple Feature Importance Analysis')
    parser.add_argument('--root_path', type=str, 
                        default='./dataset/sl_piliyandala',
                        help='Root path to dataset directory')
    parser.add_argument('--target', type=str, default='Solar Power Output',
                        help='Target column name')
    
    args = parser.parse_args()
    
    analyze_feature_importance(args.root_path, args.target)

