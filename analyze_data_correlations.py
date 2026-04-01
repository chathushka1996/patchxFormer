"""
Analyze feature correlations with Solar Power Output
to understand expected feature importance.

Includes both RAW and NORMALIZED data analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import os

# Create output directory
os.makedirs('./shap_results', exist_ok=True)

# Load data
train_df = pd.read_csv('./dataset/sl_piliyandala/train.csv')

print("="*70)
print("DATA ANALYSIS - Feature Correlations with Solar Power Output")
print("="*70)

# Feature columns (excluding date and target)
feature_cols = ['dayofyear', 'timeofday', 'temp', 'dew', 'humidity', 
                'winddir', 'windspeed', 'pressure', 'cloudcover']
target_col = 'Solar Power Output'

# ============================================================
# PART 1: RAW DATA ANALYSIS
# ============================================================
print("\n" + "="*70)
print("PART 1: RAW DATA ANALYSIS (Before Normalization)")
print("="*70)

# Basic statistics
print("\n1.1 RAW FEATURE STATISTICS")
print("-"*70)
print(train_df[feature_cols + [target_col]].describe().T[['mean', 'std', 'min', 'max']])

# Correlation with target (RAW)
print("\n1.2 RAW CORRELATION WITH SOLAR POWER OUTPUT")
print("-"*70)
correlations_raw = train_df[feature_cols].corrwith(train_df[target_col])
correlations_raw = correlations_raw.sort_values(key=abs, ascending=False)
print(correlations_raw.to_string())

# ============================================================
# PART 2: NORMALIZED DATA ANALYSIS (Same as model sees)
# ============================================================
print("\n" + "="*70)
print("PART 2: NORMALIZED DATA ANALYSIS (After StandardScaler)")
print("="*70)

# Apply StandardScaler (same as data_loader.py does)
scaler = StandardScaler()
all_features = feature_cols + [target_col]
normalized_data = scaler.fit_transform(train_df[all_features])
normalized_df = pd.DataFrame(normalized_data, columns=all_features)

print("\n2.1 NORMALIZED FEATURE STATISTICS")
print("-"*70)
print(normalized_df.describe().T[['mean', 'std', 'min', 'max']])

print("\n2.2 SCALER PARAMETERS (mean and std used for normalization)")
print("-"*70)
print(f"{'Feature':<20} {'Mean (μ)':<15} {'Std (σ)':<15} {'Range after norm':<20}")
print("-"*70)
for i, feat in enumerate(all_features):
    orig_range = train_df[feat].max() - train_df[feat].min()
    norm_range = normalized_df[feat].max() - normalized_df[feat].min()
    print(f"{feat:<20} {scaler.mean_[i]:<15.4f} {scaler.scale_[i]:<15.4f} [{normalized_df[feat].min():.2f}, {normalized_df[feat].max():.2f}]")

# Correlation with target (NORMALIZED)
print("\n2.3 NORMALIZED CORRELATION WITH SOLAR POWER OUTPUT")
print("-"*70)
correlations_norm = normalized_df[feature_cols].corrwith(normalized_df[target_col])
correlations_norm = correlations_norm.sort_values(key=abs, ascending=False)
print(correlations_norm.to_string())

print("\nNote: Correlations should be IDENTICAL for raw and normalized data")
print("      (StandardScaler preserves correlation structure)")

# ============================================================
# PART 3: EXPECTED IMPORTANCE RANKING
# ============================================================
print("\n" + "="*70)
print("PART 3: EXPECTED FEATURE IMPORTANCE RANKING")
print("="*70)

abs_corr = correlations_raw.abs().sort_values(ascending=False)
total_corr = abs_corr.sum()
print(f"\n{'Rank':<6}{'Feature':<15}{'Correlation':<15}{'Expected %':<15}{'Direction':<10}")
print("-"*70)
for rank, (feature, corr) in enumerate(abs_corr.items(), 1):
    pct = (corr / total_corr) * 100
    direction = "Positive" if correlations_raw[feature] > 0 else "Negative"
    print(f"{rank:<6}{feature:<15}{correlations_raw[feature]:>+.4f}{'':>8}{pct:>6.1f}%{'':>8}{direction:<10}")

# ============================================================
# PART 4: FEATURE CORRELATION MATRIX
# ============================================================
print("\n" + "="*70)
print("PART 4: FEATURE CORRELATION MATRIX (Multicollinearity Check)")
print("="*70)

corr_matrix = train_df[feature_cols].corr()

# Find highly correlated feature pairs
high_corr_pairs = []
for i in range(len(feature_cols)):
    for j in range(i+1, len(feature_cols)):
        corr = corr_matrix.iloc[i, j]
        if abs(corr) > 0.3:  # Lower threshold to catch more
            high_corr_pairs.append((feature_cols[i], feature_cols[j], corr))

print("\nCorrelated feature pairs (|r| > 0.3):")
if high_corr_pairs:
    for f1, f2, corr in sorted(high_corr_pairs, key=lambda x: abs(x[2]), reverse=True):
        print(f"  {f1} <-> {f2}: r={corr:+.4f}")
else:
    print("  No highly correlated feature pairs found.")

# ============================================================
# PART 5: VISUALIZATIONS
# ============================================================
print("\n" + "="*70)
print("PART 5: CREATING VISUALIZATIONS")
print("="*70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 1. Bar plot of correlations with target
ax1 = axes[0, 0]
colors = ['green' if c > 0 else 'red' for c in correlations_raw.values]
bars = ax1.barh(range(len(correlations_raw)), correlations_raw.values, color=colors)
ax1.set_yticks(range(len(correlations_raw)))
ax1.set_yticklabels(correlations_raw.index)
ax1.set_xlabel('Correlation with Solar Power Output')
ax1.set_title('Feature Correlations with Target')
ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
for i, v in enumerate(correlations_raw.values):
    ax1.text(v + 0.01 if v >= 0 else v - 0.05, i, f'{v:.3f}', va='center', fontsize=9)

# 2. Correlation heatmap
ax2 = axes[0, 1]
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0, ax=ax2, 
            square=True, linewidths=0.5)
ax2.set_title('Feature Correlation Matrix')

# 3. Normalized data ranges comparison
ax3 = axes[0, 2]
norm_ranges = [(normalized_df[f].max() - normalized_df[f].min()) for f in feature_cols]
ax3.barh(range(len(feature_cols)), norm_ranges, color='steelblue')
ax3.set_yticks(range(len(feature_cols)))
ax3.set_yticklabels(feature_cols)
ax3.set_xlabel('Range after Normalization (std units)')
ax3.set_title('Normalized Feature Ranges')

# 4. Scatter plot of top correlated feature vs target
ax4 = axes[1, 0]
top_feature = abs_corr.index[0]
ax4.scatter(train_df[top_feature], train_df[target_col], alpha=0.3, s=5, c='blue')
ax4.set_xlabel(top_feature)
ax4.set_ylabel('Solar Power Output')
ax4.set_title(f'{top_feature} vs Solar Power (r={correlations_raw[top_feature]:.3f})')

# 5. Distribution of Solar Power Output by time of day
ax5 = axes[1, 1]
train_df['hour'] = (train_df['timeofday'] / 3600).astype(int)
hourly_mean = train_df.groupby('hour')[target_col].mean()
ax5.bar(hourly_mean.index, hourly_mean.values, color='orange')
ax5.set_xlabel('Hour of Day')
ax5.set_ylabel('Mean Solar Power Output')
ax5.set_title('Solar Power Output by Hour')
ax5.set_xticks(range(0, 24, 2))

# 6. Weather features vs Solar Power
ax6 = axes[1, 2]
weather_features = ['temp', 'humidity', 'cloudcover', 'pressure']
weather_corrs = [correlations_raw[f] for f in weather_features]
colors = ['green' if c > 0 else 'red' for c in weather_corrs]
ax6.barh(range(len(weather_features)), weather_corrs, color=colors)
ax6.set_yticks(range(len(weather_features)))
ax6.set_yticklabels(weather_features)
ax6.set_xlabel('Correlation with Solar Power')
ax6.set_title('Weather Features Correlation')
ax6.axvline(x=0, color='black', linestyle='-', linewidth=0.5)

plt.suptitle('Data Correlation Analysis - PatchXFormer Solar Forecasting', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig('./shap_results/data_correlation_analysis.png', dpi=300, bbox_inches='tight')
plt.savefig('./shap_results/data_correlation_analysis.pdf', format='pdf', bbox_inches='tight')
print(f"Visualization saved to: ./shap_results/data_correlation_analysis.png")

# ============================================================
# PART 6: VARIANCE ANALYSIS (Critical for Model Importance)
# ============================================================
print("\n" + "="*70)
print("PART 6: FEATURE VARIANCE ANALYSIS (Critical for Model Learning)")
print("="*70)

print("\nVariance in NORMALIZED data (what the model sees):")
print(f"{'Feature':<20}{'Norm Variance':<15}{'Norm Std':<15}{'Rank':<10}")
print("-"*70)
norm_var = normalized_df[feature_cols].var().sort_values(ascending=False)
for rank, (feat, var) in enumerate(norm_var.items(), 1):
    print(f"{feat:<20}{var:<15.4f}{np.sqrt(var):<15.4f}{rank:<10}")

# ============================================================
# PART 7: TEMPORAL VARIANCE CHECK
# ============================================================
print("\n" + "="*70)
print("PART 7: TEMPORAL VARIANCE CHECK (How much do features change over time?)")
print("="*70)

print("\nFeature changes (diff) statistics:")
print(f"{'Feature':<20}{'Mean |diff|':<15}{'Std of diff':<15}{'Norm. Variability':<20}")
print("-"*70)
temporal_var = {}
for feat in feature_cols:
    diff = train_df[feat].diff().abs()
    mean_diff = diff.mean()
    std_diff = diff.std()
    # Normalize by feature std
    norm_variability = std_diff / train_df[feat].std() if train_df[feat].std() > 0 else 0
    temporal_var[feat] = norm_variability
    print(f"{feat:<20}{mean_diff:<15.4f}{std_diff:<15.4f}{norm_variability:<20.4f}")

print("\nTemporal variability ranking (how much features change timestep-to-timestep):")
sorted_temporal = sorted(temporal_var.items(), key=lambda x: x[1], reverse=True)
for rank, (feat, var) in enumerate(sorted_temporal, 1):
    print(f"  {rank}. {feat:<15} temporal_variability={var:.4f}")

# ============================================================
# PART 8: PRESSURE CORRELATION CHECK
# ============================================================
print("\n" + "="*70)
print("PART 8: PRESSURE CORRELATIONS WITH OTHER FEATURES")
print("="*70)

print("\nPressure correlations with other features:")
pressure_corrs = {}
for feat in feature_cols:
    if feat != 'pressure':
        corr = train_df['pressure'].corr(train_df[feat])
        pressure_corrs[feat] = corr
        marker = " ***" if abs(corr) > 0.3 else ""
        print(f"  pressure <-> {feat:<15}: r={corr:+.4f}{marker}")

# ============================================================
# PART 9: SUMMARY AND COMPARISON
# ============================================================
print("\n" + "="*70)
print("PART 9: SUMMARY - DATA vs SHAP COMPARISON")
print("="*70)

print("""
CORRELATION-BASED EXPECTED IMPORTANCE:
  (Based on |correlation| with Solar Power Output)
""")
for rank, (feature, corr) in enumerate(abs_corr.items(), 1):
    pct = (corr / total_corr) * 100
    print(f"  {rank}. {feature:<15} |r|={corr:.4f}  ({pct:.1f}%)")

print("""
OBSERVED SHAP IMPORTANCE (from your model):
  1. pressure       50.96%
  2. timeofday      13.59%
  3. humidity       13.17%
  4. dayofyear       6.18%
  5. cloudcover      4.85%
  6. temp            4.23%
  7. winddir         3.89%
  8. windspeed       1.76%
  9. dew             1.37%
""")

print("""
KEY INSIGHTS:
============
1. AFTER NORMALIZATION: All features have similar scale (std~1.0)
   - This means pressure's high raw range doesn't directly cause high importance
   
2. CORRELATION vs MODEL IMPORTANCE:
   - Correlation measures LINEAR relationship with target
   - Model (Transformer) can capture NON-LINEAR and TEMPORAL patterns
   
3. POSSIBLE EXPLANATIONS FOR PRESSURE DOMINANCE:
   a) Pressure may have strong TEMPORAL patterns the model learns
   b) Pressure changes may precede weather changes (predictive power)
   c) Atmospheric pressure affects air density → affects irradiance scattering
   d) Model may have learned that pressure variations indicate weather fronts
   
4. WHY CORRELATION-BASED RANKING DIFFERS:
   - Correlation is point-to-point; model uses 96 timesteps (sequence)
   - Transformers excel at capturing dependencies over time
   - Pressure variations over TIME may be more informative than single values

5. NEXT STEPS TO INVESTIGATE:
   - Check if pressure correlates with cloudcover/humidity in transitions
   - Analyze if pressure predicts FUTURE solar output better than current
   - Try removing pressure and retrain to see impact on accuracy
""")

# Save summary to CSV
summary_df = pd.DataFrame({
    'Feature': feature_cols,
    'Raw_Correlation': [correlations_raw[f] for f in feature_cols],
    'Abs_Correlation': [abs(correlations_raw[f]) for f in feature_cols],
    'Normalized_Std': [normalized_df[f].std() for f in feature_cols],
    'Temporal_Variability': [temporal_var[f] for f in feature_cols]
})
summary_df = summary_df.sort_values('Abs_Correlation', ascending=False)
summary_df.to_csv('./shap_results/data_feature_analysis.csv', index=False)
print(f"\nSummary saved to: ./shap_results/data_feature_analysis.csv")
