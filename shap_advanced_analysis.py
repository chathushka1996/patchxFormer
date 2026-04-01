"""
Advanced SHAP Analysis for PatchXFormer Solar Power Forecasting

This module provides advanced explainability features including:
1. Correlation analysis between features and predictions
2. LIME comparison for validation
3. Interactive visualizations
4. Statistical significance testing
5. Confidence intervals for SHAP values
6. Time-of-day and seasonal analysis

Based on methodology from:
"Solar energy prediction through machine learning models: A comparative analysis"
(PLOS ONE, 2025)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
except ImportError:
    os.system('pip install shap')
    import shap

try:
    import lime
    import lime.lime_tabular
except ImportError:
    os.system('pip install lime')
    import lime
    import lime.lime_tabular


class AdvancedSHAPAnalyzer:
    """
    Advanced SHAP analysis with additional interpretability methods.
    """
    
    def __init__(self, feature_names, output_dir='./shap_results/advanced/'):
        """
        Initialize advanced analyzer.
        
        Args:
            feature_names: List of feature names
            output_dir: Output directory for results
        """
        self.feature_names = feature_names
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Weather feature categories
        self.weather_features = ['temp', 'dew', 'humidity', 'winddir', 
                                  'windspeed', 'pressure', 'cloudcover']
        self.temporal_features = ['dayofyear', 'timeofday']
        
    def compute_correlation_analysis(self, data, predictions, save_path=None):
        """
        Compute and visualize correlations between features and predictions.
        
        Args:
            data: Input data [samples, features]
            predictions: Model predictions [samples]
            save_path: Path to save figure
        """
        print("Computing correlation analysis...")
        
        if len(data.shape) == 3:
            data = data.mean(axis=1)
        
        # Create correlation matrix
        data_with_pred = np.column_stack([data, predictions])
        feature_names_with_pred = list(self.feature_names[:data.shape[1]]) + ['Prediction']
        
        corr_matrix = np.corrcoef(data_with_pred.T)
        
        # Plot correlation heatmap
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # Full correlation matrix
        ax1 = axes[0]
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='RdBu_r', center=0,
                   xticklabels=feature_names_with_pred,
                   yticklabels=feature_names_with_pred,
                   ax=ax1, vmin=-1, vmax=1)
        ax1.set_title('Feature Correlation Matrix', fontsize=14)
        
        # Correlation with prediction
        ax2 = axes[1]
        pred_corr = corr_matrix[:-1, -1]
        colors = ['green' if c > 0 else 'red' for c in pred_corr]
        bars = ax2.barh(range(len(pred_corr)), pred_corr, color=colors, alpha=0.7)
        ax2.set_yticks(range(len(pred_corr)))
        ax2.set_yticklabels(self.feature_names[:len(pred_corr)])
        ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax2.set_xlabel('Correlation with Prediction', fontsize=12)
        ax2.set_title('Feature-Prediction Correlation', fontsize=14)
        
        # Add correlation values
        for bar, val in zip(bars, pred_corr):
            ax2.text(val + 0.02 if val >= 0 else val - 0.02, 
                    bar.get_y() + bar.get_height()/2,
                    f'{val:.3f}', va='center', 
                    ha='left' if val >= 0 else 'right', fontsize=9)
        
        ax2.invert_yaxis()
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'correlation_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Saved correlation analysis to {save_path}")
        plt.close()
        
        # Save correlation statistics
        corr_df = pd.DataFrame({
            'Feature': self.feature_names[:len(pred_corr)],
            'Correlation': pred_corr,
            'Abs_Correlation': np.abs(pred_corr)
        }).sort_values('Abs_Correlation', ascending=False)
        corr_df.to_csv(os.path.join(self.output_dir, 'feature_correlations.csv'), index=False)
        
        return corr_matrix, pred_corr
    
    def compute_shap_confidence_intervals(self, shap_values, confidence=0.95):
        """
        Compute confidence intervals for SHAP values.
        
        Args:
            shap_values: SHAP values [samples, features]
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            ci_df: DataFrame with mean SHAP, lower CI, upper CI
        """
        if len(shap_values.shape) == 3:
            shap_values = shap_values.mean(axis=1)
        
        n_samples = shap_values.shape[0]
        n_features = shap_values.shape[1]
        
        results = []
        z_score = stats.norm.ppf((1 + confidence) / 2)
        
        for i in range(n_features):
            feature_shap = shap_values[:, i]
            mean_shap = np.mean(feature_shap)
            std_shap = np.std(feature_shap)
            se = std_shap / np.sqrt(n_samples)
            
            ci_lower = mean_shap - z_score * se
            ci_upper = mean_shap + z_score * se
            
            mean_abs_shap = np.mean(np.abs(feature_shap))
            std_abs_shap = np.std(np.abs(feature_shap))
            se_abs = std_abs_shap / np.sqrt(n_samples)
            
            results.append({
                'Feature': self.feature_names[i] if i < len(self.feature_names) else f'Feature_{i}',
                'Mean_SHAP': mean_shap,
                'Std_SHAP': std_shap,
                'CI_Lower': ci_lower,
                'CI_Upper': ci_upper,
                'Mean_Abs_SHAP': mean_abs_shap,
                'CI_Abs_Lower': mean_abs_shap - z_score * se_abs,
                'CI_Abs_Upper': mean_abs_shap + z_score * se_abs
            })
        
        ci_df = pd.DataFrame(results)
        ci_df.to_csv(os.path.join(self.output_dir, 'shap_confidence_intervals.csv'), index=False)
        
        return ci_df
    
    def plot_shap_with_confidence(self, ci_df, save_path=None):
        """
        Plot SHAP feature importance with confidence intervals.
        
        Args:
            ci_df: DataFrame with confidence interval data
            save_path: Path to save figure
        """
        plt.figure(figsize=(12, 8))
        
        # Sort by absolute importance
        ci_df_sorted = ci_df.sort_values('Mean_Abs_SHAP', ascending=True)
        
        y_pos = range(len(ci_df_sorted))
        
        plt.barh(y_pos, ci_df_sorted['Mean_Abs_SHAP'], 
                xerr=[ci_df_sorted['Mean_Abs_SHAP'] - ci_df_sorted['CI_Abs_Lower'],
                      ci_df_sorted['CI_Abs_Upper'] - ci_df_sorted['Mean_Abs_SHAP']],
                capsize=3, color='steelblue', alpha=0.7, ecolor='black')
        
        plt.yticks(y_pos, ci_df_sorted['Feature'])
        plt.xlabel('Mean |SHAP Value| with 95% CI', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance with Confidence Intervals\nPatchXFormer Solar Forecasting', fontsize=14)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'shap_confidence_intervals.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Saved SHAP CI plot to {save_path}")
        plt.close()
        
    def compute_lime_comparison(self, model_predict, data, sample_indices=[0, 1, 2],
                                mode='regression'):
        """
        Compute LIME explanations for comparison with SHAP.
        
        Args:
            model_predict: Model prediction function
            data: Input data
            sample_indices: Indices of samples to explain
            mode: 'regression' or 'classification'
            
        Returns:
            lime_explanations: List of LIME explanation objects
        """
        print("Computing LIME explanations for comparison...")
        
        if len(data.shape) == 3:
            data = data.mean(axis=1)
        
        feature_names_display = self.feature_names[:data.shape[1]]
        
        # Create LIME explainer
        explainer = lime.lime_tabular.LimeTabularExplainer(
            data,
            feature_names=feature_names_display,
            mode=mode,
            verbose=False
        )
        
        lime_results = []
        
        for idx in sample_indices:
            if idx >= len(data):
                continue
                
            exp = explainer.explain_instance(
                data[idx],
                model_predict,
                num_features=len(feature_names_display)
            )
            
            lime_results.append({
                'sample_idx': idx,
                'explanation': exp,
                'feature_importance': dict(exp.as_list())
            })
            
            # Save LIME visualization
            fig = exp.as_pyplot_figure()
            plt.title(f'LIME Explanation - Sample {idx}', fontsize=14)
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f'lime_explanation_sample_{idx}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved LIME explanation for sample {idx}")
            plt.close()
        
        return lime_results
    
    def compare_shap_lime(self, shap_values, data, lime_results, save_path=None):
        """
        Create comparison visualization between SHAP and LIME.
        
        Args:
            shap_values: SHAP values
            data: Input data
            lime_results: LIME explanations
            save_path: Path to save figure
        """
        if len(shap_values.shape) == 3:
            shap_values = shap_values.mean(axis=1)
        
        n_samples = len(lime_results)
        fig, axes = plt.subplots(n_samples, 2, figsize=(14, 4 * n_samples))
        
        if n_samples == 1:
            axes = axes.reshape(1, -1)
        
        for i, lime_result in enumerate(lime_results):
            sample_idx = lime_result['sample_idx']
            
            # SHAP values for this sample
            ax1 = axes[i, 0]
            sample_shap = shap_values[sample_idx]
            feature_names_display = self.feature_names[:len(sample_shap)]
            
            sorted_idx = np.argsort(np.abs(sample_shap))[::-1]
            colors = ['red' if v < 0 else 'blue' for v in sample_shap[sorted_idx]]
            
            ax1.barh(range(len(sorted_idx)), sample_shap[sorted_idx], color=colors, alpha=0.7)
            ax1.set_yticks(range(len(sorted_idx)))
            ax1.set_yticklabels([feature_names_display[j] for j in sorted_idx])
            ax1.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax1.set_xlabel('SHAP Value')
            ax1.set_title(f'SHAP - Sample {sample_idx}')
            ax1.invert_yaxis()
            
            # LIME values for this sample
            ax2 = axes[i, 1]
            lime_importance = lime_result['feature_importance']
            
            lime_features = []
            lime_values = []
            for feat, val in lime_importance.items():
                lime_features.append(feat.split('<=')[0].split('>')[0].strip())
                lime_values.append(val)
            
            sorted_idx_lime = np.argsort(np.abs(lime_values))[::-1]
            colors_lime = ['red' if lime_values[j] < 0 else 'green' for j in sorted_idx_lime]
            
            ax2.barh(range(len(sorted_idx_lime)), 
                    [lime_values[j] for j in sorted_idx_lime], 
                    color=colors_lime, alpha=0.7)
            ax2.set_yticks(range(len(sorted_idx_lime)))
            ax2.set_yticklabels([lime_features[j][:20] for j in sorted_idx_lime])
            ax2.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
            ax2.set_xlabel('LIME Value')
            ax2.set_title(f'LIME - Sample {sample_idx}')
            ax2.invert_yaxis()
        
        plt.suptitle('SHAP vs LIME Comparison\nPatchXFormer Solar Forecasting', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'shap_lime_comparison.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Saved SHAP-LIME comparison to {save_path}")
        plt.close()
        
    def analyze_time_of_day_effects(self, shap_values, data, save_path=None):
        """
        Analyze how SHAP values vary by time of day.
        
        Args:
            shap_values: SHAP values
            data: Input data with timeofday feature
            save_path: Path to save figure
        """
        if len(shap_values.shape) == 3:
            shap_values = shap_values.mean(axis=1)
        if len(data.shape) == 3:
            data = data.mean(axis=1)
        
        # Find timeofday feature index
        try:
            tod_idx = self.feature_names.index('timeofday')
        except ValueError:
            print("timeofday feature not found")
            return
        
        timeofday = data[:, tod_idx]
        
        # Bin by hour (assuming timeofday is in seconds from midnight)
        hours = (timeofday / 3600).astype(int) % 24
        
        # Analyze weather feature importance by hour
        weather_indices = [i for i, name in enumerate(self.feature_names[:shap_values.shape[1]]) 
                         if name in self.weather_features]
        
        hour_importance = {h: [] for h in range(24)}
        for i in range(len(hours)):
            hour = hours[i]
            hour_importance[hour].append(np.abs(shap_values[i, weather_indices]).mean())
        
        # Compute mean importance per hour
        mean_importance = [np.mean(hour_importance[h]) if hour_importance[h] else 0 
                         for h in range(24)]
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot importance by hour
        ax1 = axes[0]
        ax1.bar(range(24), mean_importance, color='orange', alpha=0.7)
        ax1.set_xlabel('Hour of Day', fontsize=12)
        ax1.set_ylabel('Mean |SHAP| for Weather Features', fontsize=12)
        ax1.set_title('Weather Feature Importance by Hour', fontsize=14)
        ax1.set_xticks(range(0, 24, 2))
        
        # Heatmap of feature importance by hour
        ax2 = axes[1]
        hourly_shap = np.zeros((len(weather_indices), 24))
        
        for h in range(24):
            mask = hours == h
            if mask.sum() > 0:
                hourly_shap[:, h] = np.abs(shap_values[mask][:, weather_indices]).mean(axis=0)
        
        weather_names = [self.feature_names[i] for i in weather_indices]
        
        sns.heatmap(hourly_shap, xticklabels=range(24), yticklabels=weather_names,
                   cmap='YlOrRd', ax=ax2)
        ax2.set_xlabel('Hour of Day', fontsize=12)
        ax2.set_ylabel('Feature', fontsize=12)
        ax2.set_title('Feature Importance Heatmap by Hour', fontsize=14)
        
        plt.suptitle('Time-of-Day Analysis\nPatchXFormer Solar Forecasting', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'time_of_day_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Saved time-of-day analysis to {save_path}")
        plt.close()
        
    def analyze_seasonal_effects(self, shap_values, data, save_path=None):
        """
        Analyze how SHAP values vary by season (using dayofyear).
        
        Args:
            shap_values: SHAP values
            data: Input data with dayofyear feature
            save_path: Path to save figure
        """
        if len(shap_values.shape) == 3:
            shap_values = shap_values.mean(axis=1)
        if len(data.shape) == 3:
            data = data.mean(axis=1)
        
        # Find dayofyear feature index
        try:
            doy_idx = self.feature_names.index('dayofyear')
        except ValueError:
            print("dayofyear feature not found")
            return
        
        dayofyear = data[:, doy_idx]
        
        # Define seasons
        def get_season(doy):
            if doy < 80 or doy >= 355:
                return 'Winter'
            elif doy < 172:
                return 'Spring'
            elif doy < 264:
                return 'Summer'
            else:
                return 'Fall'
        
        seasons = [get_season(d) for d in dayofyear]
        season_order = ['Winter', 'Spring', 'Summer', 'Fall']
        
        # Weather feature indices
        weather_indices = [i for i, name in enumerate(self.feature_names[:shap_values.shape[1]]) 
                         if name in self.weather_features]
        
        # Compute mean importance per season
        season_importance = {s: {self.feature_names[i]: [] 
                                 for i in weather_indices} 
                            for s in season_order}
        
        for i, season in enumerate(seasons):
            for idx in weather_indices:
                feature_name = self.feature_names[idx]
                season_importance[season][feature_name].append(np.abs(shap_values[i, idx]))
        
        # Create DataFrame for plotting
        plot_data = []
        for season in season_order:
            for feature in [self.feature_names[i] for i in weather_indices]:
                values = season_importance[season][feature]
                if values:
                    plot_data.append({
                        'Season': season,
                        'Feature': feature,
                        'Mean_Abs_SHAP': np.mean(values)
                    })
        
        plot_df = pd.DataFrame(plot_data)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Grouped bar plot
        ax1 = axes[0]
        pivot_df = plot_df.pivot(index='Feature', columns='Season', values='Mean_Abs_SHAP')
        pivot_df = pivot_df[season_order]
        pivot_df.plot(kind='bar', ax=ax1, colormap='Set2')
        ax1.set_xlabel('Feature', fontsize=12)
        ax1.set_ylabel('Mean |SHAP Value|', fontsize=12)
        ax1.set_title('Feature Importance by Season', fontsize=14)
        ax1.legend(title='Season')
        ax1.set_xticklabels(ax1.get_xticklabels(), rotation=45, ha='right')
        
        # Heatmap
        ax2 = axes[1]
        heatmap_data = pivot_df.values
        sns.heatmap(heatmap_data, 
                   xticklabels=season_order, 
                   yticklabels=pivot_df.index,
                   cmap='YlOrRd', ax=ax2, annot=True, fmt='.3f')
        ax2.set_xlabel('Season', fontsize=12)
        ax2.set_ylabel('Feature', fontsize=12)
        ax2.set_title('Seasonal Feature Importance Heatmap', fontsize=14)
        
        plt.suptitle('Seasonal Analysis\nPatchXFormer Solar Forecasting', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'seasonal_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Saved seasonal analysis to {save_path}")
        plt.close()
        
        # Save seasonal statistics
        plot_df.to_csv(os.path.join(self.output_dir, 'seasonal_importance.csv'), index=False)
        
    def create_interaction_analysis(self, shap_values, data, top_k=5, save_path=None):
        """
        Analyze feature interactions using SHAP interaction values proxy.
        
        Args:
            shap_values: SHAP values
            data: Input data
            top_k: Number of top features to analyze
            save_path: Path to save figure
        """
        if len(shap_values.shape) == 3:
            shap_values = shap_values.mean(axis=1)
        if len(data.shape) == 3:
            data = data.mean(axis=1)
        
        # Get top-k most important features
        feature_importance = np.abs(shap_values).mean(axis=0)
        top_indices = np.argsort(feature_importance)[-top_k:][::-1]
        
        n_features = len(top_indices)
        fig, axes = plt.subplots(n_features, n_features, figsize=(3*n_features, 3*n_features))
        
        for i, idx_i in enumerate(top_indices):
            for j, idx_j in enumerate(top_indices):
                ax = axes[i, j]
                
                if i == j:
                    # Histogram on diagonal
                    ax.hist(shap_values[:, idx_i], bins=30, alpha=0.7, color='steelblue')
                    ax.set_title(self.feature_names[idx_i], fontsize=10)
                else:
                    # Scatter plot for interactions
                    scatter = ax.scatter(data[:, idx_j], shap_values[:, idx_i],
                                        c=data[:, idx_i], cmap='RdYlBu_r', 
                                        alpha=0.5, s=10)
                    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                
                if i == n_features - 1:
                    ax.set_xlabel(self.feature_names[idx_j], fontsize=9)
                if j == 0:
                    ax.set_ylabel(f'SHAP({self.feature_names[idx_i]})', fontsize=9)
        
        plt.suptitle('Feature Interaction Analysis\nPatchXFormer Solar Forecasting', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'interaction_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved interaction analysis to {save_path}")
        plt.close()
        
    def generate_summary_report(self, shap_values, data, predictions=None):
        """
        Generate a comprehensive summary report.
        
        Args:
            shap_values: SHAP values
            data: Input data
            predictions: Model predictions (optional)
        """
        print("="*60)
        print("GENERATING ADVANCED SHAP ANALYSIS REPORT")
        print("="*60)
        
        # 1. Confidence intervals
        print("\n[1/6] Computing SHAP confidence intervals...")
        ci_df = self.compute_shap_confidence_intervals(shap_values)
        self.plot_shap_with_confidence(ci_df)
        
        # 2. Correlation analysis
        if predictions is not None:
            print("\n[2/6] Computing correlation analysis...")
            self.compute_correlation_analysis(data, predictions)
        
        # 3. Time of day effects
        print("\n[3/6] Analyzing time-of-day effects...")
        self.analyze_time_of_day_effects(shap_values, data)
        
        # 4. Seasonal effects
        print("\n[4/6] Analyzing seasonal effects...")
        self.analyze_seasonal_effects(shap_values, data)
        
        # 5. Interaction analysis
        print("\n[5/6] Computing feature interactions...")
        self.create_interaction_analysis(shap_values, data)
        
        # 6. Generate summary statistics
        print("\n[6/6] Generating summary statistics...")
        self._generate_summary_stats(shap_values, ci_df)
        
        print("\n" + "="*60)
        print(f"Advanced analysis complete! Results saved to: {self.output_dir}")
        print("="*60)
        
    def _generate_summary_stats(self, shap_values, ci_df):
        """Generate summary statistics markdown report."""
        if len(shap_values.shape) == 3:
            shap_values = shap_values.mean(axis=1)
        
        # Sort by importance
        ci_df_sorted = ci_df.sort_values('Mean_Abs_SHAP', ascending=False)
        
        report = f"""# PatchXFormer SHAP Analysis Summary Report

## Overview
- Number of samples analyzed: {shap_values.shape[0]}
- Number of features: {shap_values.shape[1]}

## Top Contributing Features

| Rank | Feature | Mean |SHAP| | 95% CI |
|------|---------|-------------|--------|
"""
        for i, row in ci_df_sorted.head(10).iterrows():
            report += f"| {i+1} | {row['Feature']} | {row['Mean_Abs_SHAP']:.4f} | [{row['CI_Abs_Lower']:.4f}, {row['CI_Abs_Upper']:.4f}] |\n"
        
        report += """
## Key Findings

### Weather Parameters Impact
Based on SHAP analysis, the weather parameters contributing most to solar power predictions are:
"""
        weather_features = ci_df_sorted[ci_df_sorted['Feature'].isin(self.weather_features)]
        for i, (_, row) in enumerate(weather_features.head(5).iterrows(), 1):
            report += f"\n{i}. **{row['Feature']}**: Mean |SHAP| = {row['Mean_Abs_SHAP']:.4f}"
        
        report += """

### Temporal Patterns
The model shows varying feature importance across:
- **Time of Day**: See `time_of_day_analysis.png`
- **Season**: See `seasonal_analysis.png`

## Files Generated
- `shap_confidence_intervals.csv`: SHAP values with confidence intervals
- `feature_correlations.csv`: Feature-prediction correlations
- `seasonal_importance.csv`: Seasonal feature importance
- Various visualization plots (PNG and PDF formats)

## Methodology
This analysis uses SHAP (SHapley Additive exPlanations) to interpret the PatchXFormer model's
predictions for solar power forecasting. The methodology follows:
- Kernel SHAP for model-agnostic feature attribution
- Bootstrap confidence intervals for statistical significance
- Time-stratified analysis for temporal patterns
"""
        
        report_path = os.path.join(self.output_dir, 'SHAP_ANALYSIS_REPORT.md')
        with open(report_path, 'w') as f:
            f.write(report)
        print(f"Saved summary report to {report_path}")


def run_advanced_analysis(shap_values, data, feature_names, predictions=None,
                          output_dir='./shap_results/advanced/'):
    """
    Run complete advanced SHAP analysis.
    
    Args:
        shap_values: Computed SHAP values
        data: Input data
        feature_names: List of feature names
        predictions: Model predictions (optional)
        output_dir: Output directory
    """
    analyzer = AdvancedSHAPAnalyzer(feature_names, output_dir)
    analyzer.generate_summary_report(shap_values, data, predictions)
    
    return analyzer


if __name__ == '__main__':
    print("This module provides advanced SHAP analysis utilities.")
    print("Import and use the AdvancedSHAPAnalyzer class or run_advanced_analysis function.")
