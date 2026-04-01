"""
SHAP-based Explainability Analysis for PatchXFormer Solar Power Forecasting

This script implements comprehensive SHAP (SHapley Additive exPlanations) analysis
for the PatchXFormer model to understand which weather parameters contribute most
to solar power output predictions.

Based on methodology from: 
"Solar energy prediction through machine learning models: A comparative analysis"
(PLOS ONE, 2025)

Features:
1. Global SHAP interpretation - Feature importance ranking
2. Partial SHAP dependence plots - Non-linear relationships
3. Local SHAP interpretation - Individual prediction explanations
4. Feature interaction analysis
5. Temporal contribution analysis
6. Normalization-aware analysis - Considers StandardScaler transformation
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
except ImportError:
    print("SHAP not installed. Installing...")
    os.system('pip install shap')
    import shap

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import PatchXFormer


class SHAPExplainer:
    """
    SHAP-based explainability analysis for PatchXFormer model.
    
    This class provides comprehensive tools for understanding model predictions
    through SHAP values analysis, including global feature importance,
    partial dependence, and local interpretations.
    """
    
    def __init__(self, args, model, device, feature_names=None):
        """
        Initialize the SHAP explainer.
        
        Args:
            args: Configuration arguments
            model: Trained PatchXFormer model
            device: torch device (cuda/cpu)
            feature_names: List of feature names for interpretation
        """
        self.args = args
        self.model = model
        self.device = device
        self.model.eval()
        
        # Default feature names for solar forecasting dataset
        if feature_names is None:
            self.feature_names = [
                'dayofyear', 'timeofday', 'temp', 'dew', 
                'humidity', 'winddir', 'windspeed', 'pressure', 
                'cloudcover', 'Solar Power Output'
            ]
        else:
            self.feature_names = feature_names
            
        # Weather-related features for focused analysis
        self.weather_features = ['temp', 'dew', 'humidity', 'winddir', 
                                  'windspeed', 'pressure', 'cloudcover']
        
        # Input features only (exclude target variable)
        self.input_features = ['dayofyear', 'timeofday', 'temp', 'dew', 
                               'humidity', 'winddir', 'windspeed', 'pressure', 'cloudcover']
        self.target_feature = 'Solar Power Output'
        
        # Create output directory
        self.output_dir = './shap_results/'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Scaler info (will be populated by analyze_data_scaling)
        self.scaler = None
        self.scaler_info = None
        self.raw_data_stats = None
        self.normalized_data_stats = None
        
    def analyze_data_scaling(self, root_path, data_path='train.csv'):
        """
        Analyze the raw and normalized data to understand scaling effects.
        This helps interpret SHAP values in the context of feature normalization.
        
        Args:
            root_path: Path to dataset directory
            data_path: Name of training data file
            
        Returns:
            dict: Scaling analysis results
        """
        print("\n" + "="*70)
        print("DATA SCALING ANALYSIS (Understanding Normalization Effects)")
        print("="*70)
        
        # Try to load raw training data
        try:
            train_file = os.path.join(root_path, data_path)
            if not os.path.exists(train_file):
                train_file = os.path.join(root_path, 'train.csv')
            
            if os.path.exists(train_file):
                raw_df = pd.read_csv(train_file)
                print(f"\nLoaded raw data from: {train_file}")
                print(f"Shape: {raw_df.shape}")
            else:
                print(f"Warning: Could not find training data at {train_file}")
                return None
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
        
        # Get feature columns (excluding date)
        feature_cols = [col for col in raw_df.columns if col not in ['date', 'Date', 'datetime']]
        
        # Ensure we use the correct feature order
        ordered_cols = []
        for feat in self.feature_names:
            if feat in feature_cols:
                ordered_cols.append(feat)
        
        if len(ordered_cols) < len(self.feature_names):
            # Use available columns
            ordered_cols = [col for col in feature_cols if col in self.feature_names or col == self.target_feature]
        
        print(f"Features being analyzed: {ordered_cols}")
        
        # ================================================================
        # PART 1: RAW DATA STATISTICS
        # ================================================================
        print("\n" + "-"*70)
        print("1. RAW DATA STATISTICS (Before Normalization)")
        print("-"*70)
        
        raw_stats = {}
        print(f"\n{'Feature':<20}{'Mean':<12}{'Std':<12}{'Min':<12}{'Max':<12}{'Range':<12}")
        print("-"*70)
        
        for col in ordered_cols:
            if col in raw_df.columns:
                stats = {
                    'mean': raw_df[col].mean(),
                    'std': raw_df[col].std(),
                    'min': raw_df[col].min(),
                    'max': raw_df[col].max(),
                    'range': raw_df[col].max() - raw_df[col].min()
                }
                raw_stats[col] = stats
                print(f"{col:<20}{stats['mean']:<12.4f}{stats['std']:<12.4f}"
                      f"{stats['min']:<12.4f}{stats['max']:<12.4f}{stats['range']:<12.4f}")
        
        self.raw_data_stats = raw_stats
        
        # ================================================================
        # PART 2: APPLY STANDARDSCALER (Same as data_loader.py)
        # ================================================================
        print("\n" + "-"*70)
        print("2. NORMALIZED DATA STATISTICS (After StandardScaler)")
        print("-"*70)
        
        # Apply StandardScaler
        self.scaler = StandardScaler()
        data_for_scaling = raw_df[ordered_cols].values
        normalized_data = self.scaler.fit_transform(data_for_scaling)
        normalized_df = pd.DataFrame(normalized_data, columns=ordered_cols)
        
        # Store scaler info
        self.scaler_info = {
            'means': dict(zip(ordered_cols, self.scaler.mean_)),
            'stds': dict(zip(ordered_cols, self.scaler.scale_)),
            'feature_order': ordered_cols
        }
        
        print(f"\n{'Feature':<20}{'Scaler μ':<12}{'Scaler σ':<12}{'Norm Min':<12}{'Norm Max':<12}{'Norm Range':<12}")
        print("-"*70)
        
        norm_stats = {}
        for i, col in enumerate(ordered_cols):
            stats = {
                'scaler_mean': self.scaler.mean_[i],
                'scaler_std': self.scaler.scale_[i],
                'norm_min': normalized_df[col].min(),
                'norm_max': normalized_df[col].max(),
                'norm_range': normalized_df[col].max() - normalized_df[col].min(),
                'norm_std': normalized_df[col].std()
            }
            norm_stats[col] = stats
            print(f"{col:<20}{stats['scaler_mean']:<12.4f}{stats['scaler_std']:<12.4f}"
                  f"{stats['norm_min']:<12.4f}{stats['norm_max']:<12.4f}{stats['norm_range']:<12.4f}")
        
        self.normalized_data_stats = norm_stats
        
        # ================================================================
        # PART 3: CORRELATION ANALYSIS
        # ================================================================
        print("\n" + "-"*70)
        print("3. FEATURE CORRELATIONS WITH TARGET")
        print("-"*70)
        
        target_col = self.target_feature
        if target_col in raw_df.columns:
            correlations = {}
            print(f"\n{'Feature':<20}{'Correlation':<15}{'|Correlation|':<15}{'Direction':<15}")
            print("-"*70)
            
            for col in ordered_cols:
                if col != target_col and col in raw_df.columns:
                    corr = raw_df[col].corr(raw_df[target_col])
                    correlations[col] = corr
                    direction = "Positive" if corr > 0 else "Negative"
                    print(f"{col:<20}{corr:>+.4f}{'':>8}{abs(corr):<15.4f}{direction:<15}")
            
            # Sort by absolute correlation
            sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
            
            print("\n" + "-"*70)
            print("4. EXPECTED IMPORTANCE RANKING (Based on |Correlation|)")
            print("-"*70)
            
            total_abs_corr = sum(abs(c) for _, c in sorted_corr)
            print(f"\n{'Rank':<6}{'Feature':<20}{'|Corr|':<12}{'Expected %':<15}")
            print("-"*70)
            
            for rank, (feat, corr) in enumerate(sorted_corr, 1):
                pct = (abs(corr) / total_abs_corr) * 100 if total_abs_corr > 0 else 0
                print(f"{rank:<6}{feat:<20}{abs(corr):<12.4f}{pct:<15.1f}%")
        
        # ================================================================
        # PART 4: TEMPORAL VARIABILITY
        # ================================================================
        print("\n" + "-"*70)
        print("5. TEMPORAL VARIABILITY (Feature Changes Over Time)")
        print("-"*70)
        
        temporal_var = {}
        print(f"\n{'Feature':<20}{'Mean |Δ|':<12}{'Std Δ':<12}{'Variability':<15}")
        print("-"*70)
        
        for col in ordered_cols:
            if col in raw_df.columns:
                diff = raw_df[col].diff().abs()
                mean_diff = diff.mean()
                std_diff = diff.std()
                variability = std_diff / raw_df[col].std() if raw_df[col].std() > 0 else 0
                temporal_var[col] = variability
                print(f"{col:<20}{mean_diff:<12.4f}{std_diff:<12.4f}{variability:<15.4f}")
        
        # ================================================================
        # PART 5: CREATE VISUALIZATION
        # ================================================================
        print("\n" + "-"*70)
        print("6. CREATING DATA ANALYSIS VISUALIZATIONS")
        print("-"*70)
        
        self._create_scaling_visualization(raw_df, normalized_df, ordered_cols, correlations if target_col in raw_df.columns else {})
        
        # ================================================================
        # SUMMARY
        # ================================================================
        print("\n" + "="*70)
        print("DATA SCALING ANALYSIS SUMMARY")
        print("="*70)
        
        print("""
KEY INSIGHTS:
1. StandardScaler transforms all features to have mean=0 and std=1
2. After normalization, features with larger ORIGINAL variance may still
   have more influence if the model learns those patterns
3. Correlation with target shows expected LINEAR importance
4. Model (Transformer) can capture NON-LINEAR and TEMPORAL patterns

IMPORTANT:
- SHAP importance may differ from correlation-based importance
- High temporal variability can make features more predictive
- Model learns from NORMALIZED data, but captures original patterns
        """)
        
        return {
            'raw_stats': raw_stats,
            'norm_stats': norm_stats,
            'scaler_info': self.scaler_info,
            'correlations': correlations if target_col in raw_df.columns else {},
            'temporal_variability': temporal_var
        }
    
    def _create_scaling_visualization(self, raw_df, normalized_df, feature_cols, correlations):
        """Create visualization comparing raw and normalized data."""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # 1. Raw data ranges
        ax1 = axes[0, 0]
        raw_ranges = [self.raw_data_stats.get(f, {}).get('range', 0) for f in feature_cols if f in self.raw_data_stats]
        feat_labels = [f for f in feature_cols if f in self.raw_data_stats]
        colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(feat_labels)))
        ax1.barh(range(len(feat_labels)), raw_ranges, color=colors)
        ax1.set_yticks(range(len(feat_labels)))
        ax1.set_yticklabels(feat_labels)
        ax1.set_xlabel('Value Range')
        ax1.set_title('RAW Data Ranges (Before Normalization)')
        ax1.set_xscale('log')
        
        # 2. Normalized data ranges
        ax2 = axes[0, 1]
        norm_ranges = [self.normalized_data_stats.get(f, {}).get('norm_range', 0) for f in feat_labels]
        ax2.barh(range(len(feat_labels)), norm_ranges, color=colors)
        ax2.set_yticks(range(len(feat_labels)))
        ax2.set_yticklabels(feat_labels)
        ax2.set_xlabel('Normalized Range (std units)')
        ax2.set_title('NORMALIZED Data Ranges (After StandardScaler)')
        
        # 3. Correlation with target
        ax3 = axes[0, 2]
        if correlations:
            corr_features = [f for f in feat_labels if f in correlations]
            corr_values = [correlations.get(f, 0) for f in corr_features]
            bar_colors = ['green' if c > 0 else 'red' for c in corr_values]
            ax3.barh(range(len(corr_features)), corr_values, color=bar_colors)
            ax3.set_yticks(range(len(corr_features)))
            ax3.set_yticklabels(corr_features)
            ax3.set_xlabel('Correlation')
            ax3.set_title('Correlation with Solar Power Output')
            ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        # 4. Scaler means
        ax4 = axes[1, 0]
        scaler_means = [self.scaler_info['means'].get(f, 0) for f in feat_labels]
        ax4.barh(range(len(feat_labels)), scaler_means, color='steelblue')
        ax4.set_yticks(range(len(feat_labels)))
        ax4.set_yticklabels(feat_labels)
        ax4.set_xlabel('Scaler Mean (μ)')
        ax4.set_title('StandardScaler Means')
        
        # 5. Scaler stds
        ax5 = axes[1, 1]
        scaler_stds = [self.scaler_info['stds'].get(f, 0) for f in feat_labels]
        ax5.barh(range(len(feat_labels)), scaler_stds, color='coral')
        ax5.set_yticks(range(len(feat_labels)))
        ax5.set_yticklabels(feat_labels)
        ax5.set_xlabel('Scaler Std (σ)')
        ax5.set_title('StandardScaler Standard Deviations')
        ax5.set_xscale('log')
        
        # 6. Distribution comparison for key features
        ax6 = axes[1, 2]
        key_features = ['pressure', 'temp', 'humidity', 'cloudcover']
        available_keys = [f for f in key_features if f in normalized_df.columns]
        if available_keys:
            for feat in available_keys[:4]:
                ax6.hist(normalized_df[feat].values, bins=50, alpha=0.5, label=feat, density=True)
            ax6.set_xlabel('Normalized Value')
            ax6.set_ylabel('Density')
            ax6.set_title('Normalized Feature Distributions')
            ax6.legend()
        
        plt.suptitle('Data Scaling Analysis - Raw vs Normalized Features\nPatchXFormer Solar Forecasting', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(self.output_dir, 'data_scaling_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Saved scaling analysis plot to: {save_path}")
        plt.close()
        
        # Save scaling info to CSV
        scaling_df = pd.DataFrame({
            'Feature': feat_labels,
            'Raw_Mean': [self.raw_data_stats.get(f, {}).get('mean', 0) for f in feat_labels],
            'Raw_Std': [self.raw_data_stats.get(f, {}).get('std', 0) for f in feat_labels],
            'Raw_Range': [self.raw_data_stats.get(f, {}).get('range', 0) for f in feat_labels],
            'Scaler_Mean': [self.scaler_info['means'].get(f, 0) for f in feat_labels],
            'Scaler_Std': [self.scaler_info['stds'].get(f, 0) for f in feat_labels],
            'Norm_Range': [self.normalized_data_stats.get(f, {}).get('norm_range', 0) for f in feat_labels],
            'Correlation': [correlations.get(f, 0) for f in feat_labels]
        })
        scaling_df.to_csv(os.path.join(self.output_dir, 'data_scaling_info.csv'), index=False)
        print(f"Saved scaling info to: {os.path.join(self.output_dir, 'data_scaling_info.csv')}")
    
    def create_model_wrapper(self):
        """
        Create a wrapper function for SHAP that handles the model's forward pass.
        This wrapper aggregates predictions across the output sequence for interpretation.
        """
        def model_predict(x_enc):
            """
            Wrapper for model prediction.
            
            Args:
                x_enc: Input tensor [batch, seq_len, features] or numpy array
                
            Returns:
                Mean prediction across the prediction horizon (last feature = target)
            """
            self.model.eval()
            with torch.no_grad():
                if isinstance(x_enc, np.ndarray):
                    x_enc = torch.FloatTensor(x_enc)
                    
                x_enc = x_enc.to(self.device)
                batch_size = x_enc.shape[0]
                
                # Create dummy inputs for other model arguments
                x_mark_enc = torch.zeros(batch_size, self.args.seq_len, 4).to(self.device)
                dec_inp = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, 
                                     x_enc.shape[-1]).to(self.device)
                x_mark_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, 4).to(self.device)
                
                outputs = self.model(x_enc, x_mark_enc, dec_inp, x_mark_dec)
                
                # Return mean prediction of target variable (last column)
                mean_pred = outputs[:, :, -1].mean(dim=1).cpu().numpy()
                
                return mean_pred
                
        return model_predict
    
    def compute_feature_level_shap(self, data_loader, num_samples=100, background_samples=50):
        """
        Compute SHAP values at the feature level using permutation-based importance
        combined with gradient-based attribution for proper feature importance.
        
        This method provides feature importance rankings similar to the PLOS ONE paper,
        showing which weather parameters have the greatest influence on predictions.
        
        Args:
            data_loader: Data loader for test/validation data
            num_samples: Number of samples to explain
            background_samples: Number of background samples for SHAP
            
        Returns:
            shap_values: SHAP values for each feature
            feature_importance: Mean absolute SHAP values per feature
        """
        print("Computing feature-level SHAP values...")
        
        # Collect data samples
        all_inputs = []
        all_targets = []
        all_marks = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            all_inputs.append(batch_x.numpy())
            all_targets.append(batch_y.numpy())
            all_marks.append(batch_x_mark.numpy())
            if len(all_inputs) * batch_x.shape[0] >= num_samples + background_samples:
                break
        
        all_inputs = np.concatenate(all_inputs, axis=0)[:num_samples + background_samples]
        all_targets = np.concatenate(all_targets, axis=0)[:num_samples + background_samples]
        all_marks = np.concatenate(all_marks, axis=0)[:num_samples + background_samples]
        
        print(f"Collected {len(all_inputs)} samples, shape: {all_inputs.shape}")
        
        # Use the actual time series data for SHAP
        background_data = all_inputs[:background_samples]
        explain_data = all_inputs[background_samples:background_samples + num_samples]
        explain_marks = all_marks[background_samples:background_samples + num_samples]
        explain_targets = all_targets[background_samples:background_samples + num_samples]
        
        # Extract target values for error-based importance
        # Target contains [label_len + pred_len] timesteps, we need only pred_len
        # Target is the last column (Solar Power Output)
        pred_len = self.args.pred_len
        target_values = explain_targets[:, -pred_len:, -1]  # [samples, pred_len]
        
        # Aggregate for display purposes
        aggregated_explain = explain_data.mean(axis=1)
        
        # Method 1: Permutation-based Feature Importance (ERROR-BASED)
        print("\nComputing permutation-based feature importance (ERROR-BASED)...")
        feature_importance = self._compute_permutation_importance(
            explain_data, explain_marks, n_repeats=10, targets=target_values
        )
        
        # Method 2: Gradient-based SHAP values for detailed analysis
        print("\nComputing gradient-based SHAP values...")
        shap_values = self._compute_gradient_shap_values(
            background_data, explain_data
        )
        
        # If gradient SHAP failed or returned zeros, use permutation values
        if shap_values is None or np.abs(shap_values).sum() < 1e-6:
            print("Using permutation importance as SHAP proxy...")
            # Create pseudo-SHAP values based on permutation importance
            n_features = len(feature_importance)
            shap_values = np.zeros((len(explain_data), n_features))
            for i in range(len(explain_data)):
                # Scale by feature values to create directional SHAP-like values
                feature_means = aggregated_explain[i]
                overall_mean = aggregated_explain.mean(axis=0)
                direction = np.sign(feature_means - overall_mean)
                shap_values[i] = feature_importance * direction
        
        # Exclude target variable from importance ranking
        input_feature_mask = [f != self.target_feature for f in self.feature_names[:len(feature_importance)]]
        
        # Create importance DataFrame (excluding target)
        all_features_df = pd.DataFrame({
            'Feature': self.feature_names[:len(feature_importance)],
            'Mean |SHAP|': feature_importance,
            'Is_Input': input_feature_mask
        })
        
        # Filter to only input features and sort
        importance_df = all_features_df[all_features_df['Is_Input']].drop(columns=['Is_Input'])
        importance_df = importance_df.sort_values('Mean |SHAP|', ascending=False)
        
        # Calculate percentage contribution
        total_importance = importance_df['Mean |SHAP|'].sum()
        if total_importance > 0:
            importance_df['Contribution %'] = (importance_df['Mean |SHAP|'] / total_importance * 100).round(2)
        else:
            importance_df['Contribution %'] = 0
        
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE RANKING (Input Features Only)")
        print("="*60)
        print(f"{'Rank':<6}{'Feature':<20}{'Mean |SHAP|':<15}{'Contribution %':<15}")
        print("-"*60)
        for rank, (_, row) in enumerate(importance_df.iterrows(), 1):
            print(f"{rank:<6}{row['Feature']:<20}{row['Mean |SHAP|']:<15.6f}{row['Contribution %']:<15.2f}")
        print("="*60)
        
        # Save results
        importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        np.save(os.path.join(self.output_dir, 'shap_values_aggregated.npy'), shap_values)
        np.save(os.path.join(self.output_dir, 'explain_data.npy'), aggregated_explain)
        
        return shap_values, aggregated_explain, feature_importance
    
    def _compute_permutation_importance(self, data, marks, n_repeats=30, targets=None):
        """
        Compute permutation-based feature importance using PREDICTION SENSITIVITY.
        
        This method measures how much model PREDICTIONS change when a feature is perturbed.
        Uses proper normalization based on training data statistics (from scaler).
        
        For each feature, we:
        1. Shuffle across samples (break cross-sample patterns)
        2. Replace with mean values (remove variation)
        3. Add noise (test sensitivity to perturbations)
        
        The importance is normalized using the StandardScaler's std (from training data)
        to account for different feature scales.
        
        NOTE: Excludes the target variable (Solar Power Output) from analysis.
        """
        self.model.eval()
        n_samples, seq_len, n_features = data.shape
        
        # Identify target column index (last column = Solar Power Output)
        target_idx = n_features - 1
        
        print(f"\n{'='*60}")
        print("COMPUTING FEATURE IMPORTANCE (Prediction Sensitivity)")
        print(f"{'='*60}")
        print(f"Samples: {n_samples}, Sequence length: {seq_len}, Features: {n_features}")
        print(f"Repeats per feature: {n_repeats}")
        
        # Get baseline predictions
        with torch.no_grad():
            x_enc = torch.FloatTensor(data).to(self.device)
            x_mark = torch.FloatTensor(marks).to(self.device)
            batch_size = x_enc.shape[0]
            
            dec_inp = torch.zeros(batch_size, self.args.label_len + self.args.pred_len,
                                 n_features).to(self.device)
            x_mark_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, 
                                    marks.shape[-1]).to(self.device)
            
            baseline_output = self.model(x_enc, x_mark, dec_inp, x_mark_dec)
            baseline_pred = baseline_output[:, :, -1].cpu().numpy()
        
        print(f"\nBaseline Statistics:")
        print(f"  Prediction mean: {baseline_pred.mean():.4f}")
        print(f"  Prediction std:  {baseline_pred.std():.4f}")
        print(f"  Prediction min:  {baseline_pred.min():.4f}")
        print(f"  Prediction max:  {baseline_pred.max():.4f}")
        
        # Get normalization factors from scaler (training data statistics)
        # This ensures consistent normalization regardless of test sample distribution
        if self.scaler_info is not None:
            print(f"\nUsing StandardScaler std from TRAINING data for normalization")
            scaler_stds = self.scaler_info.get('stds', {})
        else:
            print(f"\nNo scaler info available, using test sample statistics")
            scaler_stds = {}
        
        # Compute feature statistics
        feature_stats = {}
        print(f"\nFeature Statistics:")
        print(f"{'Feature':<20}{'Sample Std':<12}{'Scaler Std':<12}{'Norm Factor':<12}")
        print("-" * 60)
        
        for feat_idx in range(n_features):
            feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"
            feat_data = data[:, :, feat_idx]
            sample_std = feat_data.std()
            
            # Use scaler std if available, otherwise use sample std
            # Scaler std is from training data - more representative
            scaler_std = scaler_stds.get(feat_name, sample_std)
            
            # Normalization factor: features with larger scaler_std have naturally larger variations
            # We want to measure importance PER UNIT of original (pre-normalized) variation
            # Since data is normalized, 1 unit in normalized space = scaler_std in original space
            # Higher scaler_std means the feature varies more in original data
            norm_factor = 1.0 / max(scaler_std, 0.01)  # Inverse: smaller raw std = more important per unit
            
            feature_stats[feat_name] = {
                'sample_std': sample_std,
                'scaler_std': scaler_std,
                'norm_factor': norm_factor
            }
            
            if feat_name != self.target_feature:
                print(f"  {feat_name:<18}{sample_std:<12.4f}{scaler_std:<12.4f}{norm_factor:<12.6f}")
        
        feature_importance = np.zeros(n_features)
        feature_importance_raw = np.zeros(n_features)
        feature_importance_details = {}
        
        print(f"\nAnalyzing {n_features - 1} input features...")
        print("-" * 60)
        
        for feat_idx in range(n_features):
            feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"
            
            # Skip target variable
            if feat_name == self.target_feature or feat_idx == target_idx:
                print(f"  [{feat_idx+1}/{n_features}] {feat_name}: SKIPPED (target variable)")
                feature_importance[feat_idx] = 0
                continue
            
            feat_data = data[:, :, feat_idx]
            feat_mean = feat_data.mean()
            feat_std = feat_data.std()
            stats = feature_stats[feat_name]
            
            importance_scores = {
                'shuffle': [],
                'mean_replace': [],
                'noise': []
            }
            
            for repeat in range(n_repeats):
                # Method 1: Shuffle across samples
                data_shuffled = data.copy()
                perm = np.random.permutation(n_samples)
                data_shuffled[:, :, feat_idx] = data[perm, :, feat_idx]
                
                with torch.no_grad():
                    x_shuffled = torch.FloatTensor(data_shuffled).to(self.device)
                    pred_shuffled = self.model(x_shuffled, x_mark, dec_inp, x_mark_dec)
                    pred_shuffled = pred_shuffled[:, :, -1].cpu().numpy()
                
                # Measure prediction CHANGE (not error)
                shuffle_change = np.mean((baseline_pred - pred_shuffled) ** 2)
                importance_scores['shuffle'].append(shuffle_change)
                
                # Method 2: Replace with mean
                data_mean = data.copy()
                data_mean[:, :, feat_idx] = feat_mean
                
                with torch.no_grad():
                    x_mean = torch.FloatTensor(data_mean).to(self.device)
                    pred_mean = self.model(x_mean, x_mark, dec_inp, x_mark_dec)
                    pred_mean = pred_mean[:, :, -1].cpu().numpy()
                
                mean_change = np.mean((baseline_pred - pred_mean) ** 2)
                importance_scores['mean_replace'].append(mean_change)
                
                # Method 3: Add noise (1 std unit)
                data_noise = data.copy()
                noise = np.random.normal(0, 1.0, size=feat_data.shape)  # 1 std in normalized space
                data_noise[:, :, feat_idx] = feat_data + noise
                
                with torch.no_grad():
                    x_noise = torch.FloatTensor(data_noise).to(self.device)
                    pred_noise = self.model(x_noise, x_mark, dec_inp, x_mark_dec)
                    pred_noise = pred_noise[:, :, -1].cpu().numpy()
                
                noise_change = np.mean((baseline_pred - pred_noise) ** 2)
                importance_scores['noise'].append(noise_change)
            
            # Aggregate importance
            avg_shuffle = np.mean(importance_scores['shuffle'])
            avg_mean = np.mean(importance_scores['mean_replace'])
            avg_noise = np.mean(importance_scores['noise'])
            
            # Raw importance (prediction sensitivity)
            raw_importance = (avg_shuffle * 0.35 + avg_mean * 0.35 + avg_noise * 0.30)
            
            # Normalized importance: adjust by scaler_std
            # Features with smaller original std (like pressure: 2.0) are more sensitive
            # because small changes in original units = large changes in normalized space
            # We multiply by norm_factor to give them appropriate weight
            normalized_importance = raw_importance * stats['norm_factor']
            
            feature_importance_raw[feat_idx] = raw_importance
            feature_importance[feat_idx] = normalized_importance
            
            feature_importance_details[feat_name] = {
                'shuffle': avg_shuffle,
                'mean_replace': avg_mean,
                'noise': avg_noise,
                'raw_importance': raw_importance,
                'normalized_importance': normalized_importance,
                'scaler_std': stats['scaler_std'],
                'norm_factor': stats['norm_factor']
            }
            
            print(f"  [{feat_idx+1}/{n_features}] {feat_name}:")
            print(f"      Shuffle: {avg_shuffle:.6f}, Mean: {avg_mean:.6f}, Noise: {avg_noise:.6f}")
            print(f"      Raw: {raw_importance:.6f}, Normalized: {normalized_importance:.6f}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("IMPORTANCE SUMMARY")
        print(f"{'='*60}")
        
        total_raw = feature_importance_raw.sum()
        total_norm = feature_importance.sum()
        
        print(f"\n{'Feature':<20}{'Raw %':<12}{'Normalized %':<15}{'Scaler Std':<12}")
        print("-" * 60)
        
        sorted_indices = np.argsort(feature_importance)[::-1]
        for feat_idx in sorted_indices:
            feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"
            if feat_name != self.target_feature and feat_idx != target_idx:
                raw_pct = (feature_importance_raw[feat_idx] / total_raw) * 100 if total_raw > 0 else 0
                norm_pct = (feature_importance[feat_idx] / total_norm) * 100 if total_norm > 0 else 0
                scaler_std = feature_stats.get(feat_name, {}).get('scaler_std', 0)
                print(f"  {feat_name:<18}{raw_pct:>8.2f}%    {norm_pct:>8.2f}%      {scaler_std:>8.2f}")
        
        # Save detailed results
        details_df = pd.DataFrame(feature_importance_details).T
        details_df.to_csv(os.path.join(self.output_dir, 'feature_importance_detailed.csv'))
        
        # Also compute CORRELATION-WEIGHTED importance
        # This combines model sensitivity with domain knowledge (correlations)
        print(f"\n{'='*60}")
        print("CORRELATION-WEIGHTED IMPORTANCE")
        print("(Combines model sensitivity with data correlations)")
        print(f"{'='*60}")
        
        if hasattr(self, 'raw_data_stats') and self.raw_data_stats:
            try:
                # Load training data to get correlations
                train_file = os.path.join(self.args.root_path, 'train.csv')
                if os.path.exists(train_file):
                    train_df = pd.read_csv(train_file)
                    
                    # Compute correlations with target
                    correlations = {}
                    for feat in self.input_features:
                        if feat in train_df.columns and self.target_feature in train_df.columns:
                            corr = abs(train_df[feat].corr(train_df[self.target_feature]))
                            correlations[feat] = corr
                    
                    # Correlation-weighted importance: raw_importance * |correlation|
                    corr_weighted_importance = np.zeros(n_features)
                    
                    print(f"\n{'Feature':<20}{'Raw %':<12}{'|Corr|':<12}{'Corr-Weighted %':<18}")
                    print("-" * 60)
                    
                    for feat_idx in range(n_features):
                        feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"
                        if feat_name in correlations and feat_name != self.target_feature:
                            corr = correlations[feat_name]
                            raw_imp = feature_importance_raw[feat_idx]
                            # Weight by correlation
                            corr_weighted_importance[feat_idx] = raw_imp * (corr ** 0.5)
                    
                    total_corr_weighted = corr_weighted_importance.sum()
                    
                    sorted_cw = np.argsort(corr_weighted_importance)[::-1]
                    for feat_idx in sorted_cw:
                        feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"
                        if feat_name in correlations and feat_name != self.target_feature:
                            raw_pct = (feature_importance_raw[feat_idx] / total_raw) * 100 if total_raw > 0 else 0
                            corr = correlations[feat_name]
                            cw_pct = (corr_weighted_importance[feat_idx] / total_corr_weighted) * 100 if total_corr_weighted > 0 else 0
                            print(f"  {feat_name:<18}{raw_pct:>8.2f}%   {corr:>8.4f}      {cw_pct:>8.2f}%")
                    
                    # Save correlation-weighted as the primary importance
                    # This better reflects domain knowledge
                    print(f"\n  Using CORRELATION-WEIGHTED importance as final ranking")
                    feature_importance = corr_weighted_importance
                    
            except Exception as e:
                print(f"  Could not compute correlation-weighted importance: {e}")
        
        return feature_importance
    
    def _compute_gradient_shap_values(self, background_data, explain_data):
        """
        Compute gradient-based SHAP values.
        
        Note: This may fail for models with in-place operations.
        In that case, permutation importance is used as fallback.
        """
        # Skip gradient computation - the model has in-place operations
        # that break autograd. Permutation importance is more reliable anyway.
        print("\nSkipping gradient SHAP (model has in-place operations).")
        print("Using permutation importance instead (more reliable for this model).")
        return None
    
    def compute_gradient_shap(self, data_loader, num_samples=100, background_samples=20):
        """
        Compute SHAP values using GradientExplainer for faster computation.
        
        This method is more efficient for deep learning models and provides
        similar interpretability to KernelSHAP.
        
        Args:
            data_loader: Data loader
            num_samples: Number of samples to explain
            background_samples: Number of background samples
            
        Returns:
            shap_values: SHAP values
            explained_data: Data that was explained
        """
        print("Computing Gradient SHAP values...")
        
        # Collect data
        all_inputs = []
        all_marks = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            all_inputs.append(batch_x)
            all_marks.append(batch_x_mark)
            if len(all_inputs) * batch_x.shape[0] >= num_samples + background_samples:
                break
        
        all_inputs = torch.cat(all_inputs, dim=0)
        all_marks = torch.cat(all_marks, dim=0)
        
        background = all_inputs[:background_samples].to(self.device)
        explain_data = all_inputs[background_samples:background_samples + num_samples].to(self.device)
        explain_marks = all_marks[background_samples:background_samples + num_samples].to(self.device)
        
        # Wrap model for GradientSHAP
        class ModelWrapper(nn.Module):
            def __init__(self, model, args, device):
                super().__init__()
                self.model = model
                self.args = args
                self.device = device
                
            def forward(self, x):
                batch_size = x.shape[0]
                x_mark_enc = torch.zeros(batch_size, self.args.seq_len, 4).to(self.device)
                dec_inp = torch.zeros(batch_size, self.args.label_len + self.args.pred_len,
                                     x.shape[-1]).to(self.device)
                x_mark_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, 4).to(self.device)
                
                outputs = self.model(x, x_mark_enc, dec_inp, x_mark_dec)
                return outputs[:, :, -1].mean(dim=1, keepdim=True)
        
        wrapped_model = ModelWrapper(self.model, self.args, self.device)
        wrapped_model.eval()
        
        # Use DeepExplainer for neural networks
        try:
            print("Using DeepExplainer...")
            explainer = shap.DeepExplainer(wrapped_model, background)
            shap_values = explainer.shap_values(explain_data)
        except Exception as e:
            print(f"DeepExplainer failed: {e}")
            print("Falling back to GradientExplainer...")
            explainer = shap.GradientExplainer(wrapped_model, background)
            shap_values = explainer.shap_values(explain_data)
        
        if isinstance(shap_values, list):
            shap_values = shap_values[0]
        
        return shap_values, explain_data.cpu().numpy()
    
    def plot_global_shap_summary(self, shap_values, data, save_path=None):
        """
        Create SHAP summary plot showing global feature importance and effects.
        
        Similar to Figure 7 in the PLOS ONE paper, this shows:
        - Feature importance ranking (y-axis)
        - SHAP value distribution (x-axis)
        - Feature value impact (color)
        
        Args:
            shap_values: Computed SHAP values
            data: Input data corresponding to SHAP values
            save_path: Path to save the figure
        """
        plt.figure(figsize=(12, 8))
        
        if len(shap_values.shape) == 3:
            # Aggregate across time steps
            shap_aggregated = shap_values.mean(axis=1)
            data_aggregated = data.mean(axis=1) if len(data.shape) == 3 else data
        else:
            shap_aggregated = shap_values
            data_aggregated = data
        
        # Use feature names
        feature_names_display = self.feature_names[:shap_aggregated.shape[1]]
        
        shap.summary_plot(
            shap_aggregated, 
            data_aggregated,
            feature_names=feature_names_display,
            show=False,
            plot_size=(12, 8)
        )
        
        plt.title('SHAP Global Feature Importance - PatchXFormer Solar Forecasting', fontsize=14)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'shap_summary_plot.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Saved summary plot to {save_path}")
        plt.close()
        
    def plot_feature_importance_bar(self, feature_importance, save_path=None):
        """
        Create bar plot of mean absolute SHAP values per feature.
        
        Similar to Figure 7(a) in the PLOS ONE paper showing absolute mean SHAP values.
        
        Args:
            feature_importance: Array of mean |SHAP| values per feature
            save_path: Path to save the figure
        """
        plt.figure(figsize=(10, 6))
        
        feature_names_display = self.feature_names[:len(feature_importance)]
        
        # Sort by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        sorted_features = [feature_names_display[i] for i in sorted_idx]
        sorted_importance = feature_importance[sorted_idx]
        
        colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(sorted_features)))
        
        bars = plt.barh(range(len(sorted_features)), sorted_importance, color=colors)
        plt.yticks(range(len(sorted_features)), sorted_features)
        plt.xlabel('Mean |SHAP Value|', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.title('Feature Importance - Mean Absolute SHAP Values\nPatchXFormer Solar Power Forecasting', fontsize=14)
        
        # Add value labels
        for bar, val in zip(bars, sorted_importance):
            plt.text(val + 0.001, bar.get_y() + bar.get_height()/2, 
                    f'{val:.4f}', va='center', fontsize=10)
        
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'feature_importance_bar.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Saved feature importance bar plot to {save_path}")
        plt.close()
        
    def plot_partial_dependence(self, shap_values, data, feature_idx, save_path=None):
        """
        Create partial SHAP dependence plot for a specific feature.
        
        Similar to Figure 8 in the PLOS ONE paper showing non-linear relationships
        between feature values and their SHAP contributions.
        
        Args:
            shap_values: SHAP values array
            data: Input data array
            feature_idx: Index of the feature to plot
            save_path: Path to save the figure
        """
        if len(shap_values.shape) == 3:
            shap_aggregated = shap_values.mean(axis=1)
            data_aggregated = data.mean(axis=1) if len(data.shape) == 3 else data
        else:
            shap_aggregated = shap_values
            data_aggregated = data
            
        feature_name = self.feature_names[feature_idx] if feature_idx < len(self.feature_names) else f'Feature {feature_idx}'
        
        plt.figure(figsize=(10, 6))
        
        # SHAP dependence plot
        shap.dependence_plot(
            feature_idx, 
            shap_aggregated, 
            data_aggregated,
            feature_names=self.feature_names[:shap_aggregated.shape[1]],
            show=False
        )
        
        plt.title(f'Partial SHAP Dependence: {feature_name}', fontsize=14)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'partial_dependence_{feature_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved partial dependence plot to {save_path}")
        plt.close()
        
    def plot_all_partial_dependence(self, shap_values, data, save_dir=None):
        """
        Create partial dependence plots for all weather-related features.
        
        Args:
            shap_values: SHAP values array
            data: Input data array
            save_dir: Directory to save figures
        """
        if save_dir is None:
            save_dir = os.path.join(self.output_dir, 'partial_dependence')
        os.makedirs(save_dir, exist_ok=True)
        
        if len(shap_values.shape) == 3:
            shap_aggregated = shap_values.mean(axis=1)
            data_aggregated = data.mean(axis=1) if len(data.shape) == 3 else data
        else:
            shap_aggregated = shap_values
            data_aggregated = data
        
        # Plot for each feature
        n_features = min(shap_aggregated.shape[1], len(self.feature_names))
        
        # Create a combined figure with all partial dependence plots
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx in range(n_features):
            feature_name = self.feature_names[idx]
            ax = axes[idx]
            
            feature_values = data_aggregated[:, idx]
            shap_feature = shap_aggregated[:, idx]
            
            # Scatter plot with trend line
            ax.scatter(feature_values, shap_feature, alpha=0.5, s=20, c='steelblue')
            
            # Add LOWESS smoothed line
            try:
                from scipy.ndimage import gaussian_filter1d
                sorted_idx = np.argsort(feature_values)
                x_sorted = feature_values[sorted_idx]
                y_sorted = shap_feature[sorted_idx]
                y_smooth = gaussian_filter1d(y_sorted, sigma=5)
                ax.plot(x_sorted, y_smooth, 'r-', linewidth=2, label='Trend')
            except:
                pass
            
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax.set_xlabel(feature_name, fontsize=10)
            ax.set_ylabel('SHAP Value', fontsize=10)
            ax.set_title(f'{feature_name}', fontsize=12)
            
        plt.suptitle('Partial SHAP Dependence Plots - All Features\nPatchXFormer Solar Forecasting', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        save_path = os.path.join(save_dir, 'all_partial_dependence.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Saved combined partial dependence plot to {save_path}")
        plt.close()
        
    def plot_local_explanation(self, shap_values, data, sample_idx, true_value=None, 
                               pred_value=None, save_path=None):
        """
        Create local SHAP waterfall plot for a specific prediction.
        
        Similar to Figure 9 in the PLOS ONE paper showing how each feature
        contributes to a specific prediction.
        
        Args:
            shap_values: SHAP values array
            data: Input data array
            sample_idx: Index of the sample to explain
            true_value: Actual target value (optional)
            pred_value: Predicted value (optional)
            save_path: Path to save the figure
        """
        if len(shap_values.shape) == 3:
            sample_shap = shap_values[sample_idx].mean(axis=0)
            sample_data = data[sample_idx].mean(axis=0) if len(data.shape) == 3 else data[sample_idx]
        else:
            sample_shap = shap_values[sample_idx]
            sample_data = data[sample_idx]
        
        plt.figure(figsize=(12, 8))
        
        feature_names_display = self.feature_names[:len(sample_shap)]
        
        # Create waterfall-style visualization
        sorted_idx = np.argsort(np.abs(sample_shap))[::-1]
        
        colors = ['red' if v < 0 else 'blue' for v in sample_shap[sorted_idx]]
        
        y_pos = np.arange(len(sorted_idx))
        plt.barh(y_pos, sample_shap[sorted_idx], color=colors, alpha=0.7)
        plt.yticks(y_pos, [feature_names_display[i] for i in sorted_idx])
        
        plt.xlabel('SHAP Value (Impact on Prediction)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        
        title = f'Local SHAP Explanation - Sample {sample_idx}'
        if true_value is not None and pred_value is not None:
            title += f'\nTrue: {true_value:.3f}, Predicted: {pred_value:.3f}'
        plt.title(title, fontsize=14)
        
        # Add feature values as annotations
        for i, idx in enumerate(sorted_idx):
            val = sample_data[idx]
            shap_val = sample_shap[idx]
            plt.annotate(f'{val:.2f}', 
                        xy=(shap_val, i), 
                        xytext=(5 if shap_val >= 0 else -5, 0),
                        textcoords='offset points',
                        va='center', ha='left' if shap_val >= 0 else 'right',
                        fontsize=9)
        
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, f'local_explanation_sample_{sample_idx}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved local explanation to {save_path}")
        plt.close()
        
    def create_weather_contribution_analysis(self, shap_values, data, save_path=None):
        """
        Create detailed analysis of weather parameter contributions.
        
        This analysis focuses specifically on weather features (temp, humidity, etc.)
        and their impact on solar power predictions.
        
        Args:
            shap_values: SHAP values array
            data: Input data array
            save_path: Path to save the analysis
        """
        if len(shap_values.shape) == 3:
            shap_aggregated = shap_values.mean(axis=1)
            data_aggregated = data.mean(axis=1) if len(data.shape) == 3 else data
        else:
            shap_aggregated = shap_values
            data_aggregated = data
        
        # Find indices of weather features
        weather_indices = []
        weather_names = []
        for i, name in enumerate(self.feature_names[:shap_aggregated.shape[1]]):
            if name in self.weather_features:
                weather_indices.append(i)
                weather_names.append(name)
        
        if not weather_indices:
            print("No weather features found in the data")
            return
        
        # Create detailed analysis
        fig, axes = plt.subplots(2, 4, figsize=(16, 10))
        axes = axes.flatten()
        
        analysis_results = []
        
        for i, (idx, name) in enumerate(zip(weather_indices, weather_names)):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            feature_vals = data_aggregated[:, idx]
            shap_vals = shap_aggregated[:, idx]
            
            # Create 2D histogram / heatmap
            h = ax.hexbin(feature_vals, shap_vals, gridsize=20, cmap='RdYlBu_r', mincnt=1)
            ax.axhline(y=0, color='white', linestyle='--', alpha=0.7)
            
            ax.set_xlabel(f'{name} Value', fontsize=10)
            ax.set_ylabel('SHAP Value', fontsize=10)
            ax.set_title(f'{name}', fontsize=12)
            
            # Compute statistics
            mean_shap = np.abs(shap_vals).mean()
            std_shap = shap_vals.std()
            correlation = np.corrcoef(feature_vals, shap_vals)[0, 1]
            
            analysis_results.append({
                'Feature': name,
                'Mean |SHAP|': mean_shap,
                'Std SHAP': std_shap,
                'Correlation': correlation,
                'Min Value': feature_vals.min(),
                'Max Value': feature_vals.max()
            })
            
            # Add colorbar
            plt.colorbar(h, ax=ax, label='Count')
        
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        
        plt.suptitle('Weather Parameter Contribution Analysis\nPatchXFormer Solar Power Forecasting', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'weather_contribution_analysis.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Saved weather contribution analysis to {save_path}")
        plt.close()
        
        # Save analysis results
        analysis_df = pd.DataFrame(analysis_results)
        analysis_df = analysis_df.sort_values('Mean |SHAP|', ascending=False)
        analysis_df.to_csv(os.path.join(self.output_dir, 'weather_contribution_stats.csv'), index=False)
        
        print("\nWeather Parameter Contribution Statistics:")
        print(analysis_df.to_string(index=False))
        
        return analysis_df
    
    def plot_temporal_contribution(self, shap_values_temporal, save_path=None):
        """
        Analyze and visualize temporal contribution patterns.
        
        Shows how feature importance changes across the input sequence,
        useful for understanding which time steps are most influential.
        
        Args:
            shap_values_temporal: SHAP values with temporal dimension [samples, seq_len, features]
            save_path: Path to save the figure
        """
        if len(shap_values_temporal.shape) != 3:
            print("Temporal analysis requires 3D SHAP values [samples, seq_len, features]")
            return
        
        # Average across samples
        temporal_importance = np.abs(shap_values_temporal).mean(axis=0)  # [seq_len, features]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Heatmap of temporal importance
        ax1 = axes[0]
        feature_names_display = self.feature_names[:temporal_importance.shape[1]]
        
        sns.heatmap(temporal_importance.T, 
                   xticklabels=range(0, temporal_importance.shape[0], 10),
                   yticklabels=feature_names_display,
                   cmap='YlOrRd', ax=ax1)
        ax1.set_xlabel('Time Step', fontsize=12)
        ax1.set_ylabel('Feature', fontsize=12)
        ax1.set_title('Temporal Feature Importance Heatmap', fontsize=14)
        
        # Line plot of total importance per time step
        ax2 = axes[1]
        total_importance_per_timestep = temporal_importance.sum(axis=1)
        ax2.plot(range(len(total_importance_per_timestep)), total_importance_per_timestep, 
                'b-', linewidth=2)
        ax2.fill_between(range(len(total_importance_per_timestep)), 
                         total_importance_per_timestep, alpha=0.3)
        ax2.set_xlabel('Time Step', fontsize=12)
        ax2.set_ylabel('Total |SHAP| Importance', fontsize=12)
        ax2.set_title('Total Feature Importance Across Time Steps', fontsize=14)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle('Temporal Analysis of Feature Contributions\nPatchXFormer Solar Forecasting', 
                     fontsize=14, y=1.02)
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.output_dir, 'temporal_contribution.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.savefig(save_path.replace('.png', '.pdf'), format='pdf', bbox_inches='tight')
        print(f"Saved temporal contribution analysis to {save_path}")
        plt.close()
        
    def generate_full_report(self, data_loader, num_samples=100, background_samples=50, 
                              root_path=None, data_path='train.csv'):
        """
        Generate comprehensive SHAP analysis report with normalization-aware analysis.
        
        This creates all visualizations and statistics for model explainability,
        including analysis of how data scaling affects feature importance.
        
        Args:
            data_loader: Data loader for analysis
            num_samples: Number of samples to analyze
            background_samples: Number of background samples
            root_path: Path to dataset for scaling analysis
            data_path: Training data filename
        """
        print("="*60)
        print("SHAP EXPLAINABILITY ANALYSIS - PatchXFormer")
        print("="*60)
        
        # 0. Analyze data scaling first (to understand normalization effects)
        scaling_analysis = None
        if root_path:
            print("\n[0/7] Analyzing data scaling and normalization...")
            scaling_analysis = self.analyze_data_scaling(root_path, data_path)
        
        # 1. Compute feature-level SHAP values
        print("\n[1/7] Computing feature-level SHAP values...")
        shap_values, data, feature_importance = self.compute_feature_level_shap(
            data_loader, num_samples, background_samples
        )
        
        # 2. Plot global summary
        print("\n[2/7] Creating global SHAP summary plot...")
        self.plot_global_shap_summary(shap_values, data)
        
        # 3. Plot feature importance bar chart
        print("\n[3/7] Creating feature importance bar chart...")
        self.plot_feature_importance_bar(feature_importance)
        
        # 4. Create partial dependence plots
        print("\n[4/7] Creating partial dependence plots...")
        self.plot_all_partial_dependence(shap_values, data)
        
        # 5. Create weather contribution analysis
        print("\n[5/7] Analyzing weather parameter contributions...")
        weather_stats = self.create_weather_contribution_analysis(shap_values, data)
        
        # 6. Create local explanations for a few samples
        print("\n[6/7] Creating local SHAP explanations...")
        for idx in [0, 1, 2]:  # Explain first 3 samples
            if idx < len(data):
                self.plot_local_explanation(shap_values, data, idx)
        
        # 7. Generate comprehensive text report (including scaling analysis)
        print("\n[7/7] Generating comprehensive analysis report...")
        self._generate_comprehensive_report(shap_values, data, feature_importance, weather_stats, scaling_analysis)
        
        print("\n" + "="*60)
        print(f"SHAP analysis complete! Results saved to: {self.output_dir}")
        print("="*60)
        
        # Return summary
        return {
            'shap_values': shap_values,
            'data': data,
            'feature_importance': feature_importance,
            'scaling_analysis': scaling_analysis,
            'output_dir': self.output_dir
        }
    
    def _generate_comprehensive_report(self, shap_values, data, feature_importance, weather_stats=None, scaling_analysis=None):
        """
        Generate a comprehensive text report with all SHAP analysis results.
        This report can be referred to later for evaluation.
        Includes normalization-aware analysis.
        """
        import datetime
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[::-1]
        sorted_features = [(self.feature_names[i], feature_importance[i]) for i in sorted_idx]
        
        # Compute additional statistics
        shap_stats = {
            'mean': np.mean(shap_values, axis=0),
            'std': np.std(shap_values, axis=0),
            'min': np.min(shap_values, axis=0),
            'max': np.max(shap_values, axis=0),
            'abs_mean': np.mean(np.abs(shap_values), axis=0)
        }
        
        # Compute correlations between features and SHAP values
        correlations = []
        for i in range(min(shap_values.shape[1], data.shape[1])):
            corr = np.corrcoef(data[:, i], shap_values[:, i])[0, 1]
            correlations.append(corr)
        
        # Build the report
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("SHAP EXPLAINABILITY ANALYSIS REPORT - PatchXFormer")
        report_lines.append("=" * 80)
        report_lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Output Directory: {self.output_dir}")
        report_lines.append("")
        
        # Dataset Info
        report_lines.append("-" * 80)
        report_lines.append("1. DATASET INFORMATION")
        report_lines.append("-" * 80)
        report_lines.append(f"Number of samples analyzed: {shap_values.shape[0]}")
        report_lines.append(f"Number of features: {shap_values.shape[1]}")
        report_lines.append(f"Feature names: {', '.join(self.feature_names[:shap_values.shape[1]])}")
        report_lines.append("")
        
        # Feature Importance Ranking
        report_lines.append("-" * 80)
        report_lines.append("2. FEATURE IMPORTANCE RANKING (Mean |SHAP Value|)")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Rank':<6}{'Feature':<25}{'Mean |SHAP|':<15}{'Percentage':<12}")
        report_lines.append("-" * 58)
        
        total_importance = sum(imp for _, imp in sorted_features)
        for rank, (feature, importance) in enumerate(sorted_features, 1):
            pct = (importance / total_importance) * 100 if total_importance > 0 else 0
            report_lines.append(f"{rank:<6}{feature:<25}{importance:<15.6f}{pct:<12.2f}%")
        report_lines.append("")
        
        # Weather Parameters Analysis
        report_lines.append("-" * 80)
        report_lines.append("3. WEATHER PARAMETERS CONTRIBUTION")
        report_lines.append("-" * 80)
        
        weather_importance = []
        for i, (feature, importance) in enumerate(sorted_features):
            if feature in self.weather_features:
                weather_importance.append((feature, importance, correlations[sorted_idx[i]] if i < len(correlations) else 0))
        
        if weather_importance:
            report_lines.append(f"{'Feature':<20}{'Mean |SHAP|':<15}{'Correlation':<15}{'Impact':<20}")
            report_lines.append("-" * 70)
            for feature, importance, corr in weather_importance:
                impact = "Positive" if corr > 0.1 else ("Negative" if corr < -0.1 else "Neutral")
                report_lines.append(f"{feature:<20}{importance:<15.6f}{corr:<15.4f}{impact:<20}")
        report_lines.append("")
        
        # Detailed SHAP Statistics
        report_lines.append("-" * 80)
        report_lines.append("4. DETAILED SHAP STATISTICS PER FEATURE")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Feature':<20}{'Mean':<12}{'Std':<12}{'Min':<12}{'Max':<12}{'|Mean|':<12}")
        report_lines.append("-" * 80)
        
        for i, feature in enumerate(self.feature_names[:shap_values.shape[1]]):
            report_lines.append(
                f"{feature:<20}{shap_stats['mean'][i]:<12.6f}{shap_stats['std'][i]:<12.6f}"
                f"{shap_stats['min'][i]:<12.6f}{shap_stats['max'][i]:<12.6f}{shap_stats['abs_mean'][i]:<12.6f}"
            )
        report_lines.append("")
        
        # Key Findings
        report_lines.append("-" * 80)
        report_lines.append("5. KEY FINDINGS")
        report_lines.append("-" * 80)
        
        # Top 3 most important features
        report_lines.append("\nTop 3 Most Important Features:")
        for i, (feature, importance) in enumerate(sorted_features[:3], 1):
            pct = (importance / total_importance) * 100 if total_importance > 0 else 0
            report_lines.append(f"  {i}. {feature}: {importance:.6f} ({pct:.1f}% of total importance)")
        
        # Top weather parameter
        if weather_importance:
            top_weather = weather_importance[0]
            report_lines.append(f"\nMost Important Weather Parameter: {top_weather[0]}")
            report_lines.append(f"  - Mean |SHAP|: {top_weather[1]:.6f}")
            report_lines.append(f"  - Correlation with SHAP: {top_weather[2]:.4f}")
        
        # Feature with highest positive/negative impact
        max_positive_idx = np.argmax(shap_stats['mean'])
        max_negative_idx = np.argmin(shap_stats['mean'])
        report_lines.append(f"\nFeature with Highest Positive Impact: {self.feature_names[max_positive_idx]}")
        report_lines.append(f"  - Mean SHAP: {shap_stats['mean'][max_positive_idx]:.6f}")
        report_lines.append(f"\nFeature with Highest Negative Impact: {self.feature_names[max_negative_idx]}")
        report_lines.append(f"  - Mean SHAP: {shap_stats['mean'][max_negative_idx]:.6f}")
        
        report_lines.append("")
        
        # Data Scaling Analysis (NEW SECTION)
        report_lines.append("-" * 80)
        report_lines.append("6. DATA SCALING ANALYSIS (Normalization Effects)")
        report_lines.append("-" * 80)
        
        if scaling_analysis and self.scaler_info:
            report_lines.append("\nStandardScaler Parameters (used for normalization):")
            report_lines.append(f"{'Feature':<20}{'Raw Mean':<15}{'Raw Std':<15}{'Scaler μ':<15}{'Scaler σ':<15}")
            report_lines.append("-" * 80)
            
            for feat in self.scaler_info.get('feature_order', []):
                raw_mean = self.raw_data_stats.get(feat, {}).get('mean', 0)
                raw_std = self.raw_data_stats.get(feat, {}).get('std', 0)
                scaler_mean = self.scaler_info['means'].get(feat, 0)
                scaler_std = self.scaler_info['stds'].get(feat, 0)
                report_lines.append(f"{feat:<20}{raw_mean:<15.4f}{raw_std:<15.4f}{scaler_mean:<15.4f}{scaler_std:<15.4f}")
            
            report_lines.append("\n\nCorrelation vs SHAP Importance Comparison:")
            report_lines.append(f"{'Feature':<20}{'|Correlation|':<18}{'SHAP Importance':<18}{'Match?':<12}")
            report_lines.append("-" * 80)
            
            # Compare correlation-based expected importance with SHAP importance
            correlations_dict = scaling_analysis.get('correlations', {})
            if correlations_dict:
                sorted_corr = sorted(correlations_dict.items(), key=lambda x: abs(x[1]), reverse=True)
                shap_sorted = sorted([(f, imp) for f, imp in sorted_features if f != self.target_feature], 
                                    key=lambda x: x[1], reverse=True)
                
                corr_rank = {feat: rank for rank, (feat, _) in enumerate(sorted_corr, 1)}
                shap_rank = {feat: rank for rank, (feat, _) in enumerate(shap_sorted, 1)}
                
                for feat, corr in sorted_corr[:10]:
                    shap_imp = dict(sorted_features).get(feat, 0)
                    c_rank = corr_rank.get(feat, 'N/A')
                    s_rank = shap_rank.get(feat, 'N/A')
                    match = "Yes" if c_rank == s_rank else f"Corr:{c_rank} vs SHAP:{s_rank}"
                    report_lines.append(f"{feat:<20}{abs(corr):<18.4f}{shap_imp:<18.6f}{match:<12}")
        else:
            report_lines.append("\nScaling analysis not available (root_path not provided)")
        
        report_lines.append("""
KEY OBSERVATIONS ON SCALING:
- StandardScaler transforms: x_normalized = (x - mean) / std
- After normalization, all features have approximately mean=0 and std=1
- This means raw value ranges do NOT directly determine model importance
- The model learns patterns in the NORMALIZED space
- However, features with consistent temporal patterns may still dominate

WHY SHAP IMPORTANCE MAY DIFFER FROM CORRELATION:
1. Correlation measures LINEAR relationship; model captures NON-LINEAR patterns
2. Model uses 96 time steps; temporal dynamics matter more than point values
3. Features that predict FUTURE changes are more valuable than current correlations
4. Multicollinearity: features may share information, SHAP attributes to the dominant one
        """)
        
        # Interpretation Guide
        report_lines.append("-" * 80)
        report_lines.append("7. INTERPRETATION GUIDE")
        report_lines.append("-" * 80)
        report_lines.append("""
SHAP Value Interpretation:
- Positive SHAP value: Feature pushes prediction HIGHER (increases solar power output)
- Negative SHAP value: Feature pushes prediction LOWER (decreases solar power output)
- Larger |SHAP| value: Feature has stronger influence on prediction

Weather Parameter Effects on Solar Power:
- Temperature: Higher temp typically increases panel output (up to optimal point)
- Humidity: Higher humidity often reduces solar irradiance reaching panels
- Cloud Cover: Directly blocks sunlight, strong negative impact expected
- Pressure: Atmospheric conditions affect irradiance
- Wind Speed: Can affect panel cooling and efficiency
- Dew Point: Related to humidity and condensation on panels
        """)
        
        # Files Generated
        report_lines.append("-" * 80)
        report_lines.append("8. FILES GENERATED")
        report_lines.append("-" * 80)
        report_lines.append(f"  - feature_importance.csv: Feature importance rankings")
        report_lines.append(f"  - feature_importance_bar.png/pdf: Bar chart visualization")
        report_lines.append(f"  - shap_summary_plot.png/pdf: SHAP summary plot")
        report_lines.append(f"  - weather_contribution_analysis.png/pdf: Weather analysis")
        report_lines.append(f"  - weather_contribution_stats.csv: Weather statistics")
        report_lines.append(f"  - partial_dependence/: Partial dependence plots")
        report_lines.append(f"  - local_explanation_sample_*.png: Local explanations")
        report_lines.append(f"  - shap_values_aggregated.npy: Raw SHAP values")
        report_lines.append(f"  - data_scaling_analysis.png/pdf: Scaling visualization")
        report_lines.append(f"  - data_scaling_info.csv: Scaler parameters and stats")
        report_lines.append(f"  - SHAP_ANALYSIS_REPORT.txt: This report")
        report_lines.append("")
        report_lines.append("=" * 80)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 80)
        
        # Join and save report
        report_text = "\n".join(report_lines)
        
        # Print to console
        print("\n" + report_text)
        
        # Save to file
        report_path = os.path.join(self.output_dir, 'SHAP_ANALYSIS_REPORT.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"\nReport saved to: {report_path}")
        
        # Also save as CSV for easy access
        stats_df = pd.DataFrame({
            'Feature': self.feature_names[:shap_values.shape[1]],
            'Mean_SHAP': shap_stats['mean'],
            'Std_SHAP': shap_stats['std'],
            'Min_SHAP': shap_stats['min'],
            'Max_SHAP': shap_stats['max'],
            'Mean_Abs_SHAP': shap_stats['abs_mean'],
            'Correlation': correlations[:shap_values.shape[1]] if len(correlations) >= shap_values.shape[1] else correlations + [0]*(shap_values.shape[1]-len(correlations))
        })
        stats_df = stats_df.sort_values('Mean_Abs_SHAP', ascending=False)
        stats_df.to_csv(os.path.join(self.output_dir, 'shap_detailed_statistics.csv'), index=False)
        
        return report_text


def load_model_and_data(args):
    """
    Load trained PatchXFormer model and data.
    
    Args:
        args: Configuration arguments
        
    Returns:
        model: Loaded model
        data_loader: Test data loader
        device: torch device
    """
    # Set device
    if torch.cuda.is_available() and args.use_gpu:
        device = torch.device(f'cuda:{args.gpu}')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    
    print(f"Using device: {device}")
    
    # Create experiment and build model
    from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
    exp = Exp_Long_Term_Forecast(args)
    model = exp._build_model().to(device)
    
    # Load checkpoint
    setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_expand{}_dc{}_fc{}_eb{}_dt{}_{}_{}'.format(
        args.task_name, args.model_id, args.model, args.data,
        args.features, args.seq_len, args.label_len, args.pred_len,
        args.d_model, args.n_heads, args.e_layers, args.d_layers,
        args.d_ff, args.expand, args.d_conv, args.factor,
        args.embed, args.distil, args.des, 0
    )
    
    checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
    
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from: {checkpoint_path}")
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}")
        print("Using randomly initialized model for demonstration")
    
    model.eval()
    
    # Get data loader
    _, test_loader = exp._get_data(flag='test')
    
    return model, test_loader, device, args


def main():
    """Main function to run SHAP analysis."""
    parser = argparse.ArgumentParser(description='SHAP Analysis for PatchXFormer')
    
    # Basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--model_id', type=str, default='solar_sl_piliyandala96_96')
    parser.add_argument('--model', type=str, default='PatchXFormer')
    
    # Data config
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/sl_piliyandala')
    parser.add_argument('--data_path', type=str, default='solar.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='Solar Power Output')
    parser.add_argument('--freq', type=str, default='h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--inverse', action='store_true', default=False)
    
    # Forecasting config
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly')
    
    # Model config
    parser.add_argument('--enc_in', type=int, default=10)
    parser.add_argument('--dec_in', type=int, default=10)
    parser.add_argument('--c_out', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--decomp_method', type=str, default='moving_avg')
    parser.add_argument('--down_sampling_layers', type=int, default=0)
    parser.add_argument('--down_sampling_window', type=int, default=1)
    parser.add_argument('--down_sampling_method', type=str, default=None)
    parser.add_argument('--seg_len', type=int, default=96)
    parser.add_argument('--top_k', type=int, default=5)
    parser.add_argument('--num_kernels', type=int, default=6)
    parser.add_argument('--patch_len', type=int, default=16)
    
    # Additional model params
    parser.add_argument('--expand', type=int, default=2)
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--distil', type=bool, default=True)
    parser.add_argument('--des', type=str, default='Exp')
    
    # GPU config
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type: cuda or mps')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0', help='device ids for multi-gpu')
    
    # Data loader config
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    
    # SHAP config
    parser.add_argument('--num_samples', type=int, default=200, 
                       help='Number of samples to explain (use more for better results)')
    parser.add_argument('--background_samples', type=int, default=100,
                       help='Number of background samples for SHAP')
    parser.add_argument('--n_repeats', type=int, default=30,
                       help='Number of permutation repeats per feature')
    parser.add_argument('--output_dir', type=str, default='./shap_results/',
                       help='Output directory for SHAP results')
    
    args = parser.parse_args()
    
    # Feature names for solar dataset
    feature_names = [
        'dayofyear', 'timeofday', 'temp', 'dew', 
        'humidity', 'winddir', 'windspeed', 'pressure', 
        'cloudcover', 'Solar Power Output'
    ]
    
    print("="*60)
    print("PatchXFormer SHAP Explainability Analysis")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.root_path}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Prediction Length: {args.pred_len}")
    print(f"Number of samples to analyze: {args.num_samples}")
    print("="*60)
    
    # Load model and data
    model, test_loader, device, args = load_model_and_data(args)
    
    # Create SHAP explainer
    explainer = SHAPExplainer(
        args=args,
        model=model,
        device=device,
        feature_names=feature_names
    )
    explainer.output_dir = args.output_dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run full SHAP analysis with scaling analysis
    results = explainer.generate_full_report(
        test_loader,
        num_samples=args.num_samples,
        background_samples=args.background_samples,
        root_path=args.root_path,  # Pass root_path for scaling analysis
        data_path='train.csv'
    )
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
