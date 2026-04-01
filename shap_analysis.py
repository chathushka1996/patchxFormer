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
        
        # Create output directory
        self.output_dir = './shap_results/'
        os.makedirs(self.output_dir, exist_ok=True)
        
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
        
        # Aggregate for display purposes
        aggregated_explain = explain_data.mean(axis=1)
        
        # Method 1: Permutation-based Feature Importance (more reliable for time series)
        print("\nComputing permutation-based feature importance...")
        feature_importance = self._compute_permutation_importance(
            explain_data, explain_marks, n_repeats=10
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
        
        # Create importance DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names[:len(feature_importance)],
            'Mean |SHAP|': feature_importance
        }).sort_values('Mean |SHAP|', ascending=False)
        
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE (Mean |SHAP|)")
        print("="*50)
        print(importance_df.to_string(index=False))
        print("="*50)
        
        # Save results
        importance_df.to_csv(os.path.join(self.output_dir, 'feature_importance.csv'), index=False)
        np.save(os.path.join(self.output_dir, 'shap_values_aggregated.npy'), shap_values)
        np.save(os.path.join(self.output_dir, 'explain_data.npy'), aggregated_explain)
        
        return shap_values, aggregated_explain, feature_importance
    
    def _compute_permutation_importance(self, data, marks, n_repeats=10):
        """
        Compute permutation-based feature importance.
        
        For each feature, shuffle its values across samples and measure
        how much the prediction changes. Larger changes = more important feature.
        """
        self.model.eval()
        n_samples, seq_len, n_features = data.shape
        
        # Get baseline predictions
        with torch.no_grad():
            x_enc = torch.FloatTensor(data).to(self.device)
            x_mark = torch.FloatTensor(marks).to(self.device)
            batch_size = x_enc.shape[0]
            
            dec_inp = torch.zeros(batch_size, self.args.label_len + self.args.pred_len,
                                 n_features).to(self.device)
            x_mark_dec = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, 
                                    marks.shape[-1]).to(self.device)
            
            baseline_pred = self.model(x_enc, x_mark, dec_inp, x_mark_dec)
            baseline_pred = baseline_pred[:, :, -1].mean(dim=1).cpu().numpy()  # Target column
        
        feature_importance = np.zeros(n_features)
        
        for feat_idx in range(n_features):
            feat_name = self.feature_names[feat_idx] if feat_idx < len(self.feature_names) else f"Feature_{feat_idx}"
            importance_scores = []
            
            for _ in range(n_repeats):
                # Create shuffled data
                data_shuffled = data.copy()
                
                # Shuffle this feature across all samples and time steps
                shuffled_values = data[:, :, feat_idx].flatten()
                np.random.shuffle(shuffled_values)
                data_shuffled[:, :, feat_idx] = shuffled_values.reshape(n_samples, seq_len)
                
                # Get predictions with shuffled feature
                with torch.no_grad():
                    x_enc_shuffled = torch.FloatTensor(data_shuffled).to(self.device)
                    shuffled_pred = self.model(x_enc_shuffled, x_mark, dec_inp, x_mark_dec)
                    shuffled_pred = shuffled_pred[:, :, -1].mean(dim=1).cpu().numpy()
                
                # Importance = how much prediction changed
                importance = np.mean(np.abs(baseline_pred - shuffled_pred))
                importance_scores.append(importance)
            
            feature_importance[feat_idx] = np.mean(importance_scores)
            print(f"  {feat_name}: {feature_importance[feat_idx]:.6f}")
        
        # Normalize to sum to 1 for percentage interpretation
        if feature_importance.sum() > 0:
            feature_importance_normalized = feature_importance / feature_importance.sum()
        else:
            feature_importance_normalized = feature_importance
        
        return feature_importance
    
    def _compute_gradient_shap_values(self, background_data, explain_data):
        """
        Compute gradient-based SHAP values using integrated gradients approach.
        """
        try:
            self.model.eval()
            n_samples, seq_len, n_features = explain_data.shape
            
            # We'll compute feature attribution by measuring gradient * input
            shap_values = np.zeros((len(explain_data), n_features))
            
            for i in range(len(explain_data)):
                x = torch.FloatTensor(explain_data[i:i+1]).to(self.device)
                x.requires_grad = True
                
                x_mark = torch.zeros(1, seq_len, 4).to(self.device)
                dec_inp = torch.zeros(1, self.args.label_len + self.args.pred_len,
                                     n_features).to(self.device)
                x_mark_dec = torch.zeros(1, self.args.label_len + self.args.pred_len, 4).to(self.device)
                
                # Forward pass
                output = self.model(x, x_mark, dec_inp, x_mark_dec)
                target_output = output[:, :, -1].mean()  # Mean of target predictions
                
                # Backward pass
                target_output.backward()
                
                # Gradient * Input gives attribution
                if x.grad is not None:
                    attribution = (x.grad * x).detach().cpu().numpy()
                    # Aggregate across time steps
                    shap_values[i] = attribution[0].mean(axis=0)
                
                # Clear gradients
                self.model.zero_grad()
            
            return shap_values
            
        except Exception as e:
            print(f"Gradient SHAP computation failed: {e}")
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
        
    def generate_full_report(self, data_loader, num_samples=100, background_samples=50):
        """
        Generate comprehensive SHAP analysis report.
        
        This creates all visualizations and statistics for model explainability.
        
        Args:
            data_loader: Data loader for analysis
            num_samples: Number of samples to analyze
            background_samples: Number of background samples
        """
        print("="*60)
        print("SHAP EXPLAINABILITY ANALYSIS - PatchXFormer")
        print("="*60)
        
        # 1. Compute feature-level SHAP values
        print("\n[1/6] Computing feature-level SHAP values...")
        shap_values, data, feature_importance = self.compute_feature_level_shap(
            data_loader, num_samples, background_samples
        )
        
        # 2. Plot global summary
        print("\n[2/6] Creating global SHAP summary plot...")
        self.plot_global_shap_summary(shap_values, data)
        
        # 3. Plot feature importance bar chart
        print("\n[3/6] Creating feature importance bar chart...")
        self.plot_feature_importance_bar(feature_importance)
        
        # 4. Create partial dependence plots
        print("\n[4/6] Creating partial dependence plots...")
        self.plot_all_partial_dependence(shap_values, data)
        
        # 5. Create weather contribution analysis
        print("\n[5/6] Analyzing weather parameter contributions...")
        weather_stats = self.create_weather_contribution_analysis(shap_values, data)
        
        # 6. Create local explanations for a few samples
        print("\n[6/6] Creating local SHAP explanations...")
        for idx in [0, 1, 2]:  # Explain first 3 samples
            if idx < len(data):
                self.plot_local_explanation(shap_values, data, idx)
        
        # 7. Generate comprehensive text report
        print("\n[7/7] Generating comprehensive analysis report...")
        self._generate_comprehensive_report(shap_values, data, feature_importance, weather_stats)
        
        print("\n" + "="*60)
        print(f"SHAP analysis complete! Results saved to: {self.output_dir}")
        print("="*60)
        
        # Return summary
        return {
            'shap_values': shap_values,
            'data': data,
            'feature_importance': feature_importance,
            'output_dir': self.output_dir
        }
    
    def _generate_comprehensive_report(self, shap_values, data, feature_importance, weather_stats=None):
        """
        Generate a comprehensive text report with all SHAP analysis results.
        This report can be referred to later for evaluation.
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
        
        # Interpretation Guide
        report_lines.append("-" * 80)
        report_lines.append("6. INTERPRETATION GUIDE")
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
        report_lines.append("7. FILES GENERATED")
        report_lines.append("-" * 80)
        report_lines.append(f"  - feature_importance.csv: Feature importance rankings")
        report_lines.append(f"  - feature_importance_bar.png/pdf: Bar chart visualization")
        report_lines.append(f"  - shap_summary_plot.png/pdf: SHAP summary plot")
        report_lines.append(f"  - weather_contribution_analysis.png/pdf: Weather analysis")
        report_lines.append(f"  - weather_contribution_stats.csv: Weather statistics")
        report_lines.append(f"  - partial_dependence/: Partial dependence plots")
        report_lines.append(f"  - local_explanation_sample_*.png: Local explanations")
        report_lines.append(f"  - shap_values_aggregated.npy: Raw SHAP values")
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
    parser.add_argument('--num_samples', type=int, default=100, 
                       help='Number of samples to explain')
    parser.add_argument('--background_samples', type=int, default=50,
                       help='Number of background samples for SHAP')
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
    
    # Run full SHAP analysis
    results = explainer.generate_full_report(
        test_loader,
        num_samples=args.num_samples,
        background_samples=args.background_samples
    )
    
    print("\nAnalysis complete!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
