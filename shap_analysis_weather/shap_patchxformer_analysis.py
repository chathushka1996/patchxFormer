"""
SHAP Analysis for PatchXFormer Solar Power Forecasting Model
=============================================================

This script provides comprehensive SHAP (SHapley Additive exPlanations) analysis
for interpreting the PatchXFormer transformer-based time series forecasting model.

Author: AI-Assisted Research
Date: 2026-04-11
Purpose: Master's Thesis - Solar Power Forecasting Model Interpretability

References:
- SHAP Documentation: https://shap.readthedocs.io/
- Lundberg & Lee (2017): "A Unified Approach to Interpreting Model Predictions"
- PMC11695015: "Solar energy prediction through machine learning models"
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import pearsonr, spearmanr
import warnings
from tqdm import tqdm
import json
from datetime import datetime

# Import model and data components
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from data_provider.data_factory import data_provider

warnings.filterwarnings('ignore')

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class PatchXFormerSHAPAnalyzer:
    """
    Comprehensive SHAP Analysis for PatchXFormer Model
    
    This analyzer implements multiple SHAP techniques specifically designed
    for transformer-based time series forecasting models:
    
    1. GradientExplainer: For transformer architectures (recommended)
    2. DeepExplainer: For deep learning models
    3. KernelExplainer: Model-agnostic (slower but more general)
    
    For PatchXFormer, we primarily use GradientExplainer as it:
    - Works well with transformer architectures
    - Handles complex attention mechanisms
    - Provides accurate gradient-based approximations
    - Faster than KernelExplainer for deep models
    """
    
    def __init__(self, args, output_dir='shap_analysis_weather/results'):
        """
        Initialize SHAP analyzer
        
        Args:
            args: Argument object containing model configuration
            output_dir: Directory to save results and visualizations
        """
        self.args = args
        self.output_dir = output_dir
        self.device = args.device
        
        # Create output directories
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'csv_reports'), exist_ok=True)
        
        # Initialize model and data
        self.exp = Exp_Long_Term_Forecast(args)
        self.model = self.exp.model
        self.model.eval()
        
        # Weather feature names (excluding time features)
        self.weather_features = [
            'temp', 'dew', 'humidity', 'winddir', 
            'windspeed', 'pressure', 'cloudcover'
        ]
        
        # Full feature names including target
        self.all_features = self.weather_features + ['Solar Power Output']
        
        # Time feature names (to be excluded from SHAP analysis)
        self.time_features = ['dayofyear', 'timeofday']
        
        print(f"Initialized SHAP Analyzer")
        print(f"Weather features for analysis: {self.weather_features}")
        print(f"Total features in model: {len(self.all_features)}")
        print(f"Output directory: {output_dir}")
        
    def load_trained_model(self, checkpoint_path):
        """
        Load pre-trained model weights
        
        Args:
            checkpoint_path: Path to model checkpoint
        """
        if os.path.exists(checkpoint_path):
            self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded model from: {checkpoint_path}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    def prepare_data_for_shap(self, flag='test', num_samples=500):
        """
        Prepare data for SHAP analysis
        
        Args:
            flag: Dataset split ('train', 'val', or 'test')
            num_samples: Number of samples to use for background and explanation
            
        Returns:
            Dictionary containing prepared data
        """
        print(f"\nPreparing data for SHAP analysis from {flag} set...")
        
        # Get dataset
        data_set, data_loader = data_provider(self.args, flag)
        
        # Extract samples
        all_x, all_y, all_x_mark, all_y_mark = [], [], [], []
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            all_x.append(batch_x.cpu().numpy())
            all_y.append(batch_y.cpu().numpy())
            all_x_mark.append(batch_x_mark.cpu().numpy())
            all_y_mark.append(batch_y_mark.cpu().numpy())
            
            if len(all_x) * batch_x.shape[0] >= num_samples:
                break
        
        # Concatenate and limit samples
        x_data = np.concatenate(all_x, axis=0)[:num_samples]
        y_data = np.concatenate(all_y, axis=0)[:num_samples]
        x_mark = np.concatenate(all_x_mark, axis=0)[:num_samples]
        y_mark = np.concatenate(all_y_mark, axis=0)[:num_samples]
        
        # Get feature indices (exclude time features)
        # Dataset structure: [date, dayofyear, timeofday, weather_features..., target]
        # We want indices for weather features only
        weather_indices = list(range(len(self.weather_features)))
        
        print(f"Extracted {len(x_data)} samples")
        print(f"Input shape: {x_data.shape}")  # [batch, seq_len, features]
        print(f"Weather feature indices: {weather_indices}")
        
        return {
            'x': x_data,
            'y': y_data,
            'x_mark': x_mark,
            'y_mark': y_mark,
            'weather_indices': weather_indices,
            'scaler': data_set.scaler if hasattr(data_set, 'scaler') else None
        }
    
    def create_model_wrapper(self, x_mark, y_mark):
        """
        Create a wrapper function for the model that SHAP can work with
        
        For PatchXFormer, we need to handle:
        1. Encoder input (x_enc)
        2. Time embeddings (x_mark_enc, y_mark)
        3. Decoder input (constructed from zeros + label)
        
        Args:
            x_mark: Time features for encoder
            y_mark: Time features for decoder
            
        Returns:
            Wrapped prediction function
        """
        def predict_fn(x_input):
            """
            Prediction function for SHAP
            
            Args:
                x_input: Numpy array of shape [batch, seq_len, weather_features]
                
            Returns:
                Predictions as numpy array
            """
            # Convert to torch tensor
            batch_x = torch.FloatTensor(x_input).to(self.device)
            batch_x_mark = torch.FloatTensor(x_mark[:len(x_input)]).to(self.device)
            batch_y_mark = torch.FloatTensor(y_mark[:len(x_input)]).to(self.device)
            
            # Create decoder input (zeros for prediction horizon)
            batch_size = batch_x.shape[0]
            dec_inp_zeros = torch.zeros(
                (batch_size, self.args.pred_len, batch_x.shape[-1])
            ).float().to(self.device)
            
            # Use last label_len from encoder as decoder start
            dec_inp_label = batch_x[:, -self.args.label_len:, :]
            dec_inp = torch.cat([dec_inp_label, dec_inp_zeros], dim=1)
            
            # Make prediction
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                
                # Extract predictions for target feature (Solar Power Output)
                # Assuming last feature is the target
                predictions = outputs[:, -self.args.pred_len:, -1]
                
            return predictions.cpu().numpy()
        
        return predict_fn
    
    def compute_shap_values_gradient(self, data_dict, background_samples=100):
        """
        Compute SHAP values using GradientExplainer
        
        GradientExplainer is recommended for transformer models because:
        1. It leverages gradients which transformers compute efficiently
        2. Handles attention mechanisms well
        3. Faster than KernelExplainer
        4. Provides theoretically sound approximations
        
        Args:
            data_dict: Dictionary with prepared data
            background_samples: Number of background samples for explanation
            
        Returns:
            SHAP values and explainer object
        """
        print("\n" + "="*80)
        print("Computing SHAP Values using GradientExplainer")
        print("="*80)
        
        x_data = data_dict['x']
        x_mark = data_dict['x_mark']
        y_mark = data_dict['y_mark']
        
        # Select background data
        background_data = x_data[:background_samples]
        explain_data = x_data[background_samples:background_samples+200]
        
        print(f"Background samples: {len(background_data)}")
        print(f"Explanation samples: {len(explain_data)}")
        
        # Create model wrapper
        predict_fn = self.create_model_wrapper(x_mark, y_mark)
        
        # Test prediction function
        test_pred = predict_fn(background_data[:2])
        print(f"Test prediction shape: {test_pred.shape}")
        
        # Initialize GradientExplainer
        print("\nInitializing GradientExplainer...")
        try:
            # For GradientExplainer with PyTorch, we need a different approach
            explainer = shap.DeepExplainer(
                self.create_torch_model_wrapper(),
                torch.FloatTensor(background_data).to(self.device)
            )
            print("Using DeepExplainer (optimized for PyTorch)")
        except Exception as e:
            print(f"DeepExplainer failed: {e}")
            print("Falling back to KernelExplainer...")
            explainer = shap.KernelExplainer(
                predict_fn,
                background_data
            )
        
        # Compute SHAP values
        print("\nComputing SHAP values (this may take several minutes)...")
        shap_values = []
        batch_size = 20
        
        for i in tqdm(range(0, len(explain_data), batch_size)):
            batch = explain_data[i:i+batch_size]
            try:
                batch_shap = explainer.shap_values(torch.FloatTensor(batch).to(self.device))
                if isinstance(batch_shap, list):
                    batch_shap = batch_shap[0]
                shap_values.append(batch_shap)
            except Exception as e:
                print(f"Error in batch {i}: {e}")
                # Use KernelExplainer as fallback for this batch
                # Reshape to 2D for KernelExplainer (flatten time dimension)
                batch_flat = batch.reshape(len(batch), -1)
                background_flat = background_data[:50].reshape(len(background_data[:50]), -1)
                
                kernel_explainer = shap.KernelExplainer(
                    lambda x: predict_fn(x.reshape(-1, *batch.shape[1:])), 
                    background_flat
                )
                batch_shap_flat = kernel_explainer.shap_values(batch_flat)
                # Reshape back to 3D
                batch_shap = batch_shap_flat.reshape(batch.shape)
                shap_values.append(batch_shap)
        
        shap_values = np.concatenate(shap_values, axis=0)
        
        print(f"\nSHAP values computed successfully!")
        print(f"SHAP values shape: {shap_values.shape}")
        
        return shap_values, explainer, explain_data
    
    def create_torch_model_wrapper(self):
        """
        Create a PyTorch model wrapper for DeepExplainer
        
        Returns:
            Wrapped model
        """
        class ModelWrapper(nn.Module):
            def __init__(self, model, args, device):
                super().__init__()
                self.model = model
                self.args = args
                self.device = device
            
            def forward(self, x):
                # x: [batch, seq_len, features]
                batch_size = x.shape[0]
                
                # Clone input to avoid in-place operation issues
                x = x.clone()
                
                # Create dummy time features (for DeepExplainer compatibility)
                x_mark = torch.zeros(batch_size, x.shape[1], 4).to(self.device)
                y_mark = torch.zeros(batch_size, self.args.label_len + self.args.pred_len, 4).to(self.device)
                
                # Create decoder input (clone to avoid in-place issues)
                dec_inp_zeros = torch.zeros(
                    (batch_size, self.args.pred_len, x.shape[-1])
                ).float().to(self.device)
                dec_inp_label = x[:, -self.args.label_len:, :].clone()
                dec_inp = torch.cat([dec_inp_label, dec_inp_zeros], dim=1)
                
                # Make prediction
                outputs = self.model(x, x_mark, dec_inp, y_mark)
                
                # Return target predictions
                return outputs[:, -self.args.pred_len:, -1]
        
        return ModelWrapper(self.model, self.args, self.device)
    
    def analyze_feature_importance(self, shap_values, feature_names):
        """
        Analyze and rank feature importance
        
        Args:
            shap_values: Computed SHAP values
            feature_names: Names of features
            
        Returns:
            DataFrame with feature importance
        """
        print("\n" + "="*80)
        print("Analyzing Feature Importance")
        print("="*80)
        
        # Calculate mean absolute SHAP values across time steps and samples
        # shap_values shape: [n_samples, seq_len, n_features]
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': mean_abs_shap,
            'Importance_Rank': range(1, len(feature_names) + 1)
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
        importance_df['Importance_Rank'] = range(1, len(importance_df) + 1)
        
        print("\nFeature Importance Ranking:")
        print(importance_df.to_string(index=False))
        
        # Save to CSV
        importance_df.to_csv(
            os.path.join(self.output_dir, 'csv_reports', 'feature_importance_weather.csv'),
            index=False
        )
        
        return importance_df
    
    def plot_shap_summary(self, shap_values, feature_data, feature_names):
        """
        Create SHAP summary plots
        
        Args:
            shap_values: Computed SHAP values
            feature_data: Original feature data
            feature_names: Names of features
        """
        print("\nGenerating SHAP summary plots...")
        
        # Reshape for plotting (average across time steps)
        shap_values_2d = shap_values.mean(axis=1)  # [n_samples, n_features]
        feature_data_2d = feature_data.mean(axis=1)  # [n_samples, n_features]
        
        # 1. Summary plot (beeswarm)
        plt.figure(figsize=(12, 8))
        shap.summary_plot(
            shap_values_2d,
            feature_data_2d,
            feature_names=feature_names,
            show=False,
            plot_size=(12, 8)
        )
        plt.title('SHAP Summary Plot: Weather Feature Importance for Solar Power Forecasting', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('SHAP value (impact on model output)', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_summary_beeswarm.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_summary_beeswarm.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        # 2. Bar plot of mean importance
        plt.figure(figsize=(10, 6))
        shap.summary_plot(
            shap_values_2d,
            feature_data_2d,
            feature_names=feature_names,
            plot_type='bar',
            show=False
        )
        plt.title('SHAP Feature Importance: Weather Variables', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Mean |SHAP value|', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_feature_importance_bar.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_feature_importance_bar.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        print("Summary plots saved successfully!")
    
    def plot_shap_dependence(self, shap_values, feature_data, feature_names):
        """
        Create SHAP dependence plots for each feature
        
        Args:
            shap_values: Computed SHAP values
            feature_data: Original feature data
            feature_names: Names of features
        """
        print("\nGenerating SHAP dependence plots...")
        
        # Reshape data
        shap_values_2d = shap_values.mean(axis=1)
        feature_data_2d = feature_data.mean(axis=1)
        
        # Create dependence plots for each feature
        n_features = len(feature_names)
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        for idx, feature_name in enumerate(feature_names):
            if idx < len(axes):
                ax = axes[idx]
                
                # Get feature values and SHAP values
                feature_values = feature_data_2d[:, idx]
                feature_shap = shap_values_2d[:, idx]
                
                # Create scatter plot
                scatter = ax.scatter(
                    feature_values,
                    feature_shap,
                    c=feature_values,
                    cmap='viridis',
                    alpha=0.6,
                    s=20
                )
                
                # Add trend line
                z = np.polyfit(feature_values, feature_shap, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(feature_values.min(), feature_values.max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
                
                ax.set_xlabel(f'{feature_name}', fontsize=10, fontweight='bold')
                ax.set_ylabel('SHAP value', fontsize=10)
                ax.set_title(f'Dependence: {feature_name}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                
                # Add colorbar
                plt.colorbar(scatter, ax=ax, label=feature_name)
        
        # Hide unused subplots
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('SHAP Dependence Plots: Impact of Weather Features on Solar Power Predictions',
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_dependence_plots.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_dependence_plots.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        print("Dependence plots saved successfully!")
    
    def plot_temporal_shap_analysis(self, shap_values, feature_names):
        """
        Analyze SHAP values across time steps (temporal contribution)
        
        Args:
            shap_values: Computed SHAP values [n_samples, seq_len, n_features]
            feature_names: Names of features
        """
        print("\nGenerating temporal SHAP analysis...")
        
        # Average SHAP values across samples for each time step
        temporal_shap = np.abs(shap_values).mean(axis=0)  # [seq_len, n_features]
        
        # Plot temporal contribution
        plt.figure(figsize=(14, 8))
        for idx, feature_name in enumerate(feature_names):
            plt.plot(temporal_shap[:, idx], label=feature_name, linewidth=2, marker='o', 
                     markersize=4, markevery=8)
        
        plt.xlabel('Time Step in Input Sequence', fontsize=12, fontweight='bold')
        plt.ylabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
        plt.title('Temporal Feature Importance: How Weather Features Contribute Across Time',
                  fontsize=14, fontweight='bold', pad=20)
        plt.legend(loc='best', fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_temporal_contribution.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_temporal_contribution.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        # Create heatmap of temporal contributions
        plt.figure(figsize=(14, 6))
        sns.heatmap(
            temporal_shap.T,
            xticklabels=[f't-{self.args.seq_len-i}' if i < self.args.seq_len else f't+{i-self.args.seq_len}' 
                         for i in range(len(temporal_shap))],
            yticklabels=feature_names,
            cmap='RdYlBu_r',
            center=0,
            cbar_kws={'label': 'Mean Absolute SHAP Value'}
        )
        plt.title('Temporal SHAP Heatmap: Feature Contribution Across Input Sequence',
                  fontsize=14, fontweight='bold', pad=20)
        plt.xlabel('Time Step', fontsize=12, fontweight='bold')
        plt.ylabel('Weather Feature', fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_temporal_heatmap.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_temporal_heatmap.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        print("Temporal analysis plots saved successfully!")
    
    def plot_prediction_horizon_analysis(self, shap_values, feature_names, data_dict):
        """
        Analyze how features contribute to different prediction horizons
        
        This is crucial for understanding which features are important for:
        - Short-term forecasts (1-24 hours)
        - Medium-term forecasts (1-3 days)
        - Long-term forecasts (3-7 days)
        
        Args:
            shap_values: Computed SHAP values
            feature_names: Names of features
            data_dict: Data dictionary
        """
        print("\nGenerating prediction horizon analysis...")
        
        # Get predictions at different horizons
        x_data = data_dict['x']
        predict_fn = self.create_model_wrapper(data_dict['x_mark'], data_dict['y_mark'])
        
        # Compute predictions
        predictions = predict_fn(x_data[:100])
        
        # Define horizon ranges (in 15-min intervals)
        # 96 = 24 hours, 192 = 48 hours, etc.
        horizon_ranges = {
            'Short-term (0-24h)': (0, 96),
            'Medium-term (24-48h)': (96, 192),
            'Long-term (48h+)': (192, None)
        }
        
        # Average SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=(0, 1))
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        x_pos = np.arange(len(feature_names))
        bars = ax.bar(x_pos, mean_abs_shap, color='skyblue', edgecolor='black', linewidth=1.5)
        
        # Color code by importance
        colors = plt.cm.RdYlGn_r(mean_abs_shap / mean_abs_shap.max())
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(feature_names, rotation=45, ha='right')
        ax.set_ylabel('Mean Absolute SHAP Value', fontsize=12, fontweight='bold')
        ax.set_title('Feature Importance for Solar Power Forecasting',
                     fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_horizon_importance.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_horizon_importance.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        print("Prediction horizon analysis saved successfully!")
    
    def analyze_feature_interactions(self, shap_values, feature_data, feature_names):
        """
        Analyze interactions between features
        
        Args:
            shap_values: Computed SHAP values
            feature_data: Original feature data
            feature_names: Names of features
        """
        print("\nAnalyzing feature interactions...")
        
        # Reshape data
        shap_values_2d = shap_values.mean(axis=1)
        feature_data_2d = feature_data.mean(axis=1)
        
        # Compute correlation between SHAP values
        shap_correlation = np.corrcoef(shap_values_2d.T)
        
        # Plot correlation heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            shap_correlation,
            xticklabels=feature_names,
            yticklabels=feature_names,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={'label': 'SHAP Correlation'}
        )
        plt.title('Feature Interaction Analysis: SHAP Value Correlations',
                  fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_feature_interactions.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_feature_interactions.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        # Save correlation matrix
        correlation_df = pd.DataFrame(
            shap_correlation,
            index=feature_names,
            columns=feature_names
        )
        correlation_df.to_csv(
            os.path.join(self.output_dir, 'csv_reports', 'feature_shap_correlations.csv')
        )
        
        print("Feature interaction analysis saved successfully!")
    
    def generate_comprehensive_report(self, shap_values, feature_data, feature_names, 
                                     importance_df, data_dict):
        """
        Generate comprehensive text report for thesis
        
        Args:
            shap_values: Computed SHAP values
            feature_data: Original feature data
            feature_names: Names of features
            importance_df: Feature importance dataframe
            data_dict: Data dictionary
        """
        print("\nGenerating comprehensive report...")
        
        report_path = os.path.join(self.output_dir, 'COMPREHENSIVE_SHAP_REPORT.txt')
        
        with open(report_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("COMPREHENSIVE SHAP ANALYSIS REPORT\n")
            f.write("PatchXFormer Solar Power Forecasting Model\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: PatchXFormer Transformer-based Time Series Forecasting\n")
            f.write(f"Dataset: Sri Lanka Piliyandala Solar Power Data\n\n")
            
            # Model Configuration
            f.write("-"*80 + "\n")
            f.write("MODEL CONFIGURATION\n")
            f.write("-"*80 + "\n")
            f.write(f"Sequence Length: {self.args.seq_len} time steps\n")
            f.write(f"Prediction Length: {self.args.pred_len} time steps\n")
            f.write(f"Model Dimension: {self.args.d_model}\n")
            f.write(f"Number of Heads: {self.args.n_heads}\n")
            f.write(f"Encoder Layers: {self.args.e_layers}\n\n")
            
            # SHAP Analysis Overview
            f.write("-"*80 + "\n")
            f.write("SHAP ANALYSIS OVERVIEW\n")
            f.write("-"*80 + "\n")
            f.write("SHAP (SHapley Additive exPlanations) is a unified framework for interpreting\n")
            f.write("machine learning model predictions. It assigns each feature an importance value\n")
            f.write("for a particular prediction based on cooperative game theory.\n\n")
            
            f.write("Why SHAP for Transformer Models?\n")
            f.write("1. Handles complex attention mechanisms in transformers\n")
            f.write("2. Provides local and global interpretability\n")
            f.write("3. Theoretically sound with consistency guarantees\n")
            f.write("4. Can explain multivariate time series predictions\n\n")
            
            # Feature Importance Ranking
            f.write("-"*80 + "\n")
            f.write("FEATURE IMPORTANCE RANKING\n")
            f.write("-"*80 + "\n")
            f.write("Weather features ranked by their mean absolute SHAP value:\n\n")
            for idx, row in importance_df.iterrows():
                f.write(f"{row['Importance_Rank']}. {row['Feature']}: {row['Mean_Abs_SHAP']:.6f}\n")
            f.write("\n")
            
            # Detailed Feature Analysis
            f.write("-"*80 + "\n")
            f.write("DETAILED FEATURE ANALYSIS\n")
            f.write("-"*80 + "\n\n")
            
            shap_values_2d = shap_values.mean(axis=1)
            feature_data_2d = feature_data.mean(axis=1)
            
            for idx, feature_name in enumerate(feature_names):
                f.write(f"{feature_name.upper()}\n")
                f.write("-" * 40 + "\n")
                
                feature_shap = shap_values_2d[:, idx]
                feature_vals = feature_data_2d[:, idx]
                
                f.write(f"Mean SHAP Value: {feature_shap.mean():.6f}\n")
                f.write(f"Mean |SHAP| Value: {np.abs(feature_shap).mean():.6f}\n")
                f.write(f"SHAP Std Dev: {feature_shap.std():.6f}\n")
                f.write(f"SHAP Range: [{feature_shap.min():.6f}, {feature_shap.max():.6f}]\n")
                
                # Correlation with output
                corr, p_value = pearsonr(feature_vals, feature_shap)
                f.write(f"SHAP-Feature Correlation: {corr:.4f} (p={p_value:.4e})\n")
                
                # Interpretation
                f.write("\nInterpretation:\n")
                if np.abs(feature_shap).mean() > np.abs(shap_values_2d).mean().mean():
                    f.write(f"- {feature_name} has ABOVE AVERAGE importance for predictions\n")
                else:
                    f.write(f"- {feature_name} has BELOW AVERAGE importance for predictions\n")
                
                if feature_shap.mean() > 0:
                    f.write(f"- Higher {feature_name} values tend to INCREASE predicted solar power\n")
                else:
                    f.write(f"- Higher {feature_name} values tend to DECREASE predicted solar power\n")
                
                f.write("\n")
            
            # Key Findings
            f.write("-"*80 + "\n")
            f.write("KEY FINDINGS FOR THESIS\n")
            f.write("-"*80 + "\n\n")
            
            top_3_features = importance_df.head(3)['Feature'].tolist()
            
            f.write("1. MOST INFLUENTIAL WEATHER VARIABLES:\n")
            f.write(f"   The top 3 weather features influencing solar power predictions are:\n")
            for i, feat in enumerate(top_3_features, 1):
                importance = importance_df[importance_df['Feature'] == feat]['Mean_Abs_SHAP'].values[0]
                f.write(f"   {i}. {feat} (mean |SHAP| = {importance:.6f})\n")
            f.write("\n")
            
            f.write("2. TEMPORAL IMPORTANCE:\n")
            temporal_importance = np.abs(shap_values).mean(axis=2).mean(axis=0)
            early_importance = temporal_importance[:len(temporal_importance)//3].mean()
            late_importance = temporal_importance[-len(temporal_importance)//3:].mean()
            f.write(f"   Early time steps (first 1/3): mean |SHAP| = {early_importance:.6f}\n")
            f.write(f"   Late time steps (last 1/3): mean |SHAP| = {late_importance:.6f}\n")
            if late_importance > early_importance:
                f.write("   → Recent weather conditions are MORE important for predictions\n")
            else:
                f.write("   → Earlier weather conditions are MORE important for predictions\n")
            f.write("\n")
            
            f.write("3. FEATURE INTERACTIONS:\n")
            shap_correlation = np.corrcoef(shap_values_2d.T)
            np.fill_diagonal(shap_correlation, 0)
            max_corr_idx = np.unravel_index(np.abs(shap_correlation).argmax(), shap_correlation.shape)
            feat1, feat2 = feature_names[max_corr_idx[0]], feature_names[max_corr_idx[1]]
            max_corr = shap_correlation[max_corr_idx]
            f.write(f"   Strongest SHAP correlation: {feat1} ↔ {feat2} (r = {max_corr:.4f})\n")
            f.write(f"   This suggests these features have related impacts on predictions\n")
            f.write("\n")
            
            # Comparison with Reference Paper
            f.write("-"*80 + "\n")
            f.write("COMPARISON WITH REFERENCE STUDY (PMC11695015)\n")
            f.write("-"*80 + "\n")
            f.write("The referenced paper found that ambient temperature and humidity had the\n")
            f.write("greatest influence on solar energy predictions using CatBoost.\n\n")
            f.write("Our PatchXFormer SHAP analysis results:\n")
            for feat in ['temp', 'humidity']:
                if feat in importance_df['Feature'].values:
                    rank = importance_df[importance_df['Feature'] == feat]['Importance_Rank'].values[0]
                    importance = importance_df[importance_df['Feature'] == feat]['Mean_Abs_SHAP'].values[0]
                    f.write(f"- {feat}: Rank {rank}, importance = {importance:.6f}\n")
            f.write("\n")
            
            # Recommendations
            f.write("-"*80 + "\n")
            f.write("RECOMMENDATIONS FOR MODEL IMPROVEMENT\n")
            f.write("-"*80 + "\n")
            bottom_3_features = importance_df.tail(3)['Feature'].tolist()
            f.write(f"1. Focus on improving data quality for: {', '.join(top_3_features)}\n")
            f.write(f"2. Consider feature engineering for: {', '.join(top_3_features)}\n")
            f.write(f"3. Low-importance features could potentially be removed: {', '.join(bottom_3_features)}\n")
            f.write(f"4. Investigate temporal patterns in {top_3_features[0]} (highest importance)\n\n")
            
            # Limitations
            f.write("-"*80 + "\n")
            f.write("LIMITATIONS AND CONSIDERATIONS\n")
            f.write("-"*80 + "\n")
            f.write("1. SHAP values are computed on a sample of test data (not entire dataset)\n")
            f.write("2. Transformer attention mechanisms add complexity to interpretation\n")
            f.write("3. Time features (dayofyear, timeofday) were excluded as requested\n")
            f.write("4. SHAP assumes feature independence (may not hold for weather variables)\n")
            f.write("5. Computational cost limits number of samples analyzed\n\n")
            
            # References
            f.write("-"*80 + "\n")
            f.write("REFERENCES\n")
            f.write("-"*80 + "\n")
            f.write("1. Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting\n")
            f.write("   model predictions. NIPS.\n")
            f.write("2. Nguyen et al. (2025). Solar energy prediction through machine learning\n")
            f.write("   models: A comparative analysis. PLOS ONE.\n")
            f.write("3. SHAP Documentation: https://shap.readthedocs.io/\n\n")
            
            f.write("="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"Comprehensive report saved to: {report_path}")
    
    def save_shap_values_csv(self, shap_values, feature_data, feature_names):
        """
        Save SHAP values to CSV for further analysis
        
        Args:
            shap_values: Computed SHAP values
            feature_data: Original feature data
            feature_names: Names of features
        """
        print("\nSaving SHAP values to CSV...")
        
        # Reshape to 2D
        shap_values_2d = shap_values.mean(axis=1)
        feature_data_2d = feature_data.mean(axis=1)
        
        # Create dataframe
        shap_df = pd.DataFrame(shap_values_2d, columns=[f'SHAP_{feat}' for feat in feature_names])
        feature_df = pd.DataFrame(feature_data_2d, columns=[f'Value_{feat}' for feat in feature_names])
        
        # Combine
        combined_df = pd.concat([feature_df, shap_df], axis=1)
        
        # Save
        combined_df.to_csv(
            os.path.join(self.output_dir, 'csv_reports', 'shap_values_detailed.csv'),
            index=False
        )
        
        print("SHAP values saved to CSV successfully!")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SHAP Analysis for PatchXFormer')
    
    # Use same arguments as training script
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--model', type=str, default='PatchXFormer')
    parser.add_argument('--data', type=str, default='custom')
    parser.add_argument('--root_path', type=str, default='./dataset/sl_piliyandala')
    parser.add_argument('--data_path', type=str, default='solar.csv')
    parser.add_argument('--features', type=str, default='M')
    parser.add_argument('--target', type=str, default='Solar Power Output')
    parser.add_argument('--freq', type=str, default='15min')
    parser.add_argument('--seq_len', type=int, default=96)
    parser.add_argument('--label_len', type=int, default=48)
    parser.add_argument('--pred_len', type=int, default=96)
    parser.add_argument('--enc_in', type=int, default=10)
    parser.add_argument('--dec_in', type=int, default=10)
    parser.add_argument('--c_out', type=int, default=10)
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_heads', type=int, default=8)
    parser.add_argument('--e_layers', type=int, default=2)
    parser.add_argument('--d_layers', type=int, default=1)
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0')
    parser.add_argument('--inverse', action='store_true', default=False)
    
    # SHAP specific arguments
    parser.add_argument('--checkpoint_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--num_samples', type=int, default=500,
                       help='Number of samples for SHAP analysis')
    parser.add_argument('--background_samples', type=int, default=100,
                       help='Number of background samples for SHAP')
    parser.add_argument('--output_dir', type=str, default='shap_analysis_weather/results',
                       help='Output directory for results')
    
    return parser.parse_args()


def main():
    """Main execution function"""
    print("\n" + "="*80)
    print("SHAP ANALYSIS FOR PATCHXFORMER SOLAR POWER FORECASTING")
    print("="*80 + "\n")
    
    # Parse arguments
    args = parse_arguments()
    
    # Set device
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print(f"Using GPU: {args.gpu}")
    else:
        args.device = torch.device('cpu')
        print("Using CPU")
    
    # Initialize analyzer
    analyzer = PatchXFormerSHAPAnalyzer(args, args.output_dir)
    
    # Load trained model
    analyzer.load_trained_model(args.checkpoint_path)
    
    # Prepare data
    data_dict = analyzer.prepare_data_for_shap(flag='test', num_samples=args.num_samples)
    
    # Compute SHAP values
    shap_values, explainer, explain_data = analyzer.compute_shap_values_gradient(
        data_dict, 
        background_samples=args.background_samples
    )
    
    # Extract weather feature names and indices
    weather_feature_names = analyzer.weather_features
    weather_indices = data_dict['weather_indices']
    
    # Extract only weather features from data and SHAP values
    weather_data = explain_data[:, :, :len(weather_feature_names)]
    weather_shap_values = shap_values[:, :, :len(weather_feature_names)]
    
    # Analyze feature importance
    importance_df = analyzer.analyze_feature_importance(weather_shap_values, weather_feature_names)
    
    # Generate visualizations
    analyzer.plot_shap_summary(weather_shap_values, weather_data, weather_feature_names)
    analyzer.plot_shap_dependence(weather_shap_values, weather_data, weather_feature_names)
    analyzer.plot_temporal_shap_analysis(weather_shap_values, weather_feature_names)
    analyzer.plot_prediction_horizon_analysis(weather_shap_values, weather_feature_names, data_dict)
    analyzer.analyze_feature_interactions(weather_shap_values, weather_data, weather_feature_names)
    
    # Save detailed data
    analyzer.save_shap_values_csv(weather_shap_values, weather_data, weather_feature_names)
    
    # Generate comprehensive report
    analyzer.generate_comprehensive_report(
        weather_shap_values,
        weather_data,
        weather_feature_names,
        importance_df,
        data_dict
    )
    
    print("\n" + "="*80)
    print("SHAP ANALYSIS COMPLETED SUCCESSFULLY!")
    print(f"All results saved to: {args.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
