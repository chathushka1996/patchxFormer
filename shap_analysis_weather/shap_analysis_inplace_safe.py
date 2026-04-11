"""
SHAP Analysis for PatchXFormer - Inplace Operation Safe Version
================================================================

This version is specifically designed to handle models with in-place operations
that cause issues with DeepExplainer. It uses KernelExplainer with proper
data reshaping for time series data.

Usage:
    python shap_analysis_inplace_safe.py --checkpoint_path PATH/checkpoint.pth
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_provider.data_factory import data_provider
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast


class InplaceSafeSHAPAnalyzer:
    """SHAP analyzer that handles in-place operations"""
    
    def __init__(self, args, output_dir='shap_analysis_weather/results_inplace_safe'):
        self.args = args
        self.output_dir = output_dir
        self.device = args.device
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'csv_reports'), exist_ok=True)
        
        # Initialize model
        self.exp = Exp_Long_Term_Forecast(args)
        self.model = self.exp.model
        self.model.eval()
        
        self.weather_features = [
            'temp', 'dew', 'humidity', 'winddir',
            'windspeed', 'pressure', 'cloudcover'
        ]
        
        print(f"Initialized Inplace-Safe SHAP Analyzer")
        print(f"Output: {output_dir}")
    
    def load_model(self, checkpoint_path):
        """Load trained model"""
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded model from: {checkpoint_path}")
    
    def prepare_data(self, num_samples=300):
        """Prepare data for analysis"""
        print(f"\nPreparing {num_samples} samples...")
        
        data_set, data_loader = data_provider(self.args, 'test')
        
        all_x, all_y, all_x_mark, all_y_mark = [], [], [], []
        
        for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
            all_x.append(batch_x.cpu().numpy())
            all_y.append(batch_y.cpu().numpy())
            all_x_mark.append(batch_x_mark.cpu().numpy())
            all_y_mark.append(batch_y_mark.cpu().numpy())
            
            if len(all_x) * batch_x.shape[0] >= num_samples:
                break
        
        x_data = np.concatenate(all_x, axis=0)[:num_samples]
        y_data = np.concatenate(all_y, axis=0)[:num_samples]
        x_mark = np.concatenate(all_x_mark, axis=0)[:num_samples]
        y_mark = np.concatenate(all_y_mark, axis=0)[:num_samples]
        
        print(f"Data prepared: {x_data.shape}")
        
        return x_data, y_data, x_mark, y_mark
    
    def create_predict_function_with_clone(self, x_mark, y_mark):
        """Create prediction function that clones inputs to avoid in-place issues"""
        def predict(x_input):
            # Ensure input is properly shaped
            if len(x_input.shape) == 2:
                # Reshape flat input back to 3D
                n_samples = len(x_input)
                seq_len = self.args.seq_len
                n_features = x_input.shape[1] // seq_len
                x_input = x_input.reshape(n_samples, seq_len, n_features)
            
            # Convert to tensor and clone to avoid in-place issues
            batch_x = torch.FloatTensor(x_input).clone().to(self.device)
            batch_x_mark = torch.FloatTensor(x_mark[:len(x_input)]).clone().to(self.device)
            batch_y_mark = torch.FloatTensor(y_mark[:len(x_input)]).clone().to(self.device)
            
            batch_size = batch_x.shape[0]
            dec_inp_zeros = torch.zeros(
                (batch_size, self.args.pred_len, batch_x.shape[-1])
            ).float().to(self.device)
            dec_inp_label = batch_x[:, -self.args.label_len:, :].clone()
            dec_inp = torch.cat([dec_inp_label, dec_inp_zeros], dim=1)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # Average prediction over horizon for simpler analysis
                predictions = outputs[:, -self.args.pred_len:, -1].mean(dim=1)
            
            return predictions.cpu().numpy()
        
        return predict
    
    def compute_shap_kernel_safe(self, x_data, x_mark, y_mark, n_background=50, n_explain=150):
        """
        Compute SHAP using KernelExplainer with proper handling of time series
        
        Strategy: Flatten time series to 2D, compute SHAP, then interpret results
        """
        print("\n" + "="*80)
        print("Computing SHAP values using KernelExplainer (Inplace-Safe)")
        print("This may take 20-40 minutes...")
        print("="*80)
        
        # Extract weather features only and average over time for simpler analysis
        x_weather = x_data[:, :, :len(self.weather_features)]
        x_avg = x_weather.mean(axis=1)  # Average over time: [samples, features]
        
        background = x_avg[:n_background]
        explain = x_avg[n_background:n_background+n_explain]
        
        print(f"Background samples: {len(background)}, shape: {background.shape}")
        print(f"Samples to explain: {len(explain)}, shape: {explain.shape}")
        
        # Create predict function that expands averaged features back to time series
        def predict_from_avg(x_avg_input):
            # Expand averaged features back to full time series
            # Simply repeat the averaged value across all time steps
            n_samples = len(x_avg_input)
            x_expanded = np.repeat(
                x_avg_input[:, np.newaxis, :], 
                self.args.seq_len, 
                axis=1
            )
            # Add other features (assume zeros for non-weather)
            if x_data.shape[2] > len(self.weather_features):
                extra_features = np.zeros((n_samples, self.args.seq_len, 
                                          x_data.shape[2] - len(self.weather_features)))
                x_expanded = np.concatenate([x_expanded, extra_features], axis=2)
            
            return self.create_predict_function_with_clone(x_mark, y_mark)(x_expanded)
        
        # Test prediction
        test_pred = predict_from_avg(background[:2])
        print(f"Test prediction shape: {test_pred.shape}")
        print(f"Test prediction range: [{test_pred.min():.3f}, {test_pred.max():.3f}]")
        
        # Create explainer
        print("\nCreating KernelExplainer (this is the safe approach for in-place ops)...")
        explainer = shap.KernelExplainer(predict_from_avg, background)
        
        # Compute SHAP values with smaller batches
        print("\nComputing SHAP values...")
        shap_values_list = []
        batch_size = 10
        
        for i in tqdm(range(0, len(explain), batch_size)):
            batch = explain[i:i+batch_size]
            try:
                # Use fewer samples for faster computation
                batch_shap = explainer.shap_values(batch, nsamples=100, silent=True)
                shap_values_list.append(batch_shap)
            except Exception as e:
                print(f"\nError in batch {i}: {e}")
                # Try with even fewer samples
                try:
                    batch_shap = explainer.shap_values(batch, nsamples=50, silent=True)
                    shap_values_list.append(batch_shap)
                except:
                    print(f"Skipping batch {i}")
                    continue
        
        if not shap_values_list:
            raise RuntimeError("Failed to compute any SHAP values")
        
        shap_values = np.concatenate(shap_values_list, axis=0)
        
        print(f"\nSHAP values computed: {shap_values.shape}")
        print(f"SHAP range: [{shap_values.min():.6f}, {shap_values.max():.6f}]")
        
        return shap_values, explain, explainer
    
    def plot_summary(self, shap_values, data, feature_names):
        """Create summary visualizations"""
        print("\nCreating visualizations...")
        
        # 1. Summary beeswarm plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, data, feature_names=feature_names, show=False)
        plt.title('SHAP Summary: Weather Feature Importance', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_summary_beeswarm.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_summary_beeswarm.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        # 2. Bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, data, feature_names=feature_names, 
                         plot_type='bar', show=False)
        plt.title('Feature Importance Rankings', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_importance_bar.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_importance_bar.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        # 3. Individual dependence plots
        n_features = len(feature_names)
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()
        
        for idx, feature_name in enumerate(feature_names):
            if idx < len(axes):
                ax = axes[idx]
                feature_values = data[:, idx]
                feature_shap = shap_values[:, idx]
                
                scatter = ax.scatter(feature_values, feature_shap, 
                                   c=feature_values, cmap='viridis', alpha=0.6, s=20)
                
                # Trend line
                z = np.polyfit(feature_values, feature_shap, 2)
                p = np.poly1d(z)
                x_trend = np.linspace(feature_values.min(), feature_values.max(), 100)
                ax.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2)
                
                ax.set_xlabel(f'{feature_name}', fontsize=10, fontweight='bold')
                ax.set_ylabel('SHAP value', fontsize=10)
                ax.set_title(f'Dependence: {feature_name}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                plt.colorbar(scatter, ax=ax, label=feature_name)
        
        for idx in range(n_features, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle('SHAP Dependence Plots: Weather Features',
                     fontsize=16, fontweight='bold', y=1.00)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_dependence_plots.png'), 
                    dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_dependence_plots.pdf'), 
                    bbox_inches='tight')
        plt.close()
        
        print("Plots saved!")
    
    def save_results(self, shap_values, data, feature_names):
        """Save results to CSV"""
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Mean_Abs_SHAP': mean_abs_shap,
            'Rank': range(1, len(feature_names) + 1)
        })
        importance_df = importance_df.sort_values('Mean_Abs_SHAP', ascending=False)
        importance_df['Rank'] = range(1, len(importance_df) + 1)
        
        csv_path = os.path.join(self.output_dir, 'csv_reports', 'feature_importance.csv')
        importance_df.to_csv(csv_path, index=False)
        
        # Save detailed SHAP values
        shap_df = pd.DataFrame(shap_values, columns=[f'SHAP_{feat}' for feat in feature_names])
        feature_df = pd.DataFrame(data, columns=[f'Value_{feat}' for feat in feature_names])
        combined_df = pd.concat([feature_df, shap_df], axis=1)
        combined_df.to_csv(
            os.path.join(self.output_dir, 'csv_reports', 'shap_values_detailed.csv'),
            index=False
        )
        
        print(f"\nFeature Importance Rankings:")
        print(importance_df.to_string(index=False))
        print(f"\nResults saved to: {self.output_dir}")
        
        return importance_df


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Inplace-Safe SHAP Analysis')
    parser.add_argument('--checkpoint_path', type=str, required=True)
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
    parser.add_argument('--d_ff', type=int, default=512)
    parser.add_argument('--factor', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--activation', type=str, default='gelu')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--inverse', action='store_true', default=False)
    
    args = parser.parse_args()
    
    # Set device
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('cpu')
    
    print("\n" + "="*80)
    print("INPLACE-SAFE SHAP ANALYSIS FOR PATCHXFORMER")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = InplaceSafeSHAPAnalyzer(args)
    
    # Load model
    analyzer.load_model(args.checkpoint_path)
    
    # Prepare data
    x_data, y_data, x_mark, y_mark = analyzer.prepare_data(num_samples=250)
    
    # Compute SHAP
    shap_values, explain_data, explainer = analyzer.compute_shap_kernel_safe(
        x_data, x_mark, y_mark, n_background=50, n_explain=150
    )
    
    # Create plots
    analyzer.plot_summary(shap_values, explain_data, analyzer.weather_features)
    
    # Save results
    importance_df = analyzer.save_results(shap_values, explain_data, analyzer.weather_features)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print(f"Results saved to: {analyzer.output_dir}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
