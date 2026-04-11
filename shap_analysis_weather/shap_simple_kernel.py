"""
Simplified SHAP Analysis using KernelExplainer
==============================================

This script provides a simpler, more reliable (but slower) SHAP analysis
using KernelExplainer, which is model-agnostic and works with any model.

Use this if the main script has issues with GradientExplainer/DeepExplainer.
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


class SimpleSHAPAnalyzer:
    """Simplified SHAP analyzer using KernelExplainer"""
    
    def __init__(self, args, output_dir='shap_analysis_weather/results_simple'):
        self.args = args
        self.output_dir = output_dir
        self.device = args.device
        
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, 'plots'), exist_ok=True)
        
        # Initialize model
        self.exp = Exp_Long_Term_Forecast(args)
        self.model = self.exp.model
        self.model.eval()
        
        self.weather_features = [
            'temp', 'dew', 'humidity', 'winddir',
            'windspeed', 'pressure', 'cloudcover'
        ]
        
        print(f"Initialized Simple SHAP Analyzer")
        print(f"Output: {output_dir}")
    
    def load_model(self, checkpoint_path):
        """Load trained model"""
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.model.eval()
        print(f"Loaded model from: {checkpoint_path}")
    
    def prepare_data(self, num_samples=200):
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
    
    def create_predict_function(self, x_mark, y_mark):
        """Create prediction function for SHAP"""
        def predict(x_input):
            # Average predictions across sequence (simpler for SHAP)
            batch_x = torch.FloatTensor(x_input).to(self.device)
            batch_x_mark = torch.FloatTensor(x_mark[:len(x_input)]).to(self.device)
            batch_y_mark = torch.FloatTensor(y_mark[:len(x_input)]).to(self.device)
            
            batch_size = batch_x.shape[0]
            dec_inp_zeros = torch.zeros(
                (batch_size, self.args.pred_len, batch_x.shape[-1])
            ).float().to(self.device)
            dec_inp_label = batch_x[:, -self.args.label_len:, :]
            dec_inp = torch.cat([dec_inp_label, dec_inp_zeros], dim=1)
            
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # Average prediction over horizon
                predictions = outputs[:, -self.args.pred_len:, -1].mean(dim=1)
            
            return predictions.cpu().numpy()
        
        return predict
    
    def compute_shap_kernel(self, x_data, x_mark, y_mark, n_background=50, n_explain=100):
        """Compute SHAP using KernelExplainer"""
        print("\n" + "="*80)
        print("Computing SHAP values using KernelExplainer")
        print("This may take 15-30 minutes...")
        print("="*80)
        
        # Average over time dimension for simpler analysis
        x_avg = x_data.mean(axis=1)[:, :len(self.weather_features)]
        
        background = x_avg[:n_background]
        explain = x_avg[n_background:n_background+n_explain]
        
        print(f"Background samples: {len(background)}")
        print(f"Samples to explain: {len(explain)}")
        
        # Create predict function
        predict_fn = self.create_predict_function(x_mark, y_mark)
        
        # Test prediction
        test_pred = predict_fn(x_data[:2])
        print(f"Test prediction shape: {test_pred.shape}")
        
        # Create explainer
        print("\nCreating KernelExplainer...")
        explainer = shap.KernelExplainer(
            lambda x: predict_fn(
                np.repeat(x[:, np.newaxis, :], self.args.seq_len, axis=1)
            ),
            background
        )
        
        # Compute SHAP values
        print("Computing SHAP values...")
        shap_values = explainer.shap_values(explain, nsamples=100)
        
        print(f"\nSHAP values computed: {shap_values.shape}")
        
        return shap_values, explain, explainer
    
    def plot_simple_summary(self, shap_values, data, feature_names):
        """Create simple summary plots"""
        print("\nCreating visualizations...")
        
        # Summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, data, feature_names=feature_names, show=False)
        plt.title('SHAP Summary: Weather Feature Importance', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_summary.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Bar plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, data, feature_names=feature_names, plot_type='bar', show=False)
        plt.title('Feature Importance Rankings', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'plots', 'shap_importance_bar.png'), dpi=300, bbox_inches='tight')
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
        
        csv_path = os.path.join(self.output_dir, 'plots', 'feature_importance.csv')
        importance_df.to_csv(csv_path, index=False)
        
        print(f"\nFeature Importance Rankings:")
        print(importance_df.to_string(index=False))
        print(f"\nResults saved to: {csv_path}")


def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple SHAP Analysis')
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
    print("SIMPLE SHAP ANALYSIS (KernelExplainer)")
    print("="*80 + "\n")
    
    # Initialize analyzer
    analyzer = SimpleSHAPAnalyzer(args)
    
    # Load model
    analyzer.load_model(args.checkpoint_path)
    
    # Prepare data
    x_data, y_data, x_mark, y_mark = analyzer.prepare_data(num_samples=200)
    
    # Compute SHAP
    shap_values, explain_data, explainer = analyzer.compute_shap_kernel(
        x_data, x_mark, y_mark, n_background=50, n_explain=100
    )
    
    # Create plots
    analyzer.plot_simple_summary(shap_values, explain_data, analyzer.weather_features)
    
    # Save results
    analyzer.save_results(shap_values, explain_data, analyzer.weather_features)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
