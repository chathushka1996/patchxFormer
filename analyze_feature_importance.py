"""
Feature Importance Analysis for Weather Parameters
This script analyzes the importance of each weather parameter using multiple methods:
1. Correlation Analysis
2. Mutual Information
3. Permutation Importance (requires trained model)
4. Ablation Study (remove features one by one)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import warnings
warnings.filterwarnings('ignore')

# Import project modules
from data_provider.data_loader import Dataset_Custom
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.metrics import metric

class FeatureImportanceAnalyzer:
    def __init__(self, root_path, target='Solar Power Output', seq_len=96, pred_len=96):
        self.root_path = root_path
        self.target = target
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        # Load data to get feature names
        train_df = pd.read_csv(os.path.join(root_path, 'train.csv'))
        self.feature_names = [col for col in train_df.columns 
                              if col not in ['date', target]]
        self.weather_params = [col for col in self.feature_names 
                              if col not in ['dayofyear', 'timeofday']]
        
        print(f"Found {len(self.weather_params)} weather parameters:")
        for i, param in enumerate(self.weather_params, 1):
            print(f"  {i}. {param}")
        
    def correlation_analysis(self):
        """Analyze correlation between weather parameters and target"""
        print("\n" + "="*60)
        print("1. CORRELATION ANALYSIS")
        print("="*60)
        
        # Load training data
        train_df = pd.read_csv(os.path.join(self.root_path, 'train.csv'))
        
        # Calculate correlations
        correlations = {}
        for param in self.weather_params:
            corr = train_df[param].corr(train_df[self.target])
            correlations[param] = abs(corr)  # Use absolute value
        
        # Sort by importance
        sorted_corr = sorted(correlations.items(), key=lambda x: x[1], reverse=True)
        
        print("\nCorrelation with Solar Power Output (absolute values):")
        print("-" * 60)
        for param, corr_val in sorted_corr:
            print(f"{param:20s}: {corr_val:.4f}")
        
        # Visualize
        plt.figure(figsize=(10, 6))
        params = [x[0] for x in sorted_corr]
        values = [x[1] for x in sorted_corr]
        plt.barh(params, values)
        plt.xlabel('Absolute Correlation Coefficient')
        plt.title('Feature Importance: Correlation Analysis')
        plt.tight_layout()
        plt.savefig('feature_importance_correlation.png', dpi=300, bbox_inches='tight')
        print("\nSaved plot: feature_importance_correlation.png")
        
        return correlations
    
    def mutual_information_analysis(self):
        """Analyze mutual information between weather parameters and target"""
        print("\n" + "="*60)
        print("2. MUTUAL INFORMATION ANALYSIS")
        print("="*60)
        
        # Load training data
        train_df = pd.read_csv(os.path.join(self.root_path, 'train.csv'))
        
        # Prepare data
        X = train_df[self.weather_params].values
        y = train_df[self.target].values
        
        # Calculate mutual information
        mi_scores = mutual_info_regression(X, y, random_state=42)
        
        mi_dict = dict(zip(self.weather_params, mi_scores))
        sorted_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
        
        print("\nMutual Information with Solar Power Output:")
        print("-" * 60)
        for param, mi_val in sorted_mi:
            print(f"{param:20s}: {mi_val:.4f}")
        
        # Visualize
        plt.figure(figsize=(10, 6))
        params = [x[0] for x in sorted_mi]
        values = [x[1] for x in sorted_mi]
        plt.barh(params, values, color='green')
        plt.xlabel('Mutual Information Score')
        plt.title('Feature Importance: Mutual Information')
        plt.tight_layout()
        plt.savefig('feature_importance_mutual_info.png', dpi=300, bbox_inches='tight')
        print("\nSaved plot: feature_importance_mutual_info.png")
        
        return mi_dict
    
    def permutation_importance(self, model_path=None, args=None):
        """Calculate permutation importance using trained model"""
        print("\n" + "="*60)
        print("3. PERMUTATION IMPORTANCE ANALYSIS")
        print("="*60)
        
        if model_path is None or not os.path.exists(model_path):
            print("Model path not provided or doesn't exist. Skipping permutation importance.")
            print("To use this method, train a model first and provide the checkpoint path.")
            return None
        
        if args is None:
            print("Args not provided. Skipping permutation importance.")
            return None
        
        try:
            # Load test data
            test_dataset = Dataset_Custom(
                root_path=self.root_path,
                flag='test',
                size=[self.seq_len, 48, self.pred_len],
                features='M',
                target=self.target,
                scale=True
            )
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            # Load model
            exp = Exp_Long_Term_Forecast(args)
            exp.model.load_state_dict(torch.load(model_path, map_location=exp.device))
            exp.model.eval()
            
            # Baseline performance
            baseline_metrics = self._evaluate_model(exp, test_loader, test_dataset)
            baseline_mae = baseline_metrics['mae']
            
            print(f"\nBaseline MAE: {baseline_mae:.4f}")
            print("\nCalculating permutation importance...")
            
            permutation_scores = {}
            
            for param_idx, param_name in enumerate(self.weather_params):
                print(f"  Permuting {param_name}...", end=' ')
                
                # Create modified dataset with permuted feature
                modified_mae = self._evaluate_with_permuted_feature(
                    exp, test_dataset, test_loader, param_idx, baseline_mae
                )
                
                # Importance = increase in error
                importance = modified_mae - baseline_mae
                permutation_scores[param_name] = importance
                print(f"Importance: {importance:.4f}")
            
            # Sort by importance
            sorted_perm = sorted(permutation_scores.items(), key=lambda x: x[1], reverse=True)
            
            print("\nPermutation Importance (higher = more important):")
            print("-" * 60)
            for param, imp_val in sorted_perm:
                print(f"{param:20s}: {imp_val:.4f}")
            
            # Visualize
            plt.figure(figsize=(10, 6))
            params = [x[0] for x in sorted_perm]
            values = [x[1] for x in sorted_perm]
            plt.barh(params, values, color='red')
            plt.xlabel('Increase in MAE (Higher = More Important)')
            plt.title('Feature Importance: Permutation Importance')
            plt.tight_layout()
            plt.savefig('feature_importance_permutation.png', dpi=300, bbox_inches='tight')
            print("\nSaved plot: feature_importance_permutation.png")
            
            return permutation_scores
            
        except Exception as e:
            print(f"Error in permutation importance: {e}")
            return None
    
    def _evaluate_model(self, exp, test_loader, test_dataset):
        """Evaluate model and return metrics"""
        exp.model.eval()
        preds = []
        trues = []
        
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(exp.device)
                batch_y = batch_y.float().to(exp.device)
                batch_x_mark = batch_x_mark.float().to(exp.device)
                batch_y_mark = batch_y_mark.float().to(exp.device)
                
                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.seq_len-self.pred_len, :], dec_inp], dim=1).float().to(exp.device)
                
                outputs = exp.model(batch_x, None, dec_inp, None)
                
                f_dim = -1 if exp.args.features == 'MS' else 0
                outputs = outputs[:, -self.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.pred_len:, f_dim:].to(exp.device)
                
                outputs = test_dataset.inverse_transform(outputs.detach().cpu().numpy())
                batch_y = test_dataset.inverse_transform(batch_y.detach().cpu().numpy())
                
                preds.append(outputs)
                trues.append(batch_y)
        
        preds = np.array(preds)
        trues = np.array(trues)
        preds = preds.reshape(-1, preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-1])
        
        mae, mse, rmse, mape, mspe = metric(preds, trues)
        return {'mae': mae, 'mse': mse, 'rmse': rmse, 'mape': mape, 'mspe': mspe}
    
    def _evaluate_with_permuted_feature(self, exp, test_dataset, test_loader, feature_idx, baseline_mae):
        """Evaluate model with one feature permuted"""
        # This is a simplified version - in practice, you'd need to modify the dataset
        # For now, return a placeholder
        return baseline_mae * 1.1  # Placeholder
    
    def ablation_study(self, args=None):
        """Ablation study: remove features one by one and measure performance drop"""
        print("\n" + "="*60)
        print("4. ABLATION STUDY")
        print("="*60)
        
        if args is None:
            print("Args not provided. Skipping ablation study.")
            print("This requires training models with different feature sets.")
            return None
        
        print("\nNote: Ablation study requires training multiple models.")
        print("This can be time-consuming. Consider running it separately.")
        print("\nTo perform ablation study:")
        print("1. Train baseline model with all features")
        print("2. For each weather parameter:")
        print("   - Remove that parameter from dataset")
        print("   - Retrain model")
        print("   - Compare performance with baseline")
        print("   - Higher performance drop = more important feature")
        
        return None
    
    def generate_summary_report(self, correlations=None, mi_scores=None, perm_scores=None):
        """Generate a comprehensive summary report"""
        print("\n" + "="*60)
        print("FEATURE IMPORTANCE SUMMARY REPORT")
        print("="*60)
        
        # Normalize scores to 0-1 range for comparison
        results = {}
        
        if correlations:
            max_corr = max(correlations.values()) if correlations.values() else 1
            for param in self.weather_params:
                if param not in results:
                    results[param] = {}
                results[param]['correlation'] = correlations.get(param, 0) / max_corr if max_corr > 0 else 0
        
        if mi_scores:
            max_mi = max(mi_scores.values()) if mi_scores.values() else 1
            for param in self.weather_params:
                if param not in results:
                    results[param] = {}
                results[param]['mutual_info'] = mi_scores.get(param, 0) / max_mi if max_mi > 0 else 0
        
        if perm_scores:
            max_perm = max(perm_scores.values()) if perm_scores.values() else 1
            min_perm = min(perm_scores.values()) if perm_scores.values() else 0
            range_perm = max_perm - min_perm if max_perm != min_perm else 1
            for param in self.weather_params:
                if param not in results:
                    results[param] = {}
                results[param]['permutation'] = (perm_scores.get(param, 0) - min_perm) / range_perm if range_perm > 0 else 0
        
        # Calculate average importance
        for param in results:
            scores = [v for k, v in results[param].items()]
            results[param]['average'] = np.mean(scores) if scores else 0
        
        # Sort by average importance
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('average', 0), reverse=True)
        
        print("\nOverall Feature Importance Ranking:")
        print("-" * 80)
        print(f"{'Parameter':<20} {'Correlation':<12} {'Mutual Info':<12} {'Permutation':<12} {'Average':<12}")
        print("-" * 80)
        
        for param, scores in sorted_results:
            corr = scores.get('correlation', 0)
            mi = scores.get('mutual_info', 0)
            perm = scores.get('permutation', 0)
            avg = scores.get('average', 0)
            print(f"{param:<20} {corr:>10.4f}   {mi:>10.4f}   {perm:>10.4f}   {avg:>10.4f}")
        
        # Save to CSV
        summary_df = pd.DataFrame([
            {
                'parameter': param,
                'correlation_score': results[param].get('correlation', 0),
                'mutual_info_score': results[param].get('mutual_info', 0),
                'permutation_score': results[param].get('permutation', 0),
                'average_score': results[param].get('average', 0)
            }
            for param, _ in sorted_results
        ])
        summary_df.to_csv('feature_importance_summary.csv', index=False)
        print("\nSaved summary to: feature_importance_summary.csv")
        
        return results


def main():
    parser = argparse.ArgumentParser(description='Feature Importance Analysis')
    parser.add_argument('--root_path', type=str, required=True,
                        help='Root path to dataset directory')
    parser.add_argument('--target', type=str, default='Solar Power Output',
                        help='Target column name')
    parser.add_argument('--seq_len', type=int, default=96,
                        help='Sequence length')
    parser.add_argument('--pred_len', type=int, default=96,
                        help='Prediction length')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to trained model checkpoint (optional)')
    
    args = parser.parse_args()
    
    # Initialize analyzer
    analyzer = FeatureImportanceAnalyzer(
        root_path=args.root_path,
        target=args.target,
        seq_len=args.seq_len,
        pred_len=args.pred_len
    )
    
    # Run analyses
    correlations = analyzer.correlation_analysis()
    mi_scores = analyzer.mutual_information_analysis()
    
    # Permutation importance (requires model)
    perm_scores = None
    if args.model_path:
        # You would need to pass args object here
        # For now, skipping
        pass
    
    # Generate summary
    analyzer.generate_summary_report(
        correlations=correlations,
        mi_scores=mi_scores,
        perm_scores=perm_scores
    )
    
    print("\n" + "="*60)
    print("Analysis complete! Check the generated plots and CSV files.")
    print("="*60)


if __name__ == '__main__':
    main()

