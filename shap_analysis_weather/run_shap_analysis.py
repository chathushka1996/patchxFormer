"""
Helper Script to Run SHAP Analysis on PatchXFormer Model

This script provides an easy way to run SHAP analysis on your trained
PatchXFormer model for solar power forecasting.

Usage:
    python run_shap_analysis.py

Make sure you have a trained model checkpoint available.
"""

import os
import subprocess
import sys

def find_latest_checkpoint():
    """Find the latest trained model checkpoint"""
    checkpoint_dir = './drive/MyDrive/msc-val/model_log'
    
    # Alternative checkpoint locations
    alternative_dirs = [
        './checkpoints',
        './drive/MyDrive/msc-val/model_log',
        './model_checkpoints'
    ]
    
    for check_dir in alternative_dirs:
        if os.path.exists(check_dir):
            # Find all checkpoint.pth files
            for root, dirs, files in os.walk(check_dir):
                for file in files:
                    if file == 'checkpoint.pth':
                        checkpoint_path = os.path.join(root, file)
                        print(f"Found checkpoint: {checkpoint_path}")
                        return checkpoint_path
    
    return None


def run_shap_analysis():
    """Run SHAP analysis with appropriate parameters"""
    
    print("="*80)
    print("SHAP ANALYSIS RUNNER FOR PATCHXFORMER")
    print("="*80)
    print()
    
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    
    if checkpoint_path is None:
        print("ERROR: No model checkpoint found!")
        print("Please train a model first or specify checkpoint path manually.")
        print()
        print("To specify manually, run:")
        print("python shap_analysis_weather/shap_patchxformer_analysis.py --checkpoint_path YOUR_PATH")
        return
    
    print(f"Using checkpoint: {checkpoint_path}")
    print()
    
    # Run SHAP analysis for pred_len=96 (24 hours)
    print("Running SHAP analysis for 24-hour prediction (pred_len=96)...")
    print("-"*80)
    
    cmd = [
        sys.executable,
        'shap_analysis_weather/shap_patchxformer_analysis.py',
        '--checkpoint_path', checkpoint_path,
        '--pred_len', '96',
        '--seq_len', '96',
        '--d_model', '256',
        '--d_ff', '512',
        '--e_layers', '2',
        '--n_heads', '8',
        '--batch_size', '16',
        '--num_samples', '500',
        '--background_samples', '100',
        '--output_dir', 'shap_analysis_weather/results_pred96'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\nSHAP analysis completed successfully!")
        print(f"Results saved to: shap_analysis_weather/results_pred96")
    except subprocess.CalledProcessError as e:
        print(f"\nError running SHAP analysis: {e}")
        print("Please check the error messages above.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    print()
    print("="*80)


if __name__ == '__main__':
    run_shap_analysis()
