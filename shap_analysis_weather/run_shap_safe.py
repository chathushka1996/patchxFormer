"""
Updated SHAP Analysis Runner - Uses Inplace-Safe Version

This script automatically runs the inplace-safe version of SHAP analysis
which handles models with in-place operations properly.
"""

import os
import subprocess
import sys

def find_latest_checkpoint():
    """Find the latest trained model checkpoint"""
    checkpoint_dirs = [
        './checkpoints',
        './drive/MyDrive/msc-val/model_log',
        './model_checkpoints'
    ]
    
    for check_dir in checkpoint_dirs:
        if os.path.exists(check_dir):
            for root, dirs, files in os.walk(check_dir):
                for file in files:
                    if file == 'checkpoint.pth':
                        checkpoint_path = os.path.join(root, file)
                        print(f"Found checkpoint: {checkpoint_path}")
                        return checkpoint_path
    
    return None


def run_shap_analysis_safe():
    """Run inplace-safe SHAP analysis"""
    
    print("="*80)
    print("SHAP ANALYSIS RUNNER FOR PATCHXFORMER (INPLACE-SAFE VERSION)")
    print("="*80)
    print()
    
    # Find checkpoint
    checkpoint_path = find_latest_checkpoint()
    
    if checkpoint_path is None:
        print("ERROR: No model checkpoint found!")
        print("Please train a model first or specify checkpoint path manually.")
        print()
        print("To specify manually, run:")
        print("python shap_analysis_weather/shap_analysis_inplace_safe.py --checkpoint_path YOUR_PATH")
        return
    
    print(f"Using checkpoint: {checkpoint_path}")
    print()
    
    # Run SHAP analysis with inplace-safe version
    print("Running SHAP analysis (Inplace-Safe Version)...")
    print("This uses KernelExplainer which is slower but works with models that have in-place operations")
    print("-"*80)
    
    cmd = [
        sys.executable,
        'shap_analysis_weather/shap_analysis_inplace_safe.py',
        '--checkpoint_path', checkpoint_path,
        '--pred_len', '96',
        '--seq_len', '96',
        '--d_model', '256',
        '--d_ff', '512',
        '--e_layers', '2',
        '--n_heads', '8',
        '--batch_size', '16'
    ]
    
    try:
        subprocess.run(cmd, check=True)
        print("\nSHAP analysis completed successfully!")
        print(f"Results saved to: shap_analysis_weather/results_inplace_safe")
        print()
        print("Key output files:")
        print("  - plots/shap_summary_beeswarm.png")
        print("  - plots/shap_importance_bar.png")
        print("  - plots/shap_dependence_plots.png")
        print("  - csv_reports/feature_importance.csv")
        print("  - csv_reports/shap_values_detailed.csv")
    except subprocess.CalledProcessError as e:
        print(f"\nError running SHAP analysis: {e}")
        print("Please check the error messages above.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
    
    print()
    print("="*80)


if __name__ == '__main__':
    run_shap_analysis_safe()
