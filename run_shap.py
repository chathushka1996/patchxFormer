"""
Run SHAP Analysis for PatchXFormer - Python Entry Point

This script provides an easy way to run SHAP analysis on Windows.
Usage:
    python run_shap.py --dataset sl_piliyandala --pred_len 96
    python run_shap.py  # Uses default settings
"""

import os
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run SHAP Analysis for PatchXFormer')
    parser.add_argument('--dataset', type=str, default='sl_piliyandala',
                       help='Dataset name (default: sl_piliyandala)')
    parser.add_argument('--pred_len', type=int, default=96,
                       help='Prediction length (default: 96)')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to analyze (default: 100)')
    parser.add_argument('--background_samples', type=int, default=50,
                       help='Background samples for SHAP (default: 50)')
    
    args = parser.parse_args()
    
    # Model configuration based on prediction length
    config = {
        96: {'d_model': 256, 'd_ff': 512, 'e_layers': 2, 'n_heads': 8},
        192: {'d_model': 384, 'd_ff': 768, 'e_layers': 2, 'n_heads': 8},
        336: {'d_model': 512, 'd_ff': 1024, 'e_layers': 3, 'n_heads': 8},
        720: {'d_model': 512, 'd_ff': 1024, 'e_layers': 3, 'n_heads': 8},
    }
    
    model_config = config.get(args.pred_len, config[96])
    
    output_dir = f'./shap_results/{args.dataset}_pred{args.pred_len}'
    
    print("=" * 60)
    print("PatchXFormer SHAP Explainability Analysis")
    print("=" * 60)
    print(f"Dataset: {args.dataset}")
    print(f"Prediction Length: {args.pred_len}")
    print(f"Output Directory: {output_dir}")
    print("=" * 60)
    
    # Build command
    cmd = f"""python shap_analysis.py \
        --task_name long_term_forecast \
        --model_id solar_{args.dataset}96_{args.pred_len} \
        --model PatchXFormer \
        --data custom \
        --root_path ./dataset/{args.dataset} \
        --features M \
        --target "Solar Power Output" \
        --seq_len 96 \
        --label_len 48 \
        --pred_len {args.pred_len} \
        --e_layers {model_config['e_layers']} \
        --enc_in 10 \
        --dec_in 10 \
        --c_out 10 \
        --d_model {model_config['d_model']} \
        --d_ff {model_config['d_ff']} \
        --n_heads {model_config['n_heads']} \
        --dropout 0.1 \
        --num_samples {args.num_samples} \
        --background_samples {args.background_samples} \
        --output_dir {output_dir} \
        --batch_size 32 \
        --checkpoints ./checkpoints/"""
    
    # Execute
    os.system(cmd)
    
    print("\n" + "=" * 60)
    print("Analysis Complete!")
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == '__main__':
    main()
