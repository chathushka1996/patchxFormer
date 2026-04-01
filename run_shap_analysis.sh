#!/bin/bash
# SHAP-based Explainability Analysis for PatchXFormer
# This script runs comprehensive SHAP analysis on trained PatchXFormer models
# to understand which weather parameters contribute most to solar power predictions.

export CUDA_VISIBLE_DEVICES=0

# Configuration
dataset=sl_piliyandala
root_path_name=./dataset/$dataset
model_name=PatchXFormer
seq_len=96
pred_len=96

# Model configuration (should match your trained model)
d_model=256
d_ff=512
e_layers=2
n_heads=8

# SHAP analysis parameters
num_samples=100
background_samples=50
output_dir=./shap_results/${dataset}_${pred_len}

echo "=============================================="
echo "PatchXFormer SHAP Explainability Analysis"
echo "=============================================="
echo "Dataset: $dataset"
echo "Model: $model_name"
echo "Sequence Length: $seq_len"
echo "Prediction Length: $pred_len"
echo "Output Directory: $output_dir"
echo "=============================================="

# Create output directory
mkdir -p $output_dir

# Run SHAP analysis
python -u shap_analysis.py \
    --task_name long_term_forecast \
    --model_id solar_${dataset}${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --root_path $root_path_name \
    --features M \
    --target "Solar Power Output" \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --enc_in 10 \
    --dec_in 10 \
    --c_out 10 \
    --d_model $d_model \
    --d_ff $d_ff \
    --n_heads $n_heads \
    --dropout 0.1 \
    --num_samples $num_samples \
    --background_samples $background_samples \
    --output_dir $output_dir \
    --batch_size 32 \
    --checkpoints ./checkpoints/

echo "=============================================="
echo "SHAP analysis completed!"
echo "Results saved to: $output_dir"
echo "=============================================="
echo ""
echo "Generated files:"
echo "  - feature_importance.csv: Feature importance rankings"
echo "  - feature_importance_bar.png/pdf: Feature importance visualization"
echo "  - shap_summary_plot.png/pdf: SHAP summary plot"
echo "  - weather_contribution_analysis.png/pdf: Weather parameter analysis"
echo "  - weather_contribution_stats.csv: Weather statistics"
echo "  - partial_dependence/: Partial dependence plots"
echo "  - local_explanation_*.png: Local SHAP explanations"
echo "=============================================="
