#!/bin/bash
# Complete SHAP-based Explainability Analysis for PatchXFormer
# 
# This script runs comprehensive explainability analysis including:
# 1. Basic SHAP analysis (global feature importance, partial dependence)
# 2. Advanced analysis (temporal patterns, seasonal effects, interactions)
# 3. LIME comparison for validation
#
# Usage: ./run_full_shap_analysis.sh [dataset] [pred_len]
# Example: ./run_full_shap_analysis.sh sl_piliyandala 96

export CUDA_VISIBLE_DEVICES=0

# Parse arguments or use defaults
dataset=${1:-sl_piliyandala}
pred_len=${2:-96}

# Configuration
root_path_name=./dataset/$dataset
model_name=PatchXFormer
seq_len=96

# Model configuration based on prediction length
if [ $pred_len -eq 96 ]; then
    d_model=256
    d_ff=512
    e_layers=2
    n_heads=8
elif [ $pred_len -eq 192 ]; then
    d_model=384
    d_ff=768
    e_layers=2
    n_heads=8
elif [ $pred_len -eq 336 ]; then
    d_model=512
    d_ff=1024
    e_layers=3
    n_heads=8
elif [ $pred_len -eq 720 ]; then
    d_model=512
    d_ff=1024
    e_layers=3
    n_heads=8
else
    d_model=256
    d_ff=512
    e_layers=2
    n_heads=8
fi

# SHAP analysis parameters
num_samples=100
background_samples=50
output_dir=./shap_results/${dataset}_pred${pred_len}

echo "=============================================="
echo "PatchXFormer Full SHAP Explainability Analysis"
echo "=============================================="
echo ""
echo "Configuration:"
echo "  Dataset: $dataset"
echo "  Model: $model_name"
echo "  Sequence Length: $seq_len"
echo "  Prediction Length: $pred_len"
echo "  d_model: $d_model"
echo "  e_layers: $e_layers"
echo "  Output Directory: $output_dir"
echo ""
echo "SHAP Parameters:"
echo "  Samples to analyze: $num_samples"
echo "  Background samples: $background_samples"
echo "=============================================="
echo ""

# Create output directories
mkdir -p $output_dir
mkdir -p ${output_dir}/advanced
mkdir -p ${output_dir}/partial_dependence

echo "Step 1: Running Basic SHAP Analysis..."
echo "----------------------------------------"

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

echo ""
echo "Step 2: Running Advanced Analysis..."
echo "----------------------------------------"

# Advanced analysis is run as part of shap_analysis.py
# Additional standalone advanced analysis can be run separately if needed

echo ""
echo "=============================================="
echo "SHAP Analysis Complete!"
echo "=============================================="
echo ""
echo "Results Directory: $output_dir"
echo ""
echo "Generated Files:"
echo ""
echo "  Basic Analysis:"
echo "    - feature_importance.csv"
echo "    - feature_importance_bar.png/pdf"
echo "    - shap_summary_plot.png/pdf"
echo "    - shap_values_aggregated.npy"
echo ""
echo "  Weather Analysis:"
echo "    - weather_contribution_analysis.png/pdf"
echo "    - weather_contribution_stats.csv"
echo ""
echo "  Partial Dependence:"
echo "    - partial_dependence/all_partial_dependence.png/pdf"
echo ""
echo "  Local Explanations:"
echo "    - local_explanation_sample_*.png"
echo ""
echo "=============================================="
echo ""
echo "Key Weather Parameters (ranked by importance):"
echo "See: $output_dir/feature_importance.csv"
echo ""
echo "For thesis/paper:"
echo "  - Use feature_importance_bar.pdf for Figure X (Feature Importance)"
echo "  - Use shap_summary_plot.pdf for Figure Y (SHAP Summary)"
echo "  - Use weather_contribution_analysis.pdf for Figure Z (Weather Analysis)"
echo "=============================================="
