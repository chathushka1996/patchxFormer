#!/bin/bash
# ============================================================================
# Comprehensive SHAP Analysis for PatchXFormer Solar Power Forecasting
# ============================================================================
# 
# This script runs comprehensive SHAP-based explainability analysis following:
# 1. PLOS ONE: "Solar energy prediction through machine learning models" (2025)
# 2. C-SHAP for Time Series (arXiv:2504.11159v1)
# 3. Interpretable Machine Learning (Christoph Molnar)
#
# USAGE:
#   chmod +x run_comprehensive_shap.sh
#   ./run_comprehensive_shap.sh
#
# ============================================================================

export CUDA_VISIBLE_DEVICES=0

# Configuration
model_name=PatchXFormer
seq_len=96
pred_len=96
dataset=sl_piliyandala
root_path_name=./dataset/$dataset
data_path_name=solar.csv
model_id_name=solar_$dataset
data_name=custom

# Model configuration (must match trained model)
d_model=256
d_ff=512
e_layers=2
n_heads=8
d_layers=1
factor=3
enc_in=10
dec_in=10
c_out=10
dropout=0.1

# SHAP Analysis configuration
num_samples=200
n_repeats=30
batch_size=32

# Directories
checkpoints=./checkpoints/
output_dir=./shap_comprehensive_results/

# Create directories
mkdir -p $output_dir
mkdir -p ./logs

echo ""
echo "============================================================================"
echo "COMPREHENSIVE SHAP EXPLAINABILITY ANALYSIS"
echo "PatchXFormer Solar Power Forecasting"
echo "============================================================================"
echo ""
echo "Model Configuration:"
echo "  - Model: $model_name"
echo "  - Sequence Length: $seq_len"
echo "  - Prediction Length: $pred_len"
echo "  - d_model: $d_model, d_ff: $d_ff"
echo "  - e_layers: $e_layers, n_heads: $n_heads"
echo ""
echo "Analysis Configuration:"
echo "  - Number of samples: $num_samples"
echo "  - Permutation repeats: $n_repeats"
echo "  - Output directory: $output_dir"
echo ""
echo "References:"
echo "  - PLOS ONE: https://pmc.ncbi.nlm.nih.gov/articles/PMC11695015/"
echo "  - C-SHAP: https://arxiv.org/html/2504.11159v1"
echo "  - SHAP Book: https://christophm.github.io/interpretable-ml-book/shap.html"
echo ""
echo "============================================================================"
echo ""

# Run comprehensive SHAP analysis
echo "Starting comprehensive SHAP analysis..."
echo ""

python -u shap_comprehensive_analysis.py \
  --task_name long_term_forecast \
  --model_id ${model_id_name}${seq_len}_${pred_len} \
  --model $model_name \
  --data custom \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --target "Solar Power Output" \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --d_layers $d_layers \
  --factor $factor \
  --enc_in $enc_in \
  --dec_in $dec_in \
  --c_out $c_out \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --dropout $dropout \
  --embed timeF \
  --expand 2 \
  --d_conv 4 \
  --checkpoints $checkpoints \
  --batch_size $batch_size \
  --num_samples $num_samples \
  --n_repeats $n_repeats \
  --output_dir $output_dir

echo ""
echo "============================================================================"
echo "ANALYSIS COMPLETE!"
echo "============================================================================"
echo ""
echo "Output files saved to: $output_dir"
echo ""
echo "Generated visualizations (similar to PLOS ONE paper):"
echo "  - correlation_analysis.png/pdf       : Figure 3 style - Correlation matrix"
echo "  - shap_global_interpretation.png/pdf : Figure 7 style - Global SHAP values"
echo "  - shap_partial_dependence_*.png/pdf  : Figure 8 style - Partial dependence"
echo "  - shap_local_interpretation.png/pdf  : Figure 9 style - Local SHAP values"
echo "  - lime_explanation_*.png/pdf         : Figure 10 style - LIME explanations"
echo "  - shap_concept_based_analysis.png/pdf: C-SHAP methodology - Concept groups"
echo ""
echo "Generated CSV files:"
echo "  - correlation_matrix.csv             : Feature correlations"
echo "  - vif_analysis.csv                   : Variance Inflation Factor"
echo "  - feature_importance_all.csv         : All features importance"
echo "  - feature_importance_weather.csv     : Weather-only importance"
echo "  - concept_importance.csv             : Concept-level importance"
echo "  - shap_values_all_samples.csv        : Raw SHAP values"
echo ""
echo "Report:"
echo "  - COMPREHENSIVE_SHAP_REPORT.txt      : Full analysis report"
echo ""
echo "============================================================================"
