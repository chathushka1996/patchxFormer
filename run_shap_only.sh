#!/bin/bash
export CUDA_VISIBLE_DEVICES=0

# ==============================================
# SHAP Analysis ONLY (Using Existing Checkpoint)
# No training - just runs SHAP analysis
# ==============================================

# Configuration
model_name=PatchXFormer
seq_len=96
pred_len=96
dataset=sl_piliyandala
root_path_name=./dataset/$dataset
data_path_name=solar.csv
model_id_name=solar_$dataset

# Checkpoint directory (where your trained model is saved)
checkpoints=./checkpoints/

# Model hyperparameters (must match training)
d_model=256
d_ff=512
e_layers=2
n_heads=8
dropout=0.1

# Create output directory
mkdir -p ./shap_results

echo "=============================================="
echo "SHAP Analysis ONLY (Using Existing Checkpoint)"
echo "=============================================="
echo "Model: $model_name"
echo "Dataset: $dataset"
echo "Sequence Length: $seq_len"
echo "Prediction Length: $pred_len"
echo "Checkpoint Path: $checkpoints"
echo "=============================================="

# Run SHAP analysis only (no training)
python -u shap_analysis.py \
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
  --d_layers 1 \
  --factor 3 \
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --dropout $dropout \
  --embed timeF \
  --checkpoints $checkpoints \
  --gpu_type cuda \
  --num_samples 200 \
  --background_samples 100 \
  --output_dir ./shap_results/${dataset}_pred${pred_len}/

echo ""
echo "=============================================="
echo "SHAP Analysis Complete!"
echo "=============================================="
echo "Results saved to: ./shap_results/${dataset}_pred${pred_len}/"
echo "=============================================="
