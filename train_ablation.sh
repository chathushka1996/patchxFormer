#!/bin/bash

# PatchXFormer Ablation Study Training Script
# This script trains all ablation variants for comparison

export CUDA_VISIBLE_DEVICES=0
path=./drive/MyDrive/msc-val
if [ ! -d "$path/logs" ]; then
    mkdir $path/logs -p
fi

model_name=PatchXFormer_Ablation
seq_len=96
dataset=sl_piliyandala
root_path_name=./dataset/$dataset
data_path_name=solar.csv
model_id_name=ablation_solar_$dataset
data_name=custom
random_seed=2021
checkpoints=$path/model_log

echo "=========================================="
echo "PatchXFormer Ablation Study"
echo "=========================================="
echo "Dataset: $dataset"
echo "Sequence Length: $seq_len"
echo ""

# Ablation variants to test
variants=("baseline" "enhanced_patch" "freq_attention" "adaptive_norm" "cross_attn" "full")

# Forecasting horizons
horizons=(96 192 336 720)

# Hyperparameters (adjust based on your needs)
declare -A d_models
declare -A d_ffs
declare -A e_layers
declare -A batch_sizes
declare -A learning_rates
declare -A dropouts

# Hyperparameters for each horizon
d_models[96]=256
d_models[192]=384
d_models[336]=512
d_models[720]=512

d_ffs[96]=512
d_ffs[192]=768
d_ffs[336]=1024
d_ffs[720]=1024

e_layers[96]=2
e_layers[192]=2
e_layers[336]=3
e_layers[720]=3

batch_sizes[96]=16
batch_sizes[192]=12
batch_sizes[336]=8
batch_sizes[720]=6

learning_rates[96]=0.0001
learning_rates[192]=0.0001
learning_rates[336]=0.00008
learning_rates[720]=0.00005

dropouts[96]=0.1
dropouts[192]=0.1
dropouts[336]=0.12
dropouts[720]=0.15

# Train each variant for each horizon
for pred_len in "${horizons[@]}"
do
  echo ""
  echo "=========================================="
  echo "Forecasting Horizon: $pred_len steps"
  echo "=========================================="
  
  d_model=${d_models[$pred_len]}
  d_ff=${d_ffs[$pred_len]}
  e_layer=${e_layers[$pred_len]}
  batch_size=${batch_sizes[$pred_len]}
  lr=${learning_rates[$pred_len]}
  dropout=${dropouts[$pred_len]}
  
  echo "Hyperparameters: d_model=$d_model, e_layers=$e_layer, batch_size=$batch_size, lr=$lr"
  echo ""
  
  for variant in "${variants[@]}"
  do
    echo "----------------------------------------"
    echo "Training Variant: $variant"
    echo "----------------------------------------"
    
    python -u run.py \
      --task_name long_term_forecast \
      --is_training 1 \
      --root_path $root_path_name \
      --data_path $data_path_name \
      --model_id ${model_id_name}_${variant}_${seq_len}_${pred_len} \
      --model $model_name \
      --ablation_variant $variant \
      --data $data_name \
      --features M \
      --seq_len $seq_len \
      --label_len 48 \
      --pred_len $pred_len \
      --e_layers $e_layer \
      --factor 3 \
      --enc_in 10 \
      --dec_in 10 \
      --c_out 10 \
      --des "Ablation_${variant}" \
      --d_model $d_model \
      --d_ff $d_ff \
      --n_heads 8 \
      --batch_size $batch_size \
      --learning_rate $lr \
      --dropout $dropout \
      --train_epochs 15 \
      --patience 4 \
      --random_seed $random_seed \
      --itr 1 \
      --checkpoints $checkpoints
    
    if [ $? -eq 0 ]; then
      echo "✓ Successfully trained $variant for horizon $pred_len"
    else
      echo "✗ Failed to train $variant for horizon $pred_len"
    fi
    echo ""
  done
done

echo "=========================================="
echo "Ablation Study Training Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Collect results from logs"
echo "2. Compare performance across variants"
echo "3. Analyze component contributions"
echo "4. Create ablation study tables and graphs"

