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

# Function to get hyperparameters based on prediction length
get_hyperparams() {
  local pred_len=$1
  case $pred_len in
    96)
      d_model=256
      d_ff=512
      e_layer=2
      batch_size=16
      lr=0.0001
      dropout=0.1
      ;;
    192)
      d_model=384
      d_ff=768
      e_layer=2
      batch_size=12
      lr=0.0001
      dropout=0.1
      ;;
    336)
      d_model=512
      d_ff=1024
      e_layer=3
      batch_size=8
      lr=0.00008
      dropout=0.12
      ;;
    720)
      d_model=512
      d_ff=1024
      e_layer=3
      batch_size=6
      lr=0.00005
      dropout=0.15
      ;;
    *)
      d_model=256
      d_ff=512
      e_layer=2
      batch_size=16
      lr=0.0001
      dropout=0.1
      ;;
  esac
}

# Train each variant for each horizon
for pred_len in "${horizons[@]}"
do
  echo ""
  echo "=========================================="
  echo "Forecasting Horizon: $pred_len steps"
  echo "=========================================="
  
  # Get hyperparameters for this horizon
  get_hyperparams $pred_len
  
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

