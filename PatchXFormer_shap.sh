export CUDA_VISIBLE_DEVICES=0

# Configuration
model_name=PatchXFormer
seq_len=96
dataset=sl_piliyandala
root_path_name=./dataset/$dataset
data_path_name=solar.csv
model_id_name=solar_$dataset
data_name=custom
random_seed=2021

# Checkpoint directory - IMPORTANT: This is where trained models are saved
checkpoints=./checkpoints/

# Create directories
mkdir -p $checkpoints
mkdir -p ./logs
mkdir -p ./shap_results

echo "=============================================="
echo "PatchXFormer Training + SHAP Analysis"
echo "(With Normalization-Aware Feature Importance)"
echo "=============================================="
echo "Model: $model_name"
echo "Dataset: $dataset"
echo "Sequence Length: $seq_len"
echo "Checkpoints will be saved to: $checkpoints"
echo "=============================================="

# Train only pred_len=96 for SHAP analysis (faster)
# Change to: for pred_len in 96 192 336 720 if you want all
for pred_len in 96
do
  echo ""
  echo "=========================================="
  echo "Training for prediction length: $pred_len"
  echo "=========================================="
  
  # Configuration for each prediction length
  if [ $pred_len -eq 96 ]; then
    d_model=256
    d_ff=512
    e_layers=2
    n_heads=8
    batch_size=16
    train_epochs=12
    patience=4
    learning_rate=0.0001
    dropout=0.1
  elif [ $pred_len -eq 192 ]; then
    d_model=384
    d_ff=768
    e_layers=2
    n_heads=8
    batch_size=12
    train_epochs=15
    patience=4
    learning_rate=0.0001
    dropout=0.1
  elif [ $pred_len -eq 336 ]; then
    d_model=512
    d_ff=1024
    e_layers=3
    n_heads=8
    batch_size=8
    train_epochs=18
    patience=4
    learning_rate=0.00008
    dropout=0.12
  elif [ $pred_len -eq 720 ]; then
    d_model=512
    d_ff=1024
    e_layers=3
    n_heads=8
    batch_size=6
    train_epochs=20
    patience=4
    learning_rate=0.00005
    dropout=0.15
  fi
  
  echo "Config: d_model=$d_model, d_ff=$d_ff, e_layers=$e_layers, batch_size=$batch_size"
  
  # TRAINING
  python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path_name \
    --data_path $data_path_name \
    --model_id ${model_id_name}${seq_len}_${pred_len} \
    --model $model_name \
    --data custom \
    --features M \
    --seq_len $seq_len \
    --label_len 48 \
    --pred_len $pred_len \
    --e_layers $e_layers \
    --d_layers 1 \
    --factor 3 \
    --enc_in 10 \
    --dec_in 10 \
    --c_out 10 \
    --des 'Exp' \
    --d_model $d_model \
    --d_ff $d_ff \
    --n_heads $n_heads \
    --batch_size $batch_size \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --dropout $dropout \
    --checkpoints $checkpoints \
    --itr 1
  
  echo ""
  echo "Training completed for pred_len=$pred_len"
  echo "Checkpoint saved to: $checkpoints"
  
  # RUN SHAP ANALYSIS after training
  echo ""
  echo "=========================================="
  echo "Running SHAP Analysis for pred_len=$pred_len"
  echo "=========================================="
  
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
  echo "SHAP analysis completed for pred_len=$pred_len"
  echo "Results saved to: ./shap_results/${dataset}_pred${pred_len}/"
  echo "=========================================="
  
done

echo ""
echo "=============================================="
echo "All training and SHAP analysis completed!"
echo "=============================================="
echo ""
echo "Checkpoints: $checkpoints"
echo "SHAP Results: ./shap_results/"
echo ""
echo "Generated files:"
echo "  - feature_importance.csv"
echo "  - weather_contribution_stats.csv"
echo "  - data_scaling_info.csv (NEW: scaler parameters)"
echo "  - data_scaling_analysis.png/pdf (NEW: scaling visualization)"
echo "  - shap_summary_plot.png/pdf"
echo "  - feature_importance_bar.png/pdf"
echo "  - SHAP_ANALYSIS_REPORT.txt (with scaling analysis)"
echo "==============================================" 