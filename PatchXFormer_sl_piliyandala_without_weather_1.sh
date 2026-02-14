export CUDA_VISIBLE_DEVICES=0
path=./drive/MyDrive/msc-val
if [ ! -d "$path/logs" ]; then
    mkdir $path/logs -p
fi
model_name=PatchXFormer
seq_len=96
dataset=sl_piliyandala_without_weather_1
root_path_name=./dataset/$dataset
data_path_name=solar.csv
model_id_name=solar_$dataset
data_name=custom
random_seed=2021
checkpoints=$path/model_log

echo "Starting PatchXFormer evaluation on solar dataset..."
echo "Model: $model_name"
echo "Dataset: $dataset"
echo "Sequence Length: $seq_len"

for pred_len in 96 192 336 720
do
  echo "Training for prediction length: $pred_len"
  
  # Optimized configuration for stability and performance
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
  
  echo "Configuration: d_model=$d_model, e_layers=$e_layers, batch_size=$batch_size"
  
  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --factor 3 \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --des 'Exp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout $dropout \
  --itr 1
  #--checkpoints $checkpoints > $path/logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 2>&1
  
  echo "Completed training for prediction length: $pred_len"
  echo "Log saved to: $path/logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log"
  echo "----------------------------------------"
done

echo "All PatchXFormer evaluations completed!"
echo "Check logs in: $path/logs/"

