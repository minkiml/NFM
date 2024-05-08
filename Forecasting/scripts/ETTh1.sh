# sh ./Forecasting/scripts/ETThs.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=4
pred_lens=(96 192 336 720)
look_backs=(360 360 360 360) 
freq_span=-1
data_path_name='./datasets/forecasting_data'
data=ETTh1

random_seed=88
run_id=19
batch_size=840
for (( i=0; i<${runs}; i++ ));
do
    pred_len=${pred_lens[$i]}
    look_back=${look_backs[$i]}
    python -u FC_main.py \
      --seed $random_seed \
      --data_path $data_path_name \
      --id_ $run_id \
      --dataset $data \
      --look_back $look_back \
      --horizon $pred_len \
      --freq_span $freq_span \
      --vars_in_train $look_back $look_back $pred_len $look_back \
      --vars_in_test $look_back $look_back $pred_len $look_back \
      --filter_type INFF \
      --input_c 7 \
      --hidden_dim 32 \
      --hidden_factor 3 \
      --inff_siren_hidden 32\
      --inff_siren_omega 30\
      --layer_num 1 \
      --dropout 0.15 \
      --siren_hidden 32 \
      --siren_in_dim 32 \
      --siren_omega 30\
      --loss_type TFDR \
      --lamda 0.5 \
      --channel_dependence 0 \
      --n_epochs 100 \
      --scheduler 0 \
      --description "_" \
      --patience 6 \
      --gpu_dev 4 \
      --batch $batch_size --batch_testing 64 --lr_ 0.00015
done