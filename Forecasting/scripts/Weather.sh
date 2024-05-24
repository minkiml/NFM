# sh ./Forecasting/scripts/Weather.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=4
data_path_name='./datasets/forecasting_data'
data=weather
pred_lens=(96 192 336 720)
run_id=12
random_seed=88
look_backs=(720 720 720 720) 
freq_span=-1
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
      --input_c 21 \
      --hidden_dim 36 \
      --hidden_factor 3 \
      --ff_projection_ex 3 \
      --inff_siren_hidden 32\
      --inff_siren_omega 30\
      --layer_num 1 \
      --dropout 0.1 \
      --siren_hidden 32 \
      --siren_in_dim 32 \
      --siren_omega 30 \
      --loss_type TFDR \
      --channel_dependence 0 \
      --n_epochs 60 \
      --scheduler 0 \
      --warm_up 0.1 \
      --final_lr 0.00015 \
      --ref_lr 0.00025 \
      --start_lr 0.00025 \
      --description "" \
      --gpu_dev 6 \
      --patience 6 \
      --batch 1680 --batch_testing 200 --lr_ 0.00025
done

# sr mode
for (( i=0; i<${runs}; i++ ));
do
for sr in 4 6;
do
    pred_len=${pred_lens[$i]}
    look_back=${look_backs[$i]}
    half_look_back=$((look_back / ${sr}))
    half_pred_len=$((pred_len / ${sr}))
    python -u FC_main.py \
      --mode test \
      --seed $random_seed \
      --data_path $data_path_name \
      --id_ $run_id \
      --dataset $data \
      --look_back $look_back \
      --horizon $pred_len \
      --freq_span $freq_span \
      --vars_in_train $look_back $look_back $pred_len $look_back \
      --vars_in_test $look_back $half_look_back $pred_len $half_look_back \
      --filter_type INFF \
      --input_c 21 \
      --hidden_dim 36 \
      --hidden_factor 3 \
      --ff_projection_ex 3 \
      --inff_siren_hidden 32\
      --inff_siren_omega 30\
      --layer_num 1 \
      --dropout 0.1 \
      --siren_hidden 32 \
      --siren_in_dim 32 \
      --siren_omega 30 \
      --loss_type TFDR \
      --channel_dependence 0 \
      --n_epochs 40 \
      --scheduler 0 \
      --description "" \
      --gpu_dev 7 \
      --batch 1680 --batch_testing 64 --lr_ 0.0002
done
done