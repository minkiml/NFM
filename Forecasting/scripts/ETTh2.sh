# sh ./Forecasting/scripts_exp/ETThs.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=1
data_path_name='./datasets/forecasting_data'
data=ETTh2

look_back=360
freq_span=-1
random_seed=88
run_id=19
batch_size=896
for (( i=0; i<${runs}; i++ ));
do
    pred_len=96
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
      --hidden_dim 36 \
      --hidden_factor 3 \
      --ff_projection_ex 3 \
      --inff_siren_hidden 32\
      --inff_siren_omega 30\
      --layer_num 1 \
      --dropout 0.05 \
      --tau independent \
      --siren_hidden 32 \
      --siren_in_dim 32 \
      --siren_omega 30\
      --loss_type TFDR \
      --lamda 0.5 \
      --channel_dependence 0 \
      --n_epochs 40 \
      --scheduler 1 \
      --warm_up 0.2 \
      --final_lr 0.0001\
      --ref_lr 0.00025 \
      --start_lr 0.00005 \
      --description "" \
      --gpu_dev 4 \
      --patience 6 \
      --batch $batch_size --batch_testing 64 --lr_ 0.0002
done

for (( i=0; i<${runs}; i++ ));
do
    pred_len=192
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
      --hidden_dim 36 \
      --hidden_factor 3 \
      --ff_projection_ex 3 \
      --inff_siren_hidden 32\
      --inff_siren_omega 30\
      --layer_num 1 \
      --dropout 0.05 \
      --tau independent \
      --siren_hidden 32 \
      --siren_in_dim 32 \
      --siren_omega 30\
      --loss_type TFDR \
      --lamda 0.5 \
      --channel_dependence 0 \
      --n_epochs 40 \
      --scheduler 1 \
      --warm_up 0.2 \
      --final_lr 0.0001\
      --ref_lr 0.00025 \
      --start_lr 0.00005 \
      --description "" \
      --gpu_dev 4 \
      --patience 6 \
      --batch $batch_size --batch_testing 64 --lr_ 0.0002
done

for (( i=0; i<${runs}; i++ ));
do
    pred_len=336
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
      --hidden_dim 36 \
      --hidden_factor 3 \
      --ff_projection_ex 3 \
      --inff_siren_hidden 32\
      --inff_siren_omega 30\
      --layer_num 1 \
      --dropout 0.05 \
      --tau independent \
      --siren_hidden 32 \
      --siren_in_dim 32 \
      --siren_omega 30\
      --loss_type TFDR \
      --lamda 0.5 \
      --channel_dependence 0 \
      --n_epochs 40 \
      --scheduler 1 \
      --warm_up 0.2 \
      --final_lr 0.0001\
      --ref_lr 0.00025 \
      --start_lr 0.00005 \
      --description "" \
      --gpu_dev 4 \
      --patience 6 \
      --batch $batch_size --batch_testing 64 --lr_ 0.0002
done


for (( i=0; i<${runs}; i++ ));
do
    pred_len=720
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
      --hidden_dim 36 \
      --hidden_factor 3 \
      --ff_projection_ex 3 \
      --inff_siren_hidden 32\
      --inff_siren_omega 30\
      --layer_num 1 \
      --dropout 0.05 \
      --tau independent \
      --siren_hidden 32 \
      --siren_in_dim 32 \
      --siren_omega 30\
      --loss_type TFDR \
      --lamda 0.5 \
      --channel_dependence 0 \
      --n_epochs 40 \
      --scheduler 1 \
      --warm_up 0.2 \
      --final_lr 0.0001\
      --ref_lr 0.00025 \
      --start_lr 0.00005 \
      --description "" \
      --gpu_dev 4 \
      --patience 6 \
      --batch $batch_size --batch_testing 64 --lr_ 0.0002
done