# sh ./Forecasting/scripts/Elec.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=4

pred_lens=(96 192 336 720)
look_backs=(720 720 720 720)
freq_span=-1
data_path_name='./datasets/forecasting_data'
data=electricity
random_seed=88
run_id=12

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
      --input_c 321 \
      --hidden_dim 32 \
      --hidden_factor 3 \
      --inff_siren_hidden 32\
      --inff_siren_omega 35\
      --layer_num 1 \
      --dropout 0.1 \
      --siren_hidden 32 \
      --siren_in_dim 32 \
      --siren_omega 35\
      --loss_type TFDR \
      --channel_dependence 0 \
      --n_epochs 50 \
      --scheduler 1 \
      --warm_up 0.1 \
      --final_lr 0.00015 \
      --ref_lr 0.00035 \
      --start_lr 0.00035 \
      --description "_" \
      --gpu_dev 1 \
      --batch 1000 --batch_testing 16 --lr_ 0.0001
done
