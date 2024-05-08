# sh ./Classification/scripts/speechcommand_raw.sh 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=1

pred_lens=0
look_backs=16000 
freq_spans=-1
fs=16000 
data_path_name='./datasets/classification_data'
data=SpeechCommands
random_seed=88
run_id=0
test_sr=1
for (( i=0; i<${runs}; i++ ));
do

    test_lookback=$((look_backs / test_sr))
    python -u CL_main.py \
      --seed $random_seed \
      --data_path $data_path_name \
      --id_ $run_id \
      --dataset $data \
      --freq_span $freq_spans \
      --vars_in_train $fs $look_backs $pred_lens $look_backs \
      --vars_in_test $fs $test_lookback $pred_lens $test_lookback \
      --mode train \
      --filter_type INFF \
      --input_c 1 \
      --hidden_dim 32 \
      --hidden_factor 3 \
      --inff_siren_hidden 32\
      --inff_siren_omega 35\
      --layer_num 2 \
      --dropout 0.00 \
      --siren_hidden 32 \
      --siren_in_dim 32 \
      --siren_omega 35\
      --loss_type TFDR \
      --channel_dependence 1 \
      --n_epochs 300 \
      --scheduler 1 \
      --warm_up 0.1 \
      --final_lr 0.00035 \
      --ref_lr 0.0007 \
      --start_lr 0.0007 \
      --description "_" \
      --gpu_dev 2\
      --sr_train 1 \
      --sr_test $test_sr \
      --dropped_rate 0 \
      --mfcc 0 \
      --num_class 10\
      --batch 128 --batch_testing 300 --lr_ 0.0005
done