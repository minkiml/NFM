# sh ./Classification/scripts_exp/speechcommand_raw2.sh 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=1

pred_lens=0
look_backs=16000 
freq_spans=-1
fs=16000 
data_path_name='./datasets/classification_data'
data=SpeechCommands
random_seed=88
run_id=29
test_sr=1
dropouts=(0.0 0.05 0.1)
tv2=(0.25 0.3 0.4)
run_id=0
for (( i=0; i<${runs}; i++ ));
do
    tv=${tv2[$i]}
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
      --dropout 0.05 \
      --lft_norm 1 \
      --siren_hidden 32 \
      --siren_in_dim 32 \
      --siren_omega 35\
      --n_epochs 300 \
      --scheduler 1 \
      --warm_up 0.1 \
      --final_lr 0.00035 \
      --ref_lr 0.00075 \
      --start_lr 0.00075 \
      --description "" \
      --gpu_dev 2\
      --sr_train 1 \
      --sr_test $test_sr \
      --dropped_rate 0 \
      --mfcc 0 \
      --num_class 10\
      --batch 160 --batch_testing 300 --lr_ 0.0005
done