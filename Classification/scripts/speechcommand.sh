# sh ./Classification/scripts_exp/speechcommand.sh 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=3
pred_lens=0
look_backs=161
freq_spans=-1
fs=161 
data_path_name='./datasets/classification_data'
data=SpeechCommands
random_seed=99
run_id=19

test_sr=1
dropouts=(0.15 0.2 0.25)
for (( i=0; i<${runs}; i++ ));
do
    dropout=${dropouts[$i]}
    test_lookback=$((look_backs / test_sr))
    python -u CL_main.py \
      --seed $random_seed \
      --data_path $data_path_name \
      --id_ $run_id \
      --dataset $data \
      --freq_span $freq_spans \
      --vars_in_train $fs $look_backs $pred_lens $look_backs \
      --vars_in_test $fs $test_lookback $pred_lens $test_lookback \
      --filter_type INFF \
      --input_c 20 \
      --hidden_dim 32 \
      --hidden_factor 3 \
      --inff_siren_hidden 32\
      --inff_siren_omega 35\
      --layer_num 2 \
      --dropout $dropout \
      --siren_hidden 32 \
      --siren_in_dim 32 \
      --siren_omega 35\
      --init_xaviar 1 \
      --n_epochs 300 \
      --scheduler 1 \
      --warm_up 0.1 \
      --final_lr 0.00025 \
      --ref_lr 0.0005 \
      --start_lr 0.0005 \
      --description "_$dropout" \
      --gpu_dev 7 \
      --sr_train 1 \
      --sr_test $test_sr \
      --dropped_rate 0 \
      --mfcc 1 \
      --num_class 10\
      --batch 240 --batch_testing 64 --lr_ 0.0007
done
