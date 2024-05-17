# sh ./AnomalyD/scripts/SMD.sh 
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
run=1
runs=5
data_path_name='./datasets/anoamlyD_data/SMD'
data=SMD

pred_lens=0 # no m_t 
# look_backs=161
win_size=100 #16000
freqspan=-1
fs=$win_size #161 # 128
random_seed=99
run_id=16
gpu_de=4
dsr=2
input_length=$((win_size / dsr))
ars=(0.5 1 1.5 2 2.5)

for (( i=0; i<${run}; i++ ));
do
    python -u AD_main.py \
    --mode train \
    --seed $random_seed \
    --run_id $run_id \
    --data_path $data_path_name \
    --dataset $data \
    --win_size $win_size\
    --freq_span $freqspan \
    --vars_in_train $fs $input_length $pred_lens $input_length \
    --vars_in_test $fs $input_length $pred_lens $input_length \
    --DSR $dsr \
    --masking 0 \
    --input_c 38 \
    --output_c 38 \
    --anormly_ratio 0.5 \
    --filter_type INFF \
    --hidden_dim 8 \
    --hidden_factor 3 \
    --inff_siren_hidden 32 \
    --inff_siren_omega 30 \
    --layer_num 1 \
    --dropout 0.0 \
    --std 0.06 \
    --siren_hidden 32 \
    --siren_in_dim 32\
    --siren_omega 30 \
    --loss_type TFD \
    --revstat 0 \
    --num_epochs 150 \
    --gpu_dev $gpu_de \
    --batch_size 64 --lr 0.0001
done

input_length=$((win_size / dsr))

for (( i=0; i<${runs}; i++ ));
    do
        ar=${ars[$i]}
        python -u AD_main.py \
        --mode test \
        --seed $random_seed \
        --run_id $run_id \
        --data_path $data_path_name \
        --dataset $data \
        --win_size $win_size\
        --freq_span $freqspan \
        --vars_in_train $fs $input_length $pred_lens $input_length \
        --vars_in_test $fs $input_length $pred_lens $input_length \
        --DSR $dsr \
        --masking 0 \
        --input_c 38 \
        --output_c 38 \
        --anormly_ratio $ar \
        --filter_type INFF \
        --hidden_dim 8 \
        --hidden_factor 3 \
        --inff_siren_hidden 32 \
        --inff_siren_omega 30 \
        --layer_num 1 \
        --dropout 0.0 \
        --std 0.06 \
        --siren_hidden 32 \
        --siren_in_dim 32\
        --siren_omega 30 \
        --loss_type TFD \
        --revstat 0 \
        --num_epochs 1 \
        --gpu_dev $gpu_de \
        --batch_size 64 --lr 0.00005
    done
