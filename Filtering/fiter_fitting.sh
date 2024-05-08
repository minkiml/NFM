# sh ./Filtering/fiter_fitting.sh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
runs=1

look_backs=10000
freq_span=-1
omegas=30
data_path_name=''
data=sim-filter

random_seed=88
run_id=0
for (( i=0; i<${runs}; i++ ));
do
    pred_len=0
    look_back=${look_backs[$i]}
    freq_span=${freq_spans[$i]}
    omega=${omegas[$i]}
    max_mod=$((look_back / 2 - 20))
    python -u Filter_main.py \
      --seed $random_seed \
      --data_path $data_path_name \
      --id_ $run_id \
      --dataset $data \
      --freq_span $freq_span \
      --vars_in_train $look_back $look_back $pred_len $look_back \
      --vars_in_test $look_back $look_back $pred_len $look_back \
      --filter_type INFF \
      --input_c 1 \
      --hidden_dim 32 \
      --hidden_factor 3 \
      --inff_siren_hidden 32\
      --inff_siren_omega 30\
      --layer_num 1 \
      --dropout 0.30 \
      --siren_hidden 32 \
      --siren_in_dim 256 \
      --siren_omega 30\
      --loss_type TFDR \
      --channel_dependence 1 \
      --n_epochs 100 \
      --scheduler 1 \
      --final_lr 0.0001 \
      --ref_lr 0.002 \
      --start_lr 0.0005 \
      --description "_" \
      --gpu_dev 6 \
      --num_class 10\
      --max_modes $max_mod \
      --diversity 1 \
      --num_modes 20 \
      --filter_mode "Bandpass" \
      --description "Bandpass" \
      --batch 100 --batch_testing 64 --lr_ 0.0005
done
