
# print start info
TZ="Asia/HongKong"
curr_time="`date +"%Y%m%d_%H_%M_%S"`"
curr_sec=$(date +%s)
echo "Start $curr_time"
echo kill $$

export COMET_API_KEY="ff2z1CAs1CJqbr4pF9J3eI4Ui"
CUDA_VISIBLE_DEVICES=0
# build and execute command
# cmd="python -u gpt2/train_e2e.py --optim_prefix yes --preseqlen 5 --epoch 5 --learning_rate 0.00008 --mode data2text --bsz 10 --seed 101 --tuning_mode prefixtune --cache_dir ./cache"
cmd="python -u gpt2/train_e2e.py --optim_prefix yes --preseqlen 1 --epoch 1 --learning_rate 0.00008 --mode data2text --bsz 10 --seed 101 --tuning_mode prefixtune --cache_dir ./cache --max_steps 10 --eval_steps 10"
log_file="\"log/$curr_time prefix_e2e_train.log\""

# checkpoint_path="/home/ubuntu/vinc/PrefixTuning/save_e2e_models_convcheck/data2textprefixtune_y_5_act_cat_b=10-e=5_d=0.0_u=no_lr=8e-05_w=0.0_s=101_r=n_m=512_o=1_o=1"
# cmd="python -u gpt2/gen.py data2text yes valid $checkpoint_path no"
# log_file="\"log/$curr_time prefix_e2e_valid.log\""
# cmd="python -u gpt2/gen.py data2text yes test $checkpoint_path no"
# log_file="\"log/$curr_time prefix_e2e_test.log\""

final_cmd="{ $cmd; } &>> $log_file"
echo $final_cmd
eval $final_cmd

# print end info
curr_time="`date +"%Y%m%d_%H_%M_%S"`"
echo "End $curr_time"
elapsed_time=$(($(date +%s)-$curr_sec))
echo "Elapsed time ${elapsed_time}s"