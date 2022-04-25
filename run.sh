
# print exec time
now="`date +"%Y%m%d_%H_%M_%S"`";
now_format="`date +"%Y/%m/%d %T"`";
echo $now_format;





COMET_API_KEY=ff2z1CAs1CJqbr4pF9J3eI4Ui nohup time python -u gpt2/train_e2e.py --optim_prefix yes --preseqlen 5 --epoch 5 --learning_rate 0.00008 --mode data2text --bsz 10 --seed 100 --tuning_mode prefixtune --cache_dir ./cache > "log/$now prefix.log" &

# COMET_API_KEY=ff2z1CAs1CJqbr4pF9J3eI4Ui nohup time python -u gpt2/gen.py data2text yes valid /home/ubuntu/vinc/PrefixTuning/save_e2e_models_convcheck/data2textprefixtune_y_5_act_cat_b=10-e=5_d=0.0_u=no_lr=8e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 no > "log/$now valid1.log" &
# COMET_API_KEY=ff2z1CAs1CJqbr4pF9J3eI4Ui nohup time python -u gpt2/gen.py data2text yes test /home/ubuntu/vinc/PrefixTuning/save_e2e_models_convcheck/data2textprefixtune_y_5_act_cat_b=10-e=5_d=0.0_u=no_lr=8e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 no > "log/$now test1.log" &

# COMET_API_KEY=ff2z1CAs1CJqbr4pF9J3eI4Ui nohup time python -u gpt2/run_language_modeling.py --output_dir=save_e2e_models_convcheck/data2textprefixtune_y_5_act_cat_b=10-e=5_d=0.0_u=no_lr=8e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 --model_type=gpt2 --model_name_or_path=gpt2-medium --tokenizer_name=gpt2-medium --per_device_train_batch_size 10 --per_device_eval_batch_size 10 --save_steps 500000 --num_train_epochs 5 --do_train --train_data_file=data/e2e_data/src1_train.txt --do_eval --line_by_line --save_total_limit 1 --overwrite_output_dir --task_mode data2text --eval_data_file=data/e2e_data/src1_valid.txt --tuning_mode prefixtune --logging_dir save_e2e_models_convcheck/runs/data2textprefixtune_y_5_act_cat_b=10-e=5_d=0.0_u=no_lr=8e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 --train_embs no --optim_prefix yes --preseqlen 5 --prefix_mode activation --format_mode cat --gradient_accumulation_steps 1 --learning_rate 8e-05 --weight_decay 0.0 --seed 101 --disable_tqdm --mid_dim 512 --init_random no --use_dropout no --prefix_dropout 0.0 --objective_mode 1 --evaluate_during_training --eval_steps 5000 --cache_dir cache/gpt2-medium-s3 > nohup_$now.out &

# print pid
echo kill $$
echo kill $!
