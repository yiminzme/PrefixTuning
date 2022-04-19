
# print exec time
now="`date +"%Y%m%d_%H%M%S"`";
now_format="`date +"%Y/%m/%d %T"`";
echo $now_format;

# execute code in background using gpu1, and save output to nohup_now.out
COMET_API_KEY=ff2z1CAs1CJqbr4pF9J3eI4Ui nohup time python -u gpt2/run_language_modeling.py --output_dir=save_e2e_models_convcheck/data2textprefixtune_y_5_act_cat_b=10-e=5_d=0.0_u=no_lr=8e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 --model_type=gpt2 --model_name_or_path=gpt2-medium --tokenizer_name=gpt2-medium --per_device_train_batch_size 10 --per_device_eval_batch_size 10 --save_steps 500000 --num_train_epochs 5 --do_train --train_data_file=data/e2e_data/src1_train.txt --do_eval --line_by_line --save_total_limit 1 --overwrite_output_dir --task_mode data2text --eval_data_file=data/e2e_data/src1_valid.txt --tuning_mode prefixtune --logging_dir save_e2e_models_convcheck/runs/data2textprefixtune_y_5_act_cat_b=10-e=5_d=0.0_u=no_lr=8e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 --train_embs no --optim_prefix yes --preseqlen 5 --prefix_mode activation --format_mode cat --gradient_accumulation_steps 1 --learning_rate 8e-05 --weight_decay 0.0 --seed 101 --disable_tqdm --mid_dim 512 --init_random no --use_dropout no --prefix_dropout 0.0 --objective_mode 1 --evaluate_during_training --eval_steps 5000 --cache_dir cache/gpt2-medium-s3 > nohup_$now.out &

# COMET_API_KEY=ff2z1CAs1CJqbr4pF9J3eI4Ui nohup time python gpt2/run_generation.py --model_type=gpt2 --length 100 --model_name_or_path=gpt2-medium --num_return_sequences 5 --stop_token [EOS] --tokenizer_name=/home/ubuntu/vinc/PrefixTuning/save_e2e_models_convcheck/data2textprefixtune_y_5_act_cat_b=10-e=5_d=0.0_u=no_lr=8e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 --task_mode=data2text --control_mode=yes --tuning_mode prefixtune --gen_dir e2e_results_conv2 --eval_dataset valid --optim_prefix no --preseqlen 20 --prefix_mode activation --format_mode cat --prefixModel_name_or_path /home/ubuntu/vinc/PrefixTuning/save_e2e_models_convcheck/data2textprefixtune_y_5_act_cat_b=10-e=5_d=0.0_u=no_lr=8e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 --cache_dir cache/gpt2-medium-s3 > nohup_$now.out &

# COMET_API_KEY=ff2z1CAs1CJqbr4pF9J3eI4Ui nohup time python gpt2/run_generation.py --model_type=gpt2 --length 100 --model_name_or_path=gpt2-medium --num_return_sequences 5 --stop_token [EOS] --tokenizer_name=/home/ubuntu/vinc/PrefixTuning/save_e2e_models_convcheck/data2textprefixtune_y_5_act_cat_b=10-e=5_d=0.0_u=no_lr=8e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 --task_mode=data2text --control_mode=yes --tuning_mode prefixtune --gen_dir e2e_results_conv2 --eval_dataset test --optim_prefix no --preseqlen 20 --prefix_mode activation --format_mode cat --prefixModel_name_or_path /home/ubuntu/vinc/PrefixTuning/save_e2e_models_convcheck/data2textprefixtune_y_5_act_cat_b=10-e=5_d=0.0_u=no_lr=8e-05_w=0.0_s=101_r=n_m=512_o=1_o=1 --cache_dir cache/gpt2-medium-s3 > nohup_$now.out &

# print pid
echo kill $!
