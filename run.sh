#!/bin/bash
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --root_folder='' --model_choice='ffn' --ctx='cpu' --n_trials=10 > nohup_out/nn5_weekly_dataset.out 2>&1 &
# wait $!
# nohup python run_norm_w_tuning.py --dataset_name='rideshare_without_missing' --root_folder='' --model_choice='ffn' --ctx='cpu' --n_trials=10 > nohup_out/nn5_weekly_dataset.out 2>&1 &
# wait $!
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --root_folder='' --model_choice='ffn' --ctx='cpu' --n_trials=10 > nohup_out/nn5_weekly_dataset.out 2>&1 &
# wait $!

# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --root_folder='' --model_choice='ffn' --ctx='gpu' --n_trials=10 > nohup_out/current.out 2>&1 &
# wait $!
# nohup python run_nonorm_w_tuning.py --dataset_name='rideshare_without_missing' --root_folder='' --model_choice='ffn' --ctx='gpu' --n_trials=10 > nohup_out/current.out 2>&1 &
# wait $!
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --root_folder='' --model_choice='ffn' --ctx='gpu' --n_trials=10 > nohup_out/current.out 2>&1 &
# wait $!

# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=1 > nohup_out/current_gpu.out 2>&1 &
# wait $!
# nohup python run_nonorm_w_tuning.py --dataset_name='rideshare_without_missing' --model_choice='deepar' --ctx='gpu' --n_trials=1 > nohup_out/current_gpu.out 2>&1 &
# wait $!
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=1 > nohup_out/current_gpu.out 2>&1 &
# wait $!

# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='cpu' --n_trials=1 > nohup_out/current_cpu.out 2>&1 &
# wait $!
# nohup python run_norm_w_tuning.py --dataset_name='rideshare_without_missing' --model_choice='deepar' --ctx='cpu' --n_trials=1 > nohup_out/current_cpu.out 2>&1 &
# wait $!
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='cpu' --n_trials=1 > nohup_out/current_cpu.out 2>&1 &
# wait $!

# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=10 > nohup_out/current_cpu.out 2>&1 &
# wait $!
# nohup python run_norm_w_tuning.py --dataset_name='rideshare_without_missing' --model_choice='deepar' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu.out 2>&1 &
# wait $!
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=10 > nohup_out/current_cpu.out 2>&1 &
# wait $!

# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu.out 2>&1 &
# wait $!
# nohup python run_nonorm_w_tuning.py --dataset_name='rideshare_without_missing' --model_choice='transformer' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu.out 2>&1 &
# wait $!


# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu.out 2>&1 &
# wait $!
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu.out 2>&1 &
# wait $!
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu.out 2>&1 &
# wait $!
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu2.out 2>&1 &
# wait $!
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu.out 2>&1 &
# wait $!
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu5.out 2>&1 &
# wait $!

# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_ffn_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_ffn_nonorm2.out 2>&1 &

# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 > nohup_out/current_gpu_ffn_nonorm1_batch.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 > nohup_out/current_gpu_ffn_nonorm2_batch.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 --standardize > nohup_out/current_gpu_ffn_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 --standardize > nohup_out/current_gpu_ffn_nonorm2.out 2>&1 &

# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.5 --var_str=0.5 > nohup_out/current_gpu_ffn1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.5 --var_str=0.5 > nohup_out/current_gpu_ffn2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.1 --var_str=0.1 > nohup_out/current_gpu_ffn3.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.1 --var_str=0.1 > nohup_out/current_gpu_ffn4.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.01 --var_str=0.01 > nohup_out/current_gpu_ffn5.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.01 --var_str=0.01 > nohup_out/current_gpu_ffn6.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.001 --var_str=0.001 > nohup_out/current_gpu_ffn7.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.001 --var_str=0.001 > nohup_out/current_gpu_ffn8.out 2>&1 &

# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.5 --var_str=0.5 > nohup_out/current_gpu_deepar1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.5 --var_str=0.5 > nohup_out/current_gpu_deepar2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.1 --var_str=0.1 > nohup_out/current_gpu_deepar3.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.1 --var_str=0.1 > nohup_out/current_gpu_deepar4.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.01 --var_str=0.01 > nohup_out/current_gpu_deepar5.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.01 --var_str=0.01 > nohup_out/current_gpu_deepar6.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.001 --var_str=0.001 > nohup_out/current_gpu_deepar7.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.001 --var_str=0.001 > nohup_out/current_gpu_deepar8.out 2>&1 &

# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_trans_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &

# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.5 --var_str=0.5 > nohup_out/current_gpu_trans1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.5 --var_str=0.5 > nohup_out/current_gpu_trans2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.1 --var_str=0.1 > nohup_out/current_gpu_trans3.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.1 --var_str=0.1 > nohup_out/current_gpu_trans4.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.01 --var_str=0.01 > nohup_out/current_gpu_trans5.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.01 --var_str=0.01 > nohup_out/current_gpu_trans6.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.001 --var_str=0.001 > nohup_out/current_gpu_trans7.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.001 --var_str=0.001 > nohup_out/current_gpu_trans8.out 2>&1 &



# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.5 --var_str=0.000001 > nohup_out/current_gpu_ffn1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.5 --var_str=0.000001 > nohup_out/current_gpu_ffn2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.1 --var_str=0.000001 > nohup_out/current_gpu_ffn3.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.1 --var_str=0.000001 > nohup_out/current_gpu_ffn4.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.01 --var_str=0.000001 > nohup_out/current_gpu_ffn5.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.01 --var_str=0.000001 > nohup_out/current_gpu_ffn6.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.001 --var_str=0.000001 > nohup_out/current_gpu_ffn7.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0.001 --var_str=0.000001 > nohup_out/current_gpu_ffn8.out 2>&1 &

# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.5 --var_str=0.000001 > nohup_out/current_gpu_trans1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.5 --var_str=0.000001 > nohup_out/current_gpu_trans2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.1 --var_str=0.000001 > nohup_out/current_gpu_trans3.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.1 --var_str=0.000001 > nohup_out/current_gpu_trans4.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.01 --var_str=0.000001 > nohup_out/current_gpu_trans5.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.01 --var_str=0.000001 > nohup_out/current_gpu_trans6.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.001 --var_str=0.000001 > nohup_out/current_gpu_trans7.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0.001 --var_str=0.000001 > nohup_out/current_gpu_trans8.out 2>&1 &

# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_deepar_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_deepar_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_trans_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --standardize > nohup_out/current_gpu_trans_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_mqcnn_nonorm1_std.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_mqcnn_nonorm2_std.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=5  > nohup_out/current_gpu_mqcnn_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=5  > nohup_out/current_gpu_mqcnn_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_trans_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &

# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_deepar_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_deepar_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_trans_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_trans_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &


# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.5 --var_str=0.000001 > nohup_out/current_gpu_deepar1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.5 --var_str=0.000001 > nohup_out/current_gpu_deepar2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.1 --var_str=0.000001 > nohup_out/current_gpu_deepar3.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.1 --var_str=0.000001 > nohup_out/current_gpu_deepar4.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.01 --var_str=0.000001 > nohup_out/current_gpu_deepar5.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.01 --var_str=0.000001 > nohup_out/current_gpu_deepar6.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.001 --var_str=0.000001 > nohup_out/current_gpu_deepar7.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0.001 --var_str=0.000001 > nohup_out/current_gpu_deepar8.out 2>&1 &

# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 > nohup_out/current_gpu_wave_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 > nohup_out/current_gpu_wave_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 --standardize > nohup_out/current_gpu_wave_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 --standardize > nohup_out/current_gpu_wave_nonorm2.out 2>&1 &

# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 --mean_str=0.5 --var_str=0.000001 > nohup_out/current_gpu_wave1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 --mean_str=0.5 --var_str=0.000001 > nohup_out/current_gpu_wave2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 --mean_str=0.1 --var_str=0.000001 > nohup_out/current_gpu_wave3.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 --mean_str=0.1 --var_str=0.000001 > nohup_out/current_gpu_wave4.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 --mean_str=0.01 --var_str=0.000001 > nohup_out/current_gpu_wave5.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 --mean_str=0.01 --var_str=0.000001 > nohup_out/current_gpu_wave6.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 --mean_str=0.001 --var_str=0.000001 > nohup_out/current_gpu_wave7.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='wavenet' --ctx='gpu(1)' --n_trials=5 --mean_str=0.001 --var_str=0.000001 > nohup_out/current_gpu_wave8.out 2>&1 &


# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_deepar1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_deepar2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_trans1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_trans2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_ffn2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_ffn1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_mqcnn2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_mqcnn1.out 2>&1 &

# nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_deepar1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_trans1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_ffn1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_mqcnn2.out 2>&1 &

# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_deepar2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_trans1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_ffn1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_mqcnn2.out 2>&1 &

# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_deepar2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_trans2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_ffn2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_mqcnn2.out 2>&1 &


# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_ffn_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_mqcnn_nonorm1.out 2>&1 &

# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &

# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_ffn_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_trans_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_mqcnn_nonorm1.out 2>&1 &

# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &

# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_ffn_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_mqcnn_nonorm1.out 2>&1 &







# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_trans_gas_fred1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0 --var_str=0 > nohup_out/current_gpu_trans_gas_fred2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_trans_default_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu(1)' --n_trials=5 --standardize > nohup_out/current_gpu_trans_overall_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --batch_norm > nohup_out/current_gpu_trans_batch_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_scaling > nohup_out/current_gpu_trans_mean_fred1.out 2>&1 &


# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_trans_gas_nn5w1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=0 --var_str=0 > nohup_out/current_gpu_trans_gas_nn5w2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_trans_default_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu(1)' --n_trials=5 --standardize > nohup_out/current_gpu_trans_overall_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu(1)' --n_trials=5 --batch_norm > nohup_out/current_gpu_trans_batch_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu(1)' --n_trials=5 --mean_scaling > nohup_out/current_gpu_trans_mean_nn5w1.out 2>&1 &


nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='transformer' --ctx='gpu(1)' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_trans_gas_m4w1.out 2>&1 &
nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='transformer' --ctx='gpu(1)' --n_trials=5 --mean_str=0 --var_str=0 > nohup_out/current_gpu_trans_gas_m4w2.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='transformer' --ctx='gpu(1)' --n_trials=5 > nohup_out/current_gpu_trans_default_m4w1.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='transformer' --ctx='gpu(1)' --n_trials=5 --standardize > nohup_out/current_gpu_trans_overall_m4w1.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='transformer' --ctx='gpu(1)' --n_trials=5 --batch_norm > nohup_out/current_gpu_trans_batch_m4w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='transformer' --ctx='gpu(1)' --n_trials=5 --mean_scaling > nohup_out/current_gpu_trans_mean_m4w1.out 2>&1 &


# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_ffn_gas_fred1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 --mean_str=0 --var_str=0 > nohup_out/current_gpu_ffn_gas_fred2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_ffn_default_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 --standardize > nohup_out/current_gpu_ffn_overall_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 --batch_norm > nohup_out/current_gpu_ffn_batch_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 --mean_scaling > nohup_out/current_gpu_ffn_mean_fred1.out 2>&1 &


# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_ffn_gas_nn5w1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0 --var_str=0 > nohup_out/current_gpu_ffn_gas_nn5w2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_ffn_default_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 --standardize > nohup_out/current_gpu_ffn_overall_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 --batch_norm > nohup_out/current_gpu_ffn_batch_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu(1)' --n_trials=10 --mean_scaling > nohup_out/current_gpu_ffn_mean_nn5w1.out 2>&1 &


nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_ffn_gas_m4w1.out 2>&1 &
nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=0 --var_str=0 > nohup_out/current_gpu_ffn_gas_m4w2.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_ffn_default_m4w1.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --standardize > nohup_out/current_gpu_ffn_overall_m4w1.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --batch_norm > nohup_out/current_gpu_ffn_batch_m4w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_scaling > nohup_out/current_gpu_ffn_mean_m4w1.out 2>&1 &


# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_deepar_gas_fred1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0 --var_str=0 > nohup_out/current_gpu_deepar_gas_fred2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_deepar_default_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu(1)' --n_trials=5 --standardize > nohup_out/current_gpu_deepar_overall_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu(1)' --n_trials=5 --batch_norm > nohup_out/current_gpu_deepar_batch_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu(1)' --n_trials=5 --mean_scaling > nohup_out/current_gpu_deepar_mean_fred1.out 2>&1 &


# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_deepar_gas_nn5w1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=0 --var_str=0 > nohup_out/current_gpu_deepar_gas_nn5w2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_deepar_default_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_deepar_overall_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu(1)' --n_trials=5 --batch_norm > nohup_out/current_gpu_deepar_batch_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu(1)' --n_trials=5 --mean_scaling > nohup_out/current_gpu_deepar_mean_nn5w1.out 2>&1 &


nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='deepar' --ctx='gpu(1)' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_deepar_gas_m4w1.out 2>&1 &
nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='deepar' --ctx='gpu(1)' --n_trials=5 --mean_str=0 --var_str=0 > nohup_out/current_gpu_deepar_gas_m4w2.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_deepar_default_m4w1.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_deepar_overall_m4w1.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --batch_norm > nohup_out/current_gpu_deepar_batch_m4w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_scaling > nohup_out/current_gpu_deepar_mean_m4w1.out 2>&1 &


# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_mqcnn_gas_fred1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --mean_str=0 --var_str=0 > nohup_out/current_gpu_mqcnn_gas_fred2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_mqcnn_default_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_mqcnn_overall_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --batch_norm > nohup_out/current_gpu_mqcnn_batch_fred1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --mean_scaling > nohup_out/current_gpu_mqcnn_mean_fred1.out 2>&1 &


# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_mqcnn_gas_nn5w1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --mean_str=0 --var_str=0 > nohup_out/current_gpu_mqcnn_gas_nn5w2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_mqcnn_default_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_mqcnn_overall_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --batch_norm > nohup_out/current_gpu_mqcnn_batch_nn5w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=5 --mean_scaling > nohup_out/current_gpu_mqcnn_mean_nn5w1.out 2>&1 &


nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='mqcnn' --ctx='gpu(1)' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_mqcnn_gas_m4w1.out 2>&1 &
nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='mqcnn' --ctx='gpu(1)' --n_trials=5 --mean_str=0 --var_str=0 > nohup_out/current_gpu_mqcnn_gas_m4w2.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='mqcnn' --ctx='gpu(1)' --n_trials=5 > nohup_out/current_gpu_mqcnn_default_m4w1.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='mqcnn' --ctx='gpu(1)' --n_trials=5 --standardize > nohup_out/current_gpu_mqcnn_overall_m4w1.out 2>&1 &
nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='mqcnn' --ctx='gpu(1)' --n_trials=5 --batch_norm > nohup_out/current_gpu_mqcnn_batch_m4w1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='m4_weekly' --model_choice='mqcnn' --ctx='gpu(1)' --n_trials=5 --mean_scaling > nohup_out/current_gpu_mqcnn_mean_m4w1.out 2>&1 &
