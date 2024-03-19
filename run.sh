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
nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_trans1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_ffn1.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='m4_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_mqcnn2.out 2>&1 &

# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_deepar2.out 2>&1 &
nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_trans2.out 2>&1 &
# nohup python run_norm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --mean_str=-1 --var_str=-1 > nohup_out/current_gpu_ffn2.out 2>&1 &
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
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_mqcnn_nonorm1.out 2>&1 &

# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='deepar' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='transformer' --ctx='gpu' --n_trials=5 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='feedforward' --ctx='gpu' --n_trials=10 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='fred_md' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 --standardize > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &

# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='deepar' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='transformer' --ctx='gpu' --n_trials=5 > nohup_out/current_gpu_ffn_nonorm1.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='feedforward' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_trans_nonorm2.out 2>&1 &
# nohup python run_nonorm_w_tuning.py --dataset_name='nn5_weekly' --model_choice='mqcnn' --ctx='gpu' --n_trials=10 > nohup_out/current_gpu_mqcnn_nonorm1.out 2>&1 &
