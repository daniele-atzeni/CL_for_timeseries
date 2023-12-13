#!/bin/bash
nohup python run_norm_w_tuning.py --dataset_name='nn5_weekly_dataset' --root_folder='/data/r.ramjattan/CL_for_timeseries' --model_choice='ffn' --prediction_length=8 --context_length=65 --ctx='cpu' --n_trials=20 > nohup_out/nn5_weekly_dataset.out 2>&1 &
wait $!
nohup python run_norm_w_tuning.py --dataset_name='us_births_dataset' --root_folder='/data/r.ramjattan/CL_for_timeseries' --model_choice='ffn' --prediction_length=30 --context_length=9 --ctx='cpu' --n_trials=20 > nohup_out/us_births_dataset.out 2>&1 &
wait $!
nohup python run_norm_w_tuning.py --dataset_name='solar_10_minutes_dataset' --root_folder='/data/r.ramjattan/CL_for_timeseries' --model_choice='ffn' --prediction_length=1008 --context_length=50 --ctx='cpu' --n_trials=20 > nohup_out/solar_10_minutes_dataset.out 2>&1 &
wait $!
nohup python run_norm_w_tuning.py --dataset_name='weather_dataset' --root_folder='/data/r.ramjattan/CL_for_timeseries' --model_choice='ffn' --prediction_length=30 --context_length=9 --ctx='cpu' --n_trials=20 > nohup_out/weather_dataset.out 2>&1 &
wait $!
nohup python run_norm_w_tuning.py --dataset_name='sunspot_dataset_without_missing_values' --root_folder='/data/r.ramjattan/CL_for_timeseries' --model_choice='ffn' --prediction_length=30 --context_length=9 --ctx='cpu' --n_trials=20 > nohup_out/sunspot_dataset_without_missing_values.out 2>&1 &
