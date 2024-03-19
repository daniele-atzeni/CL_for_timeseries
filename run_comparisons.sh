#!/bin/bash

# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice deepar --mean_str 0.001 --var_str 0.001 > comparisons/nohup_out/deepar_nn5_0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice deepar --mean_str 0.01 --var_str 0.01 > comparisons/nohup_out/deepar_nn5_0.01.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice deepar --mean_str 0.1 --var_str 0.1 > comparisons/nohup_out/deepar_nn5_0.1.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice deepar --mean_str 0.5 --var_str 0.5 > comparisons/nohup_out/deepar_nn5_0.5.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice deepar --mean_str 0.001 --var_str 0.001 > comparisons/nohup_out/deepar_fred_0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice deepar --mean_str 0.01 --var_str 0.01 > comparisons/nohup_out/deepar_fred_0.01.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice deepar --mean_str 0.1 --var_str 0.1 > comparisons/nohup_out/deepar_fred_0.1.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice deepar --mean_str 0.5 --var_str 0.5 > comparisons/nohup_out/deepar_fred_0.5.out 2>&1 &

# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice feedforward --mean_str 0.001 --var_str 0.001 > comparisons/nohup_out/ffn_nn5_0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice feedforward --mean_str 0.01 --var_str 0.01 > comparisons/nohup_out/ffn_nn5_0.01.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice feedforward --mean_str 0.1 --var_str 0.1 > comparisons/nohup_out/ffn_nn5_0.1.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice feedforward --mean_str 0.5 --var_str 0.5 > comparisons/nohup_out/ffn_nn5_0.5.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice feedforward --mean_str 0.001 --var_str 0.001 > comparisons/nohup_out/ffn_fred_0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice feedforward --mean_str 0.01 --var_str 0.01 > comparisons/nohup_out/ffn_fred_0.01.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice feedforward --mean_str 0.1 --var_str 0.1 > comparisons/nohup_out/ffn_fred_0.1.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice feedforward --mean_str 0.5 --var_str 0.5 > comparisons/nohup_out/ffn_fred_0.5.out 2>&1 &

# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice transformer --mean_str 0.001 --var_str 0.001 > comparisons/nohup_out/trans_nn5_0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice transformer --mean_str 0.01 --var_str 0.01 > comparisons/nohup_out/trans_nn5_0.01.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice transformer --mean_str 0.1 --var_str 0.1 > comparisons/nohup_out/trans_nn5_0.1.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice transformer --mean_str 0.5 --var_str 0.5 > comparisons/nohup_out/trans_nn5_0.5.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice transformer --mean_str 0.001 --var_str 0.001 > comparisons/nohup_out/trans_fred_0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice transformer --mean_str 0.01 --var_str 0.01 > comparisons/nohup_out/trans_fred_0.01.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice transformer --mean_str 0.1 --var_str 0.1 > comparisons/nohup_out/trans_fred_0.1.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice transformer --mean_str 0.1 --var_str 1e-06 > comparisons/nohup_out/trans_fred_0.1.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice deepar --mean_str 0.1 --var_str 1e-06 > comparisons/nohup_out/deepar_fred_0.1.out 2>&1 &
nohup python compare_predictions.py --dataset_name m4_weekly --model_choice transformer --mean_str 0 --var_str 0 > comparisons/nohup_out/trans_m4_0.001.out 2>&1 &
nohup python compare_predictions.py --dataset_name fred_md --model_choice transformer --mean_str 0 --var_str 0 > comparisons/nohup_out/trans_fred_0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name m4_weekly --model_choice feedforward --mean_str 0.1 --var_str 1e-06 > comparisons/nohup_out/ffn_m4_0.1.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice feedforward --mean_str 0.001 --var_str 1e-06 > comparisons/nohup_out/ffn_fred_0.001.out 2>&1 &


# # fixed var str
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice feedforward --mean_str 0.001 --var_str 0.001 > comparisons/nohup_out/ffn_nn5_0.001_var0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice feedforward --mean_str 0.01 --var_str 0.001 > comparisons/nohup_out/ffn_nn5_0.01_var0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice feedforward --mean_str 0.1 --var_str 0.001 > comparisons/nohup_out/ffn_nn5_0.1_var0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name nn5_weekly --model_choice feedforward --mean_str 0.5 --var_str 0.001 > comparisons/nohup_out/ffn_nn5_0.5_var0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice feedforward --mean_str 0.001 --var_str 0.001 > comparisons/nohup_out/ffn_fred_0.001_var0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice feedforward --mean_str 0.01 --var_str 0.001 > comparisons/nohup_out/ffn_fred_0.01_var0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice feedforward --mean_str 0.1 --var_str 0.001 > comparisons/nohup_out/ffn_fred_0.1_var0.001.out 2>&1 &
# nohup python compare_predictions.py --dataset_name fred_md --model_choice feedforward --mean_str 0.5 --var_str 0.001 > comparisons/nohup_out/ffn_fred_0.5_var0.001.out 2>&1 &