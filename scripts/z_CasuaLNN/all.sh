#!/bin/bash

# 设置固定迭代次数
itr=1

# 设置序列分解方法，可选值: "moving_avg" 或 "dft_decomp"
# 您只需修改此处的值，即可全局生效
DECOMP_METHOD="dft_decomp"  # 例如，此处设置为 dft_decomp

# 运行 ETTh1 系列实验
python -u run_MS_IPM_ETTh1_backup3_inOutPatch.py --model_id ETTh1_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTh1_backup3_inOutPatch.py --model_id ETTh1_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTh1_backup3_inOutPatch.py --model_id ETTh1_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTh1_backup3_inOutPatch.py --model_id ETTh1_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# 运行 ETTh2 系列实验
python -u run_MS_IPM_ETTh2_backup3_inOutPatch.py --model_id ETTh2_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTh2_backup3_inOutPatch.py --model_id ETTh2_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTh2_backup3_inOutPatch.py --model_id ETTh2_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTh2_backup3_inOutPatch.py --model_id ETTh2_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# 运行 ETTm1 系列实验
python -u run_MS_IPM_ETTm1_backup3_inOutPatch.py --model_id ETTm1_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTm1_backup3_inOutPatch.py --model_id ETTm1_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTm1_backup3_inOutPatch.py --model_id ETTm1_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTm1_backup3_inOutPatch.py --model_id ETTm1_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# 运行 ETTm2 系列实验
python -u run_MS_IPM_ETTm2_backup3_inOutPatch.py --model_id ETTm2_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTm2_backup3_inOutPatch.py --model_id ETTm2_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTm2_backup3_inOutPatch.py --model_id ETTm2_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_ETTm2_backup3_inOutPatch.py --model_id ETTm2_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# 运行 Electricity 系列实验
python -u run_MS_IPM_Electricity_backup3_inOutPatch.py --model_id Electricity_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_Electricity_backup3_inOutPatch.py --model_id Electricity_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_Electricity_backup3_inOutPatch.py --model_id Electricity_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_Electricity_backup3_inOutPatch.py --model_id Electricity_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# 运行 Traffic 系列实验
python -u run_MS_IPM_Traffic_backup3_inOutPatch.py --model_id Traffic_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_Traffic_backup3_inOutPatch.py --model_id Traffic_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_Traffic_backup3_inOutPatch.py --model_id Traffic_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_Traffic_backup3_inOutPatch.py --model_id Traffic_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# 运行 Weather 系列实验
python -u run_MS_IPM_Weather_backup3_inOutPatch.py --model_id Weather_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_Weather_backup3_inOutPatch.py --model_id Weather_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_Weather_backup3_inOutPatch.py --model_id Weather_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_Weather_backup3_inOutPatch.py --model_id Weather_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# 运行 PEMS 系列实验
python -u run_MS_IPM_PEMS03_backup3_inOutPatch.py --data_path PEMS03.npz --model_id PEMS03 --enc_in 358 --dec_in 358 --c_out 358 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_MS_IPM_PEMS08_backup3_inOutPatch.py --data_path PEMS08.npz --model_id PEMS08 --enc_in 170 --dec_in 170 --c_out 170 --itr $itr --decomp_method $DECOMP_METHOD