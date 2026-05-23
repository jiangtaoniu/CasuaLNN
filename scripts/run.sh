#!/bin/bash

# =================================================================================================
# Master script for running all CasuaLNN experiments.
#
# This script executes a series of experiments for the CasuaLNN model across various datasets.
# Using the unified run.py script.
# =================================================================================================

itr=1
DECOMP_METHOD="dft_decomp"

# --- ETTh1 Dataset ---
echo "Running experiments for ETTh1 dataset..."
python -u run.py --data ETTh1 --root_path ./dataset/ETT/ --data_path ETTh1.csv --model_id ETTh1_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTh1 --root_path ./dataset/ETT/ --data_path ETTh1.csv --model_id ETTh1_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTh1 --root_path ./dataset/ETT/ --data_path ETTh1.csv --model_id ETTh1_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTh1 --root_path ./dataset/ETT/ --data_path ETTh1.csv --model_id ETTh1_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- ETTh2 Dataset ---
echo "Running experiments for ETTh2 dataset..."
python -u run.py --data ETTh2 --root_path ./dataset/ETT/ --data_path ETTh2.csv --model_id ETTh2_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTh2 --root_path ./dataset/ETT/ --data_path ETTh2.csv --model_id ETTh2_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTh2 --root_path ./dataset/ETT/ --data_path ETTh2.csv --model_id ETTh2_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTh2 --root_path ./dataset/ETT/ --data_path ETTh2.csv --model_id ETTh2_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- ETTm1 Dataset ---
echo "Running experiments for ETTm1 dataset..."
python -u run.py --data ETTm1 --root_path ./dataset/ETT/ --data_path ETTm1.csv --model_id ETTm1_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTm1 --root_path ./dataset/ETT/ --data_path ETTm1.csv --model_id ETTm1_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTm1 --root_path ./dataset/ETT/ --data_path ETTm1.csv --model_id ETTm1_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTm1 --root_path ./dataset/ETT/ --data_path ETTm1.csv --model_id ETTm1_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- ETTm2 Dataset ---
echo "Running experiments for ETTm2 dataset..."
python -u run.py --data ETTm2 --root_path ./dataset/ETT/ --data_path ETTm2.csv --model_id ETTm2_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTm2 --root_path ./dataset/ETT/ --data_path ETTm2.csv --model_id ETTm2_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTm2 --root_path ./dataset/ETT/ --data_path ETTm2.csv --model_id ETTm2_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data ETTm2 --root_path ./dataset/ETT/ --data_path ETTm2.csv --model_id ETTm2_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- Electricity Dataset ---
echo "Running experiments for Electricity dataset..."
python -u run.py --data custom --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data custom --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data custom --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data custom --root_path ./dataset/electricity/ --data_path electricity.csv --model_id Electricity_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- Traffic Dataset ---
echo "Running experiments for Traffic dataset..."
python -u run.py --data custom --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data custom --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data custom --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data custom --root_path ./dataset/traffic/ --data_path traffic.csv --model_id Traffic_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- Weather Dataset ---
echo "Running experiments for Weather dataset..."
python -u run.py --data custom --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data custom --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data custom --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data custom --root_path ./dataset/weather/ --data_path weather.csv --model_id Weather_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- PEMS Datasets ---
echo "Running experiments for PEMS datasets..."
python -u run.py --data PEMS --root_path ./dataset/PEMS/ --data_path PEMS03.npz --model_id PEMS03 --enc_in 358 --dec_in 358 --c_out 358 --itr $itr --decomp_method $DECOMP_METHOD
python -u run.py --data PEMS --root_path ./dataset/PEMS/ --data_path PEMS08.npz --model_id PEMS08 --enc_in 170 --dec_in 170 --c_out 170 --itr $itr --decomp_method $DECOMP_METHOD

echo "All experiments completed."
