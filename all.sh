#!/bin/bash

# =================================================================================================
# Master script for running all CasuaLNN experiments.
#
# This script executes a series of experiments for the CasuaLNN model across various datasets.
# You can configure the global parameters below before running.
# =================================================================================================

# --- Global Configuration ---

# Set the number of iterations for each experiment.
# A value of 1 is typically used for a single run, but can be increased for stability testing.
itr=1

# Set the series decomposition method.
# This choice affects the model's internal processing.
# Options: "moving_avg", "dft_decomp"
# To apply a method globally, modify the value here.
DECOMP_METHOD="dft_decomp" # Example: Set to dft_decomp

# =================================================================================================
# Experiment Suites
# =================================================================================================

# --- ETTh1 Dataset ---
echo "Running experiments for ETTh1 dataset..."
python -u run_CasuaLNN_ETTh1.py --model_id ETTh1_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTh1.py --model_id ETTh1_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTh1.py --model_id ETTh1_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTh1.py --model_id ETTh1_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- ETTh2 Dataset ---
echo "Running experiments for ETTh2 dataset..."
python -u run_CasuaLNN_ETTh2.py --model_id ETTh2_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTh2.py --model_id ETTh2_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTh2.py --model_id ETTh2_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTh2.py --model_id ETTh2_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- ETTm1 Dataset ---
echo "Running experiments for ETTm1 dataset..."
python -u run_CasuaLNN_ETTm1.py --model_id ETTm1_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTm1.py --model_id ETTm1_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTm1.py --model_id ETTm1_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTm1.py --model_id ETTm1_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- ETTm2 Dataset ---
echo "Running experiments for ETTm2 dataset..."
python -u run_CasuaLNN_ETTm2.py --model_id ETTm2_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTm2.py --model_id ETTm2_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTm2.py --model_id ETTm2_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_ETTm2.py --model_id ETTm2_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- Electricity Dataset ---
echo "Running experiments for Electricity dataset..."
python -u run_CasuaLNN_Electricity.py --model_id Electricity_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_Electricity.py --model_id Electricity_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_Electricity.py --model_id Electricity_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_Electricity.py --model_id Electricity_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- Traffic Dataset ---
echo "Running experiments for Traffic dataset..."
python -u run_CasuaLNN_Traffic.py --model_id Traffic_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_Traffic.py --model_id Traffic_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_Traffic.py --model_id Traffic_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_Traffic.py --model_id Traffic_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- Weather Dataset ---
echo "Running experiments for Weather dataset..."
python -u run_CasuaLNN_Weather.py --model_id Weather_96_96 --pred_len 96 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLN_Weather.py --model_id Weather_96_192 --pred_len 192 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_Weather.py --model_id Weather_96_336 --pred_len 336 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_Weather.py --model_id Weather_96_720 --pred_len 720 --itr $itr --decomp_method $DECOMP_METHOD

# --- PEMS Datasets ---
# Note: PEMS datasets have specific input/output dimensions, which are set accordingly.
echo "Running experiments for PEMS datasets..."
python -u run_CasuaLNN_PEMS03.py --data_path PEMS03.npz --model_id PEMS03 --enc_in 358 --dec_in 358 --c_out 358 --itr $itr --decomp_method $DECOMP_METHOD
python -u run_CasuaLNN_PEMS08.py --data_path PEMS08.npz --model_id PEMS08 --enc_in 170 --dec_in 170 --c_out 170 --itr $itr --decomp_method $DECOMP_METHOD

echo "All experiments completed."
