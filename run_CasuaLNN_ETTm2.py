import argparse
import os
import torch
import random
import numpy as np

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_classification import Exp_Classification
from exp.exp_anomaly_detection import Exp_Anomaly_Detection


def main():
    """
    Main entry point for the experiment.
    
    This function performs the following steps:
    1. Sets a random seed for reproducibility.
    2. Parses command-line arguments.
    3. Configures the GPU device.
    4. Dynamically selects and instantiates the appropriate experiment class based on the task name.
    5. Generates a unique experiment ID and runs the training and/or testing process.
    """
    # Set a fixed seed for reproducibility across runs.
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Time-Series-Library for Long-Term Forecasting')

    # =====================================================================================
    # 1. Basic & Experiment Setup
    # =====================================================================================
    parser.add_argument('--is_training', type=int, default=1, help='1 for training and testing, 0 for testing only.')
    parser.add_argument('--model_id', type=str, default='ETTm2_96_96', help='A unique identifier for the experiment.')
    parser.add_argument('--model', type=str, default='CasuaLNN', help='Model name, e.g., CasuaLNN, TimeMixer, etc.')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='Task name. Options: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='Location to save model checkpoints.')
    parser.add_argument('--des', type=str, default='Exp', help='A description for the experiment.')
    parser.add_argument('--itr', type=int, default=1, help='Number of times to repeat the experiment.')
    parser.add_argument('--comment', type=str, default='none', help='Additional comment for logging purposes.')

    # =====================================================================================
    # 2. Data Loader Setup
    # =====================================================================================
    parser.add_argument('--data', type=str, default='ETTm2', help='Dataset name.')
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='Root directory of the data file.')
    parser.add_argument('--data_path', type=str, default='ETTm2.csv', help='Name of the data file.')
    parser.add_argument('--features', type=str, default='M',
                        help='Forecasting task type: M for multivariate, S for univariate, MS for multivariate-to-univariate.')
    parser.add_argument('--target', type=str, default='OT', help='Target feature column name for S or MS tasks.')
    parser.add_argument('--freq', type=str, default='h',
                        help='Frequency for time features encoding. Options: [s, t, h, d, b, w, m]. \'h\' for hourly.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of worker threads for the data loader.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training.')
    parser.add_argument('--drop_last', action='store_true', help='Whether to drop the last incomplete batch.', default=True)
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="Specifies how many times to augment the data.")

    # =====================================================================================
    # 3. Common Forecasting Task Setup
    # =====================================================================================
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length (look-back window).')
    parser.add_argument('--label_len', type=int, default=0, help='Start token length for the decoder (used in models like Autoformer).')
    parser.add_argument('--pred_len', type=int, default=96, help='Prediction horizon, the length of the forecast sequence.')
    parser.add_argument('--inverse', action='store_true', help='Denormalize the output data for evaluation.', default=False)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='Seasonal pattern for the M4 dataset.')

    # =====================================================================================
    # 4. General Model Architecture Setup
    # =====================================================================================
    parser.add_argument('--d_model', type=int, default=32, help='Dimension of the model\'s hidden states.')
    parser.add_argument('--d_ff', type=int, default=32, help='Dimension of the feed-forward network\'s intermediate layer.')
    parser.add_argument('--enc_in', type=int, default=7, help='Input size for the encoder (number of features/variables).')
    parser.add_argument('--dec_in', type=int, default=7, help='Input size for the decoder.')
    parser.add_argument('--c_out', type=int, default=7, help='Output size (number of target variables).')
    parser.add_argument('--n_heads', type=int, default=8, help='Number of heads in the multi-head attention mechanism.')
    parser.add_argument('--e_layers', type=int, default=2, help='Number of encoder layers.')
    parser.add_argument('--d_layers', type=int, default=1, help='Number of decoder layers.')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate.')
    parser.add_argument('--activation', type=str, default='gelu', help='Activation function.')
    parser.add_argument('--output_attention', action='store_true', help='Whether the encoder should output attention weights.')
    parser.add_argument('--embed', type=str, default='timeF', help='Time feature encoding method. Options: [timeF, fixed, learned].')
    parser.add_argument('--distil', action='store_false', help='Use distillation in the encoder (for Informer/Autoformer).', default=True)
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='Series decomposition method. Options: [moving_avg, dft_decomp].')
    parser.add_argument('--moving_avg', type=int, default=25, help='Window size for the moving average decomposition.')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='Number of down-sampling layers (determines the number of scales).')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='Window size for down-sampling.')
    parser.add_argument('--down_sampling_method', type=str, default='avg', help='Down-sampling method. Options: [avg, max, conv].')

    # =====================================================================================
    # 5. Model-Specific Hyperparameters
    # =====================================================================================
    # --- CasuaLNN / MS-IPM ---
    parser.add_argument('--patch_len', type=int, default=16, help='Length of patches.')
    parser.add_argument('--stride', type=int, default=8, help='Stride between patches.')
    parser.add_argument('--e_layers_ipt', type=int, default=2, help='Number of layers in the IPT block encoder.')
    parser.add_argument('--e_layers_pdm', type=int, default=1, help='Number of layers in the PDM block or TemporalEncoder.')
    parser.add_argument('--use_enhanced_embedding', action='store_true', default=False, help='Use CNN-enhanced patch embedding.')
    parser.add_argument('--use_temporal_encoder', action='store_true', default=False, help='Use an explicit inter-patch temporal encoder.')
    parser.add_argument('--use_dynamic_ode', action='store_true', default=True, help='Enable the dynamic ODE (iT->LNN) architecture.')
    parser.add_argument('--lq_param_scaling', type=float, default=5.0, help='Scaling factor for generated ODE parameters (tanh(x) * C).')
    parser.add_argument('--e_layers_controller', type=int, default=1, help='Number of layers in the iT controller.')
    parser.add_argument('--lambda_causal', type=float, default=1.0, help='Weight for the DAG (h(A)) constraint loss.')
    parser.add_argument('--lambda_l1', type=float, default=0.1, help='Weight for the L1 sparsity loss on the causal graph.')
    # --- LNN Internals (Usually not changed) ---
    parser.add_argument('--lq_units', type=int, default=512, help='Number of hidden units in the CfC cell.')
    parser.add_argument('--lq_proj_size', type=int, default=512, help='Projection size in the CfC cell.')
    parser.add_argument('--lq_backbone_units', type=int, default=64, help='Number of units in the CfC backbone network.')
    parser.add_argument('--lq_backbone_layers', type=int, default=1, help='Number of layers in the CfC backbone network.')
    parser.add_argument('--lq_activation', type=str, default='gelu', help='Activation function for the CfC backbone.')
    # --- Other Models ---
    parser.add_argument('--factor', type=int, default=3, help='[Informer] ProbSparse Attention factor.')
    parser.add_argument('--top_k', type=int, default=5, help='[TimesNet] Top-k frequencies for period selection.')
    parser.add_argument('--num_kernels', type=int, default=6, help='[TimesNet] Number of kernels in the Inception block.')
    parser.add_argument('--channel_independence', type=int, default=0, help='0 for channel-dependent, 1 for channel-independent models.')
    parser.add_argument('--gcn_depth', type=int, default=2, help='[GNN-based] GCN depth.')
    parser.add_argument('--propalpha', type=float, default=0.3, help='[GNN-based] GCN propagation alpha value.')
    parser.add_-argument('--node_dim', type=int, default=10, help='[GNN-based] Node embedding dimension.')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128], help='[Stationary] Projector hidden dimensions.')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='[Stationary] Number of projector hidden layers.')
    parser.add_argument('--fusion_method', type=str, default='cross_attention', help='[Custom] Spatio-temporal feature fusion method.')

    # =====================================================================================
    # 6. Training & Optimization Setup
    # =====================================================================================
    parser.add_argument('--loss', type=str, default='mse', help='Loss function.')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate for the optimizer.')
    parser.add_argument('--lradj', type=str, default='TST', help='Learning rate adjustment strategy.')
    parser.add_argument('--train_epochs', type=int, default=1, help='Number of training epochs.')
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping.')
    parser.add_argument('--use_amp', action='store_true', help='Use Automatic Mixed Precision training.', default=False)
    parser.add_argument('--pct_start', type=float, default=0.3, help='pct_start parameter for OneCycleLR scheduler.')
    parser.add_argument('--use_norm', type=int, default=1, help='Enable normalization (e.g., RevIN).')
    parser.add_argument('--use_dtw', action='store_true', default=False, help='Use Dynamic Time Warping (DTW) for evaluation.')

    # =====================================================================================
    # 7. Hardware Setup
    # =====================================================================================
    parser.add_argument('--use_gpu', type=bool, default=True, help='Enable GPU usage.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use.')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='GPU type. Options: [cuda, mps].')
    parser.add_argument('--use_multi_gpu', action='store_true', help='Enable multi-GPU usage.', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='Comma-separated list of device IDs for multi-GPU.')

    # =====================================================================================
    # 8. Other Task-Specific Parameters
    # =====================================================================================
    parser.add_argument('--mask_rate', type=float, default=0.25, help='Masking ratio for imputation tasks.')
    parser.add_argument('--anomaly_ratio', type=float, default=1.0, help='Anomaly ratio for anomaly detection tasks.')
    parser.add_argument('--num_class', type=int, default=2, help='Number of classes for classification tasks.')

    args = parser.parse_args()

    # Configure GPU settings
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        if args.gpu_type == 'mps':
            if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
                print("Warning: MPS not available on this system. Falling back to CPU.")
                args.use_gpu = False
        elif args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

    print('Experiment Configuration:')
    print(args)

    # Map task name to the corresponding experiment class
    task_map = {
        'long_term_forecast': Exp_Long_Term_Forecast,
        'short_term_forecast': Exp_Short_Term_Forecast,
        'imputation': Exp_Imputation,
        'anomaly_detection': Exp_Anomaly_Detection,
        'classification': Exp_Classification,
    }
    Exp = task_map.get(args.task_name, Exp_Long_Term_Forecast)

    if args.is_training:
        for ii in range(args.itr):
            # Define a unique setting string for this experiment run.
            # emb: 'E' for Enhanced, 'L' for Linear
            # temp: 'Y' for Temporal Encoder, 'N' for No
            setting = (
                f'{args.model_id}_{args.model}_{args.data}_' 
                f'sl{args.seq_len}_pl{args.pred_len}_' 
                f'dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_' 
                f'patch{args.patch_len}_stride{args.stride}_' 
                f'elipt{args.e_layers_ipt}_elpdm{args.e_layers_pdm}_' 
                f'emb{"E" if args.use_enhanced_embedding else "L"}_' 
                f'temp{"Y" if args.use_temporal_encoder else "N"}_' 
                f'{args.des}_{ii}')

            exp = Exp(args)
            print(f'>>>>>>> Starting Training: {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)
            print(f'>>>>>>> Starting Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            if args.use_gpu:
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = (
            f'{args.model_id}_{args.model}_{args.data}_' 
            f'sl{args.seq_len}_pl{args.pred_len}_' 
            f'dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_' 
            f'patch{args.patch_len}_stride{args.stride}_' 
            f'elipt{args.e_layers_ipt}_elpdm{args.e_layers_pdm}_' 
            f'emb{"E" if args.use_enhanced_embedding else "L"}_' 
            f'temp{"Y" if args.use_temporal_encoder else "N"}_' 
            f'{args.des}_{ii}')

        exp = Exp(args)
        print(f'>>>>>>> Starting Testing: {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        if args.use_gpu:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
