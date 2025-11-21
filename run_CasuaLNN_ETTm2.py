# run.py

import argparse
import os
import torch
import random
import numpy as np

# 导入所有实验类
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_short_term_forecasting import Exp_Short_Term_Forecast
from exp.exp_imputation import Exp_Imputation
from exp.exp_classification import Exp_Classification
from exp.exp_anomaly_detection import Exp_Anomaly_Detection


def main():
    """
    项目主入口函数：
    1. 设置随机种子以保证实验可复现性。
    2. 解析命令行参数。
    3. 配置GPU设备。
    4. 根据任务名称动态选择并实例化对应的实验类。
    5. 生成唯一的实验ID并执行训练和/或测试流程。
    """
    # -- 设置随机种子以保证实验可复现 --
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Time-Series-Library for Long-Term Forecasting')

    # =====================================================================================
    # 1. 基础与实验设置 (Basic & Experiment Setup)
    # =====================================================================================
    parser.add_argument('--is_training', type=int, default=1, help='1: 训练并测试, 0: 仅测试')
    parser.add_argument('--model_id', type=str, default='ETTm2_96_96', # <-- [修改]
                        help='实验ID标识符, e.g., an identifier for the model experiment')
    parser.add_argument('--model', type=str, default='CasuaLNN',
                        help='model name, options: [CasuaLNN, TimeMixer, iTransformer, PatchTST, TimesNet, etc.]')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='任务名称, 选项: [long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点保存位置')
    parser.add_argument('--des', type=str, default='Exp', help='实验描述')
    parser.add_argument('--itr', type=int, default=1, help='实验重复次数')
    parser.add_argument('--comment', type=str, default='none', help='用于日志记录的额外备注')

    # =====================================================================================
    # 2. 数据加载器设置 (Data Loader Setup)
    # =====================================================================================
    parser.add_argument('--data', type=str, default='ETTm2', help='数据集类型') # <-- [修改]
    parser.add_argument('--root_path', type=str, default='./dataset/ETT/', help='数据文件根目录') # <-- [修改]
    parser.add_argument('--data_path', type=str, default='ETTm2.csv', help='数据文件名') # <-- [修改]
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务类型: M:多元预测多元, S:单源预测单源, MS:多元预测单源')
    parser.add_argument('--target', type=str, default='OT', help='在S或MS任务中的目标特征列名')
    parser.add_argument('--freq', type=str, default='t',
                        help='时间特征编码频率, 选项:[s, t, h, d, b, w, m]')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载器的工作线程数')
    parser.add_argument('--batch_size', type=int, default=128, help='训练批次大小') # <-- [修改]
    parser.add_argument('--drop_last', action='store_true', help='是否丢弃最后一个不完整的批次', default=True)
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")

    # =====================================================================================
    # 3. 预测任务通用设置 (Common Forecasting Task Setup)
    # =====================================================================================
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度 (look-back window)')
    parser.add_argument('--label_len', type=int, default=0, help='解码器起始Token长度 (用于Autoformer等模型)') # <-- [修改]
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度 (prediction horizon)') # <-- [修改]
    parser.add_argument('--inverse', action='store_true', help='对输出数据进行反归一化以评估', default=False)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='M4数据集的季节性模式')

    # =====================================================================================
    # 4. 模型架构通用设置 (General Model Architecture Setup)
    # =====================================================================================
    parser.add_argument('--d_model', type=int, default=32, help='模型隐藏维度') # <-- [修改]
    parser.add_argument('--d_ff', type=int, default=32, help='前馈网络中间层维度') # <-- [修改]
    parser.add_argument('--enc_in', type=int, default=7, help='编码器输入尺寸 (特征/变量数)') # <-- [修改]
    parser.add_argument('--dec_in', type=int, default=7, help='解码器输入尺寸 (恢复此参数以兼容旧模型)') # <-- [修改]
    parser.add_argument('--c_out', type=int, default=7, help='输出尺寸') # <-- [修改]
    parser.add_argument('--n_heads', type=int, default=8, help='多头注意力机制的头数')
    parser.add_argument('--e_layers', type=int, default=2, help='编码器层数 (通用)')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout率')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--output_attention', action='store_true', help='编码器是否输出注意力权重')
    parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码方式 [timeF, fixed, learned]')
    parser.add_argument('--distil', action='store_false', help='在编码器中使用蒸馏操作 (Informer/Autoformer)',
                        default=True)
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='序列分解方法 [moving_avg, dft_decomp]')
    parser.add_argument('--moving_avg', type=int, default=25, help='移动平均窗口大小')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='降采样层数 (决定了尺度数量)') # <-- [修改]
    parser.add_argument('--down_sampling_window', type=int, default=2, help='降采样窗口大小')
    parser.add_argument('--down_sampling_method', type=str, default='avg', help='降采样方法 [avg, max, conv]')

    # =====================================================================================
    # 5. 特定模型超参数 (Model-Specific Hyperparameters)
    # =====================================================================================
    parser.add_argument('--patch_len', type=int, default=16, help='[MS-IPM/PatchTST] Patch的长度')
    parser.add_argument('--stride', type=int, default=8, help='[MS-IPM/PatchTST] Patch的步长')
    parser.add_argument('--e_layers_ipt', type=int, default=2, help='[MS-IPM] IPT模块内部Encoder的层数')
    parser.add_argument('--e_layers_pdm', type=int, default=1,
                        help='[MS-IPM] PDM深度混合模块或TemporalEncoder的堆叠次数')
    parser.add_argument('--use_enhanced_embedding', action='store_true', default=False,
                        help='[MS-IPM 方案A] 使用CNN增强的Patch Embedding')
    parser.add_argument('--use_temporal_encoder', action='store_true', default=False,
                        help='[MS-IPM 方案B] 使用显式的Patch间时序建模模块')
    parser.add_argument('--factor', type=int, default=3, help='[Informer] ProbSparse Attention因子')
    parser.add_argument('--top_k', type=int, default=5, help='[TimesNet] 用于选择周期的Top-k频率')
    parser.add_argument('--num_kernels', type=int, default=6, help='[TimesNet] Inception块的核数量')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='0: 通道依赖, 1: 通道独立 (用于TimeMixer等)')
    parser.add_argument('--gcn_depth', type=int, default=2, help='[GNN-based] GCN深度')
    parser.add_argument('--propalpha', type=float, default=0.3, help='[GNN-based] GCN传播alpha值')
    parser.add_argument('--node_dim', type=int, default=10, help='[GNN-based] 节点嵌入维度')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='[Stationary] 投影仪隐藏层维度')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='[Stationary] 投影仪隐藏层数量')
    parser.add_argument('--fusion_method', type=str, default='cross_attention', help='(自定义模型) 时空特征融合方法')
    # --- v17/v21 共享参数 ---
    parser.add_argument('--use_dynamic_ode', action='store_true', default=True,
                        help='[MS-IPM v21] 启用动态ODE (iT->LNN) 架构')
    parser.add_argument('--lq_param_scaling', type=float, default=5.0,
                        help='[MS-IPM v21] 缩放因子，用于约束生成的ODE参数 (tanh(x) * C)')
    parser.add_argument('--e_layers_controller', type=int, default=1,
                        help='[MS-IPM v21] iT 控制器 (iT_controller) 的层数')

    # [!!! 核心 v21 修复版: 新参数 !!!]
    parser.add_argument('--lambda_causal', type=float, default=1.0,
                        help='[CausalAttn v21] DAG (h(A)) 约束损失的权重')
    parser.add_argument('--lambda_l1', type=float, default=0.1,
                        help='[CausalAttn v21] 作用于因果图的 L1 稀疏性损失权重')
    # [!!! 修复: 移除了无效的 gumbel_tau !!!]



    # (LNN 内部参数, 未修改)
    parser.add_argument('--lq_units', type=int, default=512,
                        help='[DynamicLiquid] CfC 隐藏单元数 (建议与 d_model 保持一致)')
    parser.add_argument('--lq_proj_size', type=int, default=512,
                        help='[DynamicLiquid] CfC 投影 (输出) 维度 (建议与 d_model 保持一致)')
    parser.add_argument('--lq_backbone_units', type=int, default=64,
                        help='[DynamicLiquid] CfC 内部骨干网络单元数')
    parser.add_argument('--lq_backbone_layers', type=int, default=1,
                        help='[DynamicLiquid] CfC 内部骨干网络层数')
    parser.add_argument('--lq_activation', type=str, default='gelu',
                        help='[DynamicLiquid] CfC 内部骨干网络激活函数')

    # =====================================================================================
    # 6. 训练与优化设置 (Training & Optimization Setup)
    # =====================================================================================
    parser.add_argument('--loss', type=str, default='mse', help='损失函数')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='优化器学习率') # <-- [修改]
    parser.add_argument('--lradj', type=str, default='TST', help='学习率调整策略')
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--patience', type=int, default=10, help='早停耐心轮数') # <-- [修改]
    parser.add_argument('--use_amp', action='store_true', help='使用自动混合精度训练', default=False)
    parser.add_argument('--pct_start', type=float, default=0.3, help='OneCycleLR调度器的pct_start参数')
    parser.add_argument('--use_norm', type=int, default=1, help='是否在模型中使用Normalization (例如RevIN)')
    parser.add_argument('--use_dtw', action='store_true', default=False, help='在评估中使用DTW度量')

    # =====================================================================================
    # 7. 硬件设置 (Hardware Setup)
    # =====================================================================================
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='GPU类型 [cuda, mps]')
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多GPU', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多GPU设备ID')

    # =====================================================================================
    # 8. 其他任务特定参数 (Other Task-Specific Parameters)
    # =====================================================================================
    parser.add_argument('--mask_rate', type=float, default=0.25, help='imputation任务的掩码率')
    parser.add_argument('--anomaly_ratio', type=float, default=1.0, help='anomaly detection任务的异常比例')
    parser.add_argument('--num_class', type=int, default=2, help='classification任务的类别数')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu:
        if args.gpu_type == 'mps':
            if not hasattr(torch.backends, 'mps') or not torch.backends.mps.is_available():
                args.use_gpu = False
        elif args.use_multi_gpu:
            args.devices = args.devices.replace(' ', '')
            device_ids = args.devices.split(',')
            args.device_ids = [int(id_) for id_ in device_ids]
            args.gpu = args.device_ids[0]

    print('实验参数配置:')
    print(args)

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
            # 恢复了更完整的setting字符串，以包含通用参数和新的开关参数
            setting = '{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_patch{}_stride{}_elipt{}_elpdm{}_emb{}_temp{}_{}_{}'.format(
                args.model_id, args.model, args.data,
                args.seq_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
                args.patch_len, args.stride,
                args.e_layers_ipt, args.e_layers_pdm,
                'E' if args.use_enhanced_embedding else 'L',
                'Y' if args.use_temporal_encoder else 'N',
                args.des, ii)

            exp = Exp(args)
            print(f'>>>>>>> 开始训练 : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)
            print(f'>>>>>>> 开始测试 : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            if args.use_gpu:
                torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_sl{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_patch{}_stride{}_elipt{}_elpdm{}_emb{}_temp{}_{}_{}'.format(
            args.model_id, args.model, args.data,
            args.seq_len, args.pred_len,
            args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff,
            args.patch_len, args.stride,
            args.e_layers_ipt, args.e_layers_pdm,
            'E' if args.use_enhanced_embedding else 'L',
            'Y' if args.use_temporal_encoder else 'N',
            args.des, ii)

        exp = Exp(args)
        print(f'>>>>>>> 开始测试 : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        if args.use_gpu:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()