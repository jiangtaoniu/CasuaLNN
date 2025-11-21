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
    主入口函数，负责解析命令行参数、设置实验环境并启动指定的时序任务。
    """
    # 设置随机种子以保证实验可复现性
    fix_seed = 2021
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='Time-Series-Library for Long-Term Forecasting')

    # 1. 基础与实验设置 (Basic & Experiment Setup)
    parser.add_argument('--is_training', type=int, default=1, help='是否执行训练流程 (1:是, 0:否,仅测试)')
    parser.add_argument('--model_id', type=str, default='Traffic_96_96', help='实验的唯一标识符')
    parser.add_argument('--model', type=str, default='CasuaLNN',
                        help='模型名称, 例如: CasuaLNN, iTransformer, PatchTST')
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='任务名称, 可选: long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='模型检查点保存位置')
    parser.add_argument('--des', type=str, default='Exp', help='实验描述')
    parser.add_argument('--itr', type=int, default=1, help='实验重复次数')
    parser.add_argument('--comment', type=str, default='none', help='用于日志记录的额外备注')

    # 2. 数据加载器设置 (Data Loader Setup)
    parser.add_argument('--data', type=str, default='custom', help='数据集类型')
    parser.add_argument('--root_path', type=str, default='./dataset/traffic/', help='数据文件根目录')
    parser.add_argument('--data_path', type=str, default='traffic.csv', help='数据文件名')
    parser.add_argument('--features', type=str, default='M',
                        help='预测任务类型, M: 多变量->多变量, S: 单变量->单变量, MS: 多变量->单变量')
    parser.add_argument('--target', type=str, default='OT', help='在S或MS任务中的目标特征列名')
    parser.add_argument('--freq', type=str, default='t',
                        help='时间特征编码频率, 可选: s(秒), t(分), h(时), d(日), b(工作日), w(周), m(月)')
    parser.add_argument('--num_workers', type=int, default=0, help='数据加载器的工作线程数')
    parser.add_argument('--batch_size', type=int, default=8, help='训练批次大小')
    parser.add_argument('--drop_last', action='store_true', help='是否丢弃最后一个不完整的批次', default=True)
    parser.add_argument('--augmentation_ratio', type=int, default=0, help='数据增强轮数或倍数')

    # 3. 预测任务通用设置 (Common Forecasting Task Setup)
    parser.add_argument('--seq_len', type=int, default=96, help='输入序列长度 (look-back window)')
    parser.add_argument('--label_len', type=int, default=0, help='解码器预填充的起始token长度 (常用于Autoformer等)')
    parser.add_argument('--pred_len', type=int, default=96, help='预测序列长度 (prediction horizon)')
    parser.add_argument('--inverse', action='store_true', help='对输出数据进行反归一化以评估', default=False)
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='M4数据集的季节性模式')

    # 4. 模型架构通用设置
    parser.add_argument('--d_model', type=int, default=32, help='模型隐藏维度')
    parser.add_argument('--d_ff', type=int, default=64, help='前馈网络中间层维度')
    parser.add_argument('--enc_in', type=int, default=862, help='编码器输入尺寸 (特征/变量数)')
    parser.add_argument('--dec_in', type=int, default=862, help='解码器输入尺寸 (兼容部分旧模型)')
    parser.add_argument('--c_out', type=int, default=862, help='输出尺寸')
    parser.add_argument('--n_heads', type=int, default=8, help='多头注意力机制的头数')
    parser.add_argument('--e_layers', type=int, default=3, help='编码器层数')
    parser.add_argument('--d_layers', type=int, default=1, help='解码器层数')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout比率')
    parser.add_argument('--activation', type=str, default='gelu', help='激活函数')
    parser.add_argument('--output_attention', action='store_true', help='编码器是否输出注意力权重')
    parser.add_argument('--embed', type=str, default='timeF', help='时间特征编码方式, 可选: timeF, fixed, learned')
    parser.add_argument('--distil', action='store_false', help='是否在编码器中使用蒸馏操作(适用于Informer/Autoformer)',
                        default=True)
    parser.add_argument('--decomp_method', type=str, default='moving_avg', help='序列分解方法, 可选: moving_avg, dft_decomp')
    parser.add_argument('--moving_avg', type=int, default=25, help='移动平均窗口大小')
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='降采样层数')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='降采样窗口大小')
    parser.add_argument('--down_sampling_method', type=str, default='avg', help='降采样方法, 可选: avg, max, conv')

    # 5. 特定模型超参数
    parser.add_argument('--patch_len', type=int, default=16, help='Patch的长度')
    parser.add_argument('--stride', type=int, default=8, help='Patch的步长')
    parser.add_argument('--e_layers_ipt', type=int, default=2, help='IPT模块内部Encoder的层数')
    parser.add_argument('--e_layers_pdm', type=int, default=1,
                        help='PDM深度混合模块或TemporalEncoder的堆叠次数')
    parser.add_argument('--use_enhanced_embedding', action='store_true', default=False,
                        help='使用CNN增强的Patch Embedding')
    parser.add_argument('--use_temporal_encoder', action='store_true', default=False,
                        help='使用显式的Patch间时序建模模块')
    parser.add_argument('--factor', type=int, default=3, help='ProbSparse Attention中的采样因子')
    parser.add_argument('--top_k', type=int, default=5, help='用于选择周期的Top-k频率 (TimesNet)')
    parser.add_argument('--num_kernels', type=int, default=6, help='Inception块中的核数量 (TimesNet)')
    parser.add_argument('--channel_independence', type=int, default=0,
                        help='通道独立建模 (0: 依赖, 1: 独立)')
    parser.add_argument('--gcn_depth', type=int, default=2, help='GCN深度')
    parser.add_argument('--propalpha', type=float, default=0.3, help='GCN传播alpha值')
    parser.add_argument('--node_dim', type=int, default=10, help='节点嵌入维度')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='投影仪的隐藏层维度')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='投影仪的隐藏层数量')
    parser.add_argument('--fusion_method', type=str, default='cross_attention', help='时空特征融合方法')
        parser.add_argument('--use_dynamic_ode', action='store_true', default=True,
                            help='是否启用动态ODE架构 (iTransformer -> LNN)')
        parser.add_argument('--lq_param_scaling', type=float, default=5.0,
                            help='缩放因子, 用于约束生成的ODE参数, 例如: tanh(x) * C')
        parser.add_argument('--e_layers_controller', type=int, default=1,
                            help='iTransformer控制器(iT_controller)的层数')
    
        parser.add_argument('--lq_units', type=int, default=512,
                            help='CfC/Liquid模型的隐藏单元数')
        parser.add_argument('--lq_proj_size', type=int, default=512,
                            help='CfC/Liquid模型的投影(输出)维度')
        parser.add_argument('--lq_backbone_units', type=int, default=64,
                            help='CfC/Liquid模型内部骨干网络的单元数')
        parser.add_argument('--lq_backbone_layers', type=int, default=1,
                            help='CfC/Liquid模型内部骨干网络的层数')
        parser.add_argument('--lq_activation', type=str, default='gelu',
                            help='CfC/Liquid模型内部骨干网络的激活函数')

    # 6. 训练与优化设置 (Training & Optimization Setup)
    parser.add_argument('--loss', type=str, default='mse', help='损失函数')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='优化器学习率')
    parser.add_argument('--lradj', type=str, default='TST', help='学习率调整策略, 例如: TST, CosineAnnealing')
    parser.add_argument('--train_epochs', type=int, default=10, help='训练轮数')
    parser.add_argument('--patience', type=int, default=3, help='早停法耐心轮数')
    parser.add_argument('--use_amp', action='store_true', help='是否启用自动混合精度训练(AMP)', default=False)
    parser.add_argument('--pct_start', type=float, default=0.3, help='学习率调度器(如OneCycleLR)的pct_start参数')
    parser.add_argument('--use_norm', type=int, default=1, help='是否在输入端使用序列归一化(例如RevIN)')
    parser.add_argument('--use_dtw', action='store_true', default=False, help='在评估中额外计算DTW度量')

    # 7. 硬件设置 (Hardware Setup)
    parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
    parser.add_argument('--gpu', type=int, default=0, help='指定的GPU ID')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='计算后端类型, 可选: cuda, mps')
    parser.add_argument('--use_multi_gpu', action='store_true', help='是否使用多GPU', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='多GPU设备ID')

    # 8. 其他任务特定参数 (Other Task-Specific Parameters)
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
            # -- 生成唯一的实验设定标识 --
            setting = (f'{args.model_id}_{args.model}_{args.data}_'
                       f'sl{args.seq_len}_pl{args.pred_len}_'
                       f'dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_'
                       f'patch{args.patch_len}_stride{args.stride}_'
                       f'elipt{args.e_layers_ipt}_elpdm{args.e_layers_pdm}_'
                       f'emb{"E" if args.use_enhanced_embedding else "L"}_'
                       f'temp{"Y" if args.use_temporal_encoder else "N"}_'
                       f'{args.des}_{ii}')

            exp = Exp(args)
            print(f'>>>>>>> 开始训练 : {setting} >>>>>>>>>>>>>>>>>>>>>>>>>>')
            exp.train(setting)
            print(f'>>>>>>> 开始测试 : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            if args.use_gpu:
                torch.cuda.empty_cache()
    else:
        ii = 0
        # -- 生成唯一的实验设定标识 --
        setting = (f'{args.model_id}_{args.model}_{args.data}_'
                   f'sl{args.seq_len}_pl{args.pred_len}_'
                   f'dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_'
                   f'patch{args.patch_len}_stride{args.stride}_'
                   f'elipt{args.e_layers_ipt}_elpdm{args.e_layers_pdm}_'
                   f'emb{"E" if args.use_enhanced_embedding else "L"}_'
                   f'temp{"Y" if args.use_temporal_encoder else "N"}_'
                   f'{args.des}_{ii}')

        exp = Exp(args)
        print(f'>>>>>>> 开始测试 : {setting} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        if args.use_gpu:
            torch.cuda.empty_cache()


if __name__ == '__main__':
    main()