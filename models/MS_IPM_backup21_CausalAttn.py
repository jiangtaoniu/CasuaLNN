# 文件名: models/MS_IPM_backup21_CausalAttn.py
# 路径: D:\anjt\code\MS-IPM\MS-IPM\models\MS_IPM_backup21_CausalAttn.py
#
# 描述:
# 基于 MS_IPM_backup17_DynamicLiquidODE.py。
#
# [!!! backup17 -> backup21 的关键修改 !!!]
#
# 1. (因果图) Model.__init__:
#    - 新增一个可学习的、跨尺度共享的参数:
#      `self.causal_graph_logits = nn.Parameter(torch.randn(configs.enc_in, configs.enc_in))`
#
# 2. (依赖注入) Model.__init__:
#    - 在创建 `IPT_Block_Encoder` 时，将 `self.causal_graph_logits`
#      作为参数 `causal_graph_logits` 传递进去。
#
# 3. (因果控制器) IPT_Block_Encoder.__init__:
#    - 导入并使用 `CausalEncoder`, `CausalEncoderLayer`,
#      `CausalAttentionLayer`, `CausalFullAttention` 来构建 `iT_controller`。
#
# 4. (因果注意力) IPT_Block_Encoder.forward:
#    - 在调用 `self.iT_controller` 时，传入:
#      `correlation_state, _ = self.iT_controller(controller_in, causal_mask=self.causal_graph_logits)`
#
# 5. (损失) 损失计算:
#    - 相关的因果损失 (DAG + L1) 将在
#      `exp/exp_long_term_forecasting.py` 训练循环中 (修改版)
#      通过访问 `model.causal_graph_logits` 来计算和添加。

import torch
import torch.nn as nn

# 导入原始的 PatchEmbedding, PDM_Deep_Block 等
from layers.Embed import PatchEmbedding as OriginalPatchEmbedding, PositionalEmbedding
from layers.MS_IPM_layers_backup3_inOutPatch import PDM_Deep_Block, get_num_patches
from layers.StandardNorm import Normalize

# [!!! 核心修改 v21 !!!]
# 导入我们新定义的 Causal Transformer 和 Causal Attention
from layers.CausalTransformer_EncDec import CausalEncoder, CausalEncoderLayer
from layers.CausalAttention_Family import CausalFullAttention, CausalAttentionLayer

# 导入动态 LNN 编码器 (保持不变, 依赖 layers/DynamicLiquid.py)
from layers.DynamicLiquid import DynamicLiquidEncoder, DynamicLiquidLayer


# ######################################################################
# ## --- 强化Patch内部时序特征提取 ---
# ## (与 backup17 保持一致, 无需修改)
# ######################################################################
class EnhancedPatchEmbedding(nn.Module):
    """
    使用1D-CNN增强的PatchEmbedding (同 backup17)
    """

    def __init__(self, d_model, patch_len, stride, padding, dropout):
        super(EnhancedPatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))
        self.dropout = nn.Dropout(dropout)
        self.value_embedding = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=d_model // 2, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv1d(in_channels=d_model // 2, out_channels=d_model, kernel_size=3, padding=1),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(start_dim=1)
        )
        self.position_embedding = PositionalEmbedding(d_model)

    def forward(self, x):
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        B, N, Np, P = x.shape
        x_embed = self.value_embedding(x.reshape(-1, 1, P))
        x_embed = x_embed.reshape(B * N, Np, -1)
        x_out = x_embed + self.position_embedding(x_embed)
        return self.dropout(x_out), n_vars


# ######################################################################


class Model(nn.Module):
    """
    MS_IPM 主模型 (CausalAttn v21)
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in  # 变量数 N

        # 1. 多尺度金字塔 (同 backup17)
        self.down_sampling_window = configs.down_sampling_window
        if self.configs.down_sampling_method == 'avg':
            self.down_pool = torch.nn.AvgPool1d(kernel_size=self.down_sampling_window, stride=self.down_sampling_window)
        elif self.configs.down_sampling_method == 'max':
            self.down_pool = torch.nn.MaxPool1d(kernel_size=self.down_sampling_window, stride=self.down_sampling_window)

        self.normalize_layers = torch.nn.ModuleList([
            Normalize(configs.enc_in, affine=True, non_norm=configs.use_norm == 0)
            for _ in range(configs.down_sampling_layers + 1)
        ])

        # [!!! 核心修改 v21 !!!]
        # ----------------------------------------------------
        # 2. 定义跨尺度共享的、可学习的因果图 (logits)
        # 形状: [N, N], N = 变量数
        # 训练循环将访问该参数以计算因果损失
        self.causal_graph_logits = nn.Parameter(
            torch.randn(self.n_vars, self.n_vars)
        )
        # (可选) 使用更标准的初始化
        nn.init.xavier_uniform_(self.causal_graph_logits)
        print(f">>> [模型 MS_IPM_backup21] 成功初始化共享因果图 (nn.Parameter) [N, N] = [{self.n_vars}, {self.n_vars}]")
        # ----------------------------------------------------

        # 3. 尺度内编码 (IPT)
        # [!!! 核心修改 v21 !!!]
        # 实例化 IPT_Block_Encoder，并注入共享的 causal_graph_logits
        self.ipt_blocks = nn.ModuleList([
            IPT_Block_Encoder(configs,
                              patch_len=configs.patch_len,
                              stride=configs.stride,
                              scale_idx=i,
                              use_enhanced_embedding=configs.use_enhanced_embedding,
                              use_dynamic_ode=configs.use_dynamic_ode,
                              causal_graph_logits=self.causal_graph_logits  # [!!! 注入 !!!]
                              )
            for i in range(configs.down_sampling_layers + 1)
        ])

        # 4. 跨尺度混合 (同 backup17)
        self.pdm_blocks = nn.ModuleList([
            PDM_Deep_Block(configs) for _ in range(configs.e_layers_pdm)
        ])

        # 5. 预测头 (同 backup17)
        self.pred_heads = nn.ModuleList()
        for i in range(configs.down_sampling_layers + 1):
            scale_seq_len = self.seq_len // (self.down_sampling_window ** i)
            num_patches = get_num_patches(scale_seq_len, configs.patch_len, configs.stride)
            if num_patches <= 0:
                print(f"警告: Model __init__ (scale {i}) 计算出 0 个补丁。预测头将接收 0 输入。")
                head_nf = configs.d_model
            else:
                head_nf = configs.d_model * num_patches

            self.pred_heads.append(
                nn.Sequential(
                    nn.Flatten(start_dim=-2),
                    nn.Linear(head_nf, configs.pred_len)
                )
            )

    def __multi_scale_process_inputs(self, x_enc):
        # (同 backup17)
        x_enc_list = [x_enc]
        x_enc_temp = x_enc.permute(0, 2, 1)
        for _ in range(self.configs.down_sampling_layers):
            x_enc_temp = self.down_pool(x_enc_temp)
            x_enc_list.append(x_enc_temp.permute(0, 2, 1))
        return x_enc_list

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if x_enc.dim() == 4:
            x_enc = x_enc[..., 0]

        x_list = self.__multi_scale_process_inputs(x_enc)

        ipt_outputs = []
        for i, x_k in enumerate(x_list):
            x_k_norm = self.normalize_layers[i](x_k, 'norm')

            # [!!! 核心 !!!]
            # IPT_Block_Encoder 内部现在执行
            # (iT[CausalAttn] -> ParamGen -> LNN) 的复杂逻辑
            ipt_out = self.ipt_blocks[i](x_k_norm)  # output: (B, N, N_p, d_model)

            ipt_outputs.append(ipt_out)

        # PDM 模块 (同 backup17)
        pdm_inputs = [out.permute(0, 2, 1, 3) for out in ipt_outputs]  # -> list of (B, N_p, N, d_model)
        mixed_outputs = pdm_inputs
        for pdm_block in self.pdm_blocks:
            mixed_outputs = pdm_block(mixed_outputs)

        # 预测头 (同 backup17)
        y_preds = []
        for i, h_k in enumerate(mixed_outputs):
            h_k_permuted = h_k.permute(0, 2, 1, 3)  # (B, N, N_p, D)
            y_pred = self.pred_heads[i](h_k_permuted)  # -> (B, N, pred_len)
            y_pred = y_pred.permute(0, 2, 1)  # -> (B, pred_len, N)
            y_pred = self.normalize_layers[i](y_pred, 'denorm')
            y_preds.append(y_pred)

        final_pred = torch.stack(y_preds, dim=-1).mean(dim=-1)

        # 注意：因果损失 (h_A, l1) 在 `exp_long_term_forecasting.py` 的训练循环中
        # 通过访问 `self.causal_graph_logits` 独立计算。
        # 本函数 (forecast) 只需返回预测值。
        return final_pred

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        # (同 backup17)
        if self.task_name in ['long_term_forecast', 'short_term_forecast']:
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        return None


#
# [!!! 真正实现“因果控制器”的模块 (v21) !!!]
#
class IPT_Block_Encoder(nn.Module):
    """
    IPT_Block_Encoder (CausalAttn + Dynamic ODE v21)

    该模块实现了 [iT CausalController -> Parameter Generator -> LNN Executor] 的流程。

    [!!! v21 修改 !!!]
    1. __init__ 接收 `causal_graph_logits` 参数。
    2. `iT_controller` 使用 `CausalEncoder` 和 `CausalAttentionLayer` 构建。
    3. `forward` 方法将 `self.causal_graph_logits` 作为 `causal_mask`
       传递给 `self.iT_controller`。
    """

    def __init__(self, configs, patch_len=16, stride=8, scale_idx=0,
                 use_enhanced_embedding=False,
                 use_dynamic_ode=False,
                 causal_graph_logits=None  # [!!! 核心修改 v21 !!!]
                 ):

        super(IPT_Block_Encoder, self).__init__()
        self.d_model = configs.d_model
        self.use_dynamic_ode = use_dynamic_ode

        # [!!! 核心修改 v21 !!!]
        # 保存对共享因果图的引用
        self.causal_graph_logits = causal_graph_logits
        if self.causal_graph_logits is None:
            # 这是一个安全检查
            print(f"警告: IPT_Block (Scale {scale_idx}) 未收到 causal_graph_logits。CausalAttn 将被禁用。")


        # 1. Patching 模块 (同 backup17)
        if use_enhanced_embedding:
            self.patch_embedding = EnhancedPatchEmbedding(
                configs.d_model, patch_len, stride, stride, configs.dropout)
        else:
            self.patch_embedding = OriginalPatchEmbedding(
                configs.d_model, patch_len, stride, stride, configs.dropout)

        # ----------------------------------------------------
        # [!!! 核心修改 v21 !!!]
        # ----------------------------------------------------
        if self.use_dynamic_ode:
            print(f">>> [模型 MS_IPM_backup21] IPT_Block (Scale {scale_idx}) 启用 [Dynamic ODE] 结构。")
            print(f">>> [模型 MS_IPM_backup21] ... 并启用 [Causal Attention Controller]。")

            # (v17 的稳定性参数)
            self.lq_param_scaling = configs.lq_param_scaling
            self.tanh = nn.Tanh()

            # [!!! 核心修改 v21 !!!]
            # 模块 1: iT Causal Controller (使用 Causal* 类)
            self.iT_controller = CausalEncoder(
                [
                    CausalEncoderLayer(
                        CausalAttentionLayer(
                            CausalFullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                    ) for _ in range(configs.e_layers_controller)  # 使用 v17 引入的 e_layers_controller
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

            # 模块 2: Parameter Generator (同 backup17)
            self.param_gen_output_dim = configs.lq_units * 2  # k = 2 * H (用于 t_a, t_b)
            self.param_generator = nn.Linear(configs.d_model, self.param_gen_output_dim)

            # 模块 3: LNN Executor (同 backup17)
            self.LNN_executor = DynamicLiquidEncoder(
                [
                    DynamicLiquidLayer(
                        configs=configs,
                        dropout=configs.dropout
                    ) for _ in range(configs.e_layers_pdm)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )

        else:
            # 回退（Fallback）逻辑 (同 backup17)
            # [!!! 核心修改 v21 !!!]
            # 即使在回退模式下，我们仍然可以使用 Causal Attention
            print(f">>> [模型 MS_IPM_backup21] IPT_Block (Scale {scale_idx}) 仅使用 [Causal Inverted-Att] (iT) 结构。")
            self.encoder = CausalEncoder(
                [
                    CausalEncoderLayer(
                        CausalAttentionLayer(
                            CausalFullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model, configs.d_ff, dropout=configs.dropout, activation=configs.activation
                    ) for _ in range(configs.e_layers_ipt)  # 回退时使用 e_layers_ipt
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model)
            )
        # ----------------------------------------------------

    def forward(self, x):
        # x: [B, L_k, N]
        B, L_k, N = x.shape
        x = x.permute(0, 2, 1)  # -> [B, N, L_k]

        # 1. Patching (同 backup17)
        # patch_out: [B * N, N_p, d_model]
        patch_out, n_vars = self.patch_embedding(x)
        N_p = patch_out.shape[1]

        # ----------------------------------------------------
        # [!!! 核心修改 v21 !!!]
        # ----------------------------------------------------
        if self.use_dynamic_ode:

            # --- Causal Controller 路径 ---

            # 2a. 重塑 for iT Causal Controller (同 backup17)
            # [B*N, Np, D] -> [B, N, Np, D] -> [B*Np, N, D]
            controller_in = patch_out.reshape(B, N, N_p, self.d_model).permute(0, 2, 1, 3).reshape(B * N_p, N,
                                                                                                   self.d_model)

            # 3a. 运行 iT Causal Controller
            # [B*Np, N, D] -> [B*Np, N, D]
            # [!!! 核心修改 v21 !!!] 传入 causal_mask
            correlation_state, _ = self.iT_controller(
                controller_in,
                causal_mask=self.causal_graph_logits
            )

            # 4a. 运行 Parameter Generator (同 backup17)
            # [B*Np, N, D] -> [B*Np, N, k]
            unscaled_params = self.param_generator(correlation_state)
            dynamic_params = self.lq_param_scaling * self.tanh(unscaled_params)

            # 5a. 对齐参数 for LNN Executor (同 backup17)
            # [B*Np, N, k] -> [B*N, Np, k]
            dynamic_params_seq = dynamic_params.reshape(B, N_p, N, -1).permute(0, 2, 1, 3).reshape(B * N, N_p, -1)

            # --- 执行器路径 (Executor Path) ---

            # 6a. 运行 LNN Executor (同 backup17)
            # [B*N, Np, D] + [B*N, Np, k] -> [B*N, Np, D]
            lnn_output, _ = self.LNN_executor(patch_out, dynamic_params=dynamic_params_seq)

            # 7a. 重塑 (Reshape) 最终输出 (同 backup17)
            # [B*N, Np, D] -> [B, N, Np, D]
            final_out = lnn_output.reshape(B, N, N_p, self.d_model)

        else:
            # --- 回退路径 (Fallback Causal iT Path) ---

            # 2b. 重塑 for Causal iT (同 backup17)
            # [B*N, Np, D] -> [B*Np, N, D]
            ipt_in = patch_out.reshape(B, N, N_p, self.d_model).permute(0, 2, 1, 3).reshape(B * N_p, N, self.d_model)

            # 3b. 运行 Causal iT
            # [B*Np, N, D] -> [B*Np, N, D]
            # [!!! 核心修改 v21 !!!] 传入 causal_mask
            ipt_out, _ = self.encoder(
                ipt_in,
                causal_mask=self.causal_graph_logits
            )

            # 4b. 恢复 (Reshape back) (同 backup17)
            # [B*Np, N, D] -> [B, Np, N, D] -> [B, N, Np, D]
            final_out = ipt_out.reshape(B, N_p, N, self.d_model).permute(0, 2, 1, 3)

        # ----------------------------------------------------

        return final_out  # 最终输出形状: (B, N, N_p, d_model)