# G:\Code\MS-IPM\MS-IPM\layers\DynamicLiquid.py
# [!!! 新文件 !!!]
#
# 该文件定义了 LNN-iT-DynamicODE 架构的核心：
# 1. DynamicCfCCell: 一个 CfC 单元，其内部 ODE 参数 (t_a, t_b) 可以被外部信号动态调制。
# 2. DynamicCfC: 一个 RNN 封装器，负责在循环中将外部参数序列传递给 DynamicCfCCell。
# 3. DynamicLiquidEncoder: 一个 Transformer 风格的编码器，将 DynamicCfC 封装为标准层。
#
# 它依赖于您在上下文中提供的 LeCun 和 LSTMCell 实现。

import torch
from torch import nn
from typing import Optional, Union
import ncps
import numpy as np
import torch.nn.functional as F


# --- 依赖项：从上下文中复制 LeCun ---
class LeCun(nn.Module):
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x):
        return 1.7159 * self.tanh(0.666 * x)


# --- 依赖项：从上下文中复制 LSTMCell (DynamicCfC的mixed_memory需要) ---
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs, states):
        output_state, cell_state = states
        z = self.input_map(inputs) + self.recurrent_map(output_state)
        i, ig, fg, og = z.chunk(4, 1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)
        output_gate = self.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = self.tanh(new_cell) * output_gate

        return output_state, new_cell


# --- 核心修改 1: DynamicCfCCell ---
# (基于上下文中的 CfCCell 修改)
class DynamicCfCCell(nn.Module):
    def __init__(
            self,
            input_size,
            hidden_size,
            mode="default",
            backbone_activation="lecun_tanh",
            backbone_units=128,
            backbone_layers=1,
            backbone_dropout=0.0,
            sparsity_mask=None,
    ):
        super(DynamicCfCCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        allowed_modes = ["default", "pure", "no_gate"]
        if mode not in allowed_modes:
            raise ValueError(
                f"Unknown mode '{mode}', valid options are {str(allowed_modes)}"
            )
        self.sparsity_mask = (
            None
            if sparsity_mask is None
            else torch.nn.Parameter(
                data=torch.from_numpy(np.abs(sparsity_mask.T).astype(np.float32)),
                requires_grad=False,
            )
        )
        self.mode = mode

        if backbone_activation == "silu":
            backbone_activation = nn.SiLU
        elif backbone_activation == "relu":
            backbone_activation = nn.ReLU
        elif backbone_activation == "tanh":
            backbone_activation = nn.Tanh
        elif backbone_activation == "gelu":
            backbone_activation = nn.GELU
        elif backbone_activation == "lecun_tanh":
            backbone_activation = LeCun
        else:
            raise ValueError(f"Unknown activation {backbone_activation}")

        self.backbone = None
        self.backbone_layers = backbone_layers
        if backbone_layers > 0:
            layer_list = [
                nn.Linear(input_size + hidden_size, backbone_units),
                backbone_activation(),
            ]
            for i in range(1, backbone_layers):
                layer_list.append(nn.Linear(backbone_units, backbone_units))
                layer_list.append(backbone_activation())
                if backbone_dropout > 0.0:
                    layer_list.append(torch.nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        cat_shape = int(
            self.hidden_size + input_size if backbone_layers == 0 else backbone_units
        )

        self.ff1 = nn.Linear(cat_shape, hidden_size)
        if self.mode == "pure":
            self.w_tau = torch.nn.Parameter(
                data=torch.zeros(1, self.hidden_size), requires_grad=True
            )
            self.A = torch.nn.Parameter(
                data=torch.ones(1, self.hidden_size), requires_grad=True
            )
        else:
            self.ff2 = nn.Linear(cat_shape, hidden_size)
            # 这些是ODE动态的“静态”部分
            self.time_a = nn.Linear(cat_shape, hidden_size)
            self.time_b = nn.Linear(cat_shape, hidden_size)
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)

    def forward(self, input, hx, ts, dynamic_params=None):
        """
        [!!! 核心修改 !!!]
        新增 dynamic_params 参数，用于接收来自 iT Controller 的调制信号。
        dynamic_params 预期形状: [Batch, 2 * hidden_size]
        """
        x = torch.cat([input, hx], 1)
        if self.backbone_layers > 0:
            x = self.backbone(x)
        if self.sparsity_mask is not None:
            ff1 = F.linear(x, self.ff1.weight * self.sparsity_mask, self.ff1.bias)
        else:
            ff1 = self.ff1(x)

        if self.mode == "pure":
            # 纯净模式：动态调制 w_tau (时间常数)
            # (注意: "pure" 模式的动态调制更复杂，为简化，我们专注于 "default" 模式)
            # (如果需要 "pure" 模式，dynamic_params 应包含 delta_w_tau 和 delta_A)
            w_tau_final = self.w_tau
            A_final = self.A
            if dynamic_params is not None:
                # 假设 dynamic_params 包含 [delta_w_tau, delta_A]
                delta_w_tau, delta_A = dynamic_params.chunk(2, 1)
                w_tau_final = self.w_tau + delta_w_tau
                A_final = self.A + delta_A

            new_hidden = (
                    -A_final
                    * torch.exp(-ts * (torch.abs(w_tau_final) + torch.abs(ff1)))
                    * ff1
                    + A_final
            )
        else:
            # 默认模式 ("default" 或 "no_gate")
            if self.sparsity_mask is not None:
                ff2 = F.linear(x, self.ff2.weight * self.sparsity_mask, self.ff2.bias)
            else:
                ff2 = self.ff2(x)
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)

            # [!!! 核心修改 !!!]
            # 1. 计算 "静态" 的ODE参数 (t_a, t_b)
            t_a_static = self.time_a(x)
            t_b_static = self.time_b(x)

            # 2. (如果提供了) 应用 "动态" 调制
            if dynamic_params is not None:
                # 假设 dynamic_params 包含 [delta_t_a, delta_t_b]
                # 形状: [B, 2 * hidden_size]
                delta_t_a, delta_t_b = dynamic_params.chunk(2, 1)

                # 加性调制 (Additive Modulation)
                t_a_final = t_a_static + delta_t_a
                t_b_final = t_b_static + delta_t_b
            else:
                t_a_final = t_a_static
                t_b_final = t_b_static

            # 3. 使用最终的参数计算插值
            t_interp = self.sigmoid(t_a_final * ts + t_b_final)

            if self.mode == "no_gate":
                new_hidden = ff1 + t_interp * ff2
            else:  # "default" 模式
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, new_hidden


# --- 核心修改 2: DynamicCfC ---
# (基于上下文中的 CfC 修改)
class DynamicCfC(nn.Module):
    def __init__(
            self,
            input_size: Union[int, ncps.wirings.Wiring],
            units,  # 注意：在非Wired模式下，units == hidden_size
            proj_size: Optional[int] = None,
            return_sequences: bool = True,
            batch_first: bool = True,
            mixed_memory: bool = False,
            mode: str = "default",
            activation: str = "lecun_tanh",
            backbone_units: Optional[int] = None,
            backbone_layers: Optional[int] = None,
            backbone_dropout: Optional[int] = None,
    ):
        super(DynamicCfC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.proj_size = proj_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        # --- 仅保留非 Wired 模式 (根据 Liquid_Enc.py 的使用情况简化) ---
        # (WiredCfCCell 逻辑未实现动态参数，如需请参照 DynamicCfCCell 修改)
        if isinstance(units, ncps.wirings.Wiring):
            raise NotImplementedError("DynamicCfC 目前不支持 Wired 模式")

        self.wired_false = True
        backbone_units = 128 if backbone_units is None else backbone_units
        backbone_layers = 1 if backbone_layers is None else backbone_layers
        backbone_dropout = 0.0 if backbone_dropout is None else backbone_dropout
        self.state_size = units
        self.output_size = self.state_size

        # [!!! 核心修改 !!!]
        # 使用 DynamicCfCCell 而不是 CfCCell
        self.rnn_cell = DynamicCfCCell(
            input_size,
            self.wiring_or_units,  # self.wiring_or_units 就是 hidden_size
            mode,
            activation,
            backbone_units,
            backbone_layers,
            backbone_dropout,
        )

        self.use_mixed = mixed_memory
        if self.use_mixed:
            # 依赖于上面定义的 LSTMCell
            self.lstm = LSTMCell(input_size, self.state_size)

        if proj_size is None:
            self.fc = nn.Identity()
        else:
            self.fc = nn.Linear(self.output_size, self.proj_size)

    def forward(self, input, hx=None, timespans=None, dynamic_params=None):
        """
        [!!! 核心修改 !!!]
        新增 dynamic_params 参数
        预期形状: [B, L, k] (batch_first=True) 或 [L, B, k] (batch_first=False)
        其中 k = 2 * hidden_size
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim = 0 if self.batch_first else 1
        seq_dim = 1 if self.batch_first else 0
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)
            # [!!! 核心修改 !!!]
            if dynamic_params is not None:
                dynamic_params = dynamic_params.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = (
                torch.zeros((batch_size, self.state_size), device=device)
                if self.use_mixed
                else None
            )
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError("...")  # 状态检查 (同原版)
            h_state, c_state = hx if self.use_mixed else (hx, None)
            # ... (省略原版中对 hx 维度的检查) ...

        output_sequence = []
        for t in range(seq_len):
            if self.batch_first:
                inputs = input[:, t]
                ts = 1.0 if timespans is None else timespans[:, t].squeeze()
                # [!!! 核心修改 !!!]
                params_t = None if dynamic_params is None else dynamic_params[:, t]
            else:
                inputs = input[t]
                ts = 1.0 if timespans is None else timespans[t].squeeze()
                # [!!! 核心修改 !!!]
                params_t = None if dynamic_params is None else dynamic_params[t]

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs, (h_state, c_state))

            # [!!! 核心修改 !!!]
            # 将 params_t 传递给 DynamicCfCCell
            h_out, h_state = self.rnn_cell.forward(inputs, h_state, ts, dynamic_params=params_t)

            if self.return_sequences:
                output_sequence.append(self.fc(h_out))

        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = self.fc(h_out)

        hx = (h_state, c_state) if self.use_mixed else h_state

        if not is_batched:
            readout = readout.squeeze(batch_dim)
            hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]

        return readout, hx


# --- 核心修改 3: DynamicLiquidLayer 和 DynamicLiquidEncoder ---
# (基于上下文中的 Liquid_Enc.py 修改)

class DynamicLiquidLayer(nn.Module):
    """
    一个封装了 DynamicCfC 的层，带有残差连接和层归一化 (Pre-Norm)。

    输入/输出形状: [Batch, Sequence (Np), Features (D)]
    """

    def __init__(self, configs, dropout=0.1):
        super(DynamicLiquidLayer, self).__init__()
        self.d_model = configs.d_model

        self.norm = nn.LayerNorm(configs.d_model)

        # [!!! 核心修改 !!!]
        # 使用 DynamicCfC
        self.cfc = DynamicCfC(
            input_size=configs.d_model,
            units=configs.lq_units,
            proj_size=configs.lq_proj_size,
            batch_first=True,
            return_sequences=True,
            backbone_units=configs.lq_backbone_units,
            backbone_layers=configs.lq_backbone_layers,
            activation=configs.lq_activation
        )
        self.dropout = nn.Dropout(dropout)

        self.proj = nn.Linear(configs.lq_proj_size, configs.d_model) \
            if configs.lq_proj_size != configs.d_model else nn.Identity()

    def forward(self, x, dynamic_params, timespans=None):
        """
        [!!! 核心修改 !!!]
        x: [B, L, D]
        dynamic_params: [B, L, k] (k = 2 * lq_units)
        """
        x_norm = self.norm(x)

        # [!!! 核心修改 !!!]
        # 将 dynamic_params 传递给 cfc
        cfc_out, _ = self.cfc(x_norm, timespans=timespans, dynamic_params=dynamic_params)

        cfc_out = self.proj(cfc_out)

        x = x + self.dropout(cfc_out)

        return x


class DynamicLiquidEncoder(nn.Module):
    """
    一个堆叠的 DynamicLiquidLayer 编码器。
    """

    def __init__(self, layers, norm_layer=None):
        super(DynamicLiquidEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, dynamic_params, timespans=None):
        """
        [!!! 核心修改 !!!]
        x: [B, L, D]
        dynamic_params: [B, L, k]
        """
        for layer in self.layers:
            # [!!! 核心修改 !!!]
            # 传递 dynamic_params
            x = layer(x, dynamic_params=dynamic_params, timespans=timespans)

        if self.norm is not None:
            x = self.norm(x)

        return x, None  # 返回 None 以匹配 (output, attn) 的格式