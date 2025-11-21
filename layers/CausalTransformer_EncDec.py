# 文件名: layers/CausalTransformer_EncDec.py
# 路径: D:\anjt\code\MS-IPM\MS-IPM\layers\CausalTransformer_EncDec.py
#
# 描述:
# 这是 Transformer_EncDec.py 的一个最小化、修改过的版本，
# 专为 MS_IPM_backup21_CausalAttn 模型实现“因果注意力”。
#
# 关键修改:
# 1. CausalEncoderLayer:
#    - forward 方法增加了 `causal_mask` 参数，
#      并将其传递给 `self.attention` (即 CausalAttentionLayer)。
# 2. CausalEncoder:
#    - forward 方法增加了 `causal_mask` 参数，
#      并将其传递给每个 `attn_layer`。

import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalEncoderLayer(nn.Module):
    """
    修改自 Transformer_EncDec.EncoderLayer
    """
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(CausalEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        # self.attention 必须是一个 CausalAttentionLayer 实例
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, causal_mask=None, tau=None, delta=None):
        # [!!! 核心因果修改 !!!]
        # 传递 causal_mask
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            causal_mask=causal_mask, # 传递掩码
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class CausalEncoder(nn.Module):
    """
    修改自 Transformer_EncDec.Encoder
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(CausalEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, causal_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            # (iT 控制器不使用 conv_layers, 为简单起见,
            #  我们假设 self.conv_layers is None, 如 backup17 中所示)
            raise NotImplementedError("CausalEncoder 尚未实现 conv_layers 路径")
        else:
            # [!!! 核心因果修改 !!!]
            # 在循环中传递 causal_mask
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x,
                                     attn_mask=attn_mask,
                                     causal_mask=causal_mask, # 传递掩码
                                     tau=tau,
                                     delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns