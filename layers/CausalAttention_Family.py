# 文件名: layers/CausalAttention_Family.py
# 路径: D:\anjt\code\MS-IPM\MS-IPM\layers\CausalAttention_Family.py
#
# 描述:
# 这是 SelfAttention_Family.py 的一个最小化、修改过的版本，
# 专为 MS_IPM_backup21_CausalAttn 模型实现“因果注意力”。
# 仅保留 iTransformer (FullAttention) 所需的类。
#
# 关键修改:
# 1. CausalFullAttention:
#    - forward 方法增加了一个 `causal_mask` 参数。
#    - 这个 `causal_mask` (shape [N, N]) 会被广播并
#      直接加到 `scores` 矩阵 (shape [B*Np, H, N, N]) 上，
#      从而在 softmax 之前引导注意力。
# 2. CausalAttentionLayer:
#    - forward 方法也增加了 `causal_mask` 参数，
#      并将其传递给内部的 `self.inner_attention`。
#

import torch
import torch.nn as nn
import numpy as np
from math import sqrt
# 注意：我们特意移除了 'utils.masking' 的导入，
# 因为 iTransformer (mask_flag=False) 不需要它，这避免了额外的文件依赖。


class CausalFullAttention(nn.Module):
    """
    修改自 FullAttention，以接受一个可学习的因果图 (logits) 作为注意力偏置 (bias)。
    """

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(CausalFullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag  # iTransformer 会将其设为 False
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, causal_mask=None, tau=None, delta=None):
        # queries/keys/values shape: [B, L, H, E] (在 iT 中, L = N)
        # causal_mask shape: [N, N] (由 IPT_Block_Encoder 传入)

        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # scores shape: [B, H, L, S] (在 iT 中, [B*Np, H, N, N])
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # [!!! 核心因果修改 !!!]
        # 将可学习的因果图 (logits) 作为偏置项添加到 scores
        if causal_mask is not None:
            # causal_mask [N, N] -> [1, 1, N, N]
            # scores [B*Np, H, N, N]
            # 广播相加, 实现对 N x N 维度的注意力引导
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        if self.mask_flag:
            # iT 模式下 (mask_flag=False) 会跳过这里
            if attn_mask is None:
                raise ValueError("CausalFullAttention: mask_flag=True 但 attn_mask=None")
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class CausalAttentionLayer(nn.Module):
    """
    修改自 AttentionLayer，以中继 `causal_mask` 参数。
    """

    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(CausalAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        # self.inner_attention 必须是一个 CausalFullAttention 实例
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, causal_mask=None, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # [!!! 核心因果修改 !!!]
        # 将 causal_mask 传递给 CausalFullAttention
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            causal_mask=causal_mask,  # 传递掩码
            tau=tau,
            delta=delta
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn