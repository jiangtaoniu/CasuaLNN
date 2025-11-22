import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from typing import Optional, Tuple

"""
This script provides attention layers modified to incorporate a learnable causal graph
as an attention bias. It is a specialized version of `SelfAttention_Family.py`
designed for models like CasuaLNN that learn inter-variable dependencies.

Key Modifications:
1. `CausalFullAttention`: The `forward` method accepts an additional `causal_mask`
   argument. This mask (typically with shape [N, N] for N variables) is broadcast
   and added to the attention scores before the softmax, thereby biasing the
   attention mechanism based on the learned causal graph.
2. `CausalAttentionLayer`: This layer acts as a wrapper that properly forwards
   the `causal_mask` argument to its inner attention module.
"""

# Note: The import of 'utils.masking' is intentionally removed as this module is
# designed for the iTransformer architecture (where mask_flag=False), which does
# not require standard causal masking, thus avoiding extra file dependencies.


class CausalFullAttention(nn.Module):
    """
    A modification of FullAttention that accepts a learnable causal graph (logits)
    as an attention bias. This allows the model to inject prior or learned structural
    knowledge directly into the attention mechanism.
    """

    def __init__(self, mask_flag: bool = True, factor: int = 5, scale: Optional[float] = None,
                 attention_dropout: float = 0.1, output_attention: bool = False):
        """
        Args:
            mask_flag (bool): If True, applies a standard attention mask. In models like iTransformer, this is set to False.
            factor (int): Unused in this module, maintained for API consistency.
            scale (Optional[float]): Custom scaling factor for attention scores. Defaults to 1/sqrt(d_k).
            attention_dropout (float): Dropout rate for attention weights.
            output_attention (bool): If True, returns the attention weights.
        """
        super(CausalFullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor], causal_mask: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None, delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for CausalFullAttention.

        Args:
            queries/keys/values (torch.Tensor): Input tensors for attention, shape (B, L, H, E). In the iTransformer context, L represents the number of variables (N).
            attn_mask (Optional[torch.Tensor]): Standard attention mask (e.g., for padding).
            causal_mask (Optional[torch.Tensor]): The learnable causal graph, shape (N, N), which is broadcast and added as a bias to the attention scores.
            tau/delta (Optional[torch.Tensor]): Unused de-stationary factors, maintained for API consistency.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The context vector and optionally the attention weights.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Standard dot-product attention scores
        # Shape: (B, H, L, S), which in the iT context is (B*num_patches, H, N, N)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # [Causal Modification]
        # Add the learnable causal graph (logits) as a bias to the scores.
        if causal_mask is not None:
            # `causal_mask` [N, N] is broadcasted to match `scores` [B*num_patches, H, N, N].
            # This guides the attention mechanism based on the learned inter-variable dependencies.
            scores = scores + causal_mask.unsqueeze(0).unsqueeze(0)

        # Apply standard attention mask if required (typically disabled for iTransformer).
        if self.mask_flag:
            if attn_mask is None:
                raise ValueError("CausalFullAttention: mask_flag is True but no attn_mask was provided.")
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class CausalAttentionLayer(nn.Module):
    """
    A modification of the standard AttentionLayer that relays the `causal_mask`
    parameter to its inner attention module.
    """

    def __init__(self, attention: nn.Module, d_model: int, n_heads: int,
                 d_keys: Optional[int] = None, d_values: Optional[int] = None):
        """
        Args:
            attention (nn.Module): The core attention mechanism, which must be an instance
                                   of a class that accepts a `causal_mask` argument,
                                   e.g., `CausalFullAttention`.
            d_model (int): Dimension of the model's hidden states.
            n_heads (int): Number of attention heads.
            d_keys (Optional[int]): Dimension of keys per head. Defaults to d_model // n_heads.
            d_values (Optional[int]): Dimension of values per head. Defaults to d_model // n_heads.
        """
        super(CausalAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor], causal_mask: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None, delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the CausalAttentionLayer.

        Args:
            queries/keys/values (torch.Tensor): Input tensors, shape (B, L, D_model).
            attn_mask (Optional[torch.Tensor]): Standard attention mask.
            causal_mask (Optional[torch.Tensor]): The learnable causal graph to be passed to the inner attention.
            tau/delta: Optional de-stationary factors, passed to the inner attention.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The output context vector and optionally the attention weights.
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        # Project and reshape for multi-head attention
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        # [Causal Modification]
        # Pass the causal_mask to the inner attention mechanism.
        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            causal_mask=causal_mask,  # Pass the mask through
            tau=tau,
            delta=delta
        )
        
        # Concatenate heads and apply final output projection
        out = out.reshape(B, L, -1)
        return self.out_projection(out), attn
