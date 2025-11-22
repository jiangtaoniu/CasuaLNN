import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
from typing import Optional, Tuple


class DSAttention(nn.Module):
    """
    De-stationary Attention module.

    This attention mechanism adapts to non-stationary time series data by
    rescaling the pre-softmax attention scores with learned de-stationary factors,
    `tau` (for queries) and `delta` (for keys).
    """

    def __init__(self, mask_flag: bool = True, factor: int = 5, scale: Optional[float] = None,
                 attention_dropout: float = 0.1, output_attention: bool = False):
        """
        Args:
            mask_flag (bool, optional): Whether to apply masking. Defaults to True.
            factor (int, optional): Factor for ProbAttention, unused here. Defaults to 5.
            scale (Optional[float], optional): Custom scaling factor for attention scores. If None, defaults to 1/sqrt(d_k).
            attention_dropout (float, optional): Dropout rate for attention weights. Defaults to 0.1.
            output_attention (bool, optional): Whether to output attention weights. Defaults to False.
        """
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor], tau: Optional[torch.Tensor] = None,
                delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for De-stationary Attention.

        Args:
            queries (torch.Tensor): Shape (B, L, H, E).
            keys (torch.Tensor): Shape (B, S, H, E).
            values (torch.Tensor): Shape (B, S, H, D).
            attn_mask (Optional[torch.Tensor]): Attention mask.
            tau (Optional[torch.Tensor]): Learned de-stationary factor for queries. Shape (B, 1, 1, 1).
            delta (Optional[torch.Tensor]): Learned de-stationary factor for keys. Shape (B, 1, 1, S).

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The context vector and optionally the attention weights.
        """
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Rescale pre-softmax scores with de-stationary factors
        tau = 1.0 if tau is None else tau.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(1).unsqueeze(1)  # B x 1 x 1 x S
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        # Apply mask if specified
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Compute attention weights and apply to values
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    """
    Standard full self-attention mechanism, as described in "Attention is All You Need".
    """
    def __init__(self, mask_flag: bool = True, factor: int = 5, scale: Optional[float] = None,
                 attention_dropout: float = 0.1, output_attention: bool = False):
        """
        Args:
            mask_flag (bool, optional): Whether to apply masking. Defaults to True.
            factor (int, optional): Factor for ProbAttention, unused here. Defaults to 5.
            scale (Optional[float], optional): Custom scaling factor for attention scores. If None, defaults to 1/sqrt(d_k).
            attention_dropout (float, optional): Dropout rate for attention weights. Defaults to 0.1.
            output_attention (bool, optional): Whether to output attention weights. Defaults to False.
        """
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor], tau: Optional[torch.Tensor] = None,
                delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Full Attention. `tau` and `delta` are ignored.
        """
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # Standard dot-product attention
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        # Apply mask if specified
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # Compute attention weights and apply to values
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class ProbAttention(nn.Module):
    """
    Probabilistic Sparse Attention ("ProbSparse") mechanism, as proposed in the Informer paper.

    This attention mechanism reduces complexity from O(L^2) to O(L*log(L)) by only
    calculating attention scores for a subset of "important" queries, selected based
    on a sparsity measurement.
    """
    def __init__(self, mask_flag: bool = True, factor: int = 5, scale: Optional[float] = None,
                 attention_dropout: float = 0.1, output_attention: bool = False):
        """
        Args:
            mask_flag (bool, optional): Whether to apply masking. Defaults to True.
            factor (int, optional): The sparsity factor `c` for selecting top queries. Defaults to 5.
            scale (Optional[float], optional): Custom scaling factor for attention scores. If None, defaults to 1/sqrt(d_k).
            attention_dropout (float, optional): Dropout rate for attention weights. Defaults to 0.1.
            output_attention (bool, optional): Whether to output attention weights. Defaults to False.
        """
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q: torch.Tensor, K: torch.Tensor, sample_k: int, n_top: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Selects the top `n_top` queries based on a sparsity measurement and
        calculates their attention scores against all keys.

        Args:
            Q (torch.Tensor): Queries, shape (B, H, L_Q, E).
            K (torch.Tensor): Keys, shape (B, H, L_K, E).
            sample_k (int): Number of keys to sample for sparsity measurement.
            n_top (int): Number of top queries to select.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The scores for the top queries and their indices.
        """
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # Sample a subset of keys for efficient sparsity measurement
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze(-2)

        # Calculate sparsity measurement M(q,K) = max(q*k) - mean(q*k)
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        # Select the top `n_top` queries with the highest sparsity measurement
        M_top = M.topk(n_top, sorted=False)[1]

        # Use the selected queries to calculate attention scores against all keys
        Q_reduce = Q[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], M_top, :]
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))

        return Q_K, M_top

    def _get_initial_context(self, V: torch.Tensor, L_Q: int) -> torch.Tensor:
        """
        Generates an initial context vector. For non-causal attention, it's the mean of all values;
        for causal attention, it's the cumulative sum up to the current position.
        """
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # For non-causal attention, context is the mean of all values
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
        else: # Causal attention
            assert (L_Q == L_V), "Causal masking requires Q and V to have same length."
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in: torch.Tensor, V: torch.Tensor, scores: torch.Tensor,
                        index: torch.Tensor, L_Q: int, attn_mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Updates the initial context with the attention results from the selected top-k queries.
        """
        B, H, L_V, D = V.shape

        if self.mask_flag:
            # Apply ProbMask to the scores of the selected queries
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)

        # Update only the context vectors corresponding to the selected top-k queries
        context_in[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = torch.matmul(attn, V).type_as(context_in)
        
        if self.output_attention:
            # Construct a full attention map for visualization/analysis if required
            attns = (torch.ones([B, H, L_V, L_V]) / L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor], tau: Optional[torch.Tensor] = None,
                delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for Probabilistic Sparse Attention. `tau` and `delta` are ignored.
        """
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Number of keys to sample for sparsity measurement: c*ln(L_K)
        U_part = self.factor * int(np.ceil(np.log(L_K)))
        # Number of top queries to select: c*ln(L_Q)
        u = self.factor * int(np.ceil(np.log(L_Q)))

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        # Select top queries and get their scores
        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # Apply scaling factor
        scale = self.scale or 1. / sqrt(D)
        scores_top = scores_top * scale
        
        # Get initial context and update it with top-k attention
        context = self._get_initial_context(values, L_Q)
        context, attn = self._update_context(context, values, scores_top, index, L_Q, attn_mask)

        # Transpose back to (B, L, H, D) before returning
        return context.transpose(1, 2).contiguous(), attn


class AttentionLayer(nn.Module):
    """
    A standard attention layer that combines Q, K, V projections, a core attention
    mechanism, and a final output projection.
    """
    def __init__(self, attention: nn.Module, d_model: int, n_heads: int,
                 d_keys: Optional[int] = None, d_values: Optional[int] = None):
        """
        Args:
            attention (nn.Module): The core attention mechanism (e.g., FullAttention, ProbAttention).
            d_model (int): Dimension of the model's hidden states.
            n_heads (int): Number of attention heads.
            d_keys (Optional[int]): Dimension of keys per head. Defaults to d_model // n_heads.
            d_values (Optional[int]): Dimension of values per head. Defaults to d_model // n_heads.
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor], tau: Optional[torch.Tensor] = None,
                delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the attention layer.

        Args:
            queries (torch.Tensor): Shape (B, L, D_model).
            keys (torch.Tensor): Shape (B, S, D_model).
            values (torch.Tensor): Shape (B, S, D_model).
            attn_mask (Optional[torch.Tensor]): Attention mask.
            tau, delta: Optional de-stationary factors for DSAttention.

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

        # Apply inner attention mechanism
        out, attn = self.inner_attention(queries, keys, values, attn_mask, tau=tau, delta=delta)
        
        # Concatenate heads and apply final output projection
        out = out.reshape(B, L, -1)
        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    """
    An attention layer using LSH (Locality-Sensitive Hashing) Self-Attention
    from the `reformer-pytorch` library for efficient attention.
    """
    def __init__(self, attention: Optional[nn.Module] = None, d_model: int = 512, n_heads: int = 8,
                 d_keys: Optional[int] = None, d_values: Optional[int] = None,
                 causal: bool = False, bucket_size: int = 4, n_hashes: int = 4):
        """
        Args:
            attention (Optional[nn.Module]): Unused here, for API consistency.
            d_model (int): Dimension of the model's hidden states.
            n_heads (int): Number of attention heads.
            causal (bool): Whether to use causal masking.
            bucket_size (int): Size of LSH buckets.
            n_hashes (int): Number of LSH hashes.
        """
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries: torch.Tensor) -> torch.Tensor:
        """
        Pads the input sequence to a length that is a multiple of (bucket_size * 2),
        as required by the LSH attention implementation.
        """
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries # No padding needed
        else:
            # Calculate required padding length and pad with zeros
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                attn_mask: Optional[torch.Tensor], tau: Optional[torch.Tensor] = None,
                delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for ReformerLayer. Assumes queries=keys=values for self-attention.
        """
        B, N, C = queries.shape
        # Pad, apply attention, then truncate back to original length
        reformer_out = self.attn(self.fit_length(queries))[:, :N, :]
        return reformer_out, None


class TwoStageAttentionLayer(nn.Module):
    """
    The Two Stage Attention (TSA) Layer, which applies attention across both
    the time and feature dimensions.

    - Cross-Time Stage: Applies standard multi-head self-attention to each feature dimension independently.
    - Cross-Dimension Stage: Uses a set of learnable vectors (router) to aggregate and
      distribute information across the feature dimensions.
    """
    def __init__(self, configs, seg_num: int, factor: int, d_model: int,
                 n_heads: int, d_ff: Optional[int] = None, dropout: float = 0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        # Attention layer for the time dimension
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        # Attention layers for the feature dimension (sender and receiver)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=False), d_model, n_heads)
        # Learnable router vectors to facilitate cross-dimension communication
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for Two-Stage Attention.

        Args:
            x (torch.Tensor): Input tensor, shape (B, D_feature, L_segment, D_model).
        
        Returns:
            torch.Tensor: Output tensor with the same shape as input.
        """
        batch = x.shape[0]

        # --- Cross-Time Stage ---
        # Reshape to apply attention across time for each feature dimension
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, _ = self.time_attention(time_in, time_in, time_in, attn_mask=None)
        
        # Add & Norm block with FFN
        dim_in = self.norm1(time_in + self.dropout(time_enc))
        dim_in = self.norm2(dim_in + self.dropout(self.MLP1(dim_in)))

        # --- Cross-Dimension Stage ---
        # Reshape to apply attention across features for each time segment
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        
        # Use learnable routers to aggregate information from feature dimensions
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, _ = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None)
        
        # Use aggregated information to update feature dimension representations
        dim_receive, _ = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None)
        
        # Add & Norm block with FFN
        dim_enc = self.norm3(dim_send + self.dropout(dim_receive))
        dim_enc = self.norm4(dim_enc + self.dropout(self.MLP2(dim_enc)))

        # Reshape back to original input shape
        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out