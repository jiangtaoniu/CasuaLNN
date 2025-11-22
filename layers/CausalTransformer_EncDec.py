import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

"""
This script provides Transformer Encoder components (`CausalEncoder` and `CausalEncoderLayer`)
that are specifically adapted to accept and propagate a `causal_mask`. This enables them
to be used with a learnable causal graph for biasing the self-attention mechanism,
which is a core feature of the CasuaLNN model.

This is a minimized and modified version of a standard Transformer Encoder implementation.
"""

class CausalEncoderLayer(nn.Module):
    """
    A modified Transformer Encoder Layer that accepts a `causal_mask`.

    This layer integrates a causal-aware self-attention mechanism with a
    position-wise feed-forward network, including residual connections and layer normalization.
    """
    def __init__(self, attention: nn.Module, d_model: int, d_ff: Optional[int] = None,
                 dropout: float = 0.1, activation: str = "relu"):
        """
        Args:
            attention (nn.Module): The attention module to use. This is expected to be an
                                   instance of `CausalAttentionLayer`.
            d_model (int): The dimension of the model's hidden states.
            d_ff (Optional[int]): The dimension of the feed-forward network's intermediate layer.
                                  Defaults to 4 * d_model.
            dropout (float): Dropout rate.
            activation (str): The activation function ('relu' or 'gelu').
        """
        super(CausalEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = attention
        # Position-wise Feed-Forward Network implemented using 1D convolutions
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None, delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the CausalEncoderLayer.

        Args:
            x (torch.Tensor): Input tensor, shape (B, L, D).
            attn_mask (Optional[torch.Tensor]): Standard attention mask (e.g., for padding).
            causal_mask (Optional[torch.Tensor]): The learnable causal graph to be passed to the attention layer.
            tau, delta: Optional de-stationary factors, passed to the attention layer.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The output tensor and optionally the attention weights.
        """
        # --- Causal Self-Attention Block ---
        # The causal_mask is passed directly to the CausalAttentionLayer.
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            causal_mask=causal_mask, # Pass the causal mask here
            tau=tau, delta=delta
        )
        x = x + self.dropout(new_x) # Residual connection
        y = self.norm1(x) # Layer normalization

        # --- Position-wise Feed-Forward Block ---
        # The FFN is applied to the output of the attention block.
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # Second residual connection and layer normalization
        return self.norm2(x + y), attn


class CausalEncoder(nn.Module):
    """
    A stack of CausalEncoderLayers.

    This module sequentially applies multiple CausalEncoderLayers to the input data.
    """
    def __init__(self, attn_layers: List[nn.Module], conv_layers: Optional[List[nn.Module]] = None, norm_layer: Optional[nn.Module] = None):
        """
        Args:
            attn_layers (List[nn.Module]): A list of CausalEncoderLayer instances.
            conv_layers (Optional[List[nn.Module]]): Optional convolutional layers (not used in the current causal architecture).
            norm_layer (Optional[nn.Module]): A normalization layer to apply after the last attention layer.
        """
        super(CausalEncoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                causal_mask: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None, delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        Forward pass for the CausalEncoder.

        Args:
            x (torch.Tensor): Input tensor, shape (B, L, D).
            attn_mask (Optional[torch.Tensor]): Standard attention mask.
            causal_mask (Optional[torch.Tensor]): The learnable causal graph, passed to each layer.
            tau, delta: Optional de-stationary factors, passed to each layer.

        Returns:
            Tuple[torch.Tensor, List[Optional[torch.Tensor]]]: The encoded output tensor and a list of attention weights from each layer.
        """
        attns = []
        
        # This path is not used in the current Causal/iTransformer controller architecture,
        # but is kept for potential future compatibility.
        if self.conv_layers is not None:
            raise NotImplementedError("The `conv_layers` path is not implemented for CausalEncoder.")
        else:
            # Sequentially apply each CausalEncoderLayer, passing the causal_mask through.
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x,
                                     attn_mask=attn_mask,
                                     causal_mask=causal_mask, # Pass the causal mask here
                                     tau=tau,
                                     delta=delta)
                attns.append(attn)

        # Apply final normalization if specified
        if self.norm is not None:
            x = self.norm(x)

        return x, attns
