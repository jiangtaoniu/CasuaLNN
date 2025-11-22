import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class ConvLayer(nn.Module):
    """
    A 1D convolutional layer used for downsampling time series data within an encoder.
    It applies a convolution, normalization, activation, and max pooling.
    """
    def __init__(self, c_in: int):
        """
        Args:
            c_in (int): The number of input channels (features).
        """
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the convolutional downsampling layer.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, D).

        Returns:
            torch.Tensor: Output tensor of shape (B, L_new, D), where L_new is downsampled.
        """
        # Permute from (B, L, D) to (B, D, L) for Conv1d
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        # Transpose back to (B, L_new, D)
        x = x.transpose(1, 2)
        return x


class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.

    This layer consists of a self-attention mechanism followed by a position-wise
    feed-forward network. Residual connections and layer normalization are applied
    around each of the two sub-layers.
    """
    def __init__(self, attention: nn.Module, d_model: int, d_ff: Optional[int] = None,
                 dropout: float = 0.1, activation: str = "relu"):
        """
        Args:
            attention (nn.Module): The attention module to use (e.g., FullAttention).
            d_model (int): The dimension of the model's hidden states.
            d_ff (Optional[int]): The dimension of the feed-forward network's intermediate layer. Defaults to 4 * d_model.
            dropout (float): Dropout rate.
            activation (str): The activation function ('relu' or 'gelu').
        """
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None, delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the EncoderLayer.

        Args:
            x (torch.Tensor): Input tensor, shape (B, L, D).
            attn_mask (Optional[torch.Tensor]): Attention mask.
            tau, delta: Optional de-stationary factors for DSAttention.

        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: The output tensor and optionally the attention weights.
        """
        # --- Self-Attention Block ---
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x) # Residual connection
        y = self.norm1(x)           # Layer normalization

        # --- Position-wise Feed-Forward Block ---
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # Second residual connection and layer normalization
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    """
    A stack of EncoderLayers to form the full Transformer Encoder.
    """
    def __init__(self, attn_layers: List[nn.Module], conv_layers: Optional[List[nn.Module]] = None,
                 norm_layer: Optional[nn.Module] = None):
        """
        Args:
            attn_layers (List[nn.Module]): A list of EncoderLayer instances.
            conv_layers (Optional[List[nn.Module]]): Optional list of ConvLayer instances for downsampling
                                                     between attention layers.
            norm_layer (Optional[nn.Module]): A normalization layer to apply after the last encoder layer.
        """
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None, delta: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        Forward pass for the Encoder.

        Args:
            x (torch.Tensor): Input tensor, shape (B, L, D).
            attn_mask (Optional[torch.Tensor]): Attention mask.
            tau, delta: Optional de-stationary factors, passed to each layer.

        Returns:
            Tuple[torch.Tensor, List[Optional[torch.Tensor]]]: The encoded output tensor and a list of attention weights.
        """
        attns = []
        if self.conv_layers is not None:
            # Apply attention and convolution sequentially
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                # De-stationary factors are typically applied only to the first layer
                delta_i = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta_i)
                x = conv_layer(x) # Downsample
                attns.append(attn)
            # Final attention layer without downsampling
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            # Apply attention layers sequentially without convolution
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        # Apply final normalization if specified
        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    A single layer of the Transformer Decoder.

    This layer consists of a masked self-attention mechanism, a cross-attention
    mechanism (attending to the encoder's output), and a position-wise
    feed-forward network.
    """
    def __init__(self, self_attention: nn.Module, cross_attention: nn.Module, d_model: int,
                 d_ff: Optional[int] = None, dropout: float = 0.1, activation: str = "relu"):
        """
        Args:
            self_attention (nn.Module): The masked self-attention module.
            cross_attention (nn.Module): The cross-attention module.
            d_model (int): The dimension of the model's hidden states.
            d_ff (Optional[int]): The dimension of the feed-forward network's intermediate layer. Defaults to 4 * d_model.
            dropout (float): Dropout rate.
            activation (str): The activation function ('relu' or 'gelu').
        """
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the DecoderLayer.

        Args:
            x (torch.Tensor): Input tensor from the previous decoder layer, shape (B, L, D).
            cross (torch.Tensor): Output from the encoder, shape (B, S, D).
            x_mask (Optional[torch.Tensor]): Mask for self-attention (causal mask).
            cross_mask (Optional[torch.Tensor]): Mask for cross-attention (padding mask).
            tau, delta: Optional de-stationary factors for DSAttention.

        Returns:
            torch.Tensor: The output tensor from the decoder layer.
        """
        # --- Masked Self-Attention Block ---
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=None)[0])
        x = self.norm1(x)

        # --- Cross-Attention Block ---
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask, tau=tau, delta=delta)[0])
        y = self.norm2(x)

        # --- Position-wise Feed-Forward Block ---
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)


class Decoder(nn.Module):
    """
    A stack of DecoderLayers to form the full Transformer Decoder.
    """
    def __init__(self, layers: List[nn.Module], norm_layer: Optional[nn.Module] = None,
                 projection: Optional[nn.Module] = None):
        """
        Args:
            layers (List[nn.Module]): A list of DecoderLayer instances.
            norm_layer (Optional[nn.Module]): A normalization layer to apply after the last decoder layer.
            projection (Optional[nn.Module]): A final linear layer to project the output to the desired dimension.
        """
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None,
                tau: Optional[torch.Tensor] = None, delta: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the Decoder.

        Args:
            x (torch.Tensor): Input tensor for the decoder, shape (B, L, D).
            cross (torch.Tensor): Output from the encoder, shape (B, S, D).
            x_mask (Optional[torch.Tensor]): Mask for self-attention.
            cross_mask (Optional[torch.Tensor]): Mask for cross-attention.
            tau, delta: Optional de-stationary factors.

        Returns:
            torch.Tensor: The final output tensor from the decoder.
        """
        # Sequentially apply each DecoderLayer
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)

        # Apply final normalization and projection if specified
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
            
        return x