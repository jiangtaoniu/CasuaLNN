import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class my_Layernorm(nn.Module):
    """
    A special LayerNorm variant for the seasonal component of a time series.

    This layer applies standard LayerNorm and then subtracts the mean of the
    normalized output. This effectively centers the seasonal component around zero.
    """
    def __init__(self, channels: int):
        """
        Args:
            channels (int): The number of features/channels to normalize.
        """
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the custom layer normalization.

        Args:
            x (torch.Tensor): Input tensor, shape (B, L, D).

        Returns:
            torch.Tensor: Normalized and centered tensor.
        """
        x_hat = self.layernorm(x)
        # Subtract the mean across the time dimension to center the output
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block used to extract the trend component of a time series.
    """
    def __init__(self, kernel_size: int, stride: int):
        """
        Args:
            kernel_size (int): The size of the moving average window.
            stride (int): The stride of the moving average.
        """
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the moving average.

        To maintain the sequence length, the input is padded at both ends before
        applying the average pooling operation.

        Args:
            x (torch.Tensor): Input tensor, shape (B, L, D).

        Returns:
            torch.Tensor: The trend component, with the same shape as the input.
        """
        # Pad the series at both ends to maintain the original length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        
        # Apply average pooling and permute back to original shape
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Decomposes a time series into its trend and seasonal (residual) components.

    This is a core component of the Autoformer model architecture.
    """
    def __init__(self, kernel_size: int):
        """
        Args:
            kernel_size (int): The size of the moving average window used for trend extraction.
        """
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs series decomposition.

        Args:
            x (torch.Tensor): Input time series, shape (B, L, D).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the seasonal component (residual)
                                               and the trend component.
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean # Residual (seasonal component)
        return res, moving_mean


class series_decomp_multi(nn.Module):
    """
    Performs series decomposition using multiple moving average kernels, as proposed in FEDformer.

    This method applies multiple `series_decomp` blocks with different kernel sizes and
    averages their results to obtain a more robust trend and seasonal component.
    """
    def __init__(self, kernel_size: List[int]):
        """
        Args:
            kernel_size (List[int]): A list of kernel sizes for the moving average blocks.
        """
        super(series_decomp_multi, self).__init__()
        self.kernel_size = kernel_size
        self.series_decomp = nn.ModuleList([series_decomp(kernel) for kernel in kernel_size])

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs multi-scale series decomposition.

        Args:
            x (torch.Tensor): Input time series, shape (B, L, D).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the averaged seasonal component
                                               and the averaged trend component.
        """
        moving_mean = []
        res = []
        for func in self.series_decomp:
            sea, moving_avg = func(x)
            moving_mean.append(moving_avg)
            res.append(sea)

        # Average the results from all decomposition blocks
        sea = sum(res) / len(res)
        moving_mean = sum(moving_mean) / len(moving_mean)
        return sea, moving_mean


class EncoderLayer(nn.Module):
    """
    A single layer of the Autoformer Encoder, featuring a progressive decomposition architecture.

    This layer applies self-attention and a feed-forward network, with series decomposition
    interspersed to separate and process seasonal and trend components.
    """
    def __init__(self, attention: nn.Module, d_model: int, d_ff: Optional[int] = None, moving_avg: int = 25,
                 dropout: float = 0.1, activation: str = "relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for the Autoformer EncoderLayer.
        """
        # --- Auto-Correlation (Self-Attention) and First Decomposition ---
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        x, _ = self.decomp1(x) # Decompose and keep the seasonal part

        # --- Feed-Forward and Second Decomposition ---
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        res, _ = self.decomp2(x + y) # Decompose the combined result and keep the seasonal part
        
        return res, attn


class Encoder(nn.Module):
    """
    A stack of EncoderLayers to form the full Autoformer Encoder.
    """
    def __init__(self, attn_layers: List[nn.Module], conv_layers: Optional[List[nn.Module]] = None,
                 norm_layer: Optional[nn.Module] = None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, List[Optional[torch.Tensor]]]:
        """
        Forward pass for the Autoformer Encoder.
        """
        attns = []
        if self.conv_layers is not None:
            # Apply attention and optional downsampling convolution sequentially
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            # Apply attention layers sequentially
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    A single layer of the Autoformer Decoder, featuring a progressive decomposition architecture.

    This layer incorporates self-attention, cross-attention, and a feed-forward network,
    with three series decomposition blocks to progressively refine the trend component.
    """
    def __init__(self, self_attention: nn.Module, cross_attention: nn.Module, d_model: int, c_out: int,
                 d_ff: Optional[int] = None, moving_avg: int = 25, dropout: float = 0.1, activation: str = "relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Autoformer DecoderLayer.
        """
        # --- Masked Self-Attention and First Decomposition ---
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x, trend1 = self.decomp1(x) # Decompose and get seasonal part `x` and trend part `trend1`

        # --- Cross-Attention and Second Decomposition ---
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        x, trend2 = self.decomp2(x) # Decompose and get seasonal part `x` and trend part `trend2`

        # --- Feed-Forward and Third Decomposition ---
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y) # Decompose and get seasonal part `x` and trend part `trend3`

        # Aggregate the trend components from all three decomposition blocks
        residual_trend = trend1 + trend2 + trend3
        # Project the aggregated trend component
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)
        
        return x, residual_trend


class Decoder(nn.Module):
    """
    A stack of DecoderLayers to form the full Autoformer Decoder.
    """
    def __init__(self, layers: List[nn.Module], norm_layer: Optional[nn.Module] = None,
                 projection: Optional[nn.Module] = None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x: torch.Tensor, cross: torch.Tensor, x_mask: Optional[torch.Tensor] = None,
                cross_mask: Optional[torch.Tensor] = None, trend: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the Autoformer Decoder.
        """
        # Sequentially apply each DecoderLayer
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            # Progressively accumulate the trend components from each layer
            trend = trend + residual_trend

        # Apply final normalization and projection if specified
        if self.norm is not None:
            x = self.norm(x)
        if self.projection is not None:
            x = self.projection(x)
            
        return x, trend