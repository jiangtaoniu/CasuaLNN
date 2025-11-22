import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
from typing import Optional


class PositionalEmbedding(nn.Module):
    """
    Sinusoidal Positional Encoding, as described in "Attention is All You Need".

    This layer adds positional information to the input embeddings. The encodings
    are fixed (non-trainable) and are computed once during initialization.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Args:
            d_model (int): The dimension of the model's hidden states.
            max_len (int): The maximum possible sequence length.
        """
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        # Sine for even indices, cosine for odd indices
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe) # Register as a non-trainable buffer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Adds positional encoding to the input tensor.

        Args:
            x (torch.Tensor): The input tensor, shape (B, L, D).

        Returns:
            torch.Tensor: The positional encoding tensor, shape (1, L, D).
        """
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    """
    Token Embedding using a 1D convolution.

    This layer projects the input features (channels) of a time series into a
    `d_model`-dimensional space. It's often used as a value embedding layer.
    """
    def __init__(self, c_in: int, d_model: int):
        """
        Args:
            c_in (int): Number of input channels/features.
            d_model (int): Dimension of the model's hidden states.
        """
        super(TokenEmbedding, self).__init__()
        # Use circular padding for the 1D convolution
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        # Initialize weights using Kaiming normalization
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the token embedding.

        Args:
            x (torch.Tensor): Input tensor, shape (B, L, C).

        Returns:
            torch.Tensor: Embedded tensor, shape (B, L, D).
        """
        # Permute to (B, C, L) for Conv1d, then transpose back
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    """
    Fixed, non-trainable embedding layer using sinusoidal functions.

    Similar to `PositionalEmbedding`, but used for embedding categorical features
    (e.g., month, day) rather than time steps.
    """
    def __init__(self, c_in: int, d_model: int):
        """
        Args:
            c_in (int): The number of unique categories to embed.
            d_model (int): The dimension of the embedding.
        """
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.requires_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Looks up the fixed embeddings for the input indices.

        Args:
            x (torch.Tensor): A tensor of integer indices.

        Returns:
            torch.Tensor: The corresponding embeddings, detached from the computation graph.
        """
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    """
    Creates a combined embedding from multiple categorical time features.
    (e.g., month, day, weekday, hour, minute).
    """
    def __init__(self, d_model: int, embed_type: str = 'fixed', freq: str = 'h'):
        """
        Args:
            d_model (int): The dimension of the model's hidden states.
            embed_type (str, optional): 'fixed' for non-trainable sinusoidal embeddings,
                                        'standard' for standard trainable nn.Embedding. Defaults to 'fixed'.
            freq (str, optional): The frequency of the data, used to determine which time
                                  features to include (e.g., 't' includes minute). Defaults to 'h'.
        """
        super(TemporalEmbedding, self).__init__()

        # Define the vocabulary size for each time feature
        minute_size = 4  # e.g., for 15-minute intervals
        hour_size = 24
        weekday_size = 7
        day_size = 32    # Using 32 to be safe for 1-based day indexing (1-31)
        month_size = 13  # Using 13 to be safe for 1-based month indexing (1-12)

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        
        # Create embedding layers based on frequency
        if freq == 't': # 't' for minute-level frequency
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Generates temporal embeddings from input time feature markers.

        Args:
            x (torch.Tensor): A tensor of integer time features, shape (B, L, num_time_features).
                              Expected order: [month, day, weekday, hour, minute].

        Returns:
            torch.Tensor: The summed temporal embeddings, shape (B, L, D).
        """
        x = x.long()

        # Look up embeddings for each time feature and sum them
        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    """
    Creates an embedding from numerical (non-categorical) time features.
    This is typically used for features generated by `utils.timefeatures.time_features`.
    """
    def __init__(self, d_model: int, embed_type: str = 'timeF', freq: str = 'h'):
        """
        Args:
            d_model (int): The dimension of the model's hidden states.
            embed_type (str, optional): Type of embedding. Defaults to 'timeF'.
            freq (str, optional): The frequency of the data, used to determine input dimension. Defaults to 'h'.
        """
        super(TimeFeatureEmbedding, self).__init__()

        # Map frequency to the number of numerical time features
        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        # Use a simple linear layer for projection
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the linear projection to the time features.

        Args:
            x (torch.Tensor): Numerical time features, shape (B, L, num_time_features).

        Returns:
            torch.Tensor: The embedded time features, shape (B, L, D).
        """
        return self.embed(x)


class DataEmbedding(nn.Module):
    """
    A composite embedding layer for time series data.

    This class combines value embedding, positional embedding, and temporal embedding
    to create the final input representation for a Transformer-based model.
    """
    def __init__(self, c_in: int, d_model: int, embed_type: str = 'fixed', freq: str = 'h', dropout: float = 0.1):
        """
        Args:
            c_in (int): Number of input channels/features.
            d_model (int): Dimension of the model's hidden states.
            embed_type (str, optional): Type for temporal embedding ('fixed', 'timeF'). Defaults to 'fixed'.
            freq (str, optional): Frequency of the data. Defaults to 'h'.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) \
            if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Combines and returns the final data embeddings.

        Args:
            x (torch.Tensor): The raw input time series values, shape (B, L, C).
            x_mark (Optional[torch.Tensor]): The time feature markers, shape (B, L, num_time_features).

        Returns:
            torch.Tensor: The combined embeddings, shape (B, L, D).
        """
        # If time features are available, combine all three embeddings.
        # Otherwise, combine only value and positional embeddings.
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)
        
        return self.dropout(x)


class DataEmbedding_inverted(nn.Module):
    """
    An alternative data embedding for "inverted" models where the time dimension
    is treated as the channel/feature dimension.
    """
    def __init__(self, c_in: int, d_model: int, embed_type: str = 'fixed', freq: str = 'h', dropout: float = 0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Permutes the input, applies a linear projection, and applies dropout.

        Args:
            x (torch.Tensor): Raw input time series, shape (B, L, C).
            x_mark (Optional[torch.Tensor]): Time features, which can be concatenated.

        Returns:
            torch.Tensor: The embedded representation, shape (B, C, D).
        """
        # Permute from (B, L, C) to (B, C, L) to treat time as features
        x = x.permute(0, 2, 1)
        
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # Concatenate time features along the feature dimension
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        
        return self.dropout(x)


class DataEmbedding_wo_pos(nn.Module):
    """
    A variant of `DataEmbedding` that omits the positional embedding.

    This is useful for models that are inherently position-agnostic or have
    other mechanisms for handling temporal order.
    """
    def __init__(self, c_in: int, d_model: int, embed_type: str = 'fixed', freq: str = 'h', dropout: float = 0.1):
        super(DataEmbedding_wo_pos, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq) \
            if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, x_mark: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Combines value and temporal embeddings (without positional).

        Args:
            x (torch.Tensor): The raw input time series values, shape (B, L, C).
            x_mark (Optional[torch.Tensor]): The time feature markers, shape (B, L, num_time_features).

        Returns:
            torch.Tensor: The combined embeddings, shape (B, L, D).
        """
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    """
    Patch-based embedding for time series.

    This layer segments a time series into patches, projects each patch into a
    `d_model`-dimensional vector space, and adds a positional encoding.
    This is a key component for models like PatchTST.
    """
    def __init__(self, d_model: int, patch_len: int, stride: int, padding: int, dropout: float):
        """
        Args:
            d_model (int): The dimension of the model's hidden states.
            patch_len (int): The length of each patch.
            stride (int): The stride between consecutive patches.
            padding (int): Padding to apply to the input sequence.
            dropout (float): Dropout rate.
        """
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, padding))

        # Linear projection of each patch to d_model dimension
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # Positional embedding for patches
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, int]:
        """
        Applies patch embedding to the input time series.

        Args:
            x (torch.Tensor): Input time series, shape (B, N, L) where N is number of variables.

        Returns:
            tuple[torch.Tensor, int]: A tuple containing the embedded patches and the number of variables.
                                     - Embedded patches shape: (B * N, num_patches, D)
                                     - Number of variables (int)
        """
        n_vars = x.shape[1] # Number of variables
        
        # Apply padding and create patches using unfold
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride) # (B, N, num_patches, patch_len)
        
        # Reshape for embedding: treat each variable as a separate sample for batch processing
        # (B, N, num_patches, patch_len) -> (B * N, num_patches, patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        
        # Apply value and positional embeddings
        x = self.value_embedding(x) + self.position_embedding(x)
        
        return self.dropout(x), n_vars