import torch
import torch.nn as nn
from typing import Optional


class Normalize(nn.Module):
    """
    Reversible Instance Normalization (RevIN).

    A normalization layer designed for time series forecasting, as introduced in
    "Reversible Instance Normalization for Deep Time Series Forecasting" (ICLR 2022).
    It normalizes each time series instance independently and denormalizes the
    output, which helps mitigate distribution shift issues.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, affine: bool = False,
                 subtract_last: bool = False, non_norm: bool = False):
        """
        Initializes the Normalize layer.

        Args:
            num_features (int): The number of features or channels in the time series.
            eps (float, optional): A small value added for numerical stability during division. Defaults to 1e-5.
            affine (bool, optional): If True, the layer has learnable affine parameters (weight and bias)
                                     applied after normalization. Defaults to False.
            subtract_last (bool, optional): If True, normalization is done by subtracting the last time step's
                                            value instead of the mean. Defaults to False.
            non_norm (bool, optional): If True, disables normalization and denormalization entirely,
                                       making the layer an identity operation. Defaults to False.
        """
        super(Normalize, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        self.non_norm = non_norm
        
        # Initialize learnable affine parameters if specified
        if self.affine:
            self._init_params()

    def forward(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """
        Applies either normalization or denormalization based on the mode.

        Args:
            x (torch.Tensor): The input tensor.
            mode (str): The operation mode, either 'norm' for normalization
                        or 'denorm' for denormalization.

        Returns:
            torch.Tensor: The processed tensor.
        """
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError(f"Mode '{mode}' is not implemented. Use 'norm' or 'denorm'.")
        return x

    def _init_params(self):
        """Initializes the learnable affine weight and bias parameters."""
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x: torch.Tensor):
        """
        Calculates and stores the statistics (mean and stdev, or last value)
        required for normalization. These statistics are detached from the computation graph.
        """
        # Dimensions to reduce over (typically the time dimension)
        dim2reduce = tuple(range(1, x.ndim - 1))
        
        if self.subtract_last:
            # Alternative normalization: use the last time step as the reference point
            self.last = x[:, -1, :].unsqueeze(1).detach()
        else:
            # Standard instance norm: use mean and standard deviation
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the normalization to the input tensor.
        """
        if self.non_norm:
            return x # Skip normalization
            
        # Subtract mean or last value
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        
        # Divide by standard deviation
        x = x / self.stdev
        
        # Apply learnable affine transformation if enabled
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
            
        return x

    def _denormalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Applies the inverse normalization (denormalization) to the input tensor.
        This operation is the exact reverse of the normalization process.
        """
        if self.non_norm:
            return x # Skip denormalization
        
        # Reverse the affine transformation if enabled
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps) # Add epsilon for stability
        
        # Multiply by standard deviation
        x = x * self.stdev
        
        # Add back mean or last value
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
            
        return x