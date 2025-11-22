# This source code is adapted from the N-BEATS model implementation.
# The original license and copyright information are reproduced below for attribution.
#
# Original Source: Oreshkin et al., N-BEATS: Neural basis expansion analysis for
#                  interpretable time series forecasting, https://arxiv.org/abs/1905.10437
#
# Copyright 2020 Element AI Inc. All Rights Reserved.
# Licensed under the Creative Commons - Attribution-NonCommercial 4.0
# International license (CC BY-NC 4.0):
# https://creativecommons.org/licenses/by-nc/4.0/.

"""
Custom loss functions for PyTorch, primarily designed for evaluating
time series forecasting models, especially for M4 competition metrics.
"""

import torch
import torch.nn as nn
import numpy as np


def divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Performs element-wise division a/b, safely handling division by zero.

    Where the result of division would be NaN or Inf (due to zero in the denominator `b`),
    the function replaces these with 0.0.

    Args:
        a (torch.Tensor): The numerator tensor.
        b (torch.Tensor): The denominator tensor.

    Returns:
        torch.Tensor: The result of the division with NaN/Inf values replaced by 0.
    """
    result = a / b
    result[result != result] = .0  # Replace NaN with 0
    result[result == np.inf] = .0  # Replace Inf with 0
    result[result == -np.inf] = .0 # Replace -Inf with 0
    return result


class mape_loss(nn.Module):
    """
    Mean Absolute Percentage Error (MAPE) loss function.

    MAPE measures the size of the error in percentage terms. It is calculated as
    the average of the unsigned percentage error.
    """

    def __init__(self):
        super(mape_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        Calculates MAPE.

        Formula: mean(abs((forecast - target) / target)) * 100

        Args:
            insample (torch.Tensor): Not used by MAPE. Present for API consistency.
            freq (int): Not used by MAPE. Present for API consistency.
            forecast (torch.Tensor): Forecasted values. Shape: (batch_size, time_steps).
            target (torch.Tensor): Ground truth values. Shape: (batch_size, time_steps).
            mask (torch.Tensor): A 0/1 mask to indicate valid data points. Shape: (batch_size, time_steps).

        Returns:
            torch.float: The MAPE loss value.
        """
        # Calculate weights to handle potential zeros in the target
        weights = divide_no_nan(mask, target)
        return torch.mean(torch.abs((forecast - target) * weights))


class smape_loss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE) loss function.

    sMAPE is an alternative to MAPE that is symmetric in its treatment of
    over-forecasting and under-forecasting.
    """

    def __init__(self):
        super(smape_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        Calculates sMAPE as defined by Makridakis (1993).

        Formula: 200 * mean(abs(forecast - target) / (abs(forecast) + abs(target)))

        Args:
            insample (torch.Tensor): Not used by sMAPE. Present for API consistency.
            freq (int): Not used by sMAPE. Present for API consistency.
            forecast (torch.Tensor): Forecasted values. Shape: (batch_size, time_steps).
            target (torch.Tensor): Ground truth values. Shape: (batch_size, time_steps).
            mask (torch.Tensor): A 0/1 mask to indicate valid data points. Shape: (batch_size, time_steps).

        Returns:
            torch.float: The sMAPE loss value.
        """
        # The factor of 200 is used to express the result as a percentage.
        return 200 * torch.mean(divide_no_nan(torch.abs(forecast - target),
                                          (torch.abs(forecast.data) + torch.abs(target.data))) * mask)


class mase_loss(nn.Module):
    """
    Mean Absolute Scaled Error (MASE) loss function.

    MASE is a scale-free error metric that compares the forecast's mean absolute error
    to the mean absolute error of a naive seasonal forecast on the in-sample data.
    Based on: "Scaled Errors" https://robjhyndman.com/papers/mase.pdf
    """

    def __init__(self):
        super(mase_loss, self).__init__()

    def forward(self, insample: torch.Tensor, freq: int,
                forecast: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.float:
        """
        Calculates MASE.

        Formula: mean(abs(forecast - target)) / mean(abs(insample[t] - insample[t-freq]))

        Args:
            insample (torch.Tensor): In-sample (training) values used for scaling. Shape: (batch_size, time_steps_in).
            freq (int): The seasonal frequency of the time series (e.g., 12 for monthly data).
            forecast (torch.Tensor): Forecasted values. Shape: (batch_size, time_steps_out).
            target (torch.Tensor): Ground truth values. Shape: (batch_size, time_steps_out).
            mask (torch.Tensor): A 0/1 mask to indicate valid data points in the forecast. Shape: (batch_size, time_steps_out).

        Returns:
            torch.float: The MASE loss value.
        """
        # Calculate the mean absolute error of the naive seasonal forecast on the in-sample data
        # `masep` represents the scaling factor.
        masep = torch.mean(torch.abs(insample[:, freq:] - insample[:, :-freq]), dim=1)
        
        # Safely compute the inverse of the scaling factor, applying the mask
        masked_masep_inv = divide_no_nan(mask, masep[:, None]) # Add a dimension for broadcasting
        
        # Calculate the final MASE loss
        return torch.mean(torch.abs(target - forecast) * masked_masep_inv)