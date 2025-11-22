# Copyright 2022 Mathias Lechner and Ramin Hasani
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import Optional, Union, Tuple

"""
This file implements the `CfCCell`, a "Closed-form Continuous-time" recurrent
neural network cell. It is a continuous-time model that uses a closed-form
solution to its underlying neural ODE, making it computationally efficient.
"""

class LeCun(nn.Module):
    """
    LeCun's Tanh activation function, defined as 1.7159 * tanh(0.666 * x).
    This is often used in self-normalizing networks.
    """
    def __init__(self):
        super(LeCun, self).__init__()
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return 1.7159 * self.tanh(0.666 * x)


class CfCCell(nn.Module):
    """
    A "Closed-form Continuous-time" (CfC) cell.

    This class implements a single time-step of a CfC, which solves its underlying
    neural ODE in closed-form. This makes it more efficient than standard ODE solvers.
    It supports different operational modes ('default', 'pure', 'no_gate').
    """
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            mode: str = "default",
            backbone_activation: str = "lecun_tanh",
            backbone_units: int = 128,
            backbone_layers: int = 1,
            backbone_dropout: float = 0.0,
            sparsity_mask: Optional[np.ndarray] = None,
    ):
        """
        Initializes the CfCCell.

        Args:
            input_size (int): The number of input features.
            hidden_size (int): The number of hidden units.
            mode (str): The operational mode ('default', 'pure', 'no_gate').
            backbone_activation (str): The activation function for the backbone network.
            backbone_units (int): The number of units in the backbone's hidden layers.
            backbone_layers (int): The number of layers in the backbone network.
            backbone_dropout (float): The dropout rate for the backbone network.
            sparsity_mask (Optional[np.ndarray]): A non-trainable mask to enforce sparsity in connections.
        """
        super(CfCCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        
        # Define backbone network for feature extraction
        self.backbone = None
        if backbone_layers > 0:
            layer_list = [nn.Linear(input_size + hidden_size, backbone_units)]
            if backbone_activation == "silu": layer_list.append(nn.SiLU())
            elif backbone_activation == "relu": layer_list.append(nn.ReLU())
            elif backbone_activation == "tanh": layer_list.append(nn.Tanh())
            elif backbone_activation == "gelu": layer_list.append(nn.GELU())
            elif backbone_activation == "lecun_tanh": layer_list.append(LeCun())
            else: raise ValueError(f"Unknown activation {backbone_activation}")
            
            for _ in range(1, backbone_layers):
                layer_list.append(nn.Linear(backbone_units, backbone_units))
                layer_list.append(layer_list[1]) # Share activation instance
                if backbone_dropout > 0.0: layer_list.append(nn.Dropout(backbone_dropout))
            self.backbone = nn.Sequential(*layer_list)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        
        # Define linear layers for the CfC logic
        cat_shape = backbone_units if backbone_layers > 0 else input_size + hidden_size
        self.ff1 = nn.Linear(cat_shape, hidden_size)
        
        if self.mode == "pure":
            # Parameters for the "pure" mode, which has a different ODE formulation
            self.w_tau = nn.Parameter(data=torch.zeros(1, self.hidden_size), requires_grad=True)
            self.A = nn.Parameter(data=torch.ones(1, self.hidden_size), requires_grad=True)
        else: # "default" or "no_gate" modes
            self.ff2 = nn.Linear(cat_shape, hidden_size)
            self.time_a = nn.Linear(cat_shape, hidden_size)
            self.time_b = nn.Linear(cat_shape, hidden_size)
            
        self.init_weights()

    def init_weights(self):
        """Initializes the weights of the linear layers."""
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)

    def forward(self, input: torch.Tensor, hx: torch.Tensor, ts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single time step of the CfC cell.

        Args:
            input (torch.Tensor): Input tensor for the current time step, shape (B, input_size).
            hx (torch.Tensor): Hidden state from the previous time step, shape (B, hidden_size).
            ts (torch.Tensor): Timespan for the current step, shape (B,) or (B, 1).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the new hidden state and the output (which are the same).
        """
        # Concatenate input and previous hidden state
        x = torch.cat([input, hx], 1)
        if self.backbone:
            x = self.backbone(x)
        
        ff1 = self.ff1(x)

        # Ensure ts is in the correct shape (B, 1) for broadcasting
        if isinstance(ts, torch.Tensor) and ts.dim() == 1:
            ts = ts.unsqueeze(-1)

        if self.mode == "pure":
            # "Pure" mode uses an exponential decay formulation
            new_hidden = (-self.A * torch.exp(-ts * (torch.abs(self.w_tau) + torch.abs(ff1))) * ff1 + self.A)
        else: # "default" or "no_gate" modes
            ff2 = self.ff2(x)
            ff1 = self.tanh(ff1)
            ff2 = self.tanh(ff2)

            # Calculate time-dependent interpolation factor
            t_a = self.time_a(x)
            t_b = self.time_b(x)
            t_interp = self.sigmoid(t_a * ts + t_b)
            
            if self.mode == "no_gate":
                # Interpolation without an explicit gate
                new_hidden = ff1 + t_interp * ff2
            else: # "default" mode
                # Gated interpolation between the two feature transforms
                new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2
                
        return new_hidden, new_hidden
