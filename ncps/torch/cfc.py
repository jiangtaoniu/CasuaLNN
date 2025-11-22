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
from torch import nn
from typing import Optional, Union, Tuple
import ncps
from .cfc_cell import CfCCell
from .wired_cfc_cell import WiredCfCCell
from .lstm import LSTMCell

"""
This file implements the `CfC` class, a full recurrent neural network that
uses the `CfCCell` (or `WiredCfCCell`) to process sequences of data.
"""

class CfC(nn.Module):
    """
    Applies a `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ RNN
    to an input sequence.

    This class is an RNN-style wrapper for the `CfCCell`, handling the processing of
    entire sequences by iterating over time steps.

    **Examples**

    A fully-connected CfC network:
        >>> from ncps.torch import CfC
        >>>
        >>> rnn = CfC(20, 50)
        >>> x = torch.randn(2, 3, 20)  # (batch, time, features)
        >>> h0 = torch.zeros(2, 50)   # (batch, units)
        >>> output, hn = rnn(x, h0)

    A structured, "wired" CfC network:
        >>> from ncps.torch import CfC
        >>> from ncps.wirings import NCP
        >>>
        >>> wiring = NCP(inter_neurons=10, command_neurons=8, motor_neurons=4, ...)
        >>> rnn = CfC(20, wiring) # input_size=20
        >>>
        >>> x = torch.randn(2, 3, 20)   # (batch, time, features)
        >>> h0 = torch.zeros(2, wiring.units) # (batch, wiring.units)
        >>> output, hn = rnn(x, h0)
    """

    def __init__(
        self,
        input_size: int,
        units: Union[int, ncps.wirings.Wiring],
        proj_size: Optional[int] = None,
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        mode: str = "default",
        activation: str = "lecun_tanh",
        backbone_units: Optional[int] = None,
        backbone_layers: Optional[int] = None,
        backbone_dropout: Optional[float] = None,
    ):
        """
        Initializes the CfC module.

        Args:
            input_size (int): The number of input features.
            units (Union[int, ncps.wirings.Wiring]): The number of hidden units for a fully-connected network,
                                                    or a pre-configured `Wiring` object for a structured network.
            proj_size (Optional[int]): If not None, projects the output of the RNN to this dimension.
            return_sequences (bool): If True, returns the full sequence of outputs. If False, returns only the last output.
            batch_first (bool): If True, the input and output tensors are provided as (batch, seq, feature).
            mixed_memory (bool): If True, augments the CfC state with an LSTM-style memory cell.
            mode (str): The operational mode for the CfC cell ('default', 'pure', 'no_gate').
            activation (str): The activation function for the backbone network.
            backbone_units (Optional[int]): The number of units in the backbone's hidden layers.
            backbone_layers (Optional[int]): The number of layers in the backbone network.
            backbone_dropout (Optional[float]): The dropout rate for the backbone network.
        """
        super(CfC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.proj_size = proj_size
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        # Configure either a "wired" (structured) or a "fully-connected" (dense) cell
        if isinstance(units, ncps.wirings.Wiring):
            # A structured network defined by a Wiring object
            self.wired_mode = True
            if any(arg is not None for arg in [backbone_units, backbone_layers, backbone_dropout]):
                raise ValueError("Backbone arguments are not supported in wired mode.")
            self.wiring = units
            self.state_size = self.wiring.units
            self.output_size = self.wiring.output_dim
            self.rnn_cell = WiredCfCCell(input_size, self.wiring_or_units, mode)
        else:
            # A fully-connected network
            self.wired_mode = False
            self.state_size = units
            self.output_size = self.state_size
            self.rnn_cell = CfCCell(
                input_size,
                self.wiring_or_units,
                mode,
                activation,
                backbone_units or 128,
                backbone_layers or 1,
                backbone_dropout or 0.0,
            )
            
        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = LSTMCell(input_size, self.state_size)

        # Optional output projection layer
        self.fc = nn.Linear(self.output_size, self.proj_size) if proj_size is not None else nn.Identity()

    def forward(self, input: torch.Tensor, hx: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
                timespans: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the CfC RNN.

        Args:
            input (torch.Tensor): Input sequence. Shape (L, C) for unbatched, (B, L, C) if `batch_first=True`,
                                  or (L, B, C) if `batch_first=False`.
            hx (Optional): Initial hidden state. If None, initialized to zeros. If `mixed_memory=True`,
                           this should be a tuple (h0, c0).
            timespans (Optional[torch.Tensor]): The elapsed time between sequence steps. If None, assumed to be 1.0.

        Returns:
            A tuple (output, hx), where `output` is the output sequence and `hx` is the final hidden state.
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim, seq_dim = (0, 1) if self.batch_first else (1, 0)

        # Add batch dimension if input is unbatched
        if not is_batched:
            input = input.unsqueeze(batch_dim)
            if timespans is not None:
                timespans = timespans.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        # Initialize hidden state if not provided
        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = torch.zeros((batch_size, self.state_size), device=device) if self.use_mixed else None
        else:
            if self.use_mixed and isinstance(hx, torch.Tensor):
                raise RuntimeError("CfC with mixed_memory=True requires a tuple (h0, c0) as initial state.")
            h_state, c_state = hx if self.use_mixed else (hx, None)
            if not is_batched: # Unsqueeze hidden state if input was unbatched
                h_state = h_state.unsqueeze(0)
                if c_state is not None: c_state = c_state.unsqueeze(0)

        # Iterate over the sequence
        output_sequence = []
        for t in range(seq_len):
            # Slice input for the current time step
            inputs_t = input[:, t] if self.batch_first else input[t]
            ts_t = torch.ones(batch_size, device=device) if timespans is None else timespans[:, t].squeeze()

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs_t, (h_state, c_state))
            
            # Pass to the underlying RNN cell
            h_out, h_state = self.rnn_cell.forward(inputs_t, h_state, ts_t)
            
            if self.return_sequences:
                output_sequence.append(self.fc(h_out))

        # Stack outputs if returning sequences, otherwise return the last output
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = self.fc(h_out)
        
        # Package final hidden state
        final_hx = (h_state, c_state) if self.use_mixed else h_state

        # Squeeze batch dimension if input was unbatched
        if not is_batched:
            readout = readout.squeeze(batch_dim)
            final_hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]

        return readout, final_hx
