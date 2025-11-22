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

import numpy as np
import torch
from torch import nn
from typing import Optional, Union, Tuple
import ncps
from .ltc_cell import LTCCell
from .lstm import LSTMCell

"""
This file implements the `LTC` class, a full recurrent neural network that
uses the `LTCCell` to process sequences of data, handling the iteration over time.
"""

class LTC(nn.Module):
    """
    Applies a `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_
    RNN to an input sequence.

    This class is an RNN-style wrapper for the `LTCCell`, handling the processing of
    entire sequences by iterating over time steps.

    .. Note::
        For creating a wired `Neural circuit policy (NCP) <https://publik.tuwien.ac.at/files/publik_292280.pdf>`_,
        you can pass an `ncps.wirings.NCP` object as the `units` parameter.

    **Examples**

    A fully-connected LTC network:
        >>> from ncps.torch import LTC
        >>>
        >>> rnn = LTC(20, 50)
        >>> x = torch.randn(2, 3, 20)  # (batch, time, features)
        >>> h0 = torch.zeros(2, 50)   # (batch, units)
        >>> output, hn = rnn(x, h0)

    A structured LTC network (NCP):
        >>> from ncps.torch import LTC
        >>> from ncps.wirings import NCP
        >>>
        >>> wiring = NCP(inter_neurons=10, command_neurons=8, motor_neurons=4, sensory_fanout=6, inter_fanout=6, recurrent_fanout=4, motor_fanout=6)
        >>> rnn = LTC(20, wiring) # input_size=20
        >>>
        >>> x = torch.randn(2, 3, 20)   # (batch, time, features)
        >>> h0 = torch.zeros(2, 28)    # (batch, wiring.units)
        >>> output, hn = rnn(x, h0)
    """

    def __init__(
        self,
        input_size: int,
        units: Union[int, ncps.wirings.Wiring],
        return_sequences: bool = True,
        batch_first: bool = True,
        mixed_memory: bool = False,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = True,
    ):
        """
        Initializes the LTC module.

        Args:
            input_size (int): The number of input features.
            units (Union[int, ncps.wirings.Wiring]): The number of hidden units, or a pre-configured `Wiring` object for structured networks.
            return_sequences (bool): If True, returns the full sequence of outputs. If False, returns only the last output.
            batch_first (bool): If True, the input and output tensors are provided as (batch, seq, feature).
            mixed_memory (bool): If True, augments the LTC state with an LSTM-style memory cell to potentially capture longer-term dependencies.
            input_mapping (str): The mapping to apply to the input ('affine', 'linear', or 'identity').
            output_mapping (str): The mapping to apply to the output state ('affine', 'linear', or 'identity').
            ode_unfolds (int): The number of solver steps for the underlying ODE.
            epsilon (float): A small value for numerical stability.
            implicit_param_constraints (bool): If True, enforces parameter constraints implicitly via `Softplus`.
        """

        super(LTC, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        # Instantiate the wiring or a fully-connected default
        if isinstance(units, ncps.wirings.Wiring):
            wiring = units
        else:
            wiring = ncps.wirings.FullyConnected(units)
        
        # Instantiate the underlying LTCCell
        self.rnn_cell = LTCCell(
            wiring=wiring,
            in_features=input_size,
            input_mapping=input_mapping,
            output_mapping=output_mapping,
            ode_unfolds=ode_unfolds,
            epsilon=epsilon,
            implicit_param_constraints=implicit_param_constraints,
        )
        self._wiring = wiring
        
        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = LSTMCell(input_size, self.state_size)

    @property
    def state_size(self) -> int:
        """The number of neurons in the cell."""
        return self._wiring.units

    @property
    def motor_size(self) -> int:
        """The number of motor (output) neurons."""
        return self._wiring.output_dim

    @property
    def output_size(self) -> int:
        """The size of the output, which is the number of motor neurons."""
        return self.motor_size

    def forward(self, input: torch.Tensor, hx: Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]] = None,
                timespans: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass for the LTC RNN.

        Args:
            input (torch.Tensor): Input sequence.
                                  Shape (L, C) for unbatched,
                                  (B, L, C) if `batch_first=True`,
                                  or (L, B, C) if `batch_first=False`.
            hx (Optional[Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]): Initial hidden state.
                                   If None, it is initialized to zeros.
                                   If `mixed_memory=True`, this should be a tuple (h0, c0).
            timespans (Optional[torch.Tensor]): The elapsed time between sequence steps.
                                                 If None, it is assumed to be 1.0 for all steps.

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
                raise RuntimeError("LTC with mixed_memory=True requires a tuple (h0, c0) as initial state.")
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
                output_sequence.append(h_out)

        # Stack outputs if returning sequences, otherwise return the last output
        if self.return_sequences:
            stack_dim = 1 if self.batch_first else 0
            readout = torch.stack(output_sequence, dim=stack_dim)
        else:
            readout = h_out
        
        # Package final hidden state
        final_hx = (h_state, c_state) if self.use_mixed else h_state

        # Squeeze batch dimension if input was unbatched
        if not is_batched:
            readout = readout.squeeze(batch_dim)
            final_hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]

        return readout, final_hx
