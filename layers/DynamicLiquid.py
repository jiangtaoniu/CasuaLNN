"""
This file defines the core components of a dynamic ODE-based recurrent neural network,
likely inspired by Liquid Time-Constant Networks (LTCs) or Closed-form Continuous-time
(CfC) models. The key feature is that the internal ODE parameters of the recurrent
cell can be dynamically modulated by an external signal at each time step. This is a
central component of the CasuaLNN architecture, where a Transformer-based controller
generates these dynamic parameters.

The main components are:
1. DynamicCfCCell: A CfC cell whose internal ODE parameters (t_a, t_b) can be
   modulated by an external `dynamic_params` tensor.
2. DynamicCfC: An RNN-style wrapper that passes a sequence of external parameters
   to the DynamicCfCCell in a loop.
3. DynamicLiquidEncoder & DynamicLiquidLayer: Transformer-style encoder blocks that
   wrap the DynamicCfC into a standard layer with residual connections and normalization.
"""

import torch
from torch import nn
from typing import Optional, Union, Tuple
import ncps
import numpy as np
import torch.nn.functional as F


# --- Dependency: LeCun Tanh Activation ---
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


# --- Dependency: LSTMCell (for mixed_memory mode in DynamicCfC) ---
class LSTMCell(nn.Module):
    """
    A standard LSTM cell implementation, used as an optional component for
    the `mixed_memory` mode in the DynamicCfC wrapper.
    """
    def __init__(self, input_size: int, hidden_size: int):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.input_map = nn.Linear(input_size, 4 * hidden_size, bias=True)
        self.recurrent_map = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        self.init_weights()

    def init_weights(self):
        for w in self.input_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.xavier_uniform_(w)
        for w in self.recurrent_map.parameters():
            if w.dim() == 1:
                torch.nn.init.uniform_(w, -0.1, 0.1)
            else:
                torch.nn.init.orthogonal_(w)

    def forward(self, inputs: torch.Tensor, states: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_state, cell_state = states
        z = self.input_map(inputs) + self.recurrent_map(output_state)
        i, ig, fg, og = z.chunk(4, 1)

        input_activation = self.tanh(i)
        input_gate = self.sigmoid(ig)
        forget_gate = self.sigmoid(fg + 1.0)
        output_gate = self.sigmoid(og)

        new_cell = cell_state * forget_gate + input_activation * input_gate
        output_state = self.tanh(new_cell) * output_gate

        return output_state, new_cell


# --- Core Component 1: DynamicCfCCell ---
class DynamicCfCCell(nn.Module):
    """
    A "Closed-form Continuous-time" (CfC) cell whose ODE parameters can be
    dynamically modulated by an external signal at each time step. This allows
    the cell's temporal dynamics to adapt based on context provided by another model.
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
        super(DynamicCfCCell, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        
        # Backbone network for feature extraction
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
        
        # Linear layers for CfC logic
        cat_shape = backbone_units if backbone_layers > 0 else input_size + hidden_size
        self.ff1 = nn.Linear(cat_shape, hidden_size)
        self.ff2 = nn.Linear(cat_shape, hidden_size)
        # These layers generate the "static" part of the ODE parameters
        self.time_a = nn.Linear(cat_shape, hidden_size)
        self.time_b = nn.Linear(cat_shape, hidden_size)
        
        self.init_weights()

    def init_weights(self):
        for w in self.parameters():
            if w.dim() == 2 and w.requires_grad:
                torch.nn.init.xavier_uniform_(w)

    def forward(self, input: torch.Tensor, hx: torch.Tensor, ts: torch.Tensor,
                dynamic_params: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for a single time step.

        Args:
            input (torch.Tensor): Input tensor for the current time step.
            hx (torch.Tensor): Hidden state from the previous time step.
            ts (torch.Tensor): Timespan for the current step.
            dynamic_params (Optional[torch.Tensor]): External modulation signal,
                                                     shape (B, 2 * hidden_size), containing
                                                     delta_t_a and delta_t_b.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The new hidden state and the output (which are the same here).
        """
        x = torch.cat([input, hx], 1)
        if self.backbone:
            x = self.backbone(x)
        
        # Standard CfC logic with tanh activations
        ff1 = self.tanh(self.ff1(x))
        ff2 = self.tanh(self.ff2(x))

        # --- [Dynamic ODE Modification] ---
        # 1. Calculate "static" ODE parameters from the input and hidden state.
        t_a_static = self.time_a(x)
        t_b_static = self.time_b(x)

        # 2. Apply "dynamic" modulation from the external controller if provided.
        if dynamic_params is not None:
            # `dynamic_params` is expected to contain concatenated [delta_t_a, delta_t_b]
            delta_t_a, delta_t_b = dynamic_params.chunk(2, 1)
            # Additive modulation
            t_a_final = t_a_static + delta_t_a
            t_b_final = t_b_static + delta_t_b
        else:
            t_a_final = t_a_static
            t_b_final = t_b_static

        # 3. Use the final parameters to calculate the interpolation factor for the ODE solution.
        t_interp = self.sigmoid(t_a_final * ts + t_b_final)

        # 4. Calculate the new hidden state using the interpolated result.
        new_hidden = ff1 * (1.0 - t_interp) + t_interp * ff2

        return new_hidden, new_hidden


# --- Core Component 2: DynamicCfC ---
class DynamicCfC(nn.Module):
    """
    An RNN-style wrapper for the `DynamicCfCCell`.

    This module iterates over an input sequence, passing the corresponding slice
    of `dynamic_params` to the `DynamicCfCCell` at each time step. It supports
    both batched and unbatched inputs.
    """
    def __init__(
            self,
            input_size: int,
            units: int,
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
        super(DynamicCfC, self).__init__()
        self.input_size = input_size
        self.hidden_size = units
        self.batch_first = batch_first
        self.return_sequences = return_sequences

        # This dynamic version does not support the "Wired" mode.
        if isinstance(units, ncps.wirings.Wiring):
            raise NotImplementedError("DynamicCfC does not support 'Wired' mode.")
            
        # Instantiate the dynamic cell
        self.rnn_cell = DynamicCfCCell(
            input_size,
            self.hidden_size,
            mode,
            activation,
            backbone_units or 128,
            backbone_layers or 1,
            backbone_dropout or 0.0,
        )

        self.use_mixed = mixed_memory
        if self.use_mixed:
            self.lstm = LSTMCell(input_size, self.hidden_size)

        # Output projection layer
        self.fc = nn.Linear(self.hidden_size, proj_size) if proj_size is not None else nn.Identity()

    def forward(self, input: torch.Tensor, hx: Optional[torch.Tensor] = None,
                timespans: Optional[torch.Tensor] = None,
                dynamic_params: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the DynamicCfC RNN.

        Args:
            input (torch.Tensor): Input sequence, shape (B, L, C) or (L, B, C).
            hx (Optional[torch.Tensor]): Initial hidden state.
            timespans (Optional[torch.Tensor]): Timespans for each step.
            dynamic_params (Optional[torch.Tensor]): Sequence of dynamic parameters from the controller,
                                                     shape (B, L, K) or (L, B, K) where K = 2 * hidden_size.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: The output sequence and the final hidden state.
        """
        device = input.device
        is_batched = input.dim() == 3
        batch_dim, seq_dim = (0, 1) if self.batch_first else (1, 0)
        if not is_batched: # Add batch dimension if input is unbatched
            input = input.unsqueeze(batch_dim)
            if timespans is not None: timespans = timespans.unsqueeze(batch_dim)
            if dynamic_params is not None: dynamic_params = dynamic_params.unsqueeze(batch_dim)

        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        # Initialize hidden state if not provided
        if hx is None:
            h_state = torch.zeros((batch_size, self.hidden_size), device=device)
            c_state = torch.zeros((batch_size, self.hidden_size), device=device) if self.use_mixed else None
        else:
            h_state, c_state = hx if self.use_mixed else (hx, None)

        output_sequence = []
        for t in range(seq_len):
            # Slice input for the current time step
            inputs_t = input[:, t] if self.batch_first else input[t]
            ts_t = torch.ones(batch_size, device=device) if timespans is None else timespans[:, t].squeeze()
            params_t = None if dynamic_params is None else (dynamic_params[:, t] if self.batch_first else dynamic_params[t])

            if self.use_mixed:
                h_state, c_state = self.lstm(inputs_t, (h_state, c_state))

            # Pass the time-step specific dynamic parameters to the cell
            h_out, h_state = self.rnn_cell.forward(inputs_t, h_state, ts_t, dynamic_params=params_t)

            if self.return_sequences:
                output_sequence.append(self.fc(h_out))
        
        # Stack outputs if returning sequences
        stack_dim = 1 if self.batch_first else 0
        readout = torch.stack(output_sequence, dim=stack_dim) if self.return_sequences else self.fc(h_out)
        
        final_hx = (h_state, c_state) if self.use_mixed else h_state

        if not is_batched: # Squeeze batch dimension if input was unbatched
            readout = readout.squeeze(batch_dim)
            final_hx = (h_state[0], c_state[0]) if self.use_mixed else h_state[0]

        return readout, final_hx


# --- Core Component 3: DynamicLiquidEncoder and DynamicLiquidLayer ---
class DynamicLiquidLayer(nn.Module):
    """
    A Transformer-style encoder layer that wraps a `DynamicCfC` module.

    This layer includes Layer Normalization (pre-norm) and a residual connection,
    making it a compatible building block for Transformer-like encoders.
    Input/output shape: (Batch, Sequence, Features)
    """
    def __init__(self, configs, dropout: float = 0.1):
        super(DynamicLiquidLayer, self).__init__()
        self.d_model = configs.d_model
        self.norm = nn.LayerNorm(configs.d_model)

        # Instantiate the DynamicCfC module as the core of this layer
        self.cfc = DynamicCfC(
            input_size=configs.d_model,
            units=configs.lq_units,
            proj_size=configs.lq_proj_size,
            batch_first=True,
            return_sequences=True,
            backbone_units=configs.lq_backbone_units,
            backbone_layers=configs.lq_backbone_layers,
            activation=configs.lq_activation
        )
        self.dropout = nn.Dropout(dropout)
        
        # Optional projection if CFC output size differs from d_model
        self.proj = nn.Linear(configs.lq_proj_size, configs.d_model) if configs.lq_proj_size != configs.d_model else nn.Identity()

    def forward(self, x: torch.Tensor, dynamic_params: torch.Tensor,
                timespans: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass for the DynamicLiquidLayer.

        Args:
            x (torch.Tensor): Input tensor, shape (B, L, D).
            dynamic_params (torch.Tensor): Sequence of dynamic parameters, shape (B, L, K).
            timespans (Optional[torch.Tensor]): Optional timespans for each step.

        Returns:
            torch.Tensor: Output tensor after applying the dynamic CfC and residual connection.
        """
        x_norm = self.norm(x)
        # Pass dynamic parameters to the CfC module
        cfc_out, _ = self.cfc(x_norm, timespans=timespans, dynamic_params=dynamic_params)
        cfc_out = self.proj(cfc_out)
        # Apply residual connection
        x = x + self.dropout(cfc_out)
        return x


class DynamicLiquidEncoder(nn.Module):
    """
    An encoder composed of a stack of `DynamicLiquidLayer` instances.
    """
    def __init__(self, layers: List[nn.Module], norm_layer: Optional[nn.Module] = None):
        super(DynamicLiquidEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x: torch.Tensor, dynamic_params: torch.Tensor,
                timespans: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, None]:
        """
        Forward pass for the DynamicLiquidEncoder.

        Args:
            x (torch.Tensor): Input tensor, shape (B, L, D).
            dynamic_params (torch.Tensor): Sequence of dynamic parameters, shape (B, L, K).
            timespans (Optional[torch.Tensor]): Optional timespans for each step.

        Returns:
            Tuple[torch.Tensor, None]: The encoded output tensor and None (to match the
                                       (output, attn) format of other encoder layers).
        """
        # Sequentially apply each DynamicLiquidLayer, passing the dynamic parameters through
        for layer in self.layers:
            x = layer(x, dynamic_params=dynamic_params, timespans=timespans)

        if self.norm is not None:
            x = self.norm(x)

        return x, None
