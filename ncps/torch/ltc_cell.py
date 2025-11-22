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
from typing import Optional, Union

"""
This file implements the `LTCCell`, a continuous-time recurrent neural network cell
based on the "Liquid Time-Constant Networks" paper (https://ojs.aaai.org/index.php/AAAI/article/view/16936).
It models a system of neural ODEs describing the dynamics of interconnected neurons.
"""

class LTCCell(nn.Module):
    """
    A `Liquid time-constant (LTC) <https://ojs.aaai.org/index.php/AAAI/article/view/16936>`_ cell.

    This class implements a single time-step of an LTC, which is modeled as a system of
    neural Ordinary Differential Equations (ODEs). It is an RNNCell that processes single time-steps.
    To get a full RNN that can process sequences, see `ncps.torch.LTC`.

    The neural ODE is solved by an implicit solver `_ode_solver` which is unfolded for
    a fixed number of steps.
    """
    def __init__(
        self,
        wiring,
        in_features: Optional[int] = None,
        input_mapping: str = "affine",
        output_mapping: str = "affine",
        ode_unfolds: int = 6,
        epsilon: float = 1e-8,
        implicit_param_constraints: bool = False,
    ):
        """
        Initializes the LTCCell.

        Args:
            wiring: An `ncps.wirings.Wiring` object defining the network structure.
            in_features (Optional[int]): The number of input features. If not provided,
                                         the wiring must already be built.
            input_mapping (str): How to map the input. 'affine' applies a learnable weight and bias,
                                 'linear' applies only a weight, and 'identity' applies no mapping.
            output_mapping (str): How to map the output state. Same options as `input_mapping`.
            ode_unfolds (int): The number of solver steps used to approximate the continuous-time dynamics.
            epsilon (float): A small value for numerical stability.
            implicit_param_constraints (bool): If True, positivity constraints on parameters are enforced
                                               implicitly via `Softplus`. If False, they are enforced by
                                               explicitly clipping the values.
        """
        super(LTCCell, self).__init__()
        if in_features is not None:
            wiring.build(in_features)
        if not wiring.is_built():
            raise ValueError(
                "Wiring error! Unknown number of input features. Please pass the parameter 'in_features' or call 'wiring.build()' first."
            )
        self.make_positive_fn = nn.Softplus() if implicit_param_constraints else nn.Identity()
        self._implicit_param_constraints = implicit_param_constraints
        
        # Define initialization ranges for the biophysical parameters
        self._init_ranges = {
            "gleak": (0.001, 1.0),
            "vleak": (-0.2, 0.2),
            "cm": (0.4, 0.6),
            "w": (0.001, 1.0),
            "sigma": (3.0, 8.0),
            "mu": (0.3, 0.8),
            "sensory_w": (0.001, 1.0),
            "sensory_sigma": (3.0, 8.0),
            "sensory_mu": (0.3, 0.8),
        }
        self._wiring = wiring
        self._input_mapping = input_mapping
        self._output_mapping = output_mapping
        self._ode_unfolds = ode_unfolds
        self._epsilon = epsilon
        self._clip = torch.nn.ReLU()
        
        self._allocate_parameters()

    @property
    def state_size(self) -> int:
        """The number of neurons in the cell."""
        return self._wiring.units

    @property
    def sensory_size(self) -> int:
        """The number of sensory (input) neurons."""
        return self._wiring.input_dim

    @property
    def motor_size(self) -> int:
        """The number of motor (output) neurons."""
        return self._wiring.output_dim

    @property
    def output_size(self) -> int:
        """The size of the output, which is the number of motor neurons."""
        return self.motor_size

    def add_weight(self, name: str, init_value: torch.Tensor, requires_grad: bool = True) -> nn.Parameter:
        """Helper function to create and register a tensor as a learnable parameter."""
        param = torch.nn.Parameter(init_value, requires_grad=requires_grad)
        self.register_parameter(name, param)
        return param

    def _get_init_value(self, shape: tuple, param_name: str) -> torch.Tensor:
        """Initializes a parameter with random values within a predefined range."""
        minval, maxval = self._init_ranges[param_name]
        if minval == maxval:
            return torch.ones(shape) * minval
        else:
            return torch.rand(*shape) * (maxval - minval) + minval

    def _allocate_parameters(self):
        """Allocates and initializes all learnable and non-learnable parameters of the cell."""
        self._params = {}
        # Membrane leak conductance
        self._params["gleak"] = self.add_weight("gleak", self._get_init_value((self.state_size,), "gleak"))
        # Membrane leak reversal potential
        self._params["vleak"] = self.add_weight("vleak", self._get_init_value((self.state_size,), "vleak"))
        # Membrane capacitance
        self._params["cm"] = self.add_weight("cm", self._get_init_value((self.state_size,), "cm"))
        # Synaptic weights (recurrent)
        self._params["w"] = self.add_weight("w", self._get_init_value((self.state_size, self.state_size), "w"))
        # Sigmoid activation parameters (recurrent)
        self._params["sigma"] = self.add_weight("sigma", self._get_init_value((self.state_size, self.state_size), "sigma"))
        self._params["mu"] = self.add_weight("mu", self._get_init_value((self.state_size, self.state_size), "mu"))
        # Synaptic reversal potential (recurrent)
        self._params["erev"] = self.add_weight("erev", torch.Tensor(self._wiring.erev_initializer()))
        # Synaptic weights (sensory)
        self._params["sensory_w"] = self.add_weight("sensory_w", self._get_init_value((self.sensory_size, self.state_size), "sensory_w"))
        # Sigmoid activation parameters (sensory)
        self._params["sensory_sigma"] = self.add_weight("sensory_sigma", self._get_init_value((self.sensory_size, self.state_size), "sensory_sigma"))
        self._params["sensory_mu"] = self.add_weight("sensory_mu", self._get_init_value((self.sensory_size, self.state_size), "sensory_mu"))
        # Synaptic reversal potential (sensory)
        self._params["sensory_erev"] = self.add_weight("sensory_erev", torch.Tensor(self._wiring.sensory_erev_initializer()))

        # Sparsity masks (non-trainable)
        self._params["sparsity_mask"] = self.add_weight("sparsity_mask", torch.Tensor(np.abs(self._wiring.adjacency_matrix)), requires_grad=False)
        self._params["sensory_sparsity_mask"] = self.add_weight("sensory_sparsity_mask", torch.Tensor(np.abs(self._wiring.sensory_adjacency_matrix)), requires_grad=False)

        # Input/output mapping parameters
        if self._input_mapping in ["affine", "linear"]:
            self._params["input_w"] = self.add_weight("input_w", torch.ones((self.sensory_size,)))
        if self._input_mapping == "affine":
            self._params["input_b"] = self.add_weight("input_b", torch.zeros((self.sensory_size,)))
        if self._output_mapping in ["affine", "linear"]:
            self._params["output_w"] = self.add_weight("output_w", torch.ones((self.motor_size,)))
        if self._output_mapping == "affine":
            self._params["output_b"] = self.add_weight("output_b", torch.zeros((self.motor_size,)))

    def _sigmoid(self, v_pre: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """The synaptic activation function (a sigmoid)."""
        v_pre = torch.unsqueeze(v_pre, -1)  # For broadcasting over target neurons
        mues = v_pre - mu
        x = sigma * mues
        return torch.sigmoid(x)

    def _ode_solver(self, inputs: torch.Tensor, state: torch.Tensor, elapsed_time: float) -> torch.Tensor:
        """
        Semi-implicit ODE solver to update the neuron's state (membrane potential).

        This method unfolds the ODE for `self._ode_unfolds` steps to approximate the
        continuous-time dynamics.
        """
        v_pre = state

        # Pre-compute the effect of sensory inputs on the cell state
        sensory_w_activation = self.make_positive_fn(self._params["sensory_w"]) * self._sigmoid(inputs, self._params["sensory_mu"], self._params["sensory_sigma"])
        sensory_w_activation = sensory_w_activation * self._params["sensory_sparsity_mask"]
        sensory_rev_activation = sensory_w_activation * self._params["sensory_erev"]

        # Reduce over the source sensory neurons dimension to get total sensory effect
        w_numerator_sensory = torch.sum(sensory_rev_activation, dim=1)
        w_denominator_sensory = torch.sum(sensory_w_activation, dim=1)

        # Pre-compute the time-dependent part of the membrane capacitance
        cm_t = self.make_positive_fn(self._params["cm"]) / (elapsed_time / self._ode_unfolds)

        # Unfold the ODE solver for a fixed number of steps
        w_param = self.make_positive_fn(self._params["w"])
        for _ in range(self._ode_unfolds):
            # Calculate recurrent synaptic activations
            w_activation = w_param * self._sigmoid(v_pre, self._params["mu"], self._params["sigma"])
            w_activation = w_activation * self._params["sparsity_mask"]
            rev_activation = w_activation * self._params["erev"]

            # Reduce over source neurons dimension to get total recurrent effect
            w_numerator = torch.sum(rev_activation, dim=1) + w_numerator_sensory
            w_denominator = torch.sum(w_activation, dim=1) + w_denominator_sensory

            # Define the numerator and denominator of the ODE update equation
            gleak = self.make_positive_fn(self._params["gleak"])
            # Numerator combines membrane capacitance dynamics, leak currents, and synaptic currents
            numerator = cm_t * v_pre + gleak * self._params["vleak"] + w_numerator
            # Denominator combines capacitance, leak conductance, and synaptic conductances
            denominator = cm_t + gleak + w_denominator

            # Update the state (membrane potential)
            v_pre = numerator / (denominator + self._epsilon)

        return v_pre

    def _map_inputs(self, inputs: torch.Tensor) -> torch.Tensor:
        """Applies the configured linear or affine mapping to the input tensor."""
        if self._input_mapping in ["affine", "linear"]:
            inputs = inputs * self._params["input_w"]
        if self._input_mapping == "affine":
            inputs = inputs + self._params["input_b"]
        return inputs

    def _map_outputs(self, state: torch.Tensor) -> torch.Tensor:
        """Applies the configured linear or affine mapping to the output state."""
        output = state
        # Slice the state if motor_size is smaller than state_size
        if self.motor_size < self.state_size:
            output = output[:, 0:self.motor_size]

        if self._output_mapping in ["affine", "linear"]:
            output = output * self._params["output_w"]
        if self._output_mapping == "affine":
            output = output + self._params["output_b"]
        return output

    def apply_weight_constraints(self):
        """
        Explicitly clips parameter values to enforce physical constraints.
        This is used when `implicit_param_constraints` is False.
        """
        if not self._implicit_param_constraints:
            self._params["w"].data.clamp_(0)
            self._params["sensory_w"].data.clamp_(0)
            self._params["cm"].data.clamp_(0)
            self._params["gleak"].data.clamp_(0)

    def forward(self, inputs: torch.Tensor, states: torch.Tensor, elapsed_time: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Performs a single forward pass through the LTCCell for one time-step.

        Args:
            inputs (torch.Tensor): The input for the current time-step.
            states (torch.Tensor): The hidden state from the previous time-step.
            elapsed_time (float, optional): The time elapsed since the last step. Defaults to 1.0.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing the output and the new hidden state.
        """
        inputs = self._map_inputs(inputs)
        next_state = self._ode_solver(inputs, states, elapsed_time)
        outputs = self._map_outputs(next_state)
        return outputs, next_state