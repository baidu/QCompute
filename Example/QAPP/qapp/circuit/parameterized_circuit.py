# -*- coding: UTF-8 -*-
# Copyright (c) 2022 Baidu, Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Parameterized Circuit
"""

from abc import abstractmethod
import numpy as np
from QCompute.QPlatform.QRegPool import QRegPool
from .basic_circuit import BasicCircuit


class ParameterizedCircuit(BasicCircuit):
    """Parameterized Circuit class
    """
    def __init__(self, num: int, parameters: np.ndarray):
        """The constructor of the BasicCircuit class

        :param num: Number of qubits
        :param parameters: Parameters of parameterized gates
        """
        super().__init__(num)
        self._parameters = parameters.copy()
        self._param_shape = parameters.shape

    @property
    def parameters(self) -> np.ndarray:
        """Parameters of the circuit

        :return: Parameters of the circuit
        """
        return self._parameters.copy()

    def set_parameters(self, parameters: np.ndarray):
        """Sets parameters of the circuit

        :param parameters: New parameters of the circuit
        """
        assert parameters.shape == self._param_shape, "The shape of new parameters should be the same as before."
        self._parameters = parameters.copy()

    @abstractmethod
    def add_circuit(self, q: QRegPool):
        """Adds the circuit to the register

        :param q: Quantum register to which this circuit is added
        """
        raise NotImplementedError
