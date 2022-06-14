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
Basic Optimizer
"""

from abc import ABC, abstractmethod
from typing import Callable
import numpy as np
from qapp.circuit import ParameterizedCircuit


class BasicOptimizer(ABC):
    """Basic Optimizer class
    """
    def __init__(self, iterations: int, circuit: ParameterizedCircuit):
        """The constructor of the BasicOptimizer class

        :param iterations: Number of iterations
        :param circuit: Circuit whose parameters are to be optimized
        """
        self._circuit = circuit
        self._iterations = iterations
        self._loss_history = []

    def set_circuit(self, circuit: ParameterizedCircuit):
        """Sets the parameterized circuit to be optimized

        :param circuit: Parameterized Circuit to be optimized
        """
        self._circuit = circuit

    @abstractmethod
    def minimize(self, shots: int,
                 loss_func: Callable[[np.ndarray, int], float],
                 grad_func: Callable[[np.ndarray, int], np.ndarray]):
        """Minimizes the given loss function

        :param shots: Number of measurement shots
        :param loss_func: Loss function to be minimized
        :param grad_func: Function for calculating gradients
        """
        raise NotImplementedError
