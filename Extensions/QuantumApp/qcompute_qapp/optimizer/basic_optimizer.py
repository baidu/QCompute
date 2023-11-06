# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2021 Baidu, Inc. All Rights Reserved.
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
from ..circuit import ParameterizedCircuit


class BasicOptimizer(ABC):
    r"""Basic Optimizer class"""

    def __init__(self, iterations: int, circuit: ParameterizedCircuit):
        r"""The constructor of the BasicOptimizer class

        Args:
            iterations (int): Number of iterations
            circuit (ParameterizedCircuit): Circuit whose parameters are to be optimized

        """
        self._circuit = circuit
        self._iterations = iterations
        self._loss_history = []

    def set_circuit(self, circuit: ParameterizedCircuit) -> None:
        r"""Sets the parameterized circuit to be optimized

        Args:
            circuit (ParameterizedCircuit): Parameterized Circuit to be optimized

        """
        self._circuit = circuit

    @abstractmethod
    def minimize(
        self,
        shots: int,
        loss_func: Callable[[np.ndarray, int], float],
        grad_func: Callable[[np.ndarray, int], np.ndarray],
    ) -> None:
        r"""Minimizes the given loss function

        Args:
            shots (int): Number of measurement shots
            loss_func (Callable[[np.ndarray, int], float]): Loss function to be minimized
            grad_func (Callable[[np.ndarray, int], np.ndarray]): Function for calculating gradients

        """
        raise NotImplementedError
