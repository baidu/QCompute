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
Stochastic Gradient Descent
"""

from typing import Callable
import numpy as np
from tqdm import tqdm
from ..circuit import BasicCircuit
from .basic_optimizer import BasicOptimizer


class SGD(BasicOptimizer):
    r"""SGD Optimizer class
    """
    def __init__(self, iterations: int, circuit: BasicCircuit, learning_rate: float):
        r"""The constructor of the SGD class

        Args:
            iterations (int): Number of iterations
            circuit (BasicCircuit): Circuit whose parameters are to be optimized
            learning_rate (float): Learning rate

        """
        super().__init__(iterations, circuit)
        self._learning_rate = learning_rate

    def minimize(
            self, shots: int,
            loss_func: Callable[[np.ndarray, int], float],
            grad_func: Callable[[np.ndarray, int], np.ndarray]
    ) -> None:
        r"""Minimizes the given loss function

        Args:
            shots (int): Number of measurement shots
            loss_func (Callable[[np.ndarray, int], float]): Loss function to be minimized
            grad_func (Callable[[np.ndarray, int], np.ndarray]): Function for calculating gradients

        """
        self._loss_history = []
        for _ in tqdm(range(self._iterations)):
            curr_param = self._circuit.parameters
            gradient = grad_func(curr_param, shots)
            new_param = curr_param - self._learning_rate * gradient
            loss = loss_func(new_param, shots)
            self._loss_history.append(loss)
