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
Sequential Minimal Optimization
"""

import numpy as np
from tqdm import tqdm
from .basic_optimizer import BasicOptimizer
from ..circuit import BasicCircuit
from typing import Callable


class SMO(BasicOptimizer):
    r"""SMO Optimizer class

    Please see https://arxiv.org/abs/1903.12166 for details on this optimization method.
    """
    def __init__(self, iterations: int, circuit: BasicCircuit):
        r"""The constructor of the SMO class

        Args:
            iterations (int): Number of iterations
            circuit (BasicCircuit): Circuit whose parameters are to be optimized

        """
        super().__init__(iterations, circuit)

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
        curr_param = self._circuit.parameters
        for _ in tqdm(range(self._iterations)):
            for j in range(len(curr_param)):
                curr_param = self._circuit.parameters
                new_param = curr_param.copy()
                # f(0)
                new_param[j] = 0
                v1 = loss_func(new_param, shots)
                # f(pi/2)
                new_param[j] = 0.5 * np.pi
                v2 = loss_func(new_param, shots)
                # f(pi)
                new_param[j] = np.pi
                v3 = loss_func(new_param, shots)
                C = (v1 + v3) / 2
                if abs(v2 - C) < 1e-4:
                    curr_loss = C
                    self._loss_history.append(curr_loss)
                    continue
                B = np.arctan((v1 - C) / (v2 - C))
                A = (v1 - C) / np.sin(B)
                if A > 0:
                    x_new = - 0.5 * np.pi - B
                else:
                    x_new = 0.5 * np.pi - B
                curr_param[j] = x_new
                curr_loss = C - abs(A)
                self._loss_history.append(curr_loss)
                self._circuit.set_parameters(curr_param)
