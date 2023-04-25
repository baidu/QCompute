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
Powell
"""

from typing import Callable
import numpy as np
from scipy import optimize
from ..circuit import BasicCircuit
from .basic_optimizer import BasicOptimizer


class Powell(BasicOptimizer):
    r"""Powell Optimizer class
    """
    def __init__(self, iterations: int, circuit: BasicCircuit):
        r"""The constructor of the Powell class

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
            grad_func (Callable[[np.ndarray, int], np.ndarray])): Function for calculating gradients

        """
        self._loss_history = []
        curr_param = self._circuit.parameters
        opt_res = optimize.minimize(
            loss_func, curr_param, args=(shots,), method='Powell',
            options={'maxiter': self._iterations},
            callback=lambda xk: self._loss_history.append(loss_func(xk, shots))
        )
        print(opt_res.message)
