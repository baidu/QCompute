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
Simultaneous Perturbation Stochastic Approximation algorithm
"""

from typing import Callable
import numpy as np
from noisyopt import minimizeSPSA
from qapp.circuit import BasicCircuit
from .basic_optimizer import BasicOptimizer


class SPSA(BasicOptimizer):
    """SPSA Optimizer class
    """
    def __init__(self, iterations: int, circuit: BasicCircuit, a: float = 1.0, c: float = 1.0):
        """The constructor of the SPSA class

        :param iterations: Number of iterations
        :param circuit: Circuit whose parameters are to be optimized
        :param a: Scaling parameter for step size, defaults to 1.0
        :param c: Scaling parameter for evaluation step size, defaults to 1.0
        """
        super().__init__(iterations, circuit)
        self._a = a
        self._c = c

    def minimize(self, shots: int,
                 loss_func: Callable[[np.ndarray, int], float],
                 grad_func: Callable[[np.ndarray, int], np.ndarray]):
        """Minimizes the given loss function

        :param shots: Number of measurement shots
        :param loss_func: Loss function to be minimized
        :param grad_func: Function for calculating gradients
        """
        self._loss_history = []
        curr_param = self._circuit.parameters
        opt_res = minimizeSPSA(loss_func, curr_param, args=(shots,), niter=self._iterations, paired=False, a=self._a,
                               c=self._c, callback=lambda xk: self._loss_history.append(loss_func(xk, shots)))
        print(opt_res.message)
