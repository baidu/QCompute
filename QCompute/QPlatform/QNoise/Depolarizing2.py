#!/usr/bin/python3CircuitLine
# -*- coding: utf8 -*-

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
Depolarizing2
"""
import random

import numpy as np

from itertools import product

from QCompute.QPlatform.QNoise import QNoise


class Depolarizing2(QNoise):
    """
    Depolarizing2
    """

    def __init__(self, probability: float):
        super().__init__(2)
        self.probability = probability

        self.krauses = [np.kron(self.sigmaop(i), self.sigmaop(j)) for i, j in product(range(4), range(4))]
        self.krauses = list(map(lambda x: x.reshape([2, 2, 2, 2]), self.krauses))

        self.probabilities = [1.0 - 15.0 / 16.0 * probability] + [probability / 16.0] * 15       
        self.noise_class = 'mixed_unitary_noise'

        assert probability >= 0 
        assert probability <= 1

   
    def calc_batched_noise_rng(self, num: int):
        """
        calc_batched_noise_rng
        """

        return [random.choices(range(len(self.krauses)), self.probabilities)[0] for i in range(num)]

    def calc_noise_matrix(self, transfer, state, qRegList):
        """
        calc_noise_matrix
        """

        return random.choices(self.krauses, self.probabilities)[0]
