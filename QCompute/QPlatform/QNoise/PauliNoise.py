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
Pauli Noise
"""
import random

import numpy as np

from QCompute.QPlatform.QNoise import QNoise


class PauliNoise(QNoise):
    """
    Pauli Noise
    """

    def __init__(self, probability1: float, probability2: float, probability3: float):
        super().__init__(1)
        self.probability1 = probability1
        self.probability2 = probability2
        self.probability3 = probability3

        self.krauses = [self.sigmaop(0), self.sigmaop(1), self.sigmaop(2), self.sigmaop(3)]
        self.probabilities = [1 - probability1 - probability2 - probability3, probability1, probability2, probability3]

        self.noise_class = 'mixed_unitary_noise'

        assert probability1 >= 0 
        assert probability2 >= 0
        assert probability3 >= 0 
        assert probability1 + probability2 + probability3 <= 1
   


    def calc_batched_noise_rng(self, num: int):
        """
        calc_batched_noise_matrix
        """

        return [random.choices(range(len(self.krauses)), self.probabilities)[0] for i in range(num)]

    def calc_noise_matrix(self, transfer, state, qRegList):
        """
        calc_noise_matrix
        """

        return random.choices(self.krauses, self.probabilities)[0]
