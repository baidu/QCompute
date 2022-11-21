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
Phase Damping
"""
import random

import numpy as np

from QCompute.QPlatform.QNoise import QNoise


class PhaseDamping(QNoise):
    """
    Phase Damping
    Equivalent to a PhaseFlip noise 
    """

    def __init__(self, probability: float):
        super().__init__(1)
        self.probability = (1 - np.sqrt(1.0 - probability)) / 2

        self.krauses = [self.sigmaop(0), self.sigmaop(3)]
        self.probabilities = [1.0 - probability, probability]

        self.noise_class = 'mixed_unitary_noise'
        
        assert probability >= 0 
        assert probability <= 1

    
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
