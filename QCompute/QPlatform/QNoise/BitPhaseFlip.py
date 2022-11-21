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
Bit Phase Flip
"""
import random

import numpy as np

from QCompute.QPlatform.QNoise import QNoise
from QCompute.QPlatform.QNoise.BitFlip import BitFlip


class BitPhaseFlip(QNoise):
    """
    Bit Phase Flip
    """

    def __init__(self, probability: float):
        super().__init__(1)
        self.probability = probability
        self.krauses = [self.sigmaop(0), self.sigmaop(2)]
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

    if __name__ == '__main__':
        p = 0.3
        noise = BitFlip(p)        
   
        noiseList = noise.calc_batched_noise_rng(5)
        print(noiseList)
