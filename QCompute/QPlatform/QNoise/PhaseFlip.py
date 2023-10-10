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
Phase Flip
"""
FileErrorCode = 29

import random
import numpy as np
from typing import TYPE_CHECKING, List
from QCompute.QPlatform.QNoise import QNoise
from QCompute.QPlatform.QNoise.Utilities import sigma

if TYPE_CHECKING:
    from QCompute.OpenSimulator.local_baidu_sim2.Transfer import TransferProcessor


class PhaseFlip(QNoise):
    r"""
    Phase flip class.

    The Kraus operators of such noise are as follows:

    :math:`E_0 = \sqrt{1 - p} \ ID`
    
    :math:`E_1 = \sqrt{p} \ Z`

    Here, :math:`p` is the strength of noise.
    """

    def __init__(self, probability: float) -> None:
        super().__init__(1)
        self.probability = probability

        self.krauses = [sigma(0), sigma(3)]
        self.probabilities = [1.0 - probability, probability]

        self.noiseClass = 'mixed_unitary_noise'

        assert probability >= 0
        assert probability <= 1

    def calc_batched_noise_rng(self, num: int) -> List[int]:
        """
        Generate a batch of sampled random numbers.

        :param num: the number of sampled random numbers

        :return: List[int], a set of random numbers
        """

        return [random.choices(range(len(self.krauses)), self.probabilities)[0] for _ in range(num)]

    def calc_noise_matrix(self, transfer: 'TransferProcessor', state: np.ndarray, qRegList: List[int]) -> np.ndarray:
        """
        Generate a sampled Kraus operator.

        :param transfer: 'TransferProcessor', matrix-vector multiplication algorithm

        :param state: np.ndarray, current state in simulator

        :param qRegList: List[int], quantum register where the noise is added

        :return: np.ndarray, a sampled Kraus operator
        """

        return random.choices(self.krauses, self.probabilities)[0]