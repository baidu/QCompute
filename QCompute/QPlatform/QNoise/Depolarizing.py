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
Depolarizing
"""
FileErrorCode = 26

import random
import numpy as np
from typing import TYPE_CHECKING, List
from QCompute.QPlatform.QNoise import QNoise
from QCompute.QPlatform.QNoise.Utilities import sigma

if TYPE_CHECKING:
    from QCompute.OpenSimulator.local_baidu_sim2.Transfer import TransferProcessor


class Depolarizing(QNoise):
    r"""
    Depolarizing noise class.

    The Kraus operators of one qubit depolarizing noise are as follows:

    :math:`E_0 = \sqrt{1 - 3 p / 4} \ ID`

    :math:`E_1 = \sqrt{p / 4} \ X`

    :math:`E_2 = \sqrt{p / 4} \ Y`

    :math:`E_3 = \sqrt{p / 4} \ Z`

    Here, :math:`p` is the strength of noise.

    Example:

    Depolarizing(bits=2, probability=0.1)
    """

    def __init__(self, bits: int, probability: float) -> None:
        super().__init__(bits)
        self.probability = probability

        oneQubitKrauses = [sigma(index) for index in range(4)]

        self.krauses = self._tensorKrauses(oneQubitKrauses)
        self.probabilities = self._tensorProbabilitiesNonLocal(self.probability)

        self.noiseClass = 'mixed_unitary_noise'

        assert probability >= 0
        assert probability <= 1

    def calc_batched_noise_rng(self, num: int) -> List[int]:
        """
        Generate a batch of sampled random numbers.

        :param num: int, the number of sampled random numbers

        :return: List[int], the sampled random numbers
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