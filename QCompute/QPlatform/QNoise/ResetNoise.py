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
Reset Noise
"""
FileErrorCode = 30

import random
import numpy as np
from typing import TYPE_CHECKING, List, Dict

from QCompute.QPlatform.QNoise import QNoise

if TYPE_CHECKING:
    from QCompute.OpenSimulator.local_baidu_sim2.Transfer import TransferProcessor


class ResetNoise(QNoise):
    r"""
    Reset noise class.

    The Kraus operators of such noise are as follows:
    
    :math:`E_0 = \sqrt{1 - p_1 - p_2} \ ID`

    :math:`E_1 = \begin{bmatrix} \sqrt{p_1} & 0 \\ 0 & 0 \end{bmatrix}`

    :math:`E_2 = \begin{bmatrix} 0 & \sqrt{p_1} \\ 0 & 0 \end{bmatrix}`

    :math:`E_3 = \begin{bmatrix} 0 & 0 \\ \sqrt{p_2} & 0 \end{bmatrix}`

    :math:`E_4 = \begin{bmatrix} 0 & 0 \\ 0 & \sqrt{p_2} \end{bmatrix}`

    Here, :math:`p` is the strength of noise.
    """

    def __init__(self, probability1: float, probability2: float) -> None:
        super().__init__(1)
        self.probability1 = probability1
        self.probability2 = probability2

        self.krauses = [
            np.eye(2) * np.sqrt(1 - probability1 - probability2),
            np.array([[np.sqrt(probability1), 0.0], [0.0, 0.0]]), np.array(
                [[0.0, np.sqrt(probability1)], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [np.sqrt(probability2), 0.0]]), np.array(
                [[0.0, 0.0], [0.0, np.sqrt(probability2)]])        
            ]

        self.noiseClass = 'non_mixed_unitary_noise'

        assert probability1 >= 0
        assert probability2 >= 0
        assert probability1 + probability2 <= 1

    def calc_batched_noise_rng_non_mixed(self, transfer: 'TransferProcessor', stateDict: Dict[str, np.ndarray],
                                         qRegList: List[int]) -> List[int]:
        """
        Generate a batch of sampled random numbers for non-mixed-unitary noise.

        :param transfer: 'TransferProcessor', matrix-vector multiplication algorithm

        :param stateDict: Dict[str, np.ndarray], current state dict in simulator

        :param qRegList: List[int], quantum register where the noise is added

        :return: List[int], a set of random numbers
        """

        rngList = [self.calc_noise_rng_non_mixed(
            transfer, stateDict[key], qRegList) for key in stateDict.keys() for _ in key]
        return rngList

    def calc_noise_matrix(self, transfer: 'TransferProcessor', state: np.ndarray, qRegList: List[int]) -> np.ndarray:
        """
        Generate a sampled Kraus operator.

        :param transfer: 'TransferProcessor', matrix-vector multiplication algorithm

        :param state: np.ndarray, current state in simulator

        :param qRegList: List[int], quantum register where the noise is added

        :return: np.ndarray, a sampled Kraus operator
        """

        r = random.random()
        for kraus in self.krauses:
            stateCopy = transfer(state, kraus, qRegList)
            proCopy = np.vdot(stateCopy, stateCopy)

            if r < proCopy:
                return kraus / np.sqrt(proCopy)
            else:
                r = r - proCopy
        
        assert False

    def calc_noise_rng_non_mixed(self, transfer: 'TransferProcessor', state: np.ndarray, qRegList: List[int]) -> int:
        """
        Generate a sampled random number for non-mixed-unitary noise.

        :param transfer: 'TransferProcessor', matrix-vector multiplication algorithm

        :param state: np.ndarray, current state in simulator

        :param qRegList: List[int], quantum register where the noise is added

        :return: int, a sampled random number
        """

        r = random.random()
        for index in range(len(self.krauses)):
            stateCopy = transfer(state, self.krauses[index], qRegList)
            proCopy = np.vdot(stateCopy, stateCopy)

            if r < proCopy:
                return index
            else:
                r = r - proCopy
        assert False