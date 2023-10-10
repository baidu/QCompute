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
Amplitude Damping
"""
FileErrorCode = 22

import random
import numpy as np
from typing import TYPE_CHECKING, List, Dict
from QCompute.QPlatform.QNoise import QNoise

if TYPE_CHECKING:
    from QCompute.OpenSimulator.local_baidu_sim2.Transfer import TransferProcessor


class AmplitudeDamping(QNoise):
    r"""
    Amplitude damping class.

    The Kraus operators of such noise are as follows:

    :math:`E_0 = \begin{bmatrix} 1 & 0 \\ 0 & \sqrt{1 - p} \end{bmatrix}`

    :math:`E_1 = \begin{bmatrix} 0 &  \sqrt{p}\\ 0 & 0 \end{bmatrix}`

    Here, :math:`p` is the strength of noise.
    """

    def __init__(self, probability: float) -> None:
        super().__init__(1)
        self.probability = probability

        self.krauses = [np.array([[1.0, 0.0], [0.0, np.sqrt(1 - probability)]]),
                        np.array([[0.0, np.sqrt(probability)], [0.0, 0.0]])]

        self.lowerBoundList = [1 - probability, 0.0]
        self.noiseClass = self._verify_mixed_unitary_noise()

        assert probability >= 0
        assert probability <= 1

    def _verify_mixed_unitary_noise(self) -> str:
        """
        Verify the input Kraus operators are all unitary and label it.
        """

        if np.isclose(0.0, self.probability):
            return 'mixed_unitary_noise'
        else:
            return 'non_mixed_unitary_noise'

    def calc_batched_noise_rng(self, num: int) -> List[int]:
        """
        Generate a batch of sampled random numbers for mixed-unitary noise.

        :param num: int, the number of sampled random numbers

        :return: List[int], a set of random numbers
        """

        listS = self.lowerBoundList

        rngList = [random.choices(range(len(self.krauses)), listS)[
            0] for _ in range(num)]

        return rngList

    def calc_batched_noise_rng_non_mixed(self, transfer: 'TransferProcessor', stateDict: Dict[str, np.ndarray]
                                         , qRegList: List[int]) -> List[int]:
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
        Generate a sampled Kraus operator which is chosen from all Kraus operators.

        :param transfer: 'TransferProcessor', matrix-vector multiplication algorithm

        :param state: np.ndarray, current state in simulator

        :param qRegList: List[int], quantum register where the noise is added

        :return: np.ndarray, a sampled Kraus operator
        """

        if self.noiseClass == 'mixed_unitary_noise':
            return random.choices(self.krauses, self.lowerBoundList)[0]
        else:
            # joint sort lowerBoundList+krauses by lowerBoundList in descending order
            sortedLowerBoundList, sortedKrauses = zip(*sorted(zip(self.lowerBoundList, self.krauses),
                                                                key=lambda x: x[0], reverse=True))

            r = random.random()

            for kraus in sortedKrauses:
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

        # joint sort lowerBoundList+krauses+krausID by lowerBoundList in descending order
        sortedLowerBoundList, sortedKrauses, sortedKrausesID = zip(*sorted(zip(self.lowerBoundList, self.krauses,
                                                                               range(len(self.krauses))),
                                                                           key=lambda x: x[0], reverse=True))

        r = random.random()
        for bound, kraus, krausID in zip(sortedLowerBoundList, sortedKrauses, sortedKrausesID):
            if r < bound:
                return krausID
            else:
                stateCopy = transfer(state, kraus, qRegList)
                proCopy = np.vdot(stateCopy, stateCopy)

                if r < proCopy:
                    return krausID
                else:
                    r = r - proCopy
        
        assert False