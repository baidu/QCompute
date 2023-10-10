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
Customized Noise
"""
FileErrorCode = 25

from typing import TYPE_CHECKING, List, Dict
import random
import math
import numpy as np
from QCompute.QPlatform.QNoise import QNoise
from QCompute.QPlatform.QNoise.Utilities import isTracePreserving, numpyMatrixToTensorMatrix, calcKrausLowerBound

if TYPE_CHECKING:
    from QCompute.OpenSimulator.local_baidu_sim2.Transfer import TransferProcessor


class CustomizedNoise(QNoise):
    """
    Customized noise class.

    Generate a noise defined by customers which is described by Kraus operator-sum representation.

    Available for Kraus operators of numpy formation.
    """

    def __init__(self, krauses: List[np.ndarray]) -> None:
        bits = int(math.log2(math.sqrt(krauses[0].size)) + 0.5)

        expected_shape = [(2 ** bits, 2 ** bits), (2, 2) * bits]

        assert krauses[0].shape in expected_shape, 'Available for n-qubit matrix only' 

        super().__init__(bits)
        self.krauses = krauses

        self.lowerBoundList = calcKrausLowerBound(krauses)

        self.noiseClass = 'non_mixed_unitary_noise'  # A default classification, used for simulation

        assert isTracePreserving(
            krauses), 'Input Kraus operators should be trace preserving'

    def calc_batched_noise_rng_non_mixed(self, transfer: 'TransferProcessor', stateDict: Dict[str, np.ndarray],
                                         qRegList: List[int]) -> List[int]:
        """
        Generate a batch of sampled random numbers for non-mixed-unitary noise.

        :param transfer: 'TransferProcessor', matrix-vector multiplication algorithm

        :param stateDict: Dict[str, np.ndarray], current state dict in simulator

        :param qRegList: List[int], quantum register where the noise is added

        :return: List[int], a set of random numbers
        """
        if self.bits > 1 and self.krauses[0].shape[0] > 2:
            self.krauses = self.krauses = [numpyMatrixToTensorMatrix(_) for _ in self.krauses]

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

            if self.bits > 1 and self.krauses[0].shape[0] > 2:
                self.krauses = self.krauses = [numpyMatrixToTensorMatrix(_) for _ in self.krauses]

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