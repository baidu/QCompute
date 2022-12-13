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
from typing import TYPE_CHECKING, List, Dict
import random
import numpy as np
from QCompute.QPlatform.QNoise import QNoise
from QCompute.QPlatform.QNoise.Utilities import isTracePreserving, numpyMatrixToTensorMatrix, calcKrausLowerBound

if TYPE_CHECKING:   
    from QCompute.OpenSimulator.local_baidu_sim2_with_noise.Transfer import TransferProcessor


class CustomizedNoise(QNoise):
    """
    Customized noise class.

    Generate a noise defined by customers which is described by Kraus operator summation representation.

    Available for Kraus operators of numpy formation.
    """

    def __init__(self, krauses: List[np.ndarray]) -> None:

        if krauses[0].shape[0] > 2 or krauses[0].shape[1] > 2:
            binSize = str(bin(krauses[0].shape[0] - 1))[2:]  # e.g., 4 -> '11', 8 -> '111'
            bits = len(binSize)

            assert krauses[0].shape[0] == krauses[0].shape[1], 'Available for n-qubit matrix only' 
            assert sum([int(idex) for idex in binSize]) == bits, 'Available for n-qubit matrix only'
        else:
            bits = int(len(krauses[0].shape) / 2)

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
        Generate a sampled Kraus operator.

        :param transfer: 'TransferProcessor', matrix-vector multiplication algorithm
        :param state: np.ndarray, current state in simulator
        :param qRegList: List[int], quantum register where the noise is added
        :return: np.ndarray, a sampled Kraus operator
        """

        if self.noiseClass == 'mixed_unitary_noise':
            return random.choices(self.krauses, self.lowerBoundList)[0]
        else:
            # calc lower bounds for each Kraus operator and the maximum one
            listS = self.lowerBoundList
            maxLowerBound = max(listS)
            maxLowerBoundIndex = listS.index(maxLowerBound)

            if self.bits > 1 and self.krauses[0].shape[0] > 2:
                self.krauses = self.krauses = [numpyMatrixToTensorMatrix(_) for _ in self.krauses]

            r = random.random()
            stateMax = transfer(
                state, self.krauses[maxLowerBoundIndex], qRegList)
            proMax = np.vdot(stateMax, stateMax)

            if r < proMax:
                return self.krauses[maxLowerBoundIndex] / np.sqrt(proMax)

            else:
                r = r - proMax
                listIndex = [index for index in range(
                    len(self.krauses)) if index != maxLowerBoundIndex]

                for _ in listIndex:
                    stateCopy = transfer(state, self.krauses[_], qRegList)
                    proCopy = np.vdot(stateCopy, stateCopy)

                    if r < proCopy:
                        return self.krauses[_] / np.sqrt(proCopy)
                    else:
                        r = r - proCopy

    def calc_noise_rng_non_mixed(self, transfer: 'TransferProcessor', state: np.ndarray, qRegList: List[int]) -> int:
        """
        Generate a sampled random number for non-mixed-unitary noise.

        :param transfer: 'TransferProcessor', matrix-vector multiplication algorithm
        :param state: np.ndarray, current state in simulator
        :param qRegList: List[int], quantum register where the noise is added
        :return: int, a sampled random number
        """

        listS = self.lowerBoundList
        maxLowerBound = max(listS)
        maxLowerBoundIndex = listS.index(maxLowerBound)

        r = random.random()
        if r <= maxLowerBound:
            return maxLowerBoundIndex
        else:
            stateCopy = transfer(
                state, self.krauses[maxLowerBoundIndex], qRegList)
            proCopy = np.vdot(stateCopy, stateCopy)

            if r < proCopy:
                return maxLowerBoundIndex
            else:
                r = r - proCopy
                listIndex = [index for index in range(
                    len(self.krauses)) if index != maxLowerBoundIndex]

                for _ in listIndex:
                    stateCopy = transfer(state, self.krauses[_], qRegList)
                    proCopy = np.vdot(stateCopy, stateCopy)

                    if r < proCopy:
                        return _
                    else:
                        r = r - proCopy
