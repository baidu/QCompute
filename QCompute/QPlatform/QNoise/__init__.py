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
Quantum Noise
"""
FileErrorCode = 32

from typing import Optional, List
import numpy as np

from itertools import product

from QCompute.QPlatform.QNoise.Utilities import noiseTensor, numpyMatrixToTensorMatrix

class QNoise:
    """
    QNoise abstract class.

    Quantum noise arises from inherint principles of quantum mechnics. It widely exists in 
    realistic quantum operations (e.g., H and RX gates) and quantum channels (e.g., Bell states).

    A quantum noise which satifies complete positive and trace preserving can itself be represented as a quantum channel.

    Any pre-defined noise classes and the user-defined noise class must inherit it.
    """

    def __init__(self, bits: int) -> None:
        self.bits = bits

    def _tensorKrauses(self, krausList: List[np.ndarray]) -> List[np.ndarray]:
        """
        Generate Kraus operators for n tensor of an identical noise.

        :param krausList: List[np.ndarray], Kraus operators for a single noise
        """

        bits = self.bits
        krauses = krausList

        for _ in range(bits - 1):
            krauses = noiseTensor(krausList, krauses)
        return [numpyMatrixToTensorMatrix(_) for _ in krauses]


    def _tensorProbabilitiesLocal(self, probabilityList: List[float]) -> List[float]:
        """
        Generate probabilities for local noise tensor.

        :param probabilityList: List[float], the probabilities that any kraus operators occur for a mixed unitary noise
        """

        for _ in range(self.bits - 1):
            probabilityList = [probabilityList[index_1] * probabilityList[index_2]
                             for index_1, index_2 in product(range(len(probabilityList)), range(len(probabilityList)))]

        return probabilityList

    def _tensorProbabilitiesNonLocal(self, probability: float) -> List[float]:
        """
        Generate probabilities for nonlocal noise tensor.

        :param probability: float, the strength of a noise

        Works for depolarizing noise.
        """
        bits = self.bits

        probabilityEachError = probability / (4 ** bits)

        probabilities = [
            1 - (4 ** bits - 1) * probability / (4 ** bits)] + [probabilityEachError] * (4 ** bits - 1)
        return probabilities


class QNoiseDefine:
    """
    Quantum Noise Define
    """

    def __init__(self, noiseList: List[QNoise], qRegList: List[int], positionList: List[int]) -> None:
        self.noiseList = noiseList
        self.qRegList = qRegList
        self.positionList = positionList