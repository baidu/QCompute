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
from typing import List
import numpy as np


# A random FileErrorCode, need to be changed
FileErrorCode = 15

class QNoise:
    """
    Quantum Noise
    """

    def __init__(self, bits: int):
        self.bits = bits

    @staticmethod
    def sigmaop(k):
        """
        sigmaop
        """
        if k == 0:
            return np.eye(2)
        elif k == 1:
            return np.array([[0.0, 1.0], [1.0, 0.0]])
        elif k == 2:
            return np.array([[0.0, -1.0j], [1.0j, 0.0]])
        elif k == 3:
            return np.array([[1.0, 0.0], [0.0, -1.0]])

    def calcKrausLowerBound(self):
        """
        Calculate the lower bound of probabilities for sampling among a set of kraus operators
        """

        LowerBound = []
        for i in range(len(self.krauses)):   
            if self.krauses[i].shape == (2,2):
                tempOperator = np.dot(self.krauses[i].T.conjugate(), self.krauses[i])            
            elif self.krauses[i].shape == (2,2,2,2):
                tempKraus = np.reshape(self.krauses[i], [4,4])
                tempOperator = np.dot(tempKraus.T.conjugate(), tempKraus)
#            else:
#                raise Error.ArgumentError(f'Unsupported noise type {noise_type}!', ModuleErrorCode, FileErrorCode, 9)
            tempBound = min(list(np.linalg.eig(tempOperator)[0]))
            LowerBound.append(tempBound)
        
        return LowerBound



class QNoiseDefine:
    """
    Quantum Noise Define
    """

    def __init__(self, noiseList: List[QNoise], qRegList: List[int], positionList: List[int]):
        self.noiseList = noiseList
        self.qRegList = qRegList
        self.positionList = positionList


    


            
