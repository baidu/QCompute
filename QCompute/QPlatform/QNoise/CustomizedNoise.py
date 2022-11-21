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
CustomizedNoise
"""
from cmath import isclose
import random
from typing import List

import numpy as np

from QCompute.QPlatform.QNoise import QNoise

class CustomizedNoise(QNoise):
    """
    CustomizedNoise
    """

    def __init__(self, krauses: List[np.ndarray]):
        if krauses[0].shape == (2, 2):
            super().__init__(1)
        elif krauses[0].shape == (2, 2, 2, 2):
            super().__init__(2)
        else:
            assert False

        self.krauses = krauses
        self.LowerBoundList = self.calcKrausLowerBound()
        self.noise_class = self.verify_mixed_unitary_noise()

    def verify_mixed_unitary_noise(self):
        """
        verify the input kraus opretors are all unitary
        """  

        if np.isclose(1.0, sum(self.LowerBoundList)):
            self.noise_class = 'mixed_unitary_noise'
        else:
            self.noise_class = 'nonmixed_unitary_noise'

    def calc_batched_noise_rng(self, num: int):
        """
        calc a batch of samples for mixed_unitary_noise
        """   

        rngList = []

        listS = self.calcKrausLowerBound()

        rngList = [random.choices(range(len(self.krauses)), listS)[0] for i in range(num)]

        return rngList

    def calc_batched_noise_rng_nonmixed(self, transfer, stateDict, qRegList):
        """
        calc a batch of samples for nonmixed_unitary_noise
        """   
        
        rngList = []
        rngList = [self.calc_noise_rng_nonmixed(transfer, stateDict[k], qRegList) for k in stateDict.keys() for i in k]
   
        return rngList


    def calc_noise_matrix(self, transfer, state, qRegList):
        """
        calc_noise_matrix
        """
        # calc lower bounds for each kraus opeartor and its summation
        listS = self.calcKrausLowerBound()

        if self.noise_class == 'mixed_unitary_noise':

            return random.choices(self.krauses, listS)[0]
        else:
            r = random.random()    
    
            totalS = sum(listS)
            if r <= totalS:  
                for i in range(len(self.krauses)):                    
                    if r < listS[i]:
                        state_copy = transfer(state, self.krauses[i], qRegList)
                        pro_i = np.vdot(state_copy, state_copy) 
                        return self.krauses[i] / np.sqrt(pro_i)
                    else:
                        r = r - listS[i]
            else:    
                r = r - totalS                    
                for i in range(len(self.krauses)):
                    state_copy = transfer(state, self.krauses[i], qRegList)
                    pro_i = np.vdot(state_copy, state_copy) 

                    if r < pro_i - listS[i]:
                        return self.krauses[i] / np.sqrt(pro_i)
                    else:
                        r = r - (pro_i - listS[i])
            
    def calc_noise_rng_nonmixed(self, transfer, state, qRegList):
        """
        calc_noise_matrix
        """
        
        listS = self.calcKrausLowerBound()
        totalS = sum(listS)

        r = random.random()
        if r <= totalS:  
            for i in range(len(self.krauses)):                    
                if r < listS[i]:
                    return i
                else:
                    r = r - listS[i]
        else:    
            r = r - totalS                    
            for i in range(len(self.krauses)):
                state_copy = transfer(state, self.krauses[i], qRegList)
                pro_i = np.vdot(state_copy, state_copy)  

                if r < pro_i - listS[i]:
                    return i
                else:
                    r = r - (pro_i - listS[i])
