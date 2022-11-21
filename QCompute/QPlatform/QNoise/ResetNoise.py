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
import random

import numpy as np

from QCompute.QPlatform.QNoise import QNoise


class ResetNoise(QNoise):
    """
    Reset Noise
    """

    def __init__(self, probability1: float, probability2: float):
        super().__init__(1)
        self.probability1 = probability1
        self.probability2 = probability2

        self.krauses = [
            np.array([[np.sqrt(probability1), 0.0], [0.0, 0.0]]), np.array([[0.0, np.sqrt(probability1)], [0.0, 0.0]]),
            np.array([[0.0, 0.0], [np.sqrt(probability2), 0.0]]), np.array([[0.0, 0.0], [0.0, np.sqrt(probability2)]]),
            np.eye(2) * np.sqrt(1 - probability1 - probability2)
        ]

        self.noise_class = 'nonmixed_unitary_noise'

        assert probability1 >= 0 
        assert probability2 >= 0
        assert probability1 + probability2 <= 1

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
