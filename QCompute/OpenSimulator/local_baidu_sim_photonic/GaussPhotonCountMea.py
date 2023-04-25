#!/usr/bin/python3
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
Photon-count measurement
"""

from typing import Union, Tuple, List, Dict
import numpy
import math


from QCompute.OpenSimulator.local_baidu_sim_photonic.InitGaussState import MatrixType
from QCompute.OpenSimulator.local_baidu_sim_photonic.GaussTransfer import Algorithm
from QCompute.OpenSimulator.local_baidu_sim_photonic.GaussAuxiliaryCalculation import CalculateP0_and_H


class PhotonCountMeasure:
    """
    Perform photon-count measurement
    """

    def __init__(self, matrixType: MatrixType, algorithm: Algorithm) -> None:

        if matrixType == MatrixType.Dense:
            if algorithm == Algorithm.Matmul:
                self.proc = self.RunDensePhotonCountSampling
            else:
                assert False
        else:
            assert False

    def __call__(self, state_list: Union[List[numpy.ndarray], List['COO']], num_cutoff: list, shots: int) \
            -> Dict[str, int]:
        """
        To enable the object callable
        """

        return self.proc(state_list, num_cutoff, shots)

    def CalcuProbByMatmul(self, state_list: List[numpy.ndarray], num_cutoff: int) \
            -> Tuple[numpy.ndarray, List[numpy.ndarray]]:
        """
        Calculate the probabilities for a given 'num_cutoff'

        :param state_list: the first and second moments
        :param num_cutoff: resolution of count-photon detector
        :return: probability tensor and the form of output states
        """

        state_fir_mom, state_sec_mom = state_list

        (P0, H_2N) = CalculateP0_and_H(state_fir_mom, state_sec_mom, num_cutoff)
        num_mode = int(len(state_fir_mom) / 2)
        P_N = numpy.zeros(num_mode * [num_cutoff + 1])
        final_state_list = []
        probability_array = []
        # An iterator yielding pairs of tensor coordinates and values.
        for coor_tuple_P, value_P in numpy.ndenumerate(P_N):
            coor_P = numpy.array(coor_tuple_P)
            coor_H = numpy.hstack((coor_P, coor_P))
            value_H = H_2N[tuple(coor_H)]

            factorial = 1
            for index in range(num_mode):
                factorial = factorial * math.factorial(int(coor_P[index]))

            P_N[coor_tuple_P] = P0 * value_H / factorial
            if abs(P_N[coor_tuple_P]) < 1e-15:
                P_N[coor_tuple_P] = 0
            probability_array = numpy.append(probability_array, P_N[coor_tuple_P])
            final_state_list.append(coor_P)

        return probability_array, final_state_list

    def RunDensePhotonCountSampling(self, state_list: List[numpy.ndarray], cutoff_list: list, shots: int) \
            -> Dict[str, int]:
        """
        Simulate the sampling results according to calculated probability.

        :param state_list: the first and second moments
        :param cutoff_list: resolution of count-photon detector
        :return dictionary_results: sampling results
        """

        num_cutoff = cutoff_list.cutoff
        (probability_array, final_state_list) = self.CalcuProbByMatmul(state_list, num_cutoff)
        normal_prob_array = probability_array / sum(probability_array)
        counts_array = numpy.zeros_like(normal_prob_array)
        for _ in range(shots):
            index = numpy.random.choice(len(normal_prob_array), replace=True, p=normal_prob_array)
            counts_array[index] += 1
        dictionary_results = dict()
        for index in range(len(counts_array)):
            eigenstate = ''.join(str(i) for i in final_state_list[index])
            dictionary_results[eigenstate] = int(counts_array[index])
        return dictionary_results
