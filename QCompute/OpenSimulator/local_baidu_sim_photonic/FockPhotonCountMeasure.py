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


from QCompute.OpenSimulator.local_baidu_sim_photonic.InitFockState import MatrixType
from QCompute.OpenSimulator.local_baidu_sim_photonic.FockCalculateInterferometer import Algorithm
from QCompute.OpenSimulator.local_baidu_sim_photonic.FockAuxiliaryCalculation import CreateSubMatrix, RyserFormula
from QCompute.OpenSimulator.local_baidu_sim_photonic.FockCombination import FockCombination


class PhotonCountMeasure:
    """
    Perform photon-count measurement
    """

    def __init__(self, matrixType: MatrixType, algorithm: Algorithm) -> None:

        if matrixType == MatrixType.Dense:
            self.proc = self.RunDensePhotonCountSampling
        else:
            assert False

    def __call__(self, fock_state_vector: Union[numpy.ndarray, 'COO'], unitary_trans_total: Union[numpy.ndarray, 'COO'],
                 num_cutoff: list, shots: int) -> Dict[str, int]:
        """
        To enable the object callable
        """

        return self.proc(fock_state_vector, unitary_trans_total, num_cutoff, shots)

    def CalcuProbByMatmul(self, fock_state_vector: numpy.ndarray, unitary_trans_total: numpy.ndarray,
                           num_cutoff: int) -> Tuple[numpy.ndarray, List[numpy.ndarray]]:
        r"""
        Calculate the probabilities of all output states with a given 'num_cutoff'

        :param fock_state_vector: the fock state vector after adding photons
        :param unitary_trans_total: the overall unitary :math:`U` of interferometer
        :param num_cutoff: the number of photons in single shot
        :return probability_array: contains the probabilities of all output states
        :return final_state_list: contains all the form of output states
        """

        num_mode = len(fock_state_vector)
        fc = FockCombination(num_mode, fock_state_vector, num_cutoff)
        initial_state_list = fc.results_initial_state()
        final_state_list = fc.results_final_state()

        probability_array = []
        for output_state in final_state_list:
            probability_output_state = 0
            for input_state in initial_state_list:
                factorial = 1
                for index in range(num_mode):
                    factorial = factorial * math.factorial(input_state[index]) * math.factorial(output_state[index])
                unitary_st = CreateSubMatrix(unitary_trans_total, input_state, output_state)
                permanent = RyserFormula(num_cutoff, unitary_st)

                probability_output_state += abs(permanent) ** 2 / (factorial * len(initial_state_list))

            probability_array = numpy.append(probability_array, probability_output_state)

        return probability_array, final_state_list

    def RunDensePhotonCountSampling(self, fock_state_vector: numpy.ndarray, unitary_trans_total: numpy.ndarray,
                                     num_cutoff: int, shots: int) -> Dict[str, int]:
        r"""
        Simulate the sampling results according to calculated probabilities.

        :param fock_state_vector: the fock state vector after adding photons
        :param unitary_trans_total: the overall unitary :math:`U` of interferometer
        :param num_cutoff: the number of photons in single shot
        :return dictionary_results: sampling results
        """

        (probability_array, final_state_list) = \
            self.CalcuProbByMatmul(fock_state_vector, unitary_trans_total, num_cutoff)
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
