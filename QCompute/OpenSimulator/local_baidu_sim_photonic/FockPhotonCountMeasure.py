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
FileErrorCode = 16

from typing import Tuple, Dict
import numpy
import math


from QCompute.OpenSimulator.local_baidu_sim_photonic.InitFockState import MatrixType
from QCompute.OpenSimulator.local_baidu_sim_photonic.FockAuxiliaryCalculation import CreateSubMatrix, RyserFormula
from QCompute.OpenSimulator.local_baidu_sim_photonic.FockStateList import ListOfInputFockState


class PhotonCountMeasure:
    """
    Perform photon-count measurement
    """

    def __init__(self, matrixType: MatrixType) -> None:

        if matrixType == MatrixType.Dense:
            self.proc = self.RunDensePhotonCountSampling
        else:
            assert False

    def __call__(self, fock_state_vector: numpy.ndarray, unitary_trans_total: numpy.ndarray,
                 num_cutoff: list, shots: int) -> Dict[str, int]:
        """
        To enable the object callable
        """

        return self.proc(fock_state_vector, unitary_trans_total, num_cutoff, shots)

    def RunDensePhotonCountSampling(self, fock_state_vector: numpy.ndarray, unitary_trans_total: numpy.ndarray,
                                    num_cutoff: int, shots: int) -> Dict[str, int]:
        r"""
        Simulate the sampling results according to calculated probabilities.

        :param fock_state_vector: the fock state vector after adding photons
        :param unitary_trans_total: the overall unitary :math:`U` of interferometer
        :param num_cutoff: the number of photons in single shot
        :return dictionary_results: sampling results
        """
        CalcuProb = AllInOneCalcuProb(fock_state_vector, unitary_trans_total, num_cutoff)
        (probability_array, final_state_list) = CalcuProb.results_output_state()

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


class AllInOneCalcuProb:

    def __init__(self, fock_state_vector: numpy.ndarray, unitary_trans_total: numpy.ndarray, num_cutoff: int) -> None:

        self.num_mode = len(fock_state_vector)
        self.num_cutoff = num_cutoff
        self.output_state_list = []
        InputFockState = ListOfInputFockState(fock_state_vector, num_cutoff, self.num_mode)
        self.input_state_list, self.num_repetition = InputFockState.results_initial_state()
        self.input_state_list_length = len(self.input_state_list)
        self.probability_array = numpy.array([])
        self.unitary_trans_total = unitary_trans_total

    def backtrack_output_state(self, path: numpy.ndarray) -> None:
        """
        To find all output states, and calculate the probability distribution.
        """

        if len(path) == self.num_mode and sum(path) == self.num_cutoff:
            self.output_state_list.append(numpy.array(path))

            probability_output_state = 0
            output_state = numpy.copy(path)
            for index_input_state in range(self.input_state_list_length):
                input_state = self.input_state_list[index_input_state]
                unitary_st = CreateSubMatrix(self.unitary_trans_total, input_state, output_state)
                permanent = RyserFormula(self.num_cutoff, unitary_st)
                factorial = 1

                for index in range(self.num_mode):
                    factorial = factorial * math.factorial(input_state[index]) * math.factorial(output_state[index])
                probability_output_state += self.num_repetition[index_input_state] * abs(permanent) ** 2 / \
                                            (factorial * self.input_state_list_length)
            self.probability_array = numpy.append(self.probability_array, probability_output_state)

            return

        if len(path) > self.num_mode or len(path) > self.num_mode:
            return

        for counts in range(self.num_cutoff + 1):
            path.append(counts)
            self.backtrack_output_state( path)
            path.pop()

    def results_output_state(self) -> Tuple[numpy.ndarray, list]:
        """
        Return a list containing all final states
        """
        path = []
        self.backtrack_output_state(path)

        return self.probability_array, self.output_state_list