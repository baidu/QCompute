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
Homodyne measurement
"""


from typing import Union, Tuple, List, Dict

import numpy


from QCompute.OpenSimulator.local_baidu_sim_photonic.InitGaussState import MatrixType
from QCompute.OpenSimulator.local_baidu_sim_photonic.GaussTransfer import Algorithm
from QCompute.OpenSimulator.local_baidu_sim_photonic.GaussAuxiliaryCalculation import TraceOneMode, TensorProduct


class HomodyneMeasure:
    """
    Perform homodyne measurement
    """

    def __init__(self, matrixType: MatrixType, algorithm: Algorithm) -> None:

        if matrixType == MatrixType.Dense:
            if algorithm == Algorithm.Matmul:
                self.proc = self.RunMultiDenseHomoByMatmul
            else:
                assert False
        else:
            assert False

    def __call__(self, state_list: Union[List[numpy.ndarray], List['COO']], modes_array: numpy.ndarray, shots) \
            -> Dict[str, float]:
        """
        To enable the object callable
        """

        return self.proc(state_list, modes_array, shots)

    def RunSingleDenseHomoByMatmul(self, state_list: List[numpy.ndarray], mode: int, shots=1) \
            -> Tuple[float, List[numpy.ndarray]]:
        """
        Simulate the sampling process for measuring single-qumode state

        :param state_list: the first and second moments
        :param mode: measured qumode
        :param shots: 'shots' must be set to 1
        :return: sampling results, as well as the first and second moment of rest state.
        """

        state_fir_mom, state_sec_mom = state_list

        # Get the vector and the block second moment of traced and rest qumodes
        (list_vector, list_matrix) = TraceOneMode(state_fir_mom, state_sec_mom, mode)
        vector_a, vector_b = list_vector
        matrix_A, matrix_B, matrix_C = list_matrix

        # Pseudo inverse matrix is a 2 by 2 matrix in which the top-left entry is matrix_B[0, 0]
        B_one_one = matrix_B[0, 0]

        # Obtain the results of homodyne measurement
        x_value = numpy.random.normal(vector_b[0], numpy.sqrt(B_one_one), size=shots)[0]

        inverse_matrix = numpy.array([[1 / B_one_one, 0],
                                      [0, 0]])
        vector_u = numpy.array([[x_value],
                                [0]])
        matrix_C_T = numpy.transpose(matrix_C)

        # We need to update the first and second-moment
        fir_mom_rest = vector_a - numpy.matmul(numpy.matmul(matrix_C, inverse_matrix), vector_b - vector_u)
        sec_mom_rest = matrix_A - numpy.matmul(numpy.matmul(matrix_C, inverse_matrix), matrix_C_T)
        state_fir_mom, state_sec_mom = TensorProduct(fir_mom_rest, sec_mom_rest, mode)
        state_list = [state_fir_mom, state_sec_mom]

        return x_value, state_list

    def RunMultiDenseHomoByMatmul(self, state_list: List[numpy.ndarray], modes_array: numpy.ndarray, shots=1) \
            -> Dict[str, float]:
        """
        Simulate the sampling process for measuring multi-qumode state

        :param state_list: the first and second moments
        :param modes_array: measured qumodes
        :param shots: 'shots' must be set to 1
        :return dictionary_results: sampling results
        """

        dictionary_results = dict()
        for index in range(len(modes_array)):
            mode = modes_array[index]
            x_value, updated_state_list = self.RunSingleDenseHomoByMatmul(state_list, mode, shots)
            mode_str = str(mode)
            dictionary_results[mode_str] = x_value
            state_list = updated_state_list

        return dictionary_results
