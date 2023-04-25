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
Heterodyne measurement
"""
import numpy
from typing import Union, Tuple, List, Dict


from QCompute.OpenSimulator.local_baidu_sim_photonic.InitGaussState import MatrixType
from QCompute.OpenSimulator.local_baidu_sim_photonic.GaussTransfer import Algorithm
from QCompute.OpenSimulator.local_baidu_sim_photonic.GaussAuxiliaryCalculation import TraceOneMode, TensorProduct


class HeterodyneMeasure:
    """
    Perform heterodyne measurement
    """

    def __init__(self, matrixType: MatrixType, algorithm: Algorithm) -> None:

        if matrixType == MatrixType.Dense:
            if algorithm == Algorithm.Matmul:
                self.proc = self.RunMultiDenseHeteroByMatmul
            else:
                assert False
        else:
            assert False

    def __call__(self, state_list: Union[List[numpy.ndarray], List['COO']], r_and_phi_list: List[Tuple[float, float]],
                 modes_array: numpy.ndarray, shots=1) -> Dict[str, list]:
        """
        To enable the object callable
        """

        return self.proc(state_list, r_and_phi_list, modes_array, shots)

    def RunSingleDenseHeteroByMatmul(self, state_list: List[numpy.ndarray], r: float, phi: float, mode: int, shots=1) \
            -> Tuple[list, list]:
        """
        Simulate the sampling process of measuring single-qumode coherent state

        :param state_list: the first and second moments
        :param r: amplitude of the measured coherent state
        :param phi: phase of the measured coherent state
        :param mode: target qumode
        :param shots: 'shots' must be set to 1
        :return: sampling results, as well as the first and second moment of rest state.
        """

        state_fir_mom, state_sec_mom = state_list

        # Get the vector and the block second moment of traced and rest qumodes
        (list_vector, list_matrix) = TraceOneMode(state_fir_mom, state_sec_mom, mode)
        vector_a, vector_b = list_vector
        matrix_A, matrix_B, matrix_C = list_matrix

        sec_mom_coherent = numpy.eye(2)
        x_central_value = numpy.sqrt(2) * r * numpy.cos(phi)
        p_central_value = numpy.sqrt(2) * r * numpy.sin(phi)
        x_value = numpy.random.normal(x_central_value, sec_mom_coherent[0, 0], size=shots)[0]
        p_value = numpy.random.normal(p_central_value, sec_mom_coherent[1, 1], size=shots)[0]
        fir_mom_coherent = numpy.array([[x_value],
                                        [p_value]])

        matrix_C_T = numpy.transpose(matrix_C)
        inverse_matrix = numpy.linalg.inv(matrix_B + sec_mom_coherent)

        # We need to update the first and second-moment
        fir_mom_rest = vector_a - numpy.matmul(numpy.matmul(matrix_C, inverse_matrix), vector_b - fir_mom_coherent)
        sec_mom_rest = matrix_A - numpy.matmul(numpy.matmul(matrix_C, inverse_matrix), matrix_C_T)
        state_fir_mom, state_sec_mom = TensorProduct(fir_mom_rest, sec_mom_rest, mode)
        state_list = [state_fir_mom, state_sec_mom]

        return [x_value, p_value], state_list

    def RunMultiDenseHeteroByMatmul(self, state_list: List[numpy.ndarray], r_and_phi_list: list,
                                     modes_array: numpy.ndarray, shots=1) -> Dict[str, list]:
        """
        Simulate the sampling process of measuring multi-qumode coherent state

        :param state_list: the first and second moments
        :param r_and_phi_list: amplitude and phase of the measured coherent state
        :param modes_array: measured qumodes
        :param shots: 'shots' must be set to 1
        :return dictionary_results: sampling results
        """

        dictionary_results = dict()
        for index in range(len(modes_array)):
            mode = modes_array[index]
            single_r_and_phi = r_and_phi_list[index]
            phi = single_r_and_phi.phi
            r = single_r_and_phi.r
            xp_list, updated_state_list = self.RunSingleDenseHeteroByMatmul(state_list, r, phi, mode, shots)
            mode_str = str(mode)
            dictionary_results[mode_str] = xp_list
            state_list = updated_state_list

        return dictionary_results
