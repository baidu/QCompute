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
import multiprocess

FileErrorCode = 19

import numpy
import math
import copy
from typing import Union, Tuple, List, Dict


from QCompute.OpenSimulator.local_baidu_sim_photonic.InitGaussState import MatrixType


class HeterodyneMeasure:
    """
    Perform heterodyne measurement
    """

    def __init__(self, matrixType: MatrixType) -> None:

        if matrixType == MatrixType.Dense:
            self.proc = self.RunMultiDenseHeteroByMatmul
        else:
            assert False

    def __call__(self, state_list: Union[List[numpy.ndarray], List['COO']], r_and_phi_list: List[Tuple[float, float]],
                 modes_array: numpy.ndarray, shots) -> Dict[str, list]:
        """
        To enable the object callable
        """

        return self.proc(state_list, r_and_phi_list, modes_array, shots)

    def RunSingleDenseHeteroByMatmul(self, inputParam: Tuple[int, List[numpy.ndarray], float, float, int]) \
            -> Tuple[int, numpy.ndarray, list]:
        """
        Simulate the sampling process of measuring single-qumode coherent state

        :param state_list: the first and second moments
        :param r: amplitude of the measured coherent state
        :param phi: phase of the measured coherent state
        :param mode: target qumode
        :return: sampling results, as well as the first and second moment of rest state.
        """

        (index, state_list, r, phi, mode) = inputParam

        fir_mom, sec_mom = state_list

        xcoor = 2 * mode
        xprow = copy.copy(sec_mom[:, xcoor: xcoor + 2])
        matB = copy.copy(sec_mom[xcoor: xcoor + 2, xcoor: xcoor + 2])
        inverse_BplusI = numpy.linalg.inv(matB + numpy.eye(2))

        # Update the second moment
        sec_mom -= numpy.matmul(numpy.matmul(xprow, inverse_BplusI), numpy.transpose(xprow))
        sec_mom[xcoor: xcoor + 2, :] = sec_mom[:, xcoor: xcoor + 2] = 0
        sec_mom[xcoor, xcoor] = sec_mom[xcoor + 1, xcoor + 1] = 1

        # Get random x and p
        central_xp = numpy.sqrt(2) * r * numpy.array([[math.cos(phi)],
                                                      [math.sin(phi)]])
        random_xp = numpy.array([numpy.random.normal(central_xp[0, 0], scale=1.0, size=1),
                                 numpy.random.normal(central_xp[1, 0], scale=1.0, size=1)])

        # Update the first moment
        b_minus_rxp = fir_mom[xcoor: xcoor + 2] - random_xp
        fir_mom -= numpy.matmul(numpy.matmul(xprow, inverse_BplusI), b_minus_rxp)
        fir_mom[xcoor: xcoor + 2] = 0

        state_list = [fir_mom, sec_mom]

        return index, random_xp[:, 0], state_list

    def RunMultiDenseHeteroByMatmul(self, state_list: List[numpy.ndarray], r_and_phi_list: list,
                                    modes_array: numpy.ndarray, shots: int) -> Dict[str, list]:
        """
        Simulate the sampling process of measuring multi-qumode coherent state

        :param state_list: the first and second moments
        :param r_and_phi_list: amplitude and phase of the measured coherent state
        :param modes_array: measured qumodes
        :param shots: 'shots'
        :return dictionary_results: sampling results
        """
        random_xp_list = [numpy.array([0, 0])] * len(r_and_phi_list)

        for _ in range(shots):
            updated_state_list = copy.deepcopy(state_list)
            for index in range(len(modes_array)):
                mode = modes_array[index]
                single_r_and_phi = r_and_phi_list[index]
                phi = single_r_and_phi.phi
                r = single_r_and_phi.r
                index, xp_array, updated_state_list = self.RunSingleDenseHeteroByMatmul(
                    (index, updated_state_list, r, phi, mode))
                random_xp_list[index] = random_xp_list[index] + xp_array

        dictionary_results = dict()
        for index in range(len(modes_array)):
            mode_str = str(modes_array[index])
            dictionary_results[mode_str] = list(random_xp_list[index] / shots)

        return dictionary_results