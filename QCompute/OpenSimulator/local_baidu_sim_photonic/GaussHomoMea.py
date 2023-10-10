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
FileErrorCode = 20

from typing import Union, Tuple, List, Dict
import numpy
import math
import copy


from QCompute.OpenSimulator.local_baidu_sim_photonic.InitGaussState import MatrixType


class HomodyneMeasure:
    """
    Perform homodyne measurement
    """

    def __init__(self, matrixType: MatrixType) -> None:

        if matrixType == MatrixType.Dense:
            self.proc = self.RunMultiDenseHomoByMatmul
        else:
            assert False

    def __call__(self, state_list: Union[List[numpy.ndarray], List['COO']], modes_array: numpy.ndarray, shots) \
            -> Dict[str, float]:
        """
        To enable the object callable
        """

        return self.proc(state_list, modes_array, shots)

    def RunSingleDenseHomoByMatmul(self, inputParam: Tuple[int, List[numpy.ndarray], int]) \
            -> Tuple[int, float, List[numpy.ndarray]]:
        """
        Simulate the sampling process for measuring single-qumode state

        :param state_list: the first and second moments
        :param mode: measured qumode
        :param shots: 'shots' must be set to 1
        :return: sampling results, as well as the first and second moment of rest state.
        """
        (index, state_list, mode) = inputParam

        fir_mom, sec_mom = state_list

        xcoor = 2 * mode
        xrow = numpy.reshape(copy.copy(sec_mom[:, xcoor]), (len(fir_mom), 1))
        b11 = xrow[xcoor, 0]

        # Update the second moment
        sec_mom -= numpy.matmul(xrow, numpy.transpose(xrow)) / b11
        sec_mom[xcoor: xcoor + 2, :] = sec_mom[:, xcoor: xcoor + 2] = 0
        sec_mom[xcoor, xcoor] = sec_mom[xcoor + 1, xcoor + 1] = 1

        # Get the random value of x according to its covariance
        central_x = copy.copy(fir_mom[xcoor, 0])
        random_x = numpy.random.normal(central_x, math.sqrt(b11), size=1)[0]

        # Update the first moment
        x_minus_rx = fir_mom[xcoor, 0] - random_x
        fir_mom -= x_minus_rx * xrow
        fir_mom[xcoor: xcoor + 2] = 0

        state_list = [fir_mom, sec_mom]

        return index, random_x, state_list

    def RunMultiDenseHomoByMatmul(self, state_list: List[numpy.ndarray], modes_array: numpy.ndarray, shots: int) \
            -> Dict[str, float]:
        """
        Simulate the sampling process for measuring multi-qumode state

        :param state_list: the first and second moments
        :param modes_array: measured qumodes
        :param shots: 'shots'
        :return dictionary_results: sampling results
        """
        random_x_array = numpy.zeros(len(modes_array), dtype=float)

        for _ in range(shots):
            updated_state_list = copy.deepcopy(state_list)
            for index in range(len(modes_array)):
                mode = modes_array[index]
                index, x_value, updated_state_list = \
                    self.RunSingleDenseHomoByMatmul((index, updated_state_list, mode))
                random_x_array[index] = random_x_array[index] + x_value

        random_x_array /= shots

        dictionary_results = dict()
        for index in range(len(modes_array)):
            mode_str = str(modes_array[index])
            dictionary_results[mode_str] = random_x_array[index]

        return dictionary_results