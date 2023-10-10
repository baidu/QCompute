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
Several auxiliary functions for the quantum circuit based on fock states
"""
FileErrorCode = 14

import numpy
from typing import List


def CreateSubset(num_coincidence: int) -> List[list]:
    r"""
    According to a given positive integer :math:`n`,
    this function generates all subset of :math:`\left[ 1, 2, \dots, n \right]`.

    :param num_coincidence: the number of photons in single shot
    :return all_subset: all subset of :math:`\left[ 1, 2, \dots, n \right]`
    """

    num_arange = numpy.arange(num_coincidence)
    all_subset = []

    for index_1 in range(1, 2 ** num_coincidence):
        all_subset.append([])
        for index_2 in range(num_coincidence):
            if index_1 & (1 << index_2):
                all_subset[-1].append(num_arange[index_2])

    return all_subset


def RyserFormula(num_coincidence: int, U_st: numpy.ndarray) -> complex:
    r"""
    Calculate the permanent for a given submatrix

    :param num_coincidence: the number of photons in single shot
    :param U_st: the submatrix :math:`U_{\mathbf{st}}` of unitary :math:`U`,
                 where subscript :math:`\mathbf{s}` and :math:`\mathbf{t}` denote the input and output state, respectively.
    :return value_perm: permanent of input :math:`U_{\mathbf{st}}`
    """

    value_perm = 0
    set = CreateSubset(num_coincidence)

    for subset in set:
        num_elements = len(subset)
        value_times = 1
        for i in range(num_coincidence):
            value_sum = 0
            for j in subset:
                value_sum += U_st[i, j]
            value_times *= value_sum
        value_perm += value_times * (-1) ** num_elements
    value_perm *= (-1) ** num_coincidence
    return value_perm


def CreateSubMatrix(U: numpy.ndarray, input_state: numpy.ndarray, output_state: numpy.ndarray) -> numpy.ndarray:
    r"""
    Get the submatrix based on the unitary :math:`U`, input state, and output state.

    :param U: the overall unitary :math:`U` of interferometer.
    :return U_st: the submatrix :math:`U_{\mathbf{st}}` of unitary :math:`U`
    """

    in_eff_mode = numpy.nonzero(input_state)[0]
    out_eff_mode = numpy.nonzero(output_state)[0]
    row_submatrix = numpy.hstack([numpy.repeat(i, input_state[i]) for i in in_eff_mode])
    col_submatrix = numpy.hstack([numpy.repeat(i, output_state[i]) for i in out_eff_mode])
    U_st = U[row_submatrix.reshape(-1, 1), col_submatrix]

    return U_st