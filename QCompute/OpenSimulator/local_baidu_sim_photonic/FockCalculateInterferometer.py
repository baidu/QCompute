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

r"""
Calculate overall unitary :math:`U` of interferometer
"""

from enum import IntEnum, unique
from typing import Union
import numpy




@unique
class Algorithm(IntEnum):
    """
    Only the 'MATMUL' algorithm is implemented in current version.
    """

    Matmul = 0
    Einsum = Matmul + 1


def CalcuGateMatrix(gate_matrix: numpy.ndarray, modes: numpy.ndarray, n: int) -> numpy.ndarray:
    r"""
    Covert the gate matrix defined in scripy 'PhotonicFockGate'
    into an :math:`N \times N` matrix that can act on :math:`N`-qumode quantum states.

    :param gate_matrix: the matrix of single- or two-qumode gate
    :param modes: a list of target qumode(s)
    :param n: the total number of qumodes
    :return: an :math:`N \times N` matrix
    """

    pending_gate = gate_matrix.getMatrix()
    nmode_gate_matrix = numpy.eye(n, dtype=complex)
    if len(modes) == 1:
        nmode_gate_matrix[modes[0], modes[0]] = pending_gate[0, 0]
    elif len(modes) == 2:
        mode_1 = modes[0]
        mode_2 = modes[1]
        nmode_gate_matrix[mode_1, mode_1] = pending_gate[0, 0]
        nmode_gate_matrix[mode_1, mode_2] = pending_gate[0, 1]
        nmode_gate_matrix[mode_2, mode_1] = pending_gate[1, 0]
        nmode_gate_matrix[mode_2, mode_2] = pending_gate[1, 1]

    return nmode_gate_matrix


class FockStateTransferProcessor:
    r"""
    Calculate the overall unitary :math:`U` of interferometer.
    """

    def __init__(self, algorithm: Algorithm) -> None:
        if algorithm == Algorithm.Matmul:
            self.proc = self.CalcuInterByMatmul
        else:
            assert False

    def __call__(self, unitary_trans_total: Union[numpy.ndarray, 'COO'], gate_matrix: Union[numpy.ndarray, 'COO'],
                 modes: numpy.ndarray) -> Union[numpy.ndarray, 'COO']:
        """
        To enable the object callable
        """

        return self.proc(unitary_trans_total, gate_matrix, modes)

    def CalcuInterByMatmul(self, unitary_trans_total: numpy.ndarray, gate_matrix: numpy.ndarray, modes: numpy.ndarray)\
            -> numpy.ndarray:
        r"""
        Update the overall unitary :math:`U` of interferometer.

        :param unitary_trans_total: unitary :math:`U` of interferometer
        :param gate_matrix: the matrix of single- or two-qumode gate
        :param modes: a list of target qumode(s)
        :return unitary_trans_total: updated unitary :math:`U` of interferometer
        """

        unitary_trans_single = CalcuGateMatrix(gate_matrix, modes, unitary_trans_total.shape[0])
        unitary_trans_total = numpy.matmul(unitary_trans_single, unitary_trans_total)
        return unitary_trans_total
