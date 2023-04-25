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
This module transfers the gaussian state from one into another according to the sequence of gates.
"""

from enum import IntEnum, unique
from typing import List, Union
import numpy



from QCompute.OpenSimulator.local_baidu_sim_photonic.InitGaussState import MatrixType


@unique
class Algorithm(IntEnum):
    """
    Only the 'MATMUL' algorithm is implemented in current version.
    """

    Matmul = 0
    Einsum = Matmul + 1


def CalcuGateMatrix(gate: List[numpy.ndarray], modes: numpy.ndarray, n: int) -> List[numpy.ndarray]:
    r"""
    Covert the gate matrix defined in scripy 'PhotonicGaussianGate'
    into an :math:`2N \times 2N` matrix that can act on :math:`N`-qumode quantum state.

    :param gate: thr single- or two-qumode gate
    :param modes: targer mode(s)
    :param n: total number of qumodes
    :return: :math:`2N`-dimensional gate vector and :math:`2N \times 2N` gate matrix
    """

    gate_matrix, gate_vector = gate.getMatrixAndVector()
    # gate_vector = gate_list.getVector()
    nmode_gate_matrix = numpy.eye(2 * n, dtype=float)
    nmode_gate_vector = numpy.zeros((2 * n, 1), dtype=float)
    if len(modes) == 1:
        mode = modes[0]
        nmode_gate_matrix[2 * mode: 2 * mode + 2, 2 * mode: 2 * mode + 2] = gate_matrix
        nmode_gate_vector[2 * mode: 2 * mode + 2] = gate_vector
    elif len(modes) == 2:
        mode_1 = modes[0]
        mode_2 = modes[1]
        nmode_gate_matrix[2 * mode_1: 2 * mode_1 + 2, 2 * mode_1: 2 * mode_1 + 2] = gate_matrix[0: 2, 0: 2]
        nmode_gate_matrix[2 * mode_1: 2 * mode_1 + 2, 2 * mode_2: 2 * mode_2 + 2] = gate_matrix[0: 2, 2: 4]
        nmode_gate_matrix[2 * mode_2: 2 * mode_2 + 2, 2 * mode_1: 2 * mode_1 + 2] = gate_matrix[2: 4, 0: 2]
        nmode_gate_matrix[2 * mode_2: 2 * mode_2 + 2, 2 * mode_2: 2 * mode_2 + 2] = gate_matrix[2: 4, 2: 4]
        nmode_gate_vector[2 * mode_1: 2 * mode_1 + 2] = gate_vector[0: 2]
        nmode_gate_vector[2 * mode_2: 2 * mode_2 + 2] = gate_vector[2: 4]

    nmode_gate_list = [nmode_gate_matrix, nmode_gate_vector]

    return nmode_gate_list


class GaussStateTransferProcessor:
    """
    Simulate gaussian state evolution.
    """

    def __init__(self, matrixType: MatrixType, algorithm: Algorithm) -> None:

        if matrixType == MatrixType.Dense:
            if algorithm == Algorithm.Matmul:
                self.proc = self.TransferGaussStateDenseMatmul

    def __call__(self, state_list: Union[List[numpy.ndarray], List['COO']],
                 gate_list: Union[List[numpy.ndarray], List['COO']], modes: numpy.ndarray) \
            -> Union[List[numpy.ndarray], List['COO']]:
        """
        To enable the object callable
        """

        return self.proc(state_list, gate_list, modes)

    def TransferGaussStateDenseMatmul(self, state_list: List[numpy.ndarray], gate_list: List[numpy.ndarray],
                                       modes: numpy.ndarray) -> List[numpy.ndarray]:
        r"""
        Given a kind of gate, we have the following relationship,

        .. math::
            \mathbf{\bar{x}} \rightarrow \mathbf{S} \mathbf{\bar{x}} + \mathbf{d}, \\
            \mathbf{V} \rightarrow \mathbf{S} \mathbf{V} \mathbf{S}^T

        where :math:`\mathbf{\bar{x}}` and :math:`\mathbf{V}` denote the first and second moments of gaussian state,
        :math:`\mathbf{d}` and :math:`\mathbf{S}` denote the gate vector and matrix,
        and :math:`T` denotes the transposition of matrix.

        :param state_list: includes the first and second moments
        :param gate_list: includes the gate vector and matrix
        :param modes: target mode(s)
        :return state_list: updated the first and second moments
        """

        state_fir_mom, state_sec_mom = state_list

        n = int(len(state_fir_mom) / 2)
        nmode_gate_matrix, nmode_gate_vector = CalcuGateMatrix(gate_list, modes, n)

        state_fir_mom = numpy.matmul(nmode_gate_matrix, state_fir_mom) + nmode_gate_vector
        nmode_gate_matrix_T = numpy.transpose(nmode_gate_matrix)
        state_sec_mom = numpy.matmul(numpy.matmul(nmode_gate_matrix, state_sec_mom), nmode_gate_matrix_T)

        state_list = [state_fir_mom, state_sec_mom]

        return state_list
