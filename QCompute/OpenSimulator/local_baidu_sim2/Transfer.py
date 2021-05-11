#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2020 Baidu, Inc. All Rights Reserved.
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
Transfer
This module transfers the global state from one into another on its worldline, according to the sequence of gates.
Two simulating methods (SIM_METHOD) are provided:
1) einsum format (EINSUM)
2) matmul format (MATMUL)
The matmul format can often accelerate calculation.
"""

import copy
from enum import IntEnum, unique
from typing import List, Union

import numpy



from QCompute.OpenSimulator.local_baidu_sim2.InitState import MatrixType


@unique
class Algorithm(IntEnum):
    """
    Two algorithms: SIM_MED = SIM_METHOD.EINSUM and SIM_MED=SIM_METHOD.MATMUL .

       EINSUM is a common and conventional method, while

       MATMUL uses tricks that can significantly reduce calculation time.
    """

    Matmul = 0
    Einsum = Matmul + 1


def _calcEinsumIndex(bits: List[int], n: int) -> str:
    """
    Calculate einsum index.
    """

    symbols = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    assert len(bits) + n <= len(symbols)

    tens_in = symbols[:n]
    tens_out = list(tens_in)
    mat_left = ''
    mat_right = ''

    # Copying bits should NOT change the order
    bits_copy = copy.deepcopy(bits)
    for pos, idx in enumerate(reversed(bits_copy)):
        mat_left += symbols[-1 - pos]
        mat_right += tens_in[-1 - idx]
        tens_out[-1 - idx] = symbols[-1 - pos]
    tens_out = ''.join(tens_out)

    return mat_left + mat_right + ',' + tens_in + '->' + tens_out


class TransferProcessor:
    """
    Calculate state evolution by gate implementation.
    """

    def __init__(self, matrixType: MatrixType, algorithm: Algorithm) -> None:
        """
        Choose an algorithm according to the parameters.
        """

        if matrixType == MatrixType.Dense and algorithm == Algorithm.Matmul:
            self.proc = self._transferStateDenseMatmul
        elif matrixType == MatrixType.Dense and algorithm == Algorithm.Einsum:
            self.proc = self._transferStateDenseEinsum
        

    def __call__(self, state: Union[numpy.ndarray, 'COO'], gate_matrix: Union[numpy.ndarray, 'COO'], bits: List[int]) -> \
    Union[numpy.ndarray, 'COO']:
        """
        :param state:
        :param gate_matrix:
        :param bits:
        :return:
        """

        return self.proc(state, gate_matrix, bits)

    def _transferStateDenseMatmul(self, state: numpy.ndarray, gate_matrix: numpy.ndarray,
                                  bits: List[int]) -> numpy.ndarray:
        """
        Essential transfer process.
        """

        n = len(state.shape)

        # Copying bits should NOT change the order
        gate_matrix = numpy.reshape(gate_matrix, 2 * len(bits) * [2])
        # source_pos = bits
        source_pos = copy.deepcopy(bits)

        source_pos = [n - 1 - idex for idex in
                      source_pos]  # The order of qubits is reversed in the storage of the computer.

        # To find the effective qubit, the index must be subtracted by n-1.
        # For example, if my CNOT acts on [0,2], I need to act on normal CNOT on [n-1, n-3].
        # But now the CNOT matrix is reversed, so it works on [n-3, n-1].
        # It should be noted that the index is indexed from top to bottom, which is the opposite of the bit level.
        # This is no longer the bit number, but actually the index number.
        bits_len = len(bits)  # the length of bits
        two_pow_bits_len = 2 ** bits_len
        source_pos = list(reversed(source_pos))  # Reverse order
        target_pos = list(range(bits_len))
        state = numpy.moveaxis(state, source_pos,
                               target_pos)  # Axes to be contracted are moved into the first positions,
        # the order of others kept.
        # Advance the required axes, into 'ac', for 'Za,ac->Zc'
        state_new_shape = [two_pow_bits_len, 2 ** (n - bits_len)]
        state = numpy.reshape(state, state_new_shape)

        gate_new_shape = [2 ** (len(gate_matrix.shape) - bits_len), two_pow_bits_len]
        gate_matrix = numpy.reshape(gate_matrix, gate_new_shape)

        state = numpy.matmul(gate_matrix, state)

        state = numpy.reshape(state, [2] * n)  # recover the shape
        state = numpy.moveaxis(state, target_pos, source_pos)

        return state

    def _transferStateDenseEinsum(self, state: numpy.ndarray, gate_matrix: numpy.ndarray,
                                  bits: List[int]) -> numpy.ndarray:
        """
        Essential transfer process.
        """

        n = len(state.shape)

        idx = _calcEinsumIndex(bits, n)
        gate_matrix = numpy.reshape(gate_matrix, 2 * len(bits) * [2])
        state = numpy.einsum(idx, gate_matrix, state, dtype=complex, casting='no')

        return state

    
