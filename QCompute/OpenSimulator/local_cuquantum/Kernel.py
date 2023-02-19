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
Kernal
"""
import sys
import numpy as np
import random

from QCompute.OpenSimulator.local_cuquantum import DistEinsum

if 'sphinx' not in sys.modules:
    import cupy
    import cuquantum
    from cuquantum import custatevec as cusv

    CUSV_HANDLE = cusv.create()

    # single + half precision
    # NP_DATA_TYPE = np.complex64  # todo: numpy do not support half precision complex. Fix it later.
    # CUDA_DATA_TYPE = cuquantum.cudaDataType.CUDA_C_32F
    # DistEinsum.CUDA_COMPUTE_TYPE = cuquantum.ComputeType.COMPUTE_16F


    # single precision
    NP_DATA_TYPE = np.complex64
    CUDA_DATA_TYPE = cuquantum.cudaDataType.CUDA_C_32F
    DistEinsum.CUDA_COMPUTE_TYPE = cuquantum.ComputeType.COMPUTE_32F

    # double precision
    # NP_DATA_TYPE = np.complex128
    # CUDA_DATA_TYPE = cuquantum.cudaDataType.CUDA_C_64F
    # DistEinsum.CUDA_COMPUTE_TYPE = cuquantum.ComputeType.COMPUTE_64F

# SYMBOLS = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
SYMBOLS = ''.join([chr(i) for i in range(0x100, 0x10000)])


def calc_einsum_idx(bits, n):
    """
    calculate the index parameters for einsum
    """
    assert len(bits) + n <= len(SYMBOLS)

    tens_in = SYMBOLS[:n]
    tens_out = list(tens_in)
    mat_left = ''
    mat_right = ''

    for pos, idx in enumerate(reversed(bits)):
        mat_left += SYMBOLS[-1 - pos]
        mat_right += tens_in[-1 - idx]
        tens_out[-1 - idx] = SYMBOLS[-1 - pos]
    tens_out = ''.join(tens_out)

    return mat_left + mat_right + ',' + tens_in + '->' + tens_out


def transfer_state_in_buffer(n, gate_matrix, bits, expr_buffer, gate_buffer):
    """
    state transfer in the einsum commands buffer
    """
    idx = calc_einsum_idx(bits, n)
    expr_buffer.append(idx)
    gate_buffer.append(gate_matrix)


def transfer_state_flush(state, expr_buffer, gate_buffer):
    """
    flush einsum commands buffer
    """

    # nothing in buffer
    if len(expr_buffer) == 0:
        return state

    # assign unique subscripts in expr_buffer
    symbol_count = 0
    for i in range(len(expr_buffer)):
        if expr_buffer[i].index(',') == 2:
            symbol_count += 1
            expr_buffer[i] = expr_buffer[i].replace(expr_buffer[i][0], SYMBOLS[-symbol_count])
        elif expr_buffer[i].index(',') == 4:
            symbol_count += 1
            expr_buffer[i] = expr_buffer[i].replace(expr_buffer[i][0], SYMBOLS[-symbol_count])
            symbol_count += 1
            expr_buffer[i] = expr_buffer[i].replace(expr_buffer[i][1], SYMBOLS[-symbol_count])
        else:
            assert False

    # merge expr_buffer
    a, b, c = expr_buffer[0].replace('->', ',').split(',')
    expr = [a, ',', b, '->', c]
    for e in expr_buffer[1:]:
        a, b, c = e.replace('->', ',').split(',')
        t = str.maketrans(b, expr[-1])
        a = a.translate(t)
        c = c.translate(t)
        expr = [a, ','] + expr
        expr[-1] = c
    expr = ''.join(expr)

    # run combined einsum once
    state = DistEinsum.DistEinsum(expr, *reversed(gate_buffer), state)  # fine API, dtype=NP_DATA_TYPE, casting='no'
    # state = cuquantum.einsum(expr, *reversed(gate_buffer), state)  # coarse API, dtype=NP_DATA_TYPE, casting='no'

    # combined einsum
    if DistEinsum.MPI_RANK == DistEinsum.MPI_ROOT:
        assert cupy.round(cupy.linalg.norm(state), 3) == 1.0
    else:
        state is None
    # print(idx)
    # print(state)

    expr_buffer.clear()
    gate_buffer.clear()

    return state


def init_state_10(n):
    """
    initialize the state by [1, 0, ..., 0]
    """
    if DistEinsum.MPI_RANK == DistEinsum.MPI_ROOT:
        state = cupy.zeros([2] * n, NP_DATA_TYPE)
        state.reshape(-1)[0] = 1.0
    else:
        state = None
    if DistEinsum.MPI_RANK == DistEinsum.MPI_ROOT:
        assert cupy.round(cupy.linalg.norm(state), 6) == 1.0
    else:
        state is None
    # print(state)
    return state


def init_state_rand(n):
    """
    initialize the state by random unit vector
    """
    if DistEinsum.MPI_RANK == DistEinsum.MPI_ROOT:
        state = (cupy.random.random([2] * n) * 2.0 - 1.0) + (cupy.random.random([2] * n) * 2.0 - 1.0) * 1.0j
        state /= cupy.linalg.norm(state)
    else:
        state = None
    if DistEinsum.MPI_RANK == DistEinsum.MPI_ROOT:
        assert cupy.round(cupy.linalg.norm(state), 6) == 1.0
    else:
        state is None
    # print(state)
    return state


def measure_single(n, state, bit):
    """
    measure a single qubit
    use cuquantum's measure_on_z_basis
    """

    basis_bits = np.asarray([bit], dtype=np.int32)
    rnd = random.random()

    # measurement on z basis
    parity = cusv.measure_on_z_basis(
        CUSV_HANDLE,
        state.data.ptr, CUDA_DATA_TYPE,
        n,
        basis_bits.ctypes.data, 1,
        rnd,
        cusv.Collapse.NORMALIZE_AND_ZERO
    )

    return parity


def measure_all_1(n, state):
    """
    measure all qubits, version 1
    call measure_single to measure each qubits one by one
    """
    state = state.copy()

    outs = ''
    for i in range(n):
        out = measure_single(n, state,
                             i)  # After measuring bit0, bit0 collapses. It affects the subsequent bit1 measurement, but does not affect the 1000 independent measurements of the upper layer
        outs = str(out) + outs  # from low bit to high bit
    return outs


def measure_all_2(n, state):
    """
    measure all qubits, version 2
    call cuquantum's batch_measure once
    """
    state = state.copy()

    bit_ordering = np.asarray(list(reversed(range(n))), dtype=np.int32)
    rnd = random.random()

    # batch measurement
    outs = np.zeros((n,), dtype=np.int32)
    cusv.batch_measure(
        CUSV_HANDLE,
        state.data.ptr, CUDA_DATA_TYPE,
        n,
        outs.ctypes.data,
        bit_ordering.ctypes.data, n,
        rnd,
        cusv.Collapse.NORMALIZE_AND_ZERO)
    outs = ''.join(map(str, outs))

    return outs
