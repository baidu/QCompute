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
Auxiliary functions for quantum circuit based on gaussian state
"""
FileErrorCode = 18


import numpy
from typing import Tuple

def TraceOneMode(fir_mom: numpy.ndarray, sec_mom: numpy.ndarray, mode_trace: int) \
        -> Tuple[list[numpy.ndarray], list[numpy.ndarray]]:
    """
    Perform partial trace on input quantum state
    and get the first and second moments sorted in terms of the sequence of rest quantum state and traced qumode.

    :param fir_mom: the first moment
    :param sec_mom: the second moment
    :param mode_trace: index of traced qumode
    :return vector_list: the first moments of rest quantum state and traced qumode, respectively
    :return matrix_list: the second moments of rest quantum state and traced qumode, respectively
                         as well as the correlation matrix.
    """

    # Calculate the number of qumode in current quantum circuit
    num_mode = int(len(fir_mom) / 2)
    # Calculate the number of the rest qumodes
    num_mode_rest = num_mode - 1

    # The initial matrix of the first and second moment of traced qumodes
    vector_b = numpy.zeros((2, 1))
    matrix_B = numpy.zeros((2, 2))
    # The initial matrix of the first and second moment of rest qumodes
    vector_a = numpy.zeros((2 * num_mode_rest, 1))
    matrix_A = numpy.zeros((2 * num_mode_rest, 2 * num_mode_rest))
    # The initial correlated matrix of rest and trace qumodes.
    matrix_C = numpy.zeros((2 * num_mode_rest, 2))

    # An array of the sequence number of rest qumodes
    mode_all = numpy.arange(num_mode)
    mode_rest = numpy.setdiff1d(mode_all, mode_trace)

    # Obtain the matrix of block A, B, C
    for new_row in range(num_mode_rest):
        old_row = mode_rest[new_row]
        vector_a[(2 * new_row): (2 * new_row + 2), 0] = fir_mom[(2 * old_row): (2 * old_row + 2), 0]

        for new_column in range(num_mode_rest):
            old_column = mode_rest[new_column]
            matrix_A[(2 * new_row): (2 * new_row + 2), (2 * new_column): (2 * new_column + 2)] = \
                sec_mom[(2 * old_row): (2 * old_row + 2), (2 * old_column): (2 * old_column + 2)]

    vector_b[0: 2, 0] = fir_mom[(2 * mode_trace): (2 * mode_trace + 2), 0]
    matrix_B[0: 2, 0: 2] = sec_mom[(2 * mode_trace): (2 * mode_trace + 2), (2 * mode_trace): (2 * mode_trace + 2)]

    for new_row in range(num_mode_rest):
        old_row = mode_rest[new_row]
        matrix_C[(2 * new_row): (2 * new_row + 2), 0: 2] = \
            sec_mom[(2 * old_row): (2 * old_row + 2), (2 * mode_trace): (2 * mode_trace + 2)]

    vector_list = [vector_a, vector_b]
    matrix_list = [matrix_A, matrix_B, matrix_C]

    return vector_list, matrix_list


def TensorProduct(fir_mom: numpy.ndarray, sec_mom: numpy.ndarray, mode_tensor: int) \
        -> Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Calculate the tensor product on input quantum state and a single-qumode.

    :param fir_mom: the first moment
    :param sec_mom: the second moment
    :param mode_tensor: index of the single qumode
    :return: the first and second moments of the quantum state after introducing a single-qumode
    """

    num_mode = int(fir_mom.size / 2)
    num_mode_total = num_mode + 1

    if mode_tensor >= num_mode_total:
        assert False

    if num_mode == 0:
        new_fir_mom = numpy.zeros((2, 1))
        new_sec_mom = numpy.eye(2)
    else:
        if mode_tensor == num_mode_total - 1:
            new_fir_mom = numpy.append(fir_mom, numpy.array([[0],
                                                             [0]]), axis=0)
            new_sec_mom = numpy.eye(2 * num_mode_total)
            new_sec_mom[0: 2 * num_mode, 0: 2 * num_mode] = sec_mom
        else:
            mode_rearrange = numpy.arange(num_mode)
            new_fir_mom = numpy.zeros((2 * num_mode_total, 1))
            for index in range(mode_tensor, num_mode):
                mode_rearrange[index] += 1

            new_sec_mom = numpy.eye(2 * num_mode_total)
            for old_row in range(num_mode):
                new_row = mode_rearrange[old_row]
                new_fir_mom[2 * new_row: 2 * new_row + 2, 0] = fir_mom[2 * old_row: 2 * old_row + 2, 0]
                for old_col in range(num_mode):
                    new_col = mode_rearrange[old_col]
                    new_sec_mom[2 * new_row: 2 * new_row + 2, 2 * new_col: 2 * new_col + 2] = \
                        sec_mom[2 * old_row: 2 * old_row + 2, 2 * old_col: 2 * old_col + 2]

    return new_fir_mom, new_sec_mom


def CalculateP0_and_H(fir_mom: numpy.ndarray, sec_mom: numpy.ndarray, num_cutoff: int) -> Tuple[float, numpy.ndarray]:
    r"""
    Get the factor :math:`P_0` and hermite polynomials

    :param fir_mom: the first moment
    :param sec_mom: the second moment
    :param num_cutoff: the resolution of photon-count detector
    :return P0: a factor of probability tensor
    :return H_2N: hermite polynomials
    """

    (fir_mom_Q_2N, sec_mom_M_2N) = xpxp2ppxx(fir_mom, sec_mom)

    num_mode = int(len(fir_mom) / 2)
    identity_mat_N = numpy.eye(num_mode)
    identity_mat_2N = numpy.eye(2 * num_mode)
    unitary_mat_2N = 1 / numpy.sqrt(2) * numpy.block([[-1j * identity_mat_N, 1j * identity_mat_N],
                                                      [identity_mat_N, identity_mat_N]])

    I_mins_2M = identity_mat_2N - 2 * sec_mom_M_2N
    I_plus_2M = identity_mat_2N + 2 * sec_mom_M_2N

    P0 = CalculateP0(fir_mom_Q_2N, I_plus_2M)
    (R_2N, y_2N) = CalculateR_and_y(fir_mom_Q_2N, unitary_mat_2N, I_mins_2M, I_plus_2M)
    H_2N = HermitePolynomials(R_2N, y_2N, num_cutoff)

    return P0, H_2N


def CalculateP0(fir_mom_Q_2N: numpy.ndarray, I_plus_2M: numpy.ndarray) -> float:
    r"""
    Calculate the factor of probability tensor, :math:`P_0`

    :param fir_mom: the first moment
    :param I_plus_2M: an :math:`2N \times 2N` matrix
    :return P0: a factor of probability tensor
    """

    P0_1 = numpy.power(numpy.linalg.det(0.5 * I_plus_2M), -0.5)
    P0_2 = numpy.linalg.pinv(I_plus_2M)
    P0_3 = -numpy.matmul(numpy.matmul(numpy.transpose(fir_mom_Q_2N), P0_2), fir_mom_Q_2N)

    P0 = P0_1 * numpy.exp(P0_3)

    return P0[0, 0]


def CalculateR_and_y(fir_mom_Q_2N: numpy.ndarray, unitary_mat_2N: numpy.ndarray,
                      I_mins_2M: numpy.ndarray, I_plus_2M: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Calculate two auxiliary quantities :math:`R` and :math:`y`,
    which are :math:`2N \times 2N` matrix and :math:`2N`-dimensional vector, respectively

    :param fir_mom_Q_2N: the first moment
    :param unitary_mat_2N: an :math:`2N \times 2N` unitary matrix
    :param I_mins_2M: an :math:`2N \times 2N` matrix
    :param I_plus_2M: an :math:`2N \times 2N` matrix
    :return R_2N: an :math:`2N \times 2N` auxiliary matrix
    :return y_2N: an :math:`2N`-dimensional auxiliary vector
    """

    unitary_mat_T = numpy.transpose(unitary_mat_2N)
    unitary_mat_C = numpy.conjugate(unitary_mat_2N)
    unitary_mat_H = numpy.transpose(unitary_mat_C)

    R_left = numpy.matmul(unitary_mat_H, I_mins_2M)
    R_right = numpy.matmul(numpy.linalg.pinv(I_plus_2M), unitary_mat_C)

    R_2N = numpy.matmul(R_left, R_right)
    y_2N = numpy.matmul(numpy.matmul(2 * unitary_mat_T, numpy.linalg.pinv(I_mins_2M)), fir_mom_Q_2N)

    return R_2N, y_2N


def HermitePolynomials(R_2N: numpy.ndarray, y_2N: numpy.ndarray, num_cutoff: int) -> numpy.ndarray:
    r"""
    Calculate hermite polynomials

    :param R_2N: an :math:`2N \times 2N` matrix
    :param y_2N: an :math:`2N`-dimensional vector
    :param num_cutoff: the resolution of photon-count detector
    :return H_2N: hermite polynomials
    """

    dim_y = len(y_2N)
    H_2N = numpy.zeros(dim_y * [num_cutoff + 1], dtype=complex)

    for latter_coor_tuple, _ in numpy.ndenumerate(H_2N):
        latter_coor_array = numpy.array(latter_coor_tuple)
        if max(latter_coor_array) == 0:
            H_2N[latter_coor_tuple] = 1
            former_coor_array = latter_coor_array
            continue

        for index in range(dim_y - 1, -1, -1):
            if latter_coor_array[index] > former_coor_array[index]:
                k = index
                if k == dim_y - 1:
                    break
                else:
                    former_coor_array[k + 1:] = 0
                    break

        H2 = H3 = 0
        sum_R_by_y = 0
        for j in range(dim_y):
            sum_R_by_y += R_2N[k, j] * y_2N[j, 0]
            if former_coor_array[j] > 0:
                former_coor_j = numpy.copy(former_coor_array)
                former_coor_j[j] -= 1
                if k != j:
                    H2 += R_2N[k, j] * latter_coor_array[j] * H_2N[tuple(former_coor_j)]
                else:
                    H3 = R_2N[k, k] * (latter_coor_array[j] - 1) * H_2N[tuple(former_coor_j)]
            else:
                continue

        H1 = sum_R_by_y * H_2N[tuple(former_coor_array)]
        H_2N[latter_coor_tuple] = H1 - H2 - H3
        former_coor_array = latter_coor_array

    return H_2N.real


def xpxp2xxpp(fir_mom: numpy.ndarray, sec_mom: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Covert :math:`\left[ x_1, p_1, x_2, p_2, \dots, x_N, p_N \right]^T`
    into :math:`\left[ x_1, x_2, \dots, x_N, p_1, p_2, \dots, p_N \right]^T`

    :param fir_mom: original first moment
    :param sec_mom: original second moment
    :return: updated first and second moment
    """

    num_mode = int(len(fir_mom) / 2)
    permutation_mat = numpy.zeros((2 * num_mode, 2 * num_mode), dtype=int)

    for index in range(num_mode):
        permutation_mat[index, 2 * index] = permutation_mat[index + num_mode, 2 * index + 1] = 1

    permutation_mat_T = numpy.transpose(permutation_mat)

    fir_mom = numpy.matmul(permutation_mat, fir_mom)
    sec_mom = numpy.matmul(numpy.matmul(permutation_mat, sec_mom), permutation_mat_T)

    return fir_mom, sec_mom


def xpxp2ppxx(fir_mom: numpy.ndarray, sec_mom: numpy.ndarray) -> Tuple[numpy.ndarray, numpy.ndarray]:
    r"""
    Covert :math:`\left[ x_1, p_1, x_2, p_2, \dots, x_N, p_N \right]^T`
    into :math:`\left[ p_1, p_2, \dots, p_N, x_1, x_2, \dots, x_N \right]^T`

    :param fir_mom: original first moment
    :param sec_mom: original second moment
    :return: updated first and second moment
    """

    num_mode = int(len(fir_mom) / 2)
    permutation_mat = numpy.zeros((2 * num_mode, 2 * num_mode), dtype=int)

    for index in range(num_mode):
        permutation_mat[index, 2 * index + 1] = permutation_mat[index + num_mode, 2 * index] = 1

    permutation_mat_T = numpy.transpose(permutation_mat)

    fir_mom = 1 / numpy.sqrt(2) * numpy.matmul(permutation_mat, fir_mom)
    sec_mom = 1 / 2 * numpy.matmul(numpy.matmul(permutation_mat, sec_mom), permutation_mat_T)

    sec_mom = (1 - 1e-15) * sec_mom

    return fir_mom, sec_mom