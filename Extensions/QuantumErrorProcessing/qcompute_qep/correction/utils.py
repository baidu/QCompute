#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
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
Utility functions used in Quantum Error Correction.

References:

.. [R19] Joschka Roffe.
    "Quantum error correction: an introductory guide."
    Contemporary Physics 60.3 (2019): 226-245.

.. [NC10] Nielsen, Michael A., and Isaac L. Chuang.
    "Quantum Computation and Quantum Information: 10th Anniversary Edition."
    Cambridge University Press, 2010.

.. [G97] Daniel Gottesman.
    "Stabilizer Codes and Quantum Error Correction."
    PhD thesis, California Institute of Technology (1997).
"""
import numpy as np
from typing import List, Tuple
from qcompute_qep.quantum.pauli import pauli2bsf


class ColorTable:
    r"""Color table for beautifully printing quantum error correction circuits.

    Common usage:

    >>> print("{}".format(ColorTable.PHYSICAL + "Content to be Printed in Color" + ColorTable.END))
    >>> print("{}".format(ColorTable.ANCILLA + "Content to be Printed in Color" + ColorTable.END))
    """

    # https://stackoverflow.com/questions/287871/how-do-i-print-colored-text-to-the-terminal
    ORIGINAL = "\033[0;31m"     # Color code for the original qubits
    PHYSICAL = "\033[0;34m"     # Color code for the physical qubits (exclude the original qubits)
    ANCILLA = "\033[1;33m"      # Color code for the ancilla qubits
    END = '\033[0m'             # Color code for the normal text


def pauli_list_to_check_matrix(paulis: List[str]) -> np.ndarray:
    r"""Convert a list of Pauli strings to check matrix.

    The check matrix is a boolean matrix (whose elements have value `0` and `1`) of size :math:`k\times 2n`,
    where :math:`k` is the number of Pauli strings and :math:`n` is the number of single-qubit
    Pauli operators in each string.

    .. admonition:: Assumption

        The Pauli strings in the list @paulis must have the same number of single-qubit Pauli operators,
        and each Pauli operator is specified by 'I', 'X', 'Y', or 'Z'.

    :param paulis: List[str], a list of Pauli strings with the same length
    :return: np.ndarray, a :math:`k\times 2n`-dimensional check matrix

    **Examples**

        >>> cm = pauli_list_to_check_matrix(['IZZ', 'ZZI'])
        >>> print(cm)
        [[0 0 0 0 1 1]
         [0 0 0 1 1 0]]
    """
    k = len(paulis)
    n = len(paulis[0])
    check_matrix = np.zeros((k, 2 * n), dtype=np.uint)

    for i, p in enumerate(paulis):
        check_matrix[i, :] = pauli2bsf(p)

    return check_matrix


def check_matrix_to_standard_form(check_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    r"""Convert a check matrix to its standard form.

    The detailed converting procedure is summarized in Chapter 4 of [G97]_ and Section 10.5.7 of [NC10]_.
    Here we follow the notations used in the former.

    :param check_matrix: np.ndarray, a check matrix of size :math:`(n-k)\times 2n`,
            where :math:`k` is the number of logical qubits, and :math:`n` is the number of physical qubits
    :return: Tuple[np.ndarray, np.ndarray, np.ndarray, int],
                represent the standard form matrix (of size :math:`k\times 2n`),
                the logical X operators (of size :math:`k\times 2n`),
                the logical Z operators (of size :math:`k\times 2n`),
                and the rank of the X portion of the check matrix (cf. Eq. (4.1) of [G97]_).

    **Examples**

        >>> steane_code = ['XXXXIII', 'XXIIXXI', 'XIXIXIX', 'ZZZZIII', 'ZZIIZZI', 'ZIZIZIZ']
        >>> cm = pauli_list_to_check_matrix(steane_code)
        >>> print("Steane Code Check Matrix:")
        >>> print(cm)
        [[1 1 1 1 0 0 0 0 0 0 0 0 0 0]
         [1 1 0 0 1 1 0 0 0 0 0 0 0 0]
         [1 0 1 0 1 0 1 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 1 1 1 1 0 0 0]
         [0 0 0 0 0 0 0 1 1 0 0 1 1 0]
         [0 0 0 0 0 0 0 1 0 1 0 1 0 1]]
        >>> standard_form, logical_xs, logical_zs, rank = check_matrix_to_standard_form(check_matrix)
        >>> print("Steane Code Standard Form:")
        [[1 0 0 1 0 1 1 0 0 0 0 0 0 0]
         [0 1 0 1 1 0 1 0 0 0 0 0 0 0]
         [0 0 1 1 1 1 0 0 0 0 0 0 0 0]
         [0 0 0 0 0 0 0 1 1 1 1 0 0 0]
         [0 0 0 0 0 0 0 1 0 1 0 1 0 1]
         [0 0 0 0 0 0 0 0 1 1 0 0 1 1]]
        >>> print("Steane Code Logical Xs: {}".format(logical_xs))
        Steane Code Logical Xs: [[0 0 0 0 1 1 1 0 0 0 0 0 0 0]]
        >>> print("Steane Code Logical Zs: {}".format(logical_zs))
        Steane Code Logical Zs: [[0 0 0 0 0 0 0 1 1 0 0 0 0 1]]
        >>> print("Steane Code Rank: {}".format(rank))
        Steane Code Rank: 3
    """
    # Number of physical and logical qubits
    n = check_matrix.shape[1] // 2
    k = n - check_matrix.shape[0]
    std = np.copy(check_matrix)

    # Perform the first round Gaussian elimination to obtain Eq. (4.1) in [D97]_.
    r, operations = gaussian_elimination(std[:, :n])
    for op in operations:
        if op[0] == 'SWAP-ROW':
            std[[op[1], op[2]], :] = std[[op[2], op[1]], :]
        elif op[0] == 'ADD-ROW':
            std[op[1], :] = np.mod(std[op[1], :] + std[op[2], :], 2)
        elif op[0] == 'SWAP-COL':
            std[:, [op[1], op[2]]] = std[:, [op[2], op[1]]]
            std[:, [n + op[1], n + op[2]]] = std[:, [n + op[2], n + op[1]]]

    # Perform the second round Gaussian elimination to obtain Eq. (4.3) in [D97]_.
    r_e, operations = gaussian_elimination(std[r:, -(n - r):])
    assert r_e == n - k - r
    for op in operations:
        if op[0] == 'SWAP-ROW':
            std[[r + op[1], r + op[2]], :] = std[[r + op[2], r + op[1]], :]
        elif op[0] == 'ADD-ROW':
            std[r + op[1], :] = np.mod(std[r + op[1], :] + std[r + op[2], :], 2)
        elif op[0] == 'SWAP-COL':
            std[:, [-(n - r) + op[1], -(n - r) + op[2]]] = std[:, [-(n - r) + op[2], -(n - r) + op[1]]]
            std[:, [-n - (n - r) + op[1], -n - (n - r) + op[2]]] = std[:, [-n - (n - r) + op[2], -n - (n - r) + op[1]]]

    # Record the submatrices for constructing the logical operators
    e = std[r:, -k:]            # size n-k-r * k
    c_1 = std[:r, -(n - r):-k]  # size r * n-k-r
    c_2 = std[:r, -k:]          # size r * k
    a_2 = std[:r, -n - k:-n]    # size r * k

    # Construct the logical X and Z operators. See Section 4.1 of [D97]_ for more details.
    x_bar = np.concatenate([np.zeros((k, r), dtype=np.uint),
                            e.T,
                            np.identity(k, dtype=np.uint),
                            np.mod(np.dot(e.T, c_1.T) + c_2.T, 2),
                            np.zeros((k, n - r), dtype=np.uint)], axis=1)

    z_bar = np.concatenate([np.zeros((k, n), dtype=np.uint),
                            a_2.T,
                            np.zeros((k, n - k - r), dtype=np.uint),
                            np.identity(k, dtype=np.uint)], axis=1)

    return std, x_bar, z_bar, r


def gaussian_elimination(matrix: np.ndarray) -> Tuple[int, List]:
    r"""Perform Gaussian elimination on a matrix to obtain
    the `Row Echelon Form <https://en.wikipedia.org/wiki/Row_echelon_form>`_ (REF).

    Notice that when doing gaussian elimination on the X portion of the check matrix,
    the Z portion must be rearranged correspondingly.
    Thus, we also record the sequence of operations to achieve the Gaussian elimination.
    These recorded operations can later be applied to another portion of the check matrix.

    Reference: `AN ALGORITHM FOR REDUCING A MATRIX TO ROW ECHELON FORM
    <https://www.math.purdue.edu/~shao92/documents/Algorithm%20REF.pdf>`_.

    :param matrix: np.ndarray, the matrix to do Gaussian elimination
    :return: Tuple[int, List], the rank of the submatrix and the set of operations to achieve Gaussian elimination
    """
    matrix = np.copy(matrix)
    m = matrix.shape[0]
    n = matrix.shape[1]
    ops = []

    # all zeros matrix, trivial
    if np.count_nonzero(matrix) == 0:
        return 0, ops

    p_row = 0
    p_col = 0
    while True:
        # Determine the first non-zero column in the submatrix below pivot position
        # and swap columns to bring it to current column
        if np.count_nonzero(matrix[p_row:, p_col]) == 0:
            for j in range(p_col + 1, n):
                if np.count_nonzero(matrix[p_row:, j]) > 0:
                    matrix[:, [p_col, j]] = matrix[:, [j, p_col]]
                    ops.append(('SWAP-COL', p_col, j))

        # first put a 1 at matrix[p_row, j] if not already
        if matrix[p_row, p_col] != 1:
            for i in range(p_row + 1, m):
                if matrix[i, p_col] == 1:
                    matrix[[p_row, i], :] = matrix[[i, p_row], :]
                    ops.append(('SWAP-ROW', p_row, i))

        # Put zeros at each matrix[i, j] for i > p_col
        for i in range(p_row + 1, m):
            if matrix[i, p_col] == 1:
                matrix[i, :] = np.mod(matrix[i, :] + matrix[p_row, :], 2)
                ops.append(('ADD-ROW', i, p_row))

        p_row += 1
        p_col += 1
        # if no more non-zero rows below pivot, break
        non_zero_found = False
        for i in range(p_row, m):
            if np.count_nonzero(matrix[i, :]) > 0:
                non_zero_found = True
        if not non_zero_found:
            break

    rank = p_row
    # Finally convert to RREF form
    for j in range(rank):
        for i in range(j):
            if matrix[i, j] == 1:
                matrix[i, :] = np.mod(matrix[i, :] + matrix[j, :], 2)
                ops.append(('ADD-ROW', i, j))

    return rank, ops
