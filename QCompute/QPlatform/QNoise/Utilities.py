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
Utilities
"""
FileErrorCode = 31

from typing import Optional, List
import numpy as np
import math
from itertools import product

def sigma(k: int) -> np.ndarray:
    r"""
    The identity operator and pauli operators, which are usually indicated by the letter :math:`\sigma`.
    
    :param: int, the index of sigma operators which is in range(4).

    Matrix form:

    :math:`\sigma_0 \equiv ID = \begin{bmatrix}     1.0 & 0.0 \\     0.0 & 1.0   \end{bmatrix}`,

    :math:`\sigma_1 \equiv X = \begin{bmatrix}     0.0 & 1.0 \\     1.0 & 0.0    \end{bmatrix}`,

    :math:`\sigma_2 \equiv Y = \begin{bmatrix}     0.0 & -1.0j \\   1.0j & 0.0   \end{bmatrix}`,

    :math:`\sigma_3 \equiv Z = \begin{bmatrix}     1.0 & 0.0 \\     0.0 & -1.0   \end{bmatrix}`.
    """
    if k == 0:
        return np.eye(2)
    elif k == 1:
        return np.array([[0.0, 1.0], [1.0, 0.0]])
    elif k == 2:
        return np.array([[0.0, -1.0j], [1.0j, 0.0]])
    elif k == 3:
        return np.array([[1.0, 0.0], [0.0, -1.0]])

def numpyMatrixToTensorMatrix(numpyMatrix: np.ndarray) -> np.ndarray:
    """
    Covert the matrix from numpy format to tensor format. Must be square matrix.

    :param numpyMatrix: np.ndarray, a matrix of numpy format

    :return: np.ndarray, a matrix of tensor format
    """
    old_shape = numpyMatrix.shape
    new_shape = [2, 2] * (int(math.log(old_shape[0], 2)))
    return np.reshape(numpyMatrix, new_shape)


def tensorMatrixToNumpyMatrix(tensorMatrix: np.ndarray) -> np.ndarray:
    """
    Covert the matrix form tensor format to numpy format. Must be square matrix.

    :param tensorMatrix: np.ndarray, a matrix of tensor format

    :return: np.ndarray, a matrix of numpy format
    """
    bits = int(len(tensorMatrix.shape) / 2)
    new_shape = [2 ** bits] * 2
    return np.reshape(tensorMatrix, new_shape)


def noiseTensor(krausList1: List[np.ndarray], krausList2: List[np.ndarray]) -> List[np.ndarray]:
    """
    Tensor two noises, whose Kraus operators are KrausList1 and KrausList2, respectively.

    :param krausList1: List[np.ndarray], Kraus operator of noise 1

    :param krausList2: List[np.ndarray], Kraus operator of noise 2

    :return: List[np.ndarray], a list of Kraus operators
    """

    krausList = [np.kron(krausList1[index_1], krausList2[index_2])
                 for index_1, index_2 in product(range(len(krausList1)), range(len(krausList2)))]

    return krausList


def isTracePreserving(krauses: List[np.ndarray], probabilities: Optional[List[float]] = None) -> bool:
    r"""
    Verify the input Kraus operators satisfy TP (i.e., trace preserving)

    :param krauses: list(np.ndarray), the Kraus operators of a noise

    :param probabilities: Optional[List[float], default None, the probabilities correspond to the Kraus operators

    :return: bool, true for yes and false for no

    Matrix form:

    :math:`\sum_k E_k^{\dagger} E_k = ID`

    Here, :math:`E_k` is a Kraus operator, :math:`ID` is the identity matrix.
    """
    sumPositiveKraus = 0.0 + 0.0j
    for index in range(len(krauses)):
        if krauses[index].shape[0] == 2:  # whether Kraus is numpy format or tensor format
            krauses[index] = tensorMatrixToNumpyMatrix(krauses[index])

        if probabilities:
            sumPositiveKraus += probabilities[index] * \
                krauses[index].T.conjugate() @ krauses[index]
        else:
            sumPositiveKraus += krauses[index].T.conjugate() @ krauses[index]

    identityMatrix = np.eye(krauses[0].shape[1])
    return np.allclose(sumPositiveKraus, identityMatrix)


def _proportionalUnitary(Matrix: np.ndarray) -> bool:
    r"""
    Verity a matrix whether it is proportional to a unitary or not.

    :param Matrix: np.ndarray, a matrix of numpy format

    :return: bool, true for yes and false for no

    Works for non-zero numpy matrix.
    """
    temp_operator = Matrix.T.conjugate() @ Matrix
    normalized_matrix = _normMatrix(temp_operator)
    identityMatrix = np.eye(Matrix.shape[0])
    return np.allclose(normalized_matrix, identityMatrix)


def _normMatrix(Matrix: np.ndarray) -> np.ndarray:
    r"""
    Calculate a normalized Matrix.

    :param Matrix: np.ndarray, a matrix of numpy format

    :return: np.ndarray, a noramlized matrix

    :math:`\hat{M} = M / \text{Tr}[M^{\dagger} M]`

    Works for non-zero numpy matrix.
    """
    norm_matrix = np.trace(Matrix) / Matrix.shape[0]
    return Matrix / norm_matrix


def calcKrausLowerBound(krauses: List[np.ndarray]) -> List[float]:
    r"""
    Calculate the lower bound of probabilities for sampling among a set of Kraus operators.

    :param krauses: List[np.ndarray], the Kruas operator of a noise

    :math:`\lambda(E_k^{\dagger} E_k) \leq \text{Tr}[E_k^{\dagger}E_k \rho]`

    where :math:`E_k` is a Kraus operator, :math:`\rho` is an arbitrary quantum state, and 
    :math:`\lambda(E)` is the eigenvalue of matrix E.
    """
    lowerBound = []
    for _ in range(len(krauses)):
        kraus = krauses[_]
        if krauses[_].shape[0] == 2:  # whether Kraus is numpy format or tensor format
            kraus = tensorMatrixToNumpyMatrix(kraus)

        tempOperator = np.dot(
            kraus.T.conjugate(), kraus)

        tempBound = min(list(np.linalg.eig(tempOperator)[0]))
        lowerBound.append(tempBound)

    return lowerBound