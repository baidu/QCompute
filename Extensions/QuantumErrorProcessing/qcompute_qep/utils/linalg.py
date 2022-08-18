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
The ``linalg`` module provides a set of linear algebra functions for manipulating matrices.
"""
import functools
import itertools
from collections.abc import Iterable
import numpy as np
from functools import reduce

from qcompute_qep.exceptions.QEPError import ArgumentError


def normalize(A: np.ndarray, axis: int = 0) -> np.ndarray:
    r"""Normalize the given matrix along the given axis.

    If axis=0, will normalize along the row; If axis=1, will normalize along the column.

    References: https://stackoverflow.com/questions/26248654/how-to-return-0-with-divide-by-zero

    :param A: np.ndarray, the matrix to be normalized
    :param axis: int, default to :math`0`, the axis along which the normalization carries out
    :return: np.ndarray, a normalized matrix along the axis

    **Examples**

        >>> import numpy as np
        >>> A = np.asarray([[1, 3], [3, 7]])
        >>> normalize(A)
        array([[0.25, 0.3 ],
               [0.75, 0.7 ]])
        >>> normalize(A, axis=0)
        array([[0.25, 0.3 ],
               [0.75, 0.7 ]])
        >>> normalize(A, axis=1)
        array([[0.25, 0.75],
               [0.3 , 0.7 ]])
    """
    if axis == 0:
        B = A.sum(axis=axis)[np.newaxis, :]
        A = np.divide(A, B, out=np.zeros_like(A), where=B != 0)
    else:
        B = A.sum(axis=axis)[:, np.newaxis]
        A = np.divide(A, B, out=np.zeros_like(A), where=B != 0)
    return A


def dagger(A: np.ndarray) -> np.ndarray:
    r"""Compute the conjugate transpose of the matrix.

    :param A: np.ndarray, the given matrix
    :return: np.ndarray, the conjugate transposed matrix :math:`A^\dagger`
    """
    return np.array(np.conjugate(np.transpose(A)), order="C")


def vec_to_operator(vec: np.ndarray) -> np.ndarray:
    r"""Convert a vector to the operator form.

    More precisely, implement the following mappings:

    .. math::

        \vert\psi\rangle \mapsto \vert\psi\rangle\!\langle\psi\vert

        \langle\psi\vert \mapsto \vert\psi\rangle\!\langle\psi\vert

    :param vec: np.ndarray, a state vector, either in the ket form or in the bra form
    :return: np.ndarray, the corresponding operator

    **Examples**

        >>> import numpy as np
        >>> psi = np.asarray([[1, 2]])
        >>> vec_to_operator(psi)
        array([[1, 2],
               [2, 4]])
        >>> phi = np.asarray([[2], [4]])
        >>> vec_to_operator(phi)
        array([[ 4,  8],
               [ 8, 16]])
    """
    # If the input is already a square matrix, do nothing
    if vec.shape[0] == vec.shape[1]:
        return vec
    # The input is invalid
    if min(vec.shape) != 1:
        raise ArgumentError("The input is not a vector!")

    if vec.shape[0] == 1:  # in bra form
        return dagger(vec) @ vec
    else:  # in ket form
        return vec @ dagger(vec)


def basis(n: int, i: int) -> np.ndarray:
    r"""Computational basis :math:`\vert i\rangle` of an :math:`n`-qubit system.

    The computational basis is constructed in its ket notation, i.e., a column vector of length :math:`2^n`.

    :param n: int, the number of qubits in the quantum system
    :param i: int, the index of the computational basis state, which must be in range :math:`[0, 2^n - 1]`
    :return: np.ndarray, the computational basis state, which is of size :math:`2^n\times 1`

    **Examples**

        >>> import numpy as np
        >>> basis(2, 0)
        array([[1.+0.j],
               [0.+0.j],
               [0.+0.j],
               [0.+0.j]])
    """
    # the dimension of the Hilbert space
    dim = 2 ** n
    if i not in range(dim):
        raise ValueError("The computational basis {} must be in the range [0, {}].".format(i, dim - 1))

    state = np.zeros((dim, 1), dtype=complex)
    state[i] = 1.0

    return state


def tensor(*args) -> np.ndarray:
    r"""Tensor product of all matrices in the list.

    The matrices in the argument list can be given as

    + A single matrix,
    + Two or more matrices as individual inputs,
    + A list of matrices, or
    + Mixture of single matrix and list of matrices.

    **Usage**

    .. code-block:: python
        :linenos:

        tensor(A)
        tensor(A, B)
        tensor([A, B])
        tensor([A, B], C)

    :param: the list of matrices for which the tensor product is taken
    :return: np.ndarray, tensor product of matrices in the list from left to right
    """
    matrices = []
    for arg in args:
        if isinstance(arg, np.ndarray):
            matrices.append(arg)
        elif isinstance(arg, Iterable):
            matrices.extend(arg)
        else:
            raise ArgumentError("in tensor(): incorrect argument inputs for the tensor function!")

    return functools.reduce(lambda A, B: np.kron(A, B), matrices)


def expand(local_matrix: np.ndarray, indices: Iterable, n: int, dim: int = 2) -> np.ndarray:
    r"""Expand a local matrix to a global system via tensor.

    Assume :math:`k` = len(indices), the number of subsystems in @local_matrix. It must hold that :math:`k \leq n`.
    The @local_matrix operates only on the qubits specified by @indices.
    We expand it to the whole system (composed of :math:`n` subsystems, each is :math:`d`-dimensional),
    acting trivially on the remaining subsystems.

    :param local_matrix: np.ndarray, a matrix of shape :math:`(d^k, d^k)`
    :param indices: Iterable, indices of qubits on which the local_matrix acts on. Must be sorted increasingly
    :param n: int, the total number of local systems
    :param dim: int, defaults to :math:`2` (the qubit system), the dimension of each local system.
                Assume all :math:`n` local systems have the same dimension
    :return: np.ndarray, a permuted matrix of shape :math:`(d^n, d^n)`

    **Examples**

        >>> import numpy as np
        >>> A = np.random.randint(10, size=(2, 2))
        >>> expand(A, [0], n=2)
        array([[3., 0., 5., 0.],
               [0., 3., 0., 5.],
               [3., 0., 8., 0.],
               [0., 3., 0., 8.]])
        >>> expand(A, [1], n=2)
        array([[3., 5., 0., 0.],
               [3., 8., 0., 0.],
               [0., 0., 3., 5.],
               [0., 0., 3., 8.]])

    """
    k = len(indices)
    if k > n:
        raise ValueError("The number of subsystems specified must be no larger than {}".format(n))

    # Create the global matrix, acting non-trivially only on the subsystems specified by @indices
    g_op = tensor([local_matrix] + [np.identity(dim)] * (n - k))

    # Generate the subsystem permutation indices
    indices = sorted(indices)
    perm = np.repeat(-1, n)
    for (i, idx) in enumerate(indices):
        perm[idx] = i
    perm[perm < 0] = range(k, n)

    return permute_systems(g_op, perm, dim)


def permute_systems(A: np.ndarray, perm: Iterable, dim: int = 2):
    r"""Permutes the subsystems of a matrix by the indices.

    The number of subsystems is given by :math:`n` = len(@perm), while each subsystem is :math:`d` = @dim dimensional.

    **Usage**

    Assume :math:`A_0, A_1, A_2` are qubit operators and let :math:`A = A_0\otimes A_1\otimes A_2`. Then

    .. code-block:: python
        :linenos:

        permute_systems(A, (0,2,1), 2) = A_0\otimes A_2\otimes A_1

    :param A: np.ndarray, array of shape (d^n, d^n)
    :param perm: Iterable, the permutation indices
    :param dim: int, defaults to 2, the dimension of each local system.
            We assume all n local systems have the same dimension.
    :return: a permuted matrix of shape (d^n, d^n)
    """
    # Number of local systems
    n = len(perm)
    if (A.shape[0] != dim ** n) or (A.shape[1] != dim ** n):
        raise ValueError("Input matrix A must be square and have dimension {}!".format(dim ** n))
    # Step 1. Reshape the matrix to a vector
    A_p = A.reshape([dim] * 2 * n)
    # Step 2. Permute the system
    perm_col = [idx + n for idx in perm]
    A_p = np.transpose(A_p, list(itertools.chain(perm, perm_col)))
    # Step 3. Reshape back
    A_p = A_p.reshape([dim ** n, dim ** n])

    return A_p


def partial_trace(A: np.ndarray, indices: Iterable, n: int, dim: int) -> np.ndarray:
    r"""Computes the partial trace of the given matrix.
    The partial trace may be taken on any subset of the subsystems on which the matrix acts.
    Assume :math:`k` = len(indices), which is the number of subsystems to be traced out.
    It must hold that :math:`k <= n`.

    :param A: np.ndarray, a matrix of shape :math:`(d^n, d^n)`
    :param indices: Iterable, indices of the local systems on which the trace is to be applied.
    :param n: int, the total number of the local systems.
    :param dim: int, the dimension of each local system. We assume all n local systems have the same dimension.
    :return: np.ndarray, the remaining matrix after partial trace, which has shape :math:`(d^{n-k}, d^{n-k})`
    """
    # TODO: will be implemented in the next version.
    if not indices:
        return A
    if len(indices) > n:
        raise ArgumentError('The length of indices should not be larger than n.')
    if dim ** n != A.shape[0]:
        raise ArgumentError('The local systems dimension is wrong.')

    k = len(indices)
    indices.sort()
    qubit_axis = [(i, n + i) for i in indices]
    minus_factor = [(i, dim * i) for i in range(k)]
    minus_qubit_axis = [(q[0] - m[0], q[1] - m[1])
                        for q, m in zip(qubit_axis, minus_factor)]
    res = np.reshape(A, [dim, dim] * n)
    num_preserve = n - k
    for i, j in minus_qubit_axis:
        res = np.trace(res, axis1=i, axis2=j)
    if num_preserve > 1:
        res = np.reshape(res, [dim ** num_preserve] * 2)
    return res


def is_hermitian(A: np.ndarray) -> bool:
    r"""Check if the given operator is Hermitian.

    Let :math:`A` be a square linear operator. :math:`A` is Hermitian if it holds that :math:`A= A^\dagger`.

    :param A: np.ndarray, a square linear operator
    :return: bool, if A is Hermitian, return True; otherwise return False
    """
    return np.allclose(A, dagger(A), rtol=1e-05, atol=1e-08)


def is_psd(A: np.ndarray) -> bool:
    r"""Check if the given operator is positive semidefinite.

    Let :math:`A` be a square linear operator.
    :math:`A` is positive semidefinite if it holds that :math:`\langle\psi\vert A \vert\psi\rangle \geq 0`
    for all non-zero vectors :math:`\vert\psi\rangle`. Equivalently,
    :math:`A` is positive semidefinite if all its eigenvalues are non-negative.

    :param A: np.ndarray, a square linear operator
    :return: bool, if A is positive semidefinite, return True; otherwise return False
    """
    return np.all(np.linalg.eigvals(A) >= 0)
