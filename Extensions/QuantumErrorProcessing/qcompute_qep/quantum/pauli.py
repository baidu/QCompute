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
This file aims to collect functions related to the Pauli basis.
"""
from __future__ import annotations

import numpy as np
from math import log
from typing import List, Union, Tuple
import scipy
import itertools
import random
import functools

from QCompute import *
from qcompute_qep.utils.types import QProgram
from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.quantum.channel import QuantumChannel, qc_convert, PTM
from qcompute_qep.utils.linalg import tensor, is_hermitian, dagger
from copy import deepcopy
from QCompute.QPlatform.QOperation import CircuitLine

# The qubit Pauli basis. Notice that these operators are properly normalized.
QUBIT_PAULI_BASIS = {"I": np.array([[1, 0], [0, 1]]).astype(complex) / np.sqrt(2),
                     "X": np.array([[0, 1], [1, 0]]).astype(complex) / np.sqrt(2),
                     "Y": np.array([[0, -1j], [1j, 0]]).astype(complex) / np.sqrt(2),
                     "Z": np.array([[1, 0], [0, -1]]).astype(complex) / np.sqrt(2)}


class Pauli:
    r"""The Pauli Operator Class.

    The philosophy for the Pauli class is that, each Pauli operator is uniquely defined by its name,
    i.e., string of the form ``XIYX``, where the identity should not be omitted.
    Thus, when initialize a Pauli instance, we must specify its name.
    On the other hand, its matrix representation (sparse or dense) can be parsed from its name.

    We assume the LSB (the least significant bit) mode when defining the Pauli operator.
    That is, the right-most bit of string represents q[0]:

    ::

        name:           'X      I     Y      X'

        qubits:         q[3]  q[2]   q[1]   q[0]


    As so, the matrix representation of the Pauli operator ``IX`` is given by

    .. math:: \sigma_X\otimes\sigma_I = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\otimes\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
                                      = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}.

    """
    def __init__(self, name: str, sparse: bool = False):
        r"""
        init function of the Pauli operator class.
        Notice that the identity "I" in the name, if exists, should not be omitted.
        That is to say, "XX", "IXX", "XIX" and "XXI" are four completely different Pauli operators.

        Notice that we assume the LSB (the least significant bit) mode, i.e., the right-most bit represents q[0]:

        ::

            name:           'X      I     Y      X'

            qubits:         q[3]  q[2]   q[1]   q[0]

        Example: the two-qubit complete Pauli basis are:

        + "II", "IX", "IX", "IZ"
        + "XI", "XX", "XX", "XZ"
        + "YI", "YX", "YX", "YZ"
        + "ZI", "ZX", "ZX", "ZZ"

        :param name: str, the name of a Pauli operator, must be the form 'XXYX'/"XXYI"/"xxxy"/"XyxyZ".
                We convert all lower cases to the upper cases.
        :param sparse: bool, indicating the matrix of the Pauli operator should be sparse or not
        """
        # Check if name can represent a Pauli operator nor not
        for ch in name:
            if ch not in list(QUBIT_PAULI_BASIS):
                raise ArgumentError("in Pauli.__init__(): '{}' is not an valid Pauli name!".format(name))

        self._name: str = name.upper()
        self._sparse: bool = sparse

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def sparse(self):
        return self._sparse

    @sparse.setter
    def sparse(self, value):
        self._sparse = value

    @property
    def matrix(self):
        r"""
        Get the matrix representation of the Pauli operator.
        If self._sparse is True, then the matrix is represented sparsely in `scipy.sparse.csr_matrix` type.
        """
        return from_name_to_matrix(name=self._name, sparse=self._sparse)

    @property
    def size(self):
        r"""
        Get the size, i.e. the number of qubits, of the Pauli operator
        """
        return len(self._name)

    @property
    def bsf(self) -> np.ndarray:
        r"""Thr binary symplectic form of the Pauli operator.

        **Example**

        ::

            "XIZIY" -> (1 0 0 0 1 | 0 0 1 0 1)

        :return: numpy.array, (1d) Binary symplectic representation of Pauli
        """
        p = self.name
        ps = np.array(list(p))
        xs = (ps == 'X') + (ps == 'Y')
        zs = (ps == 'Z') + (ps == 'Y')
        return np.hstack((xs, zs)).astype(int)

    def sub_pauli(self, indices: List[int]) -> Pauli:
        r"""Extract the sub Pauli operator given by the qubits list.

        :param indices: List[int], list of qubit indices for which
        :return: Pauli, the Pauli operator corresponds to the qubits list.
        """
        # Check the validity of the indices of qubits
        if max(indices) >= self.size:
            raise ArgumentError("in sub_pauli(), the qubits list is out of range!")

        # Guarantee the indices are increasingly sorted
        indices = np.unique(indices)
        indices = sorted(indices)

        sub_name = ''.join([self._name[i] for i in indices])
        return Pauli(sub_name)

    def eigenvalues(self) -> np.ndarray:
        r"""Compute the eigenvalues of the Pauli operator.

        :return: np.ndarray, a diagonalized matrix storing the eigenvalues
        """
        eigs = [np.diag([1, 1]) if ch == 'I' else np.diag([1, -1]) for ch in self._name.upper()]
        return tensor(eigs)

    def meas_circuit(self, qp: QProgram, qubits: List[int] = None) -> Tuple[QProgram, np.ndarray]:
        r"""Modify the quantum circuit to implement the Pauli measurement.

        :param qp: QProgram, a quantum circuit instance that will be modified to implement the Pauli measurement
        :param qubits: List[int], the target qubit(s)
        :return: Tuple, the modified QProgram with Pauli measurement and corresponding observable operator
        """
        if qubits is None:
            qubits = [i for i in range(len(qp.Q.registerMap.keys()))]
        else:
            qubits.sort()

        qp_new = deepcopy(qp)
        for i, s in enumerate(reversed(self.name)):
            qubit_idx = qubits[i]
            if s == 'X':
                H(qp_new.Q[qubit_idx])
            elif s == 'Y':
                SDG(qp_new.Q[qubit_idx])
                H(qp_new.Q[qubit_idx])

        # Measurement in the Z basis
        qreglist, indexlist = qp_new.Q.toListPair()
        MeasureZ(qRegList=[qreglist[x] for x in qubits],
                 cRegList=[indexlist[x] for x in qubits])

        temp_list = []

        for i in self.name:
            if i == 'I':
                temp = np.diag([1, 1])
            else:
                temp = np.diag([1, -1])
            temp_list.append(temp)

        final_expval_cal = functools.reduce(np.kron, temp_list)
        return qp_new, final_expval_cal

    def preps_circuits(self, qp: QProgram, qubits: List = None) -> Tuple[List[QProgram], np.ndarray]:
        r"""Modify the quantum circuit prepare the Pauli eigenstates.

        :param qp: QProgram, a quantum circuit instance that will be modified to prepare the eigenstates
        :param qubits: List[int], the target qubit(s)
        :return: Tuple, a list of modified QPrograms such that each prepares an eigenstate
        """
        if qubits is None:
            qubits = [i for i in range(len(qp.Q.registerMap.keys()))]
        else:
            qubits.sort()

        new_qps: List[QProgram] = [deepcopy(qp)]
        for i, s in enumerate(reversed(self.name)):
            # copy all prepare circuit
            qubit_idx = qubits[i]
            copy_qps = deepcopy(new_qps)
            if s == 'I' or s == 'Z':
                new_gate1 = []
                new_gate2 = [CircuitLine(data=X, qRegList=[qubit_idx])]
            elif s == 'X':
                new_gate1 = [CircuitLine(data=H, qRegList=[qubit_idx])]
                new_gate2 = [CircuitLine(data=X, qRegList=[qubit_idx]), CircuitLine(data=H, qRegList=[qubit_idx])]
            elif s == 'Y':
                new_gate1 = [CircuitLine(data=H, qRegList=[qubit_idx]), CircuitLine(data=S, qRegList=[qubit_idx])]
                new_gate2 = [CircuitLine(data=X, qRegList=[qubit_idx]), CircuitLine(data=H, qRegList=[qubit_idx]),
                             CircuitLine(data=S, qRegList=[qubit_idx])]
            else:
                raise ArgumentError("in prep_circuits(): illegal preparation basis name {}!".format(s))

            for j in range(len(new_qps)):
                # add single-qubit gate to the front of circuit corresponding to the Pauli basis
                new_qps[j].circuit = new_gate1 + new_qps[j].circuit
                copy_qps[j].circuit = new_gate2 + copy_qps[j].circuit
            new_qps = new_qps + copy_qps

        temp_list = []
        for i in self.name:
            if i == 'I':
                temp = np.array([1, 1])
            else:
                temp = np.array([1, -1])
            temp_list.append(temp)

        final_expval_cal = functools.reduce(np.kron, temp_list)
        return new_qps, final_expval_cal


def from_name_to_matrix(name: str, sparse: bool = False) -> Union[np.ndarray, scipy.sparse.csr_matrix]:
    r"""Construct the matrix representation of the Pauli operator from its name.

    We assume the LSB (the least significant bit) mode when constructing the matrix from the Pauli string.
    That is, the right-most bit of string represents q[0]:

    ::

        name:           'X      I     Y      X'

        qubits:         q[3]  q[2]   q[1]   q[0]


    As so, the matrix of the Pauli string ``IX`` is given by

    .. math:: \sigma_X\otimes\sigma_I = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\otimes\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
                                      = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \\ 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}.

    :param name: str, the name of the Pauli operator, must be the form 'XXYX'/"XXYI"/"xxxy"/"XyxyZ".
                We convert all lower cases to the upper cases.
    :param sparse: bool, indicating the matrix of the Pauli operator should be sparse or not
    :return: Union[np.ndarray, sparse.csr_matrix], the matrix representation of the Pauli operator
    """
    pauli_operators = [QUBIT_PAULI_BASIS.get(ch) for ch in name[::-1].upper()]
    mat = tensor(pauli_operators)
    if sparse:
        return scipy.sparse.csr_matrix(mat)
    else:
        return mat


def complete_pauli_basis(n: int) -> List[Pauli]:
    r"""The complete Pauli basis of the :math:`n`-qubits system.

    The complete Pauli basis is a list of Pauli instances and is sorted in the alphabetical order of the Pauli name.
    We assume the LSB (the least significant bit) mode when constructing the complete Pauli basis.
    That is, the right-most bit of string represents q[0]:

    ::

        name:           'X      I     Y      X'

        qubits:         q[3]  q[2]   q[1]   q[0]

    As so, the set of a :math:`2`-qubit complete Pauli basis is

    ::

        {'II', 'XI', 'YI', 'ZI', 'IX', 'XX', 'YX', 'ZX', 'IY', 'XY', 'YY', 'ZY', 'IZ', 'XZ', 'YZ', 'ZZ'}

    :param n: int, the number of qubits
    :return: List[Pauli], the complete list of Pauli operators
    """
    pauli_names = itertools.product(list(QUBIT_PAULI_BASIS), repeat=n)

    pauli_basis = []
    for name in pauli_names:
        name = "".join(name)[::-1]
        pauli_basis.append(Pauli(name))

    return pauli_basis


def random_pauli_operator(n: int) -> Pauli:
    r"""Randomly generate an :math:`n`-qubit Pauli operator.

    :param n: int, number of qubits of the generated Pauli operator
    :return: Pauli, a Pauli operator instance
    """
    name = ''
    for i in range(n):
        name = name + random.choice(list(QUBIT_PAULI_BASIS.keys()))
    return Pauli(name)


def operator_to_ptm(A: np.ndarray) -> List[float]:
    r"""From the standard matrix representation to the Pauli transfer matrix.

    Notice that a quantum state is a special kind of operator with the following two constraints:
    1. the operator has trace equal to one.
    2. the operator is positive semidefinite.

    **Examples**

        In the following, we compute the PTM of the pure state :math:`\vert 0\rangle\!\langle0\vert`.

        >>> A = np.array([[1, 0], [0, 0]]).astype(complex)
        >>> ptm = operator_to_ptm(A)
        >>> print(ptm)
        [0.7071067811865475, 0.0, 0.0, 0.7071067811865475]

    :param A: np.ndarray, an :math:`n`-qubit Hermitian operator
    :return: List[float], the list of coefficients of A expanded in the Pauli matrices
    """
    if not np.log2(A.shape[0]).is_integer():
        raise ArgumentError("in operator_to_ptm(): the dimensions of the input matrix must be the power of 2!")

    n = int(np.log2(A.shape[0]))
    pauli_basis = complete_pauli_basis(n)
    coes = []
    for pauli in pauli_basis:
        coe = np.trace(A @ pauli.matrix)
        coes.append(np.real(coe))

    return coes


def ptm_to_operator(coes: Union[List[float], np.ndarray]) -> np.ndarray:
    r"""From the Pauli transfer matrix to the standard matrix representation.

    :param coes: Union[List[float], np.ndarray], the Pauli transfer matrix representation of an operator
    :return: np.ndarray, an n-qubit Hermitian linear operator

    **Examples**

        Complementary to the Examples in `operator_to_ptm`,
        in the following, we compute the pure state :math:`\vert 0\rangle\!\langle0\vert` given its PTM.

        >>> coes = [0.7071067811865475, 0.0, 0.0, 0.7071067811865475]
        >>> rho = operator_to_ptm(coes)
        >>> print(rho)
        [[1.+0.j 0.+0.j]
         [0.+0.j 0.+0.j]]
    """
    if isinstance(coes, List):
        coes = np.asarray(coes)

    # Number of qubits
    n = int(log(coes.size, 4))
    pauli_basis = complete_pauli_basis(n)
    op = np.zeros((2 ** n, 2 ** n))

    for i in range(coes.size):
        op = op + coes[i] * pauli_basis[i].matrix

    return op


def unitary_to_ptm(unitary: np.ndarray) -> QuantumChannel:
    r"""From a process unitary to its Pauli transfer matrix.

    Convert a process unitary in the matrix representation to the Pauli transfer matrix (PTM) representation.
    Assume the unitary has :math:`n` qubits, then the corresponding PTM is of size :math:`4^n\times 4^n`.

    **Examples**

        In the following, we compute the PTM of the :math:`X` gate.

        >>> X = np.array([[0, 1], [1, 0]]).astype(complex) / np.sqrt(2)
        >>> ptm = unitary_to_ptm(X)
        >>> print(ptm.data)
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j -1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j -1.+0.j]]

    :param unitary: np.ndarray, the matrix representation of the quantum process unitary
    :return: np.ndarray, the Pauli transfer matrix representation of the unitary
    """
    if not np.log2(unitary.shape[0]).is_integer():
        raise ArgumentError("in unitary_to_ptm(): the dimensions of the unitary must be the power of 2!")
    # Number of qubits
    n = int(np.log2(unitary.shape[0]))
    cpb = complete_pauli_basis(n)
    ptm = np.zeros((len(cpb), len(cpb)), dtype=float)
    for j, pauli_j in enumerate(cpb):
        for k, pauli_k in enumerate(cpb):
            ptm[j, k] = np.real(np.trace(pauli_j.matrix @ unitary @ pauli_k.matrix @ dagger(unitary)))

    return PTM(ptm)


def ptm_to_process(ptm: QuantumChannel, type: str = 'kraus') -> QuantumChannel:
    r"""From the Pauli transfer matrix of a quantum process to the given representation.

    Convert a quantum process in the Pauli transfer matrix representation to the target representation.
    Candidate target representations are:

    + str = ``superoperator``, the superoperator representation, also known as the natural representation;
    + str = ``choi``, the Choi representation;
    + str = ``kraus``, the Kraus operator representation, also known as the operator-sum representation;
    + str = ``chi``, the chi matrix representation, aka. the process matrix representation.

    :param ptm: np.ndarray, the Pauli transfer matrix representation of the quantum process
    :param type: str, default to 'kraus', type: str = 'kraus'
    :return: QuantumChannel, a quantum channel class instance
    """
    return qc_convert(ptm, type)


def bsp(a: np.ndarray, b: np.ndarray) -> int:
    r"""The binary symplectic product of two matrices.

    The binary symplectic product :math:`\odot` is defined as

    .. math::   A \odot B \equiv A \Lambda B \bmod 2,

    where

    .. math::  \Lambda = \left[\begin{matrix} 0 & I \\ I & 0 \end{matrix}\right].

    :param a: np.ndarray, LHS binary symplectic vector
    :param b: np.ndarray, RHS binary symplectic vector or matrix
    :return: int, the binary symplectic product of A with B (0 or 1)
    """
    assert np.array_equal(a % 2, a), 'BSF {} is not in binary form'.format(a)
    assert np.array_equal(b % 2, b), 'BSF {} is not in binary form'.format(b)
    a1, a2 = np.hsplit(a, 2)
    return np.hstack((a2, a1)).dot(b) % 2
