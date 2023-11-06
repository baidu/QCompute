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
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.types import QProgram
from Extensions.QuantumErrorProcessing.qcompute_qep.exceptions.QEPError import ArgumentError
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.linalg import tensor
from copy import deepcopy
from QCompute.QPlatform.QOperation import CircuitLine

# The qubit Pauli basis. Notice that these operators are properly normalized.
QUBIT_PAULI_BASIS = {
    "I": np.array([[1, 0], [0, 1]]).astype(complex) / np.sqrt(2),
    "X": np.array([[0, 1], [1, 0]]).astype(complex) / np.sqrt(2),
    "Y": np.array([[0, -1j], [1j, 0]]).astype(complex) / np.sqrt(2),
    "Z": np.array([[1, 0], [0, -1]]).astype(complex) / np.sqrt(2),
}


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
        xs = (ps == "X") + (ps == "Y")
        zs = (ps == "Z") + (ps == "Y")
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

        sub_name = "".join([self._name[i] for i in indices])
        return Pauli(sub_name)

    def eigenvalues(self) -> np.ndarray:
        r"""Compute the eigenvalues of the Pauli operator.

        :return: np.ndarray, a diagonalized matrix storing the eigenvalues
        """
        eigs = [np.diag([1, 1]) if ch == "I" else np.diag([1, -1]) for ch in self._name.upper()]
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
            if s == "X":
                H(qp_new.Q[qubit_idx])
            elif s == "Y":
                SDG(qp_new.Q[qubit_idx])
                H(qp_new.Q[qubit_idx])

        # Measurement in the Z basis
        qreglist, indexlist = qp_new.Q.toListPair()
        MeasureZ(qRegList=[qreglist[x] for x in qubits], cRegList=[indexlist[x] for x in qubits])

        temp_list = []

        for i in self.name:
            if i == "I":
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
            if s == "I" or s == "Z":
                new_gate1 = []
                new_gate2 = [CircuitLine(data=X, qRegList=[qubit_idx])]
            elif s == "X":
                new_gate1 = [CircuitLine(data=H, qRegList=[qubit_idx])]
                new_gate2 = [CircuitLine(data=X, qRegList=[qubit_idx]), CircuitLine(data=H, qRegList=[qubit_idx])]
            elif s == "Y":
                new_gate1 = [CircuitLine(data=H, qRegList=[qubit_idx]), CircuitLine(data=S, qRegList=[qubit_idx])]
                new_gate2 = [
                    CircuitLine(data=X, qRegList=[qubit_idx]),
                    CircuitLine(data=H, qRegList=[qubit_idx]),
                    CircuitLine(data=S, qRegList=[qubit_idx]),
                ]
            else:
                raise ArgumentError("in prep_circuits(): illegal preparation basis name {}!".format(s))

            for j in range(len(new_qps)):
                # add single-qubit gate to the front of circuit corresponding to the Pauli basis
                new_qps[j].circuit = new_gate1 + new_qps[j].circuit
                copy_qps[j].circuit = new_gate2 + copy_qps[j].circuit
            new_qps = new_qps + copy_qps

        temp_list = []
        for i in self.name:
            if i == "I":
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

    .. math:: \sigma_X\otimes\sigma_I
        = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}\otimes\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}
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
    name = ""
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
        >>> rho = ptm_to_operator(coes)
        >>> print(rho)
        [[1.+0.j 0.+0.j]
         [0.+0.j 0.+0.j]]
    """
    if isinstance(coes, List):
        coes = np.asarray(coes)
    # Reshape to column vector
    coes = coes.reshape((coes.size,))
    # Number of qubits
    n = int(log(coes.size, 4))
    pauli_basis = complete_pauli_basis(n)
    op = np.zeros((2**n, 2**n))

    for i in range(coes.size):
        op = op + coes[i] * pauli_basis[i].matrix

    return op


def bsp(a: np.ndarray, b: np.ndarray) -> Union[int, np.array]:
    r"""Compute the binary symplectic product of :math:`a` and :math:`b`.

    The binary symplectic product :math:`\odot` of two 1d vectors :math:`a` and :math:`b` is defined as:

    .. math::   a \odot b := a \Lambda b \bmod 2,

    where

    .. math::  \Lambda = \left[\begin{matrix} 0 & I \\ I & 0 \end{matrix}\right].

    If :math:`a` and :math:`b` are 2d matrices, then the binary symplectic product is defined
    to be the standard dot product between :math:`a\Lambda` and :math:`b`.

    :param a: np.ndarray, a 1d or 2d binary symplectic vector
    :param b: np.ndarray, a 1d or 2d binary symplectic vector
    :return: Union[int, np.array], the binary symplectic product of a with b.
                If a and b are 1d, the return value type is ``int``;
                If a and b are 2d, the return value type is ``np.array``.
    """
    assert np.array_equal(a % 2, a), "BSF {} is not in binary form".format(a)
    assert np.array_equal(b % 2, b), "BSF {} is not in binary form".format(b)
    a1, a2 = np.hsplit(a, 2)
    return (np.hstack((a2, a1)).dot(b) % 2).astype("int")


def bsf2pauli(bsf: np.ndarray) -> Union[str, List[str]]:
    r"""Convert a list of Pauli operators in binary symplectic form to string form.

    In the single-qubit case, there exists a one-one mapping between bsf and string forms:

    .. math::

            [0, 0] \equiv 0 \leftrightarrow \text{'I'},
            [1, 0] \equiv 1 \leftrightarrow \text{'X'},
            [0, 1] \equiv 2 \leftrightarrow \text{'Z'},
            [1, 1] \equiv 3 \leftrightarrow \text{'Y'}.

    where :math:`\equiv` means that we define the left binary vectors with a unique integer in decimal.

    :param bsf: np.array, the binary symplectic form of the list of Pauli operators
    :return: Union[str, List[str]], a Pauli or a list of Pauli operators in the string form

    Examples:

        >>> # Case 1: single Pauli operator
        >>> bsf = np.asarray([1, 0, 0, 0, 1, 0, 0, 1, 0, 1], dtype=int)
        >>> print(bsf2pauli(bsf))
        XIZIY
        >>> # Case 2: a list of Pauli operator
        >>> bsf = np.asarray([[1 0 0 1], [0 1 0 1]], dtype=int)
        >>> print(bsf2pauli(bsf))
        ['XZ', 'IY']
    """
    assert np.array_equal(bsf % 2, bsf), "BSF {} is not in binary form".format(bsf)

    def _2pauli(bsf_single):
        xs, zs = np.hsplit(bsf_single, 2)
        ps = (xs + zs * 2).astype(str)
        return "".join(ps).translate(str.maketrans("0123", "IXZY"))

    if bsf.ndim == 1:
        return _2pauli(bsf)
    else:
        return [_2pauli(b) for b in bsf]


def pauli2bsf(paulis: Union[str, List[str]]) -> np.array:
    r"""Convert a list of Pauli operators in string form to binary symplectic form.

    In the single-qubit case, there exists a one-one mapping between bsf and string forms:

    .. math::

            [0, 0] \leftrightarrow \text{'I'},
            [1, 0] \leftrightarrow \text{'X'},
            [0, 1] \leftrightarrow \text{'Z'},
            [1, 1] \leftrightarrow \text{'Y'}.

    :param paulis: Union[str, List[str]], a Pauli or a list of Pauli operators in the string form
    :return: np.array, the binary symplectic form of the list of Pauli operator

    Examples:

        >>> # Case 1: single Pauli operator
        >>> print(pauli.pauli2bsf('XIZIY'))
        [1 0 0 0 1 0 0 1 0 1]
        >>> # Case 2: a list of Pauli operator
        >>> print(pauli.pauli2bsf(['XZ', 'IY']))
        [[1 0 0 1]
         [0 1 0 1]]
    """

    def _2bsf(pauli):
        ps = np.array(list(pauli))
        xs = (ps == "X") + (ps == "Y")
        zs = (ps == "Z") + (ps == "Y")
        return np.hstack((xs, zs)).astype(int)

    if isinstance(paulis, str):
        return _2bsf(paulis)
    elif isinstance(paulis, List):
        return np.vstack([_2bsf(p) for p in paulis])
    else:
        raise ArgumentError("in pauli2bsf(): undefined Pauli operator type: {}".format(type(paulis)))


def mutually_commute(paulis: Union[List[str], np.ndarray]) -> bool:
    r"""Check if a list of Pauli operators mutually commute.

    Let :math:`A,B` be two :math:`n`-qubit Pauli operators. A and B *commute* if it holds that

    .. math:: [A,B]=AB-BA=0.

    A list of Pauli operators :math:`\mathcal{S}` *mutually commute*, if any two of them commute, i.e.,

    .. math:: \forall A, B\in\mathcal{S},\; [A,B]=0.

    :param paulis: Union[List[str], np.ndarray], the list of Pauli operators, either in Pauli string from
                                                or in binary symplectic form (bsf).
    :return: bool, *True* if the given list of Pauli operators mutually commute, otherwise *False*.

    **Examples**

        >>> # Use the [[4, 2, 2]] code as test: Its stabilizers mutually commute, while its logical operators do not.
        >>> import Extensions.QuantumErrorProcessing.qcompute_qep.correction
        >>> qec_code = Extensions.QuantumErrorProcessing.qcompute_qep.correction.FourTwoTwoCode()
        >>> print("The stabilizers are: {}".format(qec_code.stabilizers))
        The stabilizers are: ['XXXX', 'ZZZZ']
        >>> print("The stabilizers mutually commute? {}".format(mutually_commute(qec_code.stabilizers)))
        The stabilizers mutually commute? True
        >>> logical_operators = qec_code.logical_xs(form='str') + qec_code.logical_zs(form='str')
        >>> print("The logical operators are: {}".format(logical_operators))
        The logical operators are: ['IXXI', 'IXIX', 'ZIZI', 'ZIIZ']
        >>> print("The logical operators mutually commute? {}".format(mutually_commute(logical_operators)))
        The logical operators mutually commute? False
    """
    # Convert the list of Pauli operators the bsf form
    if isinstance(paulis, List):
        paulis = pauli2bsf(paulis)
    elif isinstance(paulis, np.ndarray):
        pass
    else:
        raise ArgumentError("in mutually_commute(): undefined Pauli operator type: {}".format(type(paulis)))

    if not np.all(bsp(paulis, paulis.T) == 0):
        return False
    else:
        return True


def mutually_anticommute(paulis: Union[List[str], np.ndarray]) -> bool:
    r"""Check if a list of Pauli operators mutually anticommute.

    Let :math:`A,B` be two :math:`n`-qubit Pauli operators. A and B *anticommute* if it holds that

    .. math:: \{A,B\}=AB+BA=0.

    A list of Pauli operators :math:`\mathcal{S}` *mutually anticommute*,
    if any two of them anticommute (excluding itself), i.e.,

    .. math:: \forall A\neq B\in\mathcal{S},\; \{A,B\}=0.

    :param paulis: Union[List[str], np.ndarray], the list of Pauli operators, either in Pauli string from
                                                or in binary symplectic form (bsf).
    :return: bool, *True* if the given list of Pauli operators mutually anticommute, otherwise *False*.

    **Examples**

        >>> # Use the five qubit code as test: Its logical operators mutually anticommute.
        >>> import Extensions.QuantumErrorProcessing.qcompute_qep.correction
        >>> qec_code = Extensions.QuantumErrorProcessing.qcompute_qep.correction.FiveQubitCode()
        >>> logical_operators = qec_code.logical_xs(form='str') + qec_code.logical_zs(form='str')
        >>> print("The logical operators are: {}".format(logical_operators))
        The logical operators are: ['ZIIZX', 'ZZZZZ']
        >>> print("The logical operators mutually anticommute? {}".format(mutually_commute(logical_operators)))
        The logical operators mutually anticommute? True
    """
    # Convert the list of Pauli operators the bsf form
    if isinstance(paulis, List):
        paulis = pauli2bsf(paulis)
    elif isinstance(paulis, np.ndarray):
        pass
    else:
        raise ArgumentError("in mutually_anticommute(): undefined Pauli operator type: {}".format(type(paulis)))

    # Force all diagonal elements to be :math:`1` since a Pauli operator commutes with itself
    val = bsp(paulis, paulis.T)
    np.fill_diagonal(val, 1)
    if not np.all(val == 1):
        return False
    else:
        return True
