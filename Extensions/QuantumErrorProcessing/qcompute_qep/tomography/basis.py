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
A collection of basis classes for the quantum tomography module.
"""
import abc
import sys
from copy import deepcopy
from typing import Tuple, List, Union, Dict
import numpy as np
import itertools

from QCompute import *
from QCompute.QPlatform.QOperation import CircuitLine

from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.utils.linalg import tensor
from qcompute_qep.quantum.pauli import complete_pauli_basis, QUBIT_PAULI_BASIS
from qcompute_qep.quantum.gellmann import GELL_MANN_BASIS
from qcompute_qep.utils.types import QProgram, number_of_qubits

SUPPORTED_MEASUREMENT_BASIS = {'Pauli', 'GellMann'}
SUPPORTED_PREPARATION_BASIS = {'Pauli', 'PauliOC'}


class Basis(abc.ABC):
    r"""The abstract `Basis` class.

    A basis is a set of ordered linear operators that spans the operator space.
    In our implementation, each `Basis` class has two properties:

        1. `name`: name of the basis (such as "Pauli");

        2. `basis`: detailed information the basis in the dictionary form `{<operator_name: operator_matrix>}`:

            1. `operator_name` is the short name of the operator.

            2. `operator_matrix` is its matrix representation.

    Example

        1. Single-qubit Pauli measurement basis, its `name` and `basis` property can be set as:

            name = "Single-qubit Pauli Basis"

            basis = {('I',): :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`,
            ('X',): :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}`,
            ('Y',): :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 0 & -j \\ j & 0 \end{bmatrix}`,
            ('Z',): :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}`}


        2. Single-qubit Pauli preparation basis, its `name` and `basis` property can be set as:

            name = "Single-qubit Pauli Preparation Basis"

            basis = {('0',): :math:`\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}`,
            ('1',): :math:`\begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}`,
            ('A',): :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}`,
            ('L',): :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & -j \\ j & 1 \end{bmatrix}`}

        3. Two-qubit GellMannMeasBasis operator basis is given by

            name = "Two-qubit GellMann Measurement Basis"

            basis = {('S12',): :math:`\begin{bmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,
            ('S13',): :math:`\begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

            ('S14',): :math:`\begin{bmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{bmatrix}`,
            ('S23',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

            ('S24',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}`,
            ('S34',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}`,

            ('A12',): :math:`\begin{bmatrix} 0 & j & 0 & 0 \\ j & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,
            ('A13',): :math:`\begin{bmatrix} 0 & 0 & j & 0 \\ 0 & 0 & 0 & 0 \\ j & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

            ('A14',): :math:`\begin{bmatrix} 0 & 0 & 0 & j \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ j & 0 & 0 & 0 \end{bmatrix}`,
            ('A23',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & j & 0 \\ 0 & j & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

            ('A24',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & j \\ 0 & 0 & 0 & 0 \\ 0 & j & 0 & 0 \end{bmatrix}`,
            ('A34',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & j \\ 0 & 0 & j & 0 \end{bmatrix}`,

            ('D00',): :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}`,
            ('D11',): :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 &
            0 \end{bmatrix}`,

            ('D22',): :math:`\frac{1}{\sqrt{3}}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & -2 & 0 \\ 0 & 0 & 0 &
            0 \end{bmatrix}`,
            ('D33',): :math:`\frac{1}{\sqrt{6}}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 &
            -3 \end{bmatrix}`}

    """
    @abc.abstractmethod
    def __init__(self):
        """
        The init function of the basis class. This method is an abstract method.
        Any class that inherits this abstract class must initialize the two properties: `name` and `basis`.
        """
        self._name = None
        self._basis = None

    @property
    def name(self):
        return self._name

    @property
    def basis(self):
        return self._basis

    def complete_basis(self, n: int) -> Dict[Tuple, np.ndarray]:
        r"""
        Given the multi-qubit `basis` for the local system,
        construct its corresponding :math:`n`-qubit complete basis of the global system using tensor.

        :param n: int, the number of qubits of the global quantum system
        :return: Dict[Tuple, np.ndarray], the complete basis extending the local basis to the global system

        The resulting complete basis is a dictionary describing the name and matrix of operators in the complete basis.

        .. note::

            We assume the LSB (least significant bit) mode, i.e., the right-most bit represents q[0]:

                name:           `X      I     Y      X`

                qubits:         q[3]  q[2]   q[1]   q[0]

            This assumption is important when constructing the complete basis.

        Example

            1. Single-qubit Pauli operator basis

                {('I',): :math:`\sigma_I`, ('X',): :math:`\sigma_X`, ('Y',): :math:`\sigma_Y`, ('Z',): :math:`\sigma_Z`}

            2. Two-qubit complete Pauli operator basis is given by

                {('I', 'I'): :math:`\sigma_I \otimes \sigma_I`, ('I', 'X'): :math:`\sigma_X \otimes \sigma_I`,
                ('I', 'Y'): :math:`\sigma_Y \otimes \sigma_I`, ('I', 'Z'): :math:`\sigma_Z \otimes \sigma_I`,

                ('X', 'I'): :math:`\sigma_I \otimes \sigma_X`, ('X', 'Y'): :math:`\sigma_X \otimes \sigma_X`,
                ('X', 'Y'): :math:`\sigma_Y \otimes \sigma_X`, ('X', 'Z'): :math:`\sigma_Z \otimes \sigma_X`,

                ('Y', 'I'): :math:`\sigma_I \otimes \sigma_Y`, ('Y', 'X'): :math:`\sigma_X \otimes \sigma_Y`,
                ('Y', 'Y'): :math:`\sigma_Y \otimes \sigma_Y`, ('Y', 'Z'): :math:`\sigma_Z \otimes \sigma_Y`,

                ('Z', 'I'): :math:`\sigma_I \otimes \sigma_Z`, ('Z', 'X'): :math:`\sigma_X \otimes \sigma_Z`,
                ('Z', 'Y'): :math:`\sigma_Y \otimes \sigma_Z`, ('Z', 'Z'): :math:`\sigma_Z \otimes \sigma_Z`}

            3. Two-qubit GellMannMeasBasis operator basis is given by

                {('S12',): :math:`\begin{bmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,
                ('S13',): :math:`\begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

                ('S14',): :math:`\begin{bmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{bmatrix}`,
                ('S23',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

                ('S24',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}`,
                ('S34',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}`,

                ('A12',): :math:`\begin{bmatrix} 0 & j & 0 & 0 \\ j & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,
                ('A13',): :math:`\begin{bmatrix} 0 & 0 & j & 0 \\ 0 & 0 & 0 & 0 \\ j & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

                ('A14',): :math:`\begin{bmatrix} 0 & 0 & 0 & j \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ j & 0 & 0 & 0 \end{bmatrix}`,
                ('A23',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & j & 0 \\ 0 & j & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

                ('A24',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & j \\ 0 & 0 & 0 & 0 \\ 0 & j & 0 & 0 \end{bmatrix}`,
                ('A34',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & j \\ 0 & 0 & j & 0 \end{bmatrix}`,

                ('D00',): :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}`,
                ('D11',): :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 &
                0 \end{bmatrix}`,

                ('D22',): :math:`\frac{1}{\sqrt{3}}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & -2 & 0 \\ 0 & 0 & 0 &
                0 \end{bmatrix}`,
                ('D33',): :math:`\frac{1}{\sqrt{6}}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 &
                -3 \end{bmatrix}`}

        """
        # Get the list of basis names and operators
        basis_names = list(self.basis.keys())
        basis_operators = list(self.basis.values())
        # Number of qubits per basis operator. Notice that each operator may be single or multi-qubit
        local_n = int(np.log2(basis_operators[0].shape[0]))

        q, r = divmod(n, local_n)
        if r != 0:
            raise ArgumentError("in complete_basis(): the given basis cannot cover all working qubits!")

        cb = dict()  # `cb` is short for `complete basis` and is a dictionary type
        cb_names = itertools.product(basis_names, repeat=q)
        for name in cb_names:
            cb_operators = [self.basis.get(ch) for ch in name]
            cb[name] = tensor(cb_operators)

        return cb

    def transition_matrix(self, n: int) -> np.ndarray:
        r"""
        Compute the transition matrix of the given basis with respect to the Pauli basis.
        Assume there are :math:`M` operators in the basis and each linear operator is :math:`n`-qubit.
        Set :math:`N=4^n`. The transition matrix :math:`A` is a :math:`M\times N` matrix.
        Let :math:`A_{jk}` be the element in the :math:`j`-th row and :math:`k`-th column, then

        .. math:: A_{jk} = {\rm Tr}[B_jP_k],

        where :math:`B_j` is the :math:`j`-th operator in the (well-ordered) basis,
        and :math:`P_k` is the :math:`k`-th Pauli operator.

        For the detailed definition of transition matrix, see Eq. (3.1) of [QGT]_.

        Reference:

        .. [QGT] Greenbaum, Daniel.
            "Introduction to quantum gate set tomography."
            arXiv preprint arXiv:1509.02921 (2015).

        :param n: int, the number of qubits of the global quantum system
        :return: np.ndarray, the transition matrix
        """
        # Get the complete basis and its corresponding operators
        cpb = self.complete_basis(n)
        cpb_operators = list(cpb.values())
        # Get the set of complete Pauli basis, whose size is :math:`4^n`
        pauli_basis = complete_pauli_basis(n)

        M = len(cpb_operators)
        N = len(pauli_basis)
        tr_matrix = np.zeros((M, N), dtype=complex)

        for j in range(M):
            for k in range(N):
                tr_matrix[j][k] = np.trace(np.dot(cpb_operators[j], pauli_basis[k].matrix))

        return tr_matrix

    def size(self, n: int = None) -> int:
        """
        Given the multi-qubit `basis` for the local system,
        compute the number of operators in the basis expanded to the :math:`n`-qubit Hilbert space,
        where :math:`n` is the number of qubits. That is to say, we have

        .. code-block:: python

            self.size(n) == len(self.complete_basis(n))

        If the variable :math:`n` is not set, return the size of the original basis.

        :param n: number of qubits of the quantum system
        :return: int, the number of operators in the complete basis
        """
        val = 0
        if n is None:
            val = len(self.basis)
        else:
            # Get the list of basis operators
            basis_operators = list(self.basis.values())
            # Number of qubits per operator. Notice that each operator may be multi-qubit
            local_n = int(np.log2(basis_operators[0].shape[0]))

            q, r = divmod(n, local_n)
            if r != 0:
                raise ArgumentError("in size(): the basis cannot cover all working qubits!")

            val = len(basis_operators) ** q

        return val

#######################################################################################################################
# The `MeasurementBasis` Abstract Class and its inherited classes
#######################################################################################################################


class MeasurementBasis(Basis):
    """The Measurement Basis abstract class.

    """
    @abc.abstractmethod
    def __init__(self):
        """
        The init function of the measurement basis. This method is an abstract method.
        Any class that inherits this abstract class must initialize the two properties: `name` and `basis`.
        """
        super().__init__()
        self._name = "Measurement Basis"

    @abc.abstractmethod
    def meas_circuits(self, qp: QProgram, qubits: List[int] = None) -> Tuple[List[QProgram], List[np.ndarray]]:
        raise NotImplementedError


class GellMannMeasBasis(MeasurementBasis):
    r"""The GellMann measurement basis class.

    The two properties: `name` and `basis` must be initialized.
    Here, `basis` is the measurement basis from the qubit Pauli operators, which is theoretically defined as

        {('S12',): :math:`\begin{bmatrix} 0 & 1 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,
        ('S13',): :math:`\begin{bmatrix} 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

        ('S14',): :math:`\begin{bmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 1 & 0 & 0 & 0 \end{bmatrix}`,
        ('S23',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

        ('S24',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}`,
        ('S34',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}`,

        ('A12',): :math:`\begin{bmatrix} 0 & j & 0 & 0 \\ j & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,
        ('A13',): :math:`\begin{bmatrix} 0 & 0 & j & 0 \\ 0 & 0 & 0 & 0 \\ j & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

        ('A14',): :math:`\begin{bmatrix} 0 & 0 & 0 & j \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ j & 0 & 0 & 0 \end{bmatrix}`,
        ('A23',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & j & 0 \\ 0 & j & 0 & 0 \\ 0 & 0 & 0 & 0 \end{bmatrix}`,

        ('A24',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & j \\ 0 & 0 & 0 & 0 \\ 0 & j & 0 & 0 \end{bmatrix}`,
        ('A34',): :math:`\begin{bmatrix} 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 & j \\ 0 & 0 & j & 0 \end{bmatrix}`,

        ('D00',): :math:`\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}`,
        ('D11',): :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & -1 & 0 & 0 \\ 0 & 0 & 0 & 0 \\ 0 & 0 & 0 &
        0 \end{bmatrix}`,

        ('D22',): :math:`\frac{1}{\sqrt{3}}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & -2 & 0 \\ 0 & 0 & 0 &
        0 \end{bmatrix}`,
        ('D33',): :math:`\frac{1}{\sqrt{6}}\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 &
        -3 \end{bmatrix}`}

        .. note::

            These operators are properly normalized.
    """
    def __init__(self):
        """
        Initializes the 2-Qubit Gell-Mann measurement basis class.
        """
        super().__init__()
        self._name = "2-Qubit Gell-Mann Measurement Basis"
        self._basis = GELL_MANN_BASIS

    def meas_circuits(self, qp: QProgram, qubits: List[int] = None) -> Tuple[List[QProgram], List[np.ndarray]]:
        """
        For each measurement operator in the complete measurement basis,
        decorate the given quantum program (without measurement) by adding the measurement specified
        by the operator to the end and construct the corresponding quantum observable.

        .. note::

            We assume the LSB (least significant bit) mode, i.e., the right-most bit represents q[0]:

                name:           `X      I     Y      X`

                qubits:         q[3]  q[2]   q[1]   q[0]

            This assumption is important when constructing the measurement circuits.

        Example

            Since the qubit is measured in `Z` basis by default,
            if we aim to measure the qubit in `X` basis, we can modify the quantum program as follows:

                0: ---H---MEAS

        :param qp: QProgram, the original quantum program (without measurement)
        :param qubits: List[int], the target qubit(s)
        :return: Tuple[List[QProgram], List[np.ndarray]], a complete set of modified quantum programs
                with the Pauli measurements appended to the end and its corresponding quantum observable

        **Example**

            >>> qps, obs = basis.meas_circuits(qp)
        """
        # Number of qubits in the quantum program
        if qubits is None:
            n = len(qp.Q.registerMap.keys())
            qubits = [i for i in range(n)]
        else:
            n = len(qubits)
            qubits.sort()

        if (n % 2 != 0):
            raise ArgumentError("n is not a odd number!!!")
        meas_qps: List[QProgram] = []
        meas_obs: List[np.ndarray] = []

        cb = self.complete_basis(n)

        for p_name in cb.keys():
            # !WARNING! DO NOT FORGET to deepcopy the original quantum program otherwise the qubits will be destroyed
            meas_qp = deepcopy(qp)
            # Store the eigenvalues for each local measurement operator.
            # !WARNING! We cannot simply compute the eigenvalues for the global measurement operator
            # using the `np.linalg.eig` function since the returned eigenvalues might be not well ordered.
            # For example, the eigenvalues of the Pauli operator `YI` should be ordered as :math:`[1, 1, -1, -1]`.
            # However, `np.linalg.eig` evaluates the eigenvalues of `YI`
            # will return :math:`[1, -1, 1, -1]`, which is incorrect.
            eigs = []
            # Map each 2 qubit Pauli measurement to the Z basis measurement
            gate_number = len(p_name)

            for i, ch in enumerate(p_name):
                # Calculate the qubit index under the LSB mode assumption
                # TODO: test if `qubits` works ??
                qubit_idx = qubits[n - i*2 - 1]
                if ch == 'S12':
                    # H \otimes I means H_Gate on 1st qubit and ID_Gate on 0th qubit
                    ID(meas_qp.Q[qubit_idx])
                    H(meas_qp.Q[qubit_idx-1])
                    eigs.append(np.diag([1, -1, 0, 0]))
                elif ch == 'S13':
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    ID(meas_qp.Q[qubit_idx])
                    H(meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    eigs.append(np.diag([1, 0, -1, 0]))
                elif ch == 'S14':
                    CX(meas_qp.Q[qubit_idx], meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    ID(meas_qp.Q[qubit_idx])
                    H(meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    CX(meas_qp.Q[qubit_idx], meas_qp.Q[qubit_idx-1])
                    eigs.append(np.diag([1, 0, 0, -1]))
                elif ch == 'S23':
                    CX(meas_qp.Q[qubit_idx], meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    ID(meas_qp.Q[qubit_idx])
                    H(meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    CX(meas_qp.Q[qubit_idx], meas_qp.Q[qubit_idx-1])
                    eigs.append(np.diag([0, 1, -1, 0]))
                elif ch == 'S24':
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    ID(meas_qp.Q[qubit_idx])
                    H(meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    eigs.append(np.diag([0, 1, 0, -1]))
                elif ch == 'S34':
                    ID(meas_qp.Q[qubit_idx])
                    H(meas_qp.Q[qubit_idx-1])
                    eigs.append(np.diag([0, 0, 1, -1]))
                if ch == 'A12':
                    # H \otimes I means H_Gate on 1st qubit and ID_Gate on 0th qubit
                    ID(meas_qp.Q[qubit_idx])
                    SDG(meas_qp.Q[qubit_idx-1])
                    H(meas_qp.Q[qubit_idx-1])
                    eigs.append(np.diag([1, -1, 0, 0]))
                elif ch == 'A13':
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    ID(meas_qp.Q[qubit_idx])
                    SDG(meas_qp.Q[qubit_idx-1])
                    H(meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    eigs.append(np.diag([1, 0, -1, 0]))
                elif ch == 'A14':
                    CX(meas_qp.Q[qubit_idx], meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    ID(meas_qp.Q[qubit_idx])
                    SDG(meas_qp.Q[qubit_idx-1])
                    H(meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    CX(meas_qp.Q[qubit_idx], meas_qp.Q[qubit_idx-1])
                    eigs.append(np.diag([1, 0, 0, -1]))
                elif ch == 'A23':
                    CX(meas_qp.Q[qubit_idx], meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    ID(meas_qp.Q[qubit_idx])
                    SDG(meas_qp.Q[qubit_idx-1])
                    H(meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    CX(meas_qp.Q[qubit_idx], meas_qp.Q[qubit_idx-1])
                    eigs.append(np.diag([0, 1, -1, 0]))
                elif ch == 'A24':
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    ID(meas_qp.Q[qubit_idx])
                    SDG(meas_qp.Q[qubit_idx-1])
                    H(meas_qp.Q[qubit_idx-1])
                    SWAP(meas_qp.Q[qubit_idx-1], meas_qp.Q[qubit_idx])
                    eigs.append(np.diag([0, 1, 0, -1]))
                elif ch == 'A34':
                    ID(meas_qp.Q[qubit_idx])
                    SDG(meas_qp.Q[qubit_idx-1])
                    H(meas_qp.Q[qubit_idx-1])
                    eigs.append(np.diag([0, 0, 1, -1]))
                elif ch == 'D00':
                    ID(meas_qp.Q[qubit_idx-1])
                    ID(meas_qp.Q[qubit_idx])
                    eigs.append(np.diag([1, 1, 1, 1])/np.sqrt(2))
                    # Should keep in mind that in the tomography module all Pauli operators are properly normalized.
                elif ch == 'D11':
                    ID(meas_qp.Q[qubit_idx-1])
                    ID(meas_qp.Q[qubit_idx])
                    eigs.append(np.diag([1, -1, 0, 0]))
                elif ch == 'D22':
                    ID(meas_qp.Q[qubit_idx-1])
                    ID(meas_qp.Q[qubit_idx])
                    eigs.append(np.diag([1, 1, -2, 0])/np.sqrt(3))
                elif ch == 'D33':
                    ID(meas_qp.Q[qubit_idx-1])
                    ID(meas_qp.Q[qubit_idx])
                    eigs.append(np.diag([1, 1, 1, -3])/np.sqrt(6))

            # Measurement in the Z basis
            MeasureZ(*meas_qp.Q.toListPair())

            meas_qps.append(meas_qp)
            meas_obs.append(tensor(eigs))

        return meas_qps, meas_obs


class PauliMeasBasis(MeasurementBasis):
    r"""The Pauli measurement basis class.

    The two properties: `name` and `basis` must be initialized.
    Here, `basis` is the measurement basis from the qubit Pauli operators, which is theoretically defined as

        {"I": :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`,
        "X": :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}`,
        "Y": :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 0 & -j \\ j & 0 \end{bmatrix}`,
        "Z": :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}`}

        .. note::

            These operators are properly normalized.
    """
    def __init__(self):
        """
        Initializes the Pauli measurement basis class.
        """
        super().__init__()
        self._name = "Single-Qubit Pauli Measurement Basis"
        self._basis = QUBIT_PAULI_BASIS

    def meas_circuits(self, qp: QEnv, qubits: List[int] = None) -> Tuple[List[QEnv], List[np.ndarray]]:
        """
        For each measurement operator in the complete measurement basis,
        decorate the given quantum program (without measurement) by adding the measurement specified
        by the operator to the end and construct the corresponding quantum observable.

        .. note::

            We assume the LSB (the least significant bit) mode, i.e., the right-most bit represents q[0]:

                name:           `X      I     Y      X`

                qubits:         q[3]  q[2]   q[1]   q[0]

            This assumption is important when constructing the measurement circuits.

        Example

            Since the qubit is measured in `Z` basis by default,
            if we aim to measure the qubit in `X` basis, we can modify the quantum program as follows:

                0: ---H---MEAS

        :param qp: QEnv, the original quantum program (without measurement)
        :param qubits: List[int], the target qubit(s)
        :return: Tuple[List[QEnv], List[np.ndarray]], a complete set of modified quantum programs
                with the Pauli measurements appended to the end and its corresponding quantum observable

        **Example**

            >>> qps, obs = basis.meas_circuits(qp)
            >>> qps, obs = basis.meas_circuits(qp, qubits=[0,2])
        """
        # Number of qubits in the quantum program
        if qubits is None:
            n = len(qp.Q.registerMap.keys())
            qubits = [i for i in range(n)]
        else:
            n = len(qubits)
            qubits.sort()

        meas_qps: List[QEnv] = []
        meas_obs: List[np.ndarray] = []

        cb = self.complete_basis(n)

        for p_name in cb.keys():
            # !WARNING! DO NOT FORGET to deepcopy the original quantum program otherwise the qubits will be destroyed
            meas_qp = deepcopy(qp)
            # Store the eigenvalues for each local measurement operator.
            # !WARNING! We cannot simply compute the eigenvalues for the global measurement operator
            # using the `np.linalg.eig` function since the returned eigenvalues might be not well ordered.
            # For example, the eigenvalues of the Pauli operator `YI` should be ordered as :math:`[1, 1, -1, -1]`.
            # However, `np.linalg.eig` evaluates the eigenvalues of `YI`
            # will return :math:`[1, -1, 1, -1]`, which is incorrect.
            eigs = []
            # Map each qubit Pauli measurement to the Z basis measurement
            for i, ch in enumerate(p_name):
                # Calculate the qubit index under the LSB mode assumption
                qubit_idx = qubits[n - i - 1]
                if ch == 'I':
                    ID(meas_qp.Q[qubit_idx])
                elif ch == 'X':
                    H(meas_qp.Q[qubit_idx])
                elif ch == 'Y':
                    SDG(meas_qp.Q[qubit_idx])
                    H(meas_qp.Q[qubit_idx])
                elif ch == 'Z':
                    pass
                else:
                    raise ArgumentError("in meas_circuits(): illegal basis name {}!".format(ch))
                # Compute the eigenvalues of the corresponding *normalized* Pauli operator.
                # Should keep in mind that in the tomography module all Pauli operators are properly normalized.
                eigs.append(np.diag([1, 1]) / np.sqrt(2) if ch == 'I' else np.diag([1, -1]) / np.sqrt(2))

            # Measurement in the Z basis
            qreglist, indexlist = meas_qp.Q.toListPair()
            MeasureZ(qRegList=[qreglist[x] for x in qubits],
                     cRegList=[indexlist[x] for x in qubits])

            meas_qps.append(meas_qp)
            meas_obs.append(tensor(eigs))

        return meas_qps, meas_obs

#######################################################################################################################
# The `PreparationBasis` Abstract Class and its inherited classes
#######################################################################################################################


class PreparationBasis(Basis):
    """
    The Preparation Basis abstract class.
    """
    def __init__(self):
        """
        The init function of the preparation basis class.  This method is an abstract method.
        Any class that inherits this abstract class must initialize the two properties: `name` and `basis`.
        """
        super().__init__()
        self._name = "Preparation Basis"

    @abc.abstractmethod
    def prep_circuits(self, qp: QProgram, qubits: List[int] = None) -> List[QProgram]:
        raise NotImplementedError


class PauliPrepBasis(PreparationBasis):
    r"""The Pauli preparation basis class.
    The two properties: `name` and `basis` must be initialized.
    Here, `basis` is the preparation basis from the qubit Pauli operators, which is theoretically defined as

        {"0": :math:`\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}`,
        "1": :math:`\begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}`,

        "A": :math:`\frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}`,
        "L": :math:`\frac{1}{2}\begin{bmatrix} 1 & -j \\ j & 1 \end{bmatrix}`}

    """
    def __init__(self):
        """
        Initializes the Pauli preparation basis class.
        """
        super().__init__()
        self._name = "Single-Qubit Pauli Preparation Basis"
        self._basis = {"0": np.array([[1, 0], [0, 0]]).astype(complex),
                       "1": np.array([[0, 0], [0, 1]]).astype(complex),
                       "A": np.array([[1, 1], [1, 1]]).astype(complex)/2,
                       "L": np.array([[1, -1j], [1j, 1]]).astype(complex)/2}

    def prep_circuits(self, qp: QEnv, qubits: List[int] = None) -> List[QEnv]:
        r"""
        For each preparation operator in the overcomplete preparation basis,
        decorate the given quantum program by adding the preparation quantum circuit
        to the beginning of the original quantum program.

        .. note::

            We assume the LSB (least significant bit) mode, i.e., the right-most bit represents q[0]:

                name:           `X      I     Y      X`

                qubits:         q[3]  q[2]   q[1]   q[0]

            This assumption is important when constructing the preparation circuits.

        Example

            If the original quantum program is the single-qubit H gate

                0: ---H---

            then the six decorated quantum programs are:

                0: ---H---              Prepare the `0` state :math:`\vert 0\rangle`

                0: ---X---H---          Prepare the `1` state :math:`\vert 1\rangle`

                0: ---H---H---          Prepare the `A` state :math:`\frac{1}{\sqrt{2}}(\vert 0\rangle+\vert 1\rangle)`

                0: ---H---S---H---      Prepare the `L` state :math:`\frac{1}{\sqrt{2}}(\vert 0\rangle+\vert i\rangle)`


        :param qp: QEnv, the original quantum program
        :param qubits: List[int], the target qubit(s)
        :return: List[QEnv], a list of QEnv objects, decorated from the original quantum program
                by adding the Pauli overcomplete preparation quantum circuits to the beginning

        **Example**

            >>> qps = basis.preparation_circuits(qp)
            >>> qps = basis.preparation_circuits(qp, qubits=[0,3])
        """
        # Number of qubits in the quantum program
        if qubits is None:
            n = len(qp.Q.registerMap.keys())
            qubits = [i for i in range(n)]
        else:
            n = len(qubits)
            qubits.sort()

        prep_qps: List[QEnv] = []

        cb = self.complete_basis(n)

        for p_name in cb.keys():
            # Deep copy the quantum program
            prep_qp = deepcopy(qp)
            p_name = "".join(p_name).upper()
            # Initialize each qubit to the desired preparation state
            for i, ch in enumerate(p_name):
                # Calculate the qubit index under the LSB mode assumption
                qubit_idx = qubits[n - i - 1]
                if ch == '0':
                    pass
                elif ch == '1':  # Execute X on the target qubit to the beginning of the quantum program
                    clX = CircuitLine(data=X, qRegList=[qubit_idx])
                    prep_qp.circuit = [clX] + prep_qp.circuit
                elif ch == 'A':  # Execute H on the target qubit to the beginning of the quantum program
                    clH = CircuitLine(data=H, qRegList=[qubit_idx])
                    prep_qp.circuit = [clH] + prep_qp.circuit
                elif ch == 'L':  # Execute H and S on the target qubit to the beginning of the quantum program
                    clH = CircuitLine(data=H, qRegList=[qubit_idx])
                    clS = CircuitLine(data=S, qRegList=[qubit_idx])
                    prep_qp.circuit = [clH, clS] + prep_qp.circuit
                else:
                    raise ArgumentError("in prep_circuits(): illegal preparation basis name {}!".format(ch))

            prep_qps.append(prep_qp)

        return prep_qps


class PauliOCPrepBasis(PreparationBasis):
    r"""The Pauli overcomplete (OC) preparation basis class.

    The two properties: `name` and `basis` must be initialized.
    Here, `basis` is the overcomplete preparation basis from the qubit Pauli operators, theoretically defined as

        {"0": :math:`\begin{bmatrix} 1 & 0 \\ 0 & 0 \end{bmatrix}`,
        "1": :math:`\begin{bmatrix} 0 & 0 \\ 0 & 1 \end{bmatrix}`,
        "A": :math:`\frac{1}{2}\begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}`,
        "D": :math:`\frac{1}{2}\begin{bmatrix} 1 & -1 \\ -1 & 1 \end{bmatrix}`,

        "L": :math:`\frac{1}{2}\begin{bmatrix} 1 & -j \\ j & 1 \end{bmatrix}`,
        "R": :math:`\frac{1}{\sqrt{2}}\begin{bmatrix} 1 & j \\ -j & 1 \end{bmatrix}`}

    """
    def __init__(self):
        """
        Initializes the Pauli overcomplete preparation basis class.
        """
        super().__init__()
        self._name = "Single-Qubit Pauli Overcomplete Preparation Basis"
        self._basis = {"0": np.array([[1, 0], [0, 0]]).astype(complex),
                       "1": np.array([[0, 0], [0, 1]]).astype(complex),
                       "A": np.array([[1, 1], [1, 1]]).astype(complex)/2,
                       "D": np.array([[1, -1], [-1, 1]]).astype(complex)/2,
                       "L": np.array([[1, -1j], [1j, 1]]).astype(complex)/2,
                       "R": np.array([[1, 1j], [-1j, 1]]).astype(complex)/2}

    def prep_circuits(self, qp: QProgram, qubits: List[int] = None) -> List[QProgram]:
        r"""
        For each preparation operator in the overcomplete preparation basis,
        decorate the given quantum program by adding the preparation quantum circuit
        to the beginning of the original quantum program.

        .. note::

            We assume the LSB (least significant bit) mode, i.e., the right-most bit represents q[0]:

                name:           `X      I     Y      X`

                qubits:         q[3]  q[2]   q[1]   q[0]

            This assumption is important when constructing the preparation circuits.

        Example

            If the original quantum program is the single-qubit H gate

                0: ---H---

            then the six decorated quantum programs are:

                0: ---H---              Prepare the `0` state :math:`\vert 0\rangle`

                0: ---X---H---          Prepare the `1` state :math:`\vert 1\rangle`

                0: ---H---H---          Prepare the `A` state :math:`\frac{1}{\sqrt{2}}(\vert 0\rangle+\vert 1\rangle)`

                0: ---X---H---H---      Prepare the `D` state :math:`\frac{1}{\sqrt{2}}(\vert 0\rangle-\vert 1\rangle)`

                0: ---H---S---H---      Prepare the `L` state :math:`\frac{1}{\sqrt{2}}(\vert 0\rangle+\vert i\rangle)`

                0: ---X---H---S---H---  Prepare the `R` state :math:`\frac{1}{\sqrt{2}}(\vert 0\rangle-\vert i\rangle)`

        :param qp: QProgram, the original quantum program
        :param qubits: List[int], the target qubit(s)
        :return: List[QProgram], a list of QProgram objects, decorated from the original quantum program
                by adding the Pauli overcomplete preparation quantum circuits to the beginning

        **Example**

            >>> qps = basis.preparation_circuits(qp)
        """
        # Number of qubits in the quantum program
        if qubits is None:
            n = len(qp.Q.registerMap.keys())
            qubits = [i for i in range(n)]
        else:
            n = len(qubits)
            qubits.sort()

        prep_qps: List[QProgram] = []

        cb = self.complete_basis(n)

        for p_name in cb.keys():
            # Deep copy the quantum program
            prep_qp = deepcopy(qp)
            p_name = "".join(p_name).upper()
            # Initialize each qubit to the desired preparation state
            for i, ch in enumerate(p_name):
                # Calculate the qubit index under the LSB mode assumption
                qubit_idx = qubits[n - i - 1]
                if ch == '0':
                    pass
                elif ch == '1':  # Execute X on the target qubit to the beginning of the quantum program
                    clX = CircuitLine(data=X, qRegList=[qubit_idx])
                    prep_qp.circuit = [clX] + prep_qp.circuit
                elif ch == 'A':  # Execute H on the target qubit to the beginning of the quantum program
                    clH = CircuitLine(data=H, qRegList=[qubit_idx])
                    prep_qp.circuit = [clH] + prep_qp.circuit
                elif ch == 'D':  # Execute X and H on the target qubit to the beginning of the quantum program
                    clX = CircuitLine(data=X, qRegList=[qubit_idx])
                    clH = CircuitLine(data=H, qRegList=[qubit_idx])
                    prep_qp.circuit = [clX, clH] + prep_qp.circuit
                elif ch == 'L':  # Execute H and S on the target qubit to the beginning of the quantum program
                    clH = CircuitLine(data=H, qRegList=[qubit_idx])
                    clS = CircuitLine(data=S, qRegList=[qubit_idx])
                    prep_qp.circuit = [clH, clS] + prep_qp.circuit
                elif ch == 'R':  # Execute X, H and S on the target qubit to the beginning of the quantum program
                    clX = CircuitLine(data=X, qRegList=[qubit_idx])
                    clH = CircuitLine(data=H, qRegList=[qubit_idx])
                    clS = CircuitLine(data=S, qRegList=[qubit_idx])
                    prep_qp.circuit = [clX, clH, clS] + prep_qp.circuit
                else:
                    raise ArgumentError("in preparation_circuits(): illegal preparation basis name {}!".format(ch))

            prep_qps.append(prep_qp)

        return prep_qps


def init_preparation_basis(val: Union[str, PreparationBasis] = None) -> PreparationBasis:
    """Initialize the preparation basis.

    The candidate basis can be `None`, can be a `PreparationBasis` instance,
    and also can be a preparation basis name such as "Pauli" and "PauliOC".
    For the first case, we initialize the preparation basis to the default `PauliPrepBasis`;
    for the second case, we do nothing;
    for the last case, we create the corresponding preparation basis class.

    :param val: Union[str, PreparationBasis], the candidate preparation basis
    :return: PreparationBasis, the initialized `PreparationBasis` class instance

    **Example**

        >>> pauli_basis = init_preparation_basis()
        >>> pauli_basis = init_preparation_basis('Pauli')
        >>> gellman_basis = init_preparation_basis('GellMann')
    """
    pb: PreparationBasis = None
    if val is None:  # if the basis is not set, use the default PauliPrepBasis
        pb = PauliPrepBasis()
    elif isinstance(val, PreparationBasis):
        pb = val
    elif isinstance(val, str):  # Construct the measurement basis from its name
        if val not in SUPPORTED_PREPARATION_BASIS:
            raise ArgumentError("in init_preparation_basis(): '{}' is not supported preparation basis!".format(val))
        else:
            pb = getattr(sys.modules[__name__], val + 'PrepBasis')()
    else:
        raise ArgumentError("in init_preparation_basis(): unsupported input value type {}!".format(type(val)))

    return pb


def init_measurement_basis(val: Union[str, MeasurementBasis] = None) -> MeasurementBasis:
    """Initialize the measurement basis.

    The candidate basis can be `None`, can be a `MeasurementBasis` instance,
    and also can be a measurement basis name such as "Pauli" and "GellMann".
    For the first case, we initialize the basis to the PauliMeasBasis;
    for the second case, we do nothing;
    for the last case, we create the corresponding basis class.

    :param val: Union[str, MeasurementBasis], the candidate measurement basis
    :return: MeasurementBasis, the initialized `MeasurementBasis` class instance

    **Example**

        >>> pauli_basis = init_measurement_basis()
        >>> pauli_basis = init_measurement_basis('Pauli')
        >>> paulioc_basis = init_measurement_basis('PauliOC')

    """
    mb: MeasurementBasis = None
    if val is None:  # if the basis is not set, use the default PauliMeasBasis
        mb = PauliMeasBasis()
    elif isinstance(val, MeasurementBasis):
        mb = val
    elif isinstance(val, str):  # Construct the measurement basis from its name
        if val not in SUPPORTED_MEASUREMENT_BASIS:
            raise ArgumentError("in init_measurement_basis(): '{}' is not supported measurement basis!".format(val))
        else:
            mb = getattr(sys.modules[__name__], val + 'MeasBasis')()
    else:
        raise ArgumentError("in init_measurement_basis(): unsupported input value type {}!".format(type(val)))

    return mb
