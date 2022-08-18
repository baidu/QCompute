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
In this script, we implement the ``Clifford`` class using its norm form, originally proposed in [S15]_.
Based on this class, we offer the ``random_clifford`` and ``complete_cliffords`` functions.

References:

.. [S15] Selinger, Peter.
        "Generators and relations for n-qubit Clifford operators."
        Logical Methods in Computer Science 11 (2015).
"""

import collections
import copy
from typing import List, Union, Tuple
import numpy as np
import itertools
from QCompute.QPlatform.QOperation import CircuitLine
from QCompute.QPlatform.QOperation import FixedGate
from QCompute.QPlatform.QRegPool import QRegStorage

import qcompute_qep.utils.circuit as circuit
import QCompute
from qcompute_qep.exceptions.QEPError import ArgumentError


class Clifford:
    r"""The Clifford Operator Class.

    We generate the :math:`n`-qubit Clifford operators from their normal forms, proposed in [S15]_.
    Each an :math:`n`-qubit Clifford operator is uniquely defined by its pattern,
    which contains five basic gates A/B/C/D/E, where

    + A = {A1, A2, A3},
    + B = {B1, B2, B3, B4},
    + C = {C1, C2}, and
    + D = {D1, D2, D3, D4},
    + E = {E1, E2, E3, E4}.

    The pattern for :math:`n`-qubit Cliffords should be like:

    ::

        [
        [['A1'], ['B1','B3'], ['C2'], ['D3','D4'], ['E2']],    N(n)
        [['A2'], ['B2'], ['C1'], ['D3'], ['E3']],              N(n-1)
        ...         ...                 ...                     ...
        ...         ...                 ...                     ...
        [[...], [...], [...], [...], [...]]                    N(1)
        ]

    User can define their own pattern to obtain the clifford operators,
    or use default settings which the pattern will be randomly defined.

    **Usage**

    .. code-block:: python
        :linenos:

        c = Clifford(n=2)

        pattern = random_pattern(n=3)
        c = Clifford(n=3)
        print(c)

    **Examples**

        >>> from qcompute_qep.quantum.clifford import Clifford
        >>> import qcompute_qep.utils.circuit as cir
        >>> import QCompute
        >>> n = 1
        >>> env = QCompute.QEnv()
        >>> q = env.Q.createList(n)
        >>> QCompute.H(q[0])
        >>> QCompute.X(q[0])
        >>> cir.print_circuit(env.circuit)
        0: ---H---X---
        >>> # Randomly generate an n-qubit Clifford gate and execute on the qubits
        >>> c = Clifford(n)
        >>> print(c)
        0: ---H---I---S---S---
        >>> c(q)
        >>> cir.print_circuit(env.circuit)
        0: ---H---X---H---I---S---S---
    """

    def __init__(self, n, pattern: List[List[List[str]]] = None) -> None:
        if pattern is None:
            pattern = random_pattern(n)
        else:
            if n > len(pattern):
                raise ArgumentError('The pattern is wrong, should be checked out.')
        self._cir = None
        self._inv_cir = None
        self._norm_form = None
        self._n = n
        self._pattern = pattern
        self._gate_set = self._basic_gate_set()

    @property
    def matrix(self) -> np.ndarray:
        r"""The matrix representation of the Clifford.
        The :math:`2^n\times 2^n` matrix in np.ndarray that represents this Clifford.
        It can be obtained using the ``circuit.circuit_to_unitary`` function.

        :return: np.ndarray, the matrix representation

        **Example**

            >>> from qcompute_qep.quantum.clifford import Clifford
            >>> n = 2
            >>> c = Clifford(n)  # randomly generate a Clifford
            >>> print(c.matrix)
            [[ 5.00000000e-01-9.21304356e-18j  9.21304356e-18-5.00000000e-01j
               5.00000000e-01+9.21304356e-18j  9.21304356e-18+5.00000000e-01j]
             [-9.21304356e-18+5.00000000e-01j -5.00000000e-01+9.21304356e-18j
              -9.21304356e-18-5.00000000e-01j -5.00000000e-01-9.21304356e-18j]
             [-9.21304356e-18-5.00000000e-01j  5.00000000e-01+9.21304356e-18j
               9.21304356e-18-5.00000000e-01j -5.00000000e-01+9.21304356e-18j]
             [ 5.00000000e-01+9.21304356e-18j -9.21304356e-18-5.00000000e-01j
              -5.00000000e-01+9.21304356e-18j  9.21304356e-18-5.00000000e-01j]]
        """
        if self.circuit:
            return circuit.layer_to_unitary(self.circuit, self._n)
        else:
            return np.identity(2 ** self._n)

    @property
    def circuit(self) -> List[CircuitLine]:
        r"""The elementary circuit representation of the Clifford.

        Convert the normal form of the Clifford operator to a quantum circuit composed of elementary gates.

        :return: List[CircuitLine], the quantum circuit representing the Clifford

        **Example**

            >>> from qcompute_qep.quantum.clifford import Clifford
            >>> from qcompute_qep.utils.circuit import print_circuit
            >>> n = 2
            >>> c = Clifford(n)  # randomly generate a Clifford
            >>> print_circuit(c.circuit)
            0: -------@---H---@---I---@---H---@---H---@---H---H---S---S---H---S---S---
                      |       |       |       |       |
            1: ---I---Z---H---Z-------Z---H---Z---H---Z---H---S---S---S---------------
        """
        if self._cir is None:
            normal_form = self.normal_form
            cir = []
            for gate, idx in normal_form:
                cir.extend(self._construct_circuit(gate, *idx))
            self._cir = cir
        if not self._cir:
            circuitline = CircuitLine()
            circuitline.data = FixedGate.getFixedGateInstance('ID')
            circuitline.qRegList = list(range(self._n))
            self._cir.append(circuitline)
        return self._cir

    @property
    def normal_form(self) -> List[Tuple[str, Tuple[int]]]:
        r"""The normal form of the Clifford.

        The normal form of the :math:`n`-qubit clifford in terms of A/B/C/D/E patterns. See [S15]_ for details.

        :return: List[Tuple[str, Tuple[int]]], the normal form of clifford

        **Examples**

            >>> from qcompute_qep.quantum.clifford import Clifford
            >>> n = 2
            >>> c = Clifford(n)
            >>> print(c.normal_form)
            [('A2', (0,)), ('C2', (0,)), ('D4', (0, 1)), ('E2', (1,)), ('A1', (0,)), ('C2', (0,)), ('E2', (0,))]
        """
        if self._norm_form is None:
            normal_form = []
            n = self._n
            pattern = self._pattern
            for i in range(n):
                L_pattern = pattern[i][:3]
                M_pattern = pattern[i][3:]
                normal_form.extend(self._L(L_pattern))
                normal_form.extend(self._M(M_pattern))
            self._norm_form = normal_form
        return self._norm_form

    def __call__(self, qRegList: List['QRegStorage'], qubits: List[int] = None, ) -> None:
        env = qRegList[0].env
        if qubits is None:
            env.circuit += self.circuit
        else:
            qubits.sort()
            if self.circuit:
                for original_cir in self.circuit:
                    circuitline = CircuitLine()
                    circuitline.data = original_cir.data
                    circuitline.qRegList = [qubits[idx] for idx in original_cir.qRegList]
                    env.circuit.append(circuitline)

    def __str__(self):
        """
        Print the quantum circuit (in elementary quantum gates) that implements the Clifford.
        """
        cir = self.circuit
        if cir:
            return circuit.print_circuit(cir, show=False)
        else:
            return circuit.print_circuit(cir, num_qubits=self._n, show=False)

    def get_inverse_circuit(self, qubits: List[int] = None, ) -> List[CircuitLine]:
        r"""Compute the inverse circuit of the clifford.

        The inverse circuit of the clifford is obtained by reversing recursively the quantum circuit representing the
        Clifford operator.

        :return: List[CircuitLine], a quantum circuit describing :math:`C^\dagger`

        **Examples**

            >>> from qcompute_qep.quantum.clifford import Clifford
            >>> from qcompute_qep.utils.circuit import print_circuit
            >>> n = 2
            >>> c = Clifford(n)
            >>> print_circuit(c.circuit)
            0: ---H---S---H---H---S---S---H---H---@---H---@---H---H---S---S---H---S---S---S---
                                                  |       |
            1: ---H-------------------------------Z---H---Z---H---S---------------------------
            >>> print_circuit(c.get_inverse_circuit())
            0: ---SDG---SDG---SDG---H---SDG---SDG---H---H---@---H---@---H---H---SDG---SDG---H---H---SDG---H---
                                                            |       |
            1: ---SDG----H----------------------------------Z---H---Z---H-------------------------------------
        """
        if self._inv_cir is None:
            self._inv_cir = circuit.inverse_layer(self.circuit)

        if qubits is not None:
            qubits.sort()
            inverse_circuit = copy.deepcopy(self._inv_cir)
            for cir in inverse_circuit:
                cir.qRegList = [qubits[idx] for idx in cir.qRegList]
            return inverse_circuit
        else:
            return self._inv_cir

    def get_inverse_matrix(self) -> np.ndarray:
        """Compute the inverse matrix of the clifford.

        :return: np.ndarray, the inverse matrix of the clifford
        """
        if self._inv_cir is None:
            self._inv_cir = circuit.inverse_layer(self.circuit)
        if self._inv_cir:
            return circuit.layer_to_unitary(self._inv_cir, self._n)
        else:
            return np.identity(2 ** self._n)

    def __print_normal_form(self):
        r"""Print the normal form of the Clifford.
        """
        # TODO: Will be implemented in the next version
        return

    @staticmethod
    def _basic_gate_set() -> 'collections.defaultdict':
        """The default gate set for clifford circuits.

        There are five types of basic gates:

        + A = {A1, A2, A3},

        + B = {B1, B2, B3, B4},

        + C = {C1, C2},

        + D = {D1, D2, D3, D4}, and

        + E = {E1, E2, E3, E4}.
        """
        gate_set = collections.defaultdict(list)

        # Here, we have simplified some of the gates
        # For example,
        # ---H---S---S---H--- =   ---X---
        #
        # ------@-------          ------@------
        #       |             =         |
        # --H---Z---H---          ------X------
        #
        # ---S---S---         =   ---Z---

        gate_set['A1'] = [(None, 0)]
        gate_set['A2'] = [('H', 0)]
        gate_set['A3'] = [('H', 0), ('S', 0), ('H', 0)]

        gate_set['B1'] = [('CX', 2), ('CX', 3), ('CX', 2), ('H', 1)]
        gate_set['B2'] = [('H', 1), ('CX', 2), ('CX', 3), ('H', 0)]
        gate_set['B3'] = [('H', 0), ('H', 1), ('S', 0), ('CX', 2), ('CX', 3), ('H', 1)]
        gate_set['B4'] = [('CX', 3), ('CX', 2), ('H', 1)]

        gate_set['C1'] = [(None, 0)]
        gate_set['C2'] = [('X', 0)]

        gate_set['D1'] = [('H', 1), ('CX', 2), ('CX', 3), ('CX', 2)]
        gate_set['D2'] = [('CX', 3), ('CX', 2)]
        gate_set['D3'] = [('H', 1), ('S', 1), ('CX', 3), ('CX', 2), ]
        gate_set['D4'] = [('H', 1), ('CX', 3), ('CX', 2)]

        gate_set['E1'] = [(None, 0)]
        gate_set['E2'] = [('S', 0)]
        gate_set['E3'] = [('Z', 0)]
        gate_set['E4'] = [('Z', 0), ('S', 0)]

        return gate_set

    @staticmethod
    def _L(L_pattern: List[List[str]]) -> List[Tuple[str, Tuple]]:
        r"""L form of clifford constructed by A, B, C basic gates.

        :param L_pattern: The pattern of L part
        :return: L form of clifford
        """

        m = len(L_pattern[1])
        L = []
        for g_list in L_pattern:
            for i, g in enumerate(g_list):
                if g[0] == 'A':
                    L.append((g, (m,)))
                elif g[0] == 'C':
                    L.append((g, (0,)))
                elif g[0] == 'B':
                    L.append((g, (m - i - 1, m - i)))
        return L

    @staticmethod
    def _M(M_pattern: List[List[str]]) -> List[Tuple[str, Tuple]]:
        r"""M form of clifford constructed by D, E basic gates.

        :param M_pattern:The pattern of M part
        :return: M form of clifford
        """

        n = len(M_pattern[0])
        M = []
        for g_list in M_pattern:
            for i, g in enumerate(g_list):
                if g[0] == 'E':
                    M.append((g, (n,)))
                elif g[0] == 'D':
                    M.append((g, (i, i + 1)))
        return M

    def _construct_circuit(self, gate_type: str, *idx: int) -> List[CircuitLine]:
        r"""Construct quantum circuit from the gate types and indices.

        :param gate_type: The basic gate convert to the elementary quantum gate
        :param idx: The qubits index
        :return:
        """
        total_cir = []
        idx = list(idx)
        gate_list = self._gate_set[gate_type]

        for gate_name, i in gate_list:
            cir = CircuitLine()
            if gate_name is not None:
                cir.data = FixedGate.getFixedGateInstance(gate_name)
                if gate_name == 'CX':
                    # Be careful about the control qubits and target qubits
                    if i == 2:
                        cir.qRegList = [idx[0], idx[1]]
                    else:
                        cir.qRegList = [idx[1], idx[0]]
                else:
                    cir.qRegList = [idx[i]]
                total_cir.append(cir)
        return total_cir


def random_pattern(n: int) -> List[List[List[str]]]:
    r"""Generate a random pattern to create clifford operator.

    See the documentation of the Clifford class for more details.

    **Usage**

    .. code-block::python
        :linenos:

        pattern = random_pattern(n=2)

    :param n: number of qubits
    :return: The pattern for initialize the Clifford class

    **Examples**

        >>> from qcompute_qep.quantum.clifford import random_pattern
        >>> n=2
        >>> pattern = random_pattern(n)
        >>> print(pattern)
        [[['A2'], ['B1'], ['C2'], ['D2'],
        ['E2']], [['A1'], [], ['C1'], [], ['E3']]]
    """
    if n <= 0:
        raise ArgumentError('n should be larger than 0')
    A_type = ['A1', 'A2', 'A3']
    B_type = ['B1', 'B2', 'B3', 'B4']
    C_type = ['C1', 'C2']
    D_type = ['D1', 'D2', 'D3', 'D4']
    E_type = ['E1', 'E2', 'E3', 'E4']
    pattern = []
    for j in range(n, 0, -1):
        k = np.random.randint(1, j + 1)
        A = [np.random.choice(A_type)]
        B = [np.random.choice(B_type) for _ in range(k - 1)]
        C = [np.random.choice(C_type)]
        D = [np.random.choice(D_type) for _ in range(j - 1)]
        E = [np.random.choice(E_type)]
        pattern.append([A, B, C, D, E])
    return pattern


def random_clifford(n: int, m: int = 1) -> List[Clifford]:
    r"""Randomly generate :math:`m` :math:`n`-qubits Clifford operators.

    Notice that when :math:`m=1`, i.e., a single-qubit Clifford is generated, the return value is still a List.

    **Usage**

    .. code-block::python
        :linenos:

        cliff = random_clifford(n=2)
        cliff_list = random_clifford(n=2,m=5)

    :param n: int, the number of qubits
    :param m: int, the number of clifford operators that will be randomly generated
    :return: Union[Clifford, List[Clifford]], a list of clifford operators

    **Examples**

        >>> from qcompute_qep.quantum.clifford import random_clifford
        >>> from qcompute_qep.utils.circuit import print_circuit
        >>> n = 2
        >>> m = 2
        >>> cliffords = random_clifford(n, m)
        >>> for c in cliffords:
        >>>     print_circuit(c.circuit)
        0: ---H---S---H---I---H---@---H---@---I---H---S---S---H---S---S---
                                  |       |
        1: ---H---S---------------Z---H---Z---H---S---S-------------------
        0: -------------------@---H---@---H---@---I---H---@---H---@---H---H---S---S---H---I---
                              |       |       |           |       |
        1: ---H---S---H---H---Z---H---Z---H---Z---H---S---Z---H---Z---H---S-------------------
    """
    if m <= 0:
        raise ArgumentError('m should not smaller than 0')

    clifford_list = []

    for i in range(m):
        pattern = random_pattern(n)
        cliff = Clifford(n, pattern)
        clifford_list.append(cliff)

    return clifford_list


def complete_cliffords(n: int, ) -> List[Clifford]:
    r"""Generate the complete set of :math:`k`-qubit Clifford operators.

    .. note::

        The number of qubits should not larger than :math:`2`, otherwise there are too many Clifford operators.
        The size of the complete set of :math:`k`-qubit Clifford operators is given by

            .. math:: C(k) = \prod_{i=1}^{k}2(4^{i}-1)4^i.

        We have :math:`C(3) = 92897280` and :math:`C(4) = 12128668876800`.

    :param n: int, the number of qubits
    :return: List[Clifford], the complete set of :math:`k`-qubit Clifford operators
    """
    if n > 2:
        ArgumentError('n should not larger than 2!')

    A = ['A1', 'A2', 'A3']
    B = ['B1', 'B2', 'B3', 'B4']
    C = ['C1', 'C2']
    D = ['D1', 'D2', 'D3', 'D4']
    E = ['E1', 'E2', 'E3', 'E4']

    # List to store the matrix of gate
    complete_cliff = []

    # get the complete clifford group by using backtrack method
    def backtrack(k, pattern):
        if k == 0:
            if pattern:
                cliff = Clifford(n, pattern)
                complete_cliff.append(cliff)
            return

        for a in A:
            for m in range(k, 0, -1):
                for B_perm in itertools.product(B, repeat=m - 1):
                    for c in C:
                        for D_perm in itertools.product(D, repeat=k - 1):
                            for e in E:
                                backtrack(k - 1, pattern + [[[a], list(B_perm), [c], list(D_perm), [e]]])
        return

    backtrack(n, [])

    return complete_cliff


if __name__ == '__main__':
    # pattern = random_pattern(2)
    # print(pattern)
    cliffs = random_clifford(3, 1)
    for cliff in cliffs:
        print(cliff._pattern)
        print(cliff)
    # print(len(complete_cliffords(2)))
