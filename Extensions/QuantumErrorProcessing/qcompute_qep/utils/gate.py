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
The built-in quantum gates in `QCompute` is very basic and low level.
In this script, we implement high level quantum gates (composed of many low level quantum gates)
that follow the same syntax of  `QCompute`.
"""
import copy
from typing import List
import numpy as np
import re
import QCompute
from QCompute.QPlatform.QOperation import QOperation, CircuitLine

import Extensions.QuantumErrorProcessing.qcompute_qep.utils.MultiCtrlGates as multictrl
from Extensions.QuantumErrorProcessing.qcompute_qep.exceptions import ArgumentError


class CPauliOP(QOperation):
    r"""Implement a high-level multi-control Pauli gate.

    This class builds a high level multi-control Pauli gate following the syntax of `QCompute`.
    Mathematically, a multi-control Pauli gate achieves the following unitary evolution:

    .. math::

        C^{m,n}_j(P)\vert c\rangle\!\vert t\rangle =
        \begin{cases}
                \vert c\rangle P\vert t\rangle, &\text{if } \vert c\rangle=\vert j \rangle;\\
                \vert c\rangle\!\vert t\rangle,  &\text{otherwise},
        \end{cases}

    where

    + :math:`m` is the number of control qubits :math:`\vert c\rangle` (specified by the argument ``reg_c``),

    + :math:`n` is the number of target qubits :math:`\vert t\rangle` (specified by the argument ``reg_t``),

    + :math:`j` is the conditional classical value (specified by the argument ``val``), and

    + :math:`P` is the target Pauli operator (specified by the argument ``pauli`` string).

    .. note::

        The qubits in the control register must be ordered, and
        we assume the LSB (the least significant bit) mode when evaluating its control value:

        ::

            reg_c:        [q[i]   q[j]  q[k]]

              bit:         b[2]   b[1]  b[0]

        For example, :math:`011 \mapsto 3` and :math:`100 \mapsto 4`.

    .. note::

        The qubits in the target register must need to be ordered, and
        we assume a one-one mapping between the set of target register and the Pauli string:

        ::

            reg_t:        [q[i]   q[j]  q[k]]

            Pauli:         'X      I     Y'

        That is, the controlled X gate will operate on q[i] and the controlled Y gate will operate on q[k].

    **Examples**

        The quantum gate :math:`C^{2,3}_0(XYZ)` could be called by

        >>> from Extensions.QuantumErrorProcessing.qcompute_qep.utils.circuit import print_circuit
        >>> qp = QCompute.QEnv()
        >>> qp.Q.createList(5)
        >>> reg_c = [qp.Q[0], qp.Q[1]]
        >>> reg_t = [qp.Q[2], qp.Q[3], qp.Q[4]]
        >>> CPauli(reg_c=reg_c, reg_t=reg_t, val=0, pauli='XYZ')
        >>> print_circuit(qp.circuit)
        0: ---●---
              |
        1: ---○---
              |
        2: ---X---
              |
        3: ---Y---
              |
        4: ---Z---
    """
    def __init__(self) -> None:
        super().__init__(name='CPauli', bits=None, matrix=None)

    def __call__(self, **kwargs) -> None:
        r"""

        :param reg_c: List[QRegStorage], the control quantum register
        :param reg_t: List[QRegStorage], the target quantum register
        :param val: int, the control value for which the Pauli gate operates
        :param pauli: str, a string such as 'XXIZ' specifies the Pauli gate
        :return: None
        """
        self.reg_c = kwargs.get('reg_c')
        self.reg_t = kwargs.get('reg_t')
        self.val = kwargs.get('val')
        self.pauli = kwargs.get('pauli')
        self.bits = len(self.reg_c) + len(self.reg_t)

        # Check the consistency of data
        if not bool(re.match(r'^[IXYZ]+$', self.pauli)):
            raise ArgumentError("CPauliOP: the Pauli string contains unrecognized Pauli character(s). "
                                "Supported Pauli characters are: 'I', 'X', 'Y', 'Z'.")
        if 2 ** len(self.reg_c) <= self.val:
            raise ArgumentError("CPauliOP: the control value {} is larger than the "
                                "maximal value that the control qubits can achieve.".format(self.val))
        if len(self.reg_t) != len(self.pauli):
            raise ArgumentError("in CPauli(): the number of qubits in the target register is not equal to "
                                "the number of qubits in the Pauli string.")

        # The control and target registers must have the same quantum environment
        env = self.reg_c[0].env
        for qubit in self.reg_c:
            if qubit.env != env:
                raise ArgumentError('CPauliOP(): The control qubits must belong to the same quantum environment!')
        for qubit in self.reg_t:
            if qubit.env != env:
                raise ArgumentError('CPauliOP(): The control and target qubits must belong '
                                    'to the same quantum environment!')

        # Create a new CircuitLine instance
        qubits = [qubit.index for qubit in self.reg_c] + [qubit.index for qubit in self.reg_t]
        env.circuit.append(CircuitLine(data=copy.copy(self), qRegList=qubits, cRegList=None))


# Create a Controlled-Pauli gate instance
CPauli = CPauliOP()


def decompose_CPauli(cp: CPauli) -> List[CircuitLine]:
    r"""Decompose the high level multi-control Pauli gate into native gates.

    This function decomposes a high level multi-control Pauli gate into many native gates originally
    supported in `QCompute`. Mathematically, a multi-control Pauli gate achieves the following unitary evolution:

    .. math::

        C^{m,n}_j(P)\vert c\rangle\!\vert t\rangle =
        \begin{cases}
                \vert c\rangle P\vert t\rangle, &\text{if } \vert c\rangle=\vert j \rangle;\\
                \vert c\rangle\!\vert t\rangle,  &\text{otherwise},
        \end{cases}

    where

    + :math:`m` is the number of control qubits :math:`\vert c\rangle` (specified by the argument ``reg_c``),

    + :math:`n` is the number of target qubits :math:`\vert t\rangle` (specified by the argument ``reg_t``),

    + :math:`j` is the conditional classical value (specified by the argument ``val``), and

    + :math:`P` is the target Pauli operator (specified by the argument ``pauli`` string).

    In the implementation, we decompose this gate to many multi-control single-qubit Pauli gate,
    conditioned on the all-:math:`1` value.
    If @val is not all-:math:`1`, we have to flip the qubits whose control values are :math:`0`,
    then do the multi-control single-qubit Pauli gate, then flip the qubits back.

    .. note::

        The qubits in the control register must be to be ordered, and
        we assume the LSB (the least significant bit) mode when evaluating its control value:

        ::

            reg_c:        [q[i]   q[j]  q[k]]

              bit:         b[2]   b[1]  b[0]

        For example, :math:`011 \mapsto 3` and :math:`100 \mapsto 4`.

    .. note::

        The qubits in the target register must be ordered, and
        we assume a one-one mapping between the set of target register and the Pauli string:

        ::

            reg_t:        [q[i]   q[j]  q[k]]

            Pauli:         'X      I     Y'

        That is, the controlled X gate will operate on q[i] and the controlled Y gate will operate on q[k].

    :param cp: CPauli, a multi-control Pauli gate instance
    :return: List[CircuitLine], a list of CircuitLine objects decomposing the multi-control Pauli gate
    """

    # Number of CircuitLines before decomposing the CPauli gate
    old_idx = len(cp.reg_c[0].env.circuit)
    # Step 1. If the control value is 0, we flip the corresponding qubit so that the control value is 1.
    # **DO** remember to flip back after finishing the controlled operation
    # Convert the control value a list of binary values
    bit_list = [int(b) for b in np.binary_repr(cp.val, width=len(cp.reg_c))]
    for qubit, bit in zip(cp.reg_c, bit_list):
        if bit % 2 == 0:
            QCompute.X(qubit)

    # Step 2. Execute the multi-control Pauli gate,
    # where the control value is all-1 and the target gate is a single-qubit Pauli gate.
    # Convert the target multi-qubit Pauli operator a list of single-qubit Pauli operators
    pauli_list = [*cp.pauli]
    for qubit, single_pauli in zip(cp.reg_t, pauli_list):
        # We do not need to implement the controlled-I gate
        if single_pauli == 'I':
            continue
        else:
            multictrl.circ_multictrl_Pauli(qubit_target=qubit, reg_ctrlling=cp.reg_c, char_Pauli=single_pauli)

    # Flip the qubit back
    for qubit, bit in zip(cp.reg_c, bit_list):
        if int(bit) % 2 == 0:
            QCompute.X(qubit)

    # Number of CircuitLines after decomposing the CPauli gate
    new_idx = len(cp.reg_c[0].env.circuit)

    # The newly added CircuitLines are exactly those that implement the CPauli gate
    return cp.reg_c[0].env.circuit[old_idx:new_idx]
