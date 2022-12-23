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

r"""
This script is an implement of block-encoding by Linear Combination of Unitary Operations (LCU).

Given a Hermitian matrix :math:`\check H\in\mathbb C^{2^n\times2^n}` represented by a linear combination of multi-qubit Paulis:

.. math::

    \check H := \sum_j a_j P_j\in\mathbb C^{2^n\times2^n},

where :math:`a_j\in\mathbb R`, :math:`P_j` is a :math:`n`-qubit Pauli matrix,
i.e. the tensor product of :math:`n` Pauli matrices.

We need to implement a circuit :math:`\operatorname{Pre}` to prepare the state
:math:`|sel\rangle := \sum_j \sqrt{a_j}|j\rangle`, i.e. :math:`\operatorname{Pre}|0\rangle = |sel\rangle`,
and the circuit :math:`\operatorname{Sel}:= \sum_j |j\rangle\langle j| \otimes P_j`.

Then we have the block-encoding:

.. math::

    (\langle 0|\otimes I_{2^n}) \cdot \operatorname{BE} \cdot
    (|0\rangle \otimes I_{2^n}) = \check H, \text{ where }
    \operatorname{BE}:=(\operatorname{Pre}^\dagger \otimes I_{2^n}) \cdot \operatorname{Sel} \cdot
    (\operatorname{Pre}\otimes I_{2^n})

and also the block-encoding for control version:

.. math::

    \begin{aligned}
    (I_2 \otimes \langle 0| \otimes I) \cdot \operatorname{BE} \cdot
    (I_2 \otimes |0\rangle \otimes I) = C(\check H), \text{ where }\\
    \operatorname{BE}:=(I_2 \otimes \operatorname{Pre}^\dagger \otimes I_{2^n}) \cdot C(\operatorname{Sel}) \cdot
    (I_2 \otimes \operatorname{Pre} \otimes I_{2^n}),
    \end{aligned}

:math:`I_{N}` is the identity matrix of dimension :math:`N`, and :math:`C(U)` is the control version for
any quantum operator :math:`U\in\mathbb C^{2^l\times2^l}`, which is defined in math:

.. math::

    C(U):=|0\rangle\langle0|\otimes I_{2^l}+|1\rangle\langle1|\otimes U\in\mathbb{C}^{2^{l+1}\times2^{l+1}}.

Reader may refer to following references for more insights.

.. [CW12] Childs, Andrew M., and Nathan Wiebe. "Hamiltonian simulation using linear combinations of unitary operations."
    arXiv preprint arXiv:1202.5822 (2012).

.. [CGJ18] Chakraborty, Shantanav, András Gilyén, and Stacey Jeffery. "The power of block-encoded matrix powers:
    improved regression techniques via faster Hamiltonian simulation." arXiv preprint arXiv:1804.01973 (2018).
"""

import re
from typing import List, Optional, Tuple

from QCompute import X
from QCompute.QPlatform.QRegPool import QRegStorage

from qcompute_qsvt.Gate.MultiCtrlGates import circ_multictrl_Z, circ_multictrl_Pauli
from qcompute_qsvt.Oracle.StatePreparation import circ_state_pre, circ_state_pre_inverse


def circ_j_ctrl_multiPauli(reg_target: List[QRegStorage], reg_ctrlling: List[QRegStorage], int_j: int,
                           str_Pauli_term: str, if_minus: bool,
                           reg_borrowed: Optional[List[QRegStorage]] = None) -> None:
    r"""A quantum circuit implementing a multictrl multi-qubit Pauli gate.

    For such multictrl multi-qubit Pauli gate:

    .. math::

        C_j((-1)^{\operatorname{IF}} P) := (-1)^{\operatorname{IF}}|j\rangle\langle j| \otimes P +
            (I_{2^m}-|j\rangle\langle j|) \otimes I_{2^n}\in\mathbb C^{2^{m+n}\times2^{m+n}}

    with :math:`(-1)^{\operatorname{IF}}=\pm1`, :math:`P` an :math:`n`-qubit Pauli gate, :math:`j=0,1,\cdots,2^m-1`,
    :math:`|j\rangle` a :math:`m`-qubit state, we need to prepare a quantum circuit implementing such quantum operation
    :math:`C_j(P)` with input a string representation for :math:`P`.

    When we call this function, we will operate a quantum circuit implementing :math:`C_j((-1)^{\operatorname{IF}}P)`
    on state :math:`|c\rangle|t\rangle` and obtain the state :math:`C_j((-1)^{\operatorname{IF}}P)|c\rangle|t\rangle`
    in register :math:`|c\rangle|t\rangle`. In math, here we have

    .. math::

        C_j((-1)^{\operatorname{IF}}P)|c\rangle|t\rangle=\begin{cases}
            |c\rangle (-1)^{\operatorname{IF}}P|t\rangle,&\text{if } c=j;\\
            |c\rangle|t\rangle,&\text{if } c\ne j\in\mathbb Z.
        \end{cases}

    :param reg_target: :math:`|t\rangle`, `List[QRegStorage]`,
        the target (:math:`n`-qubit) register of the multictrl multi-qubit Pauli gate
    :param reg_ctrlling: :math:`|c\rangle`, `List[QRegStorage]`,
        the controlling (:math:`m`-qubit) register of the multictrl multi-qubit Pauli gate
    :param int_j: :math:`j`, `int`, indicating when the control gate operates
    :param str_Pauli_term: :math:`P`, `str`,
        a string such as `\'X0Y2Z3\'` indicating which multi-qubit Pauli gate we operate
    :param if_minus: :math:`\operatorname{IF}`, `bool`,
        indicating whether the coefficient of this multi-qubit Pauli gate is negative
    :param reg_borrowed: :math:`|b\rangle`, `Optional[List[QRegStorage]]`,
        a quantum register containing several qubits as the borrowed qubits for implementing multictrl gates
    :return: **None**

    **Examples**

        The quantum gate :math:`C_5(X\otimes I\otimes Y\otimes Z)\in\mathbb C^{2^7\times 2^7}` could be called by

        >>> from QCompute import QEnv
        >>> from qcompute_qsvt.Oracle.BlockEncoding import circ_j_ctrl_multiPauli
        >>> env = QEnv()
        >>> reg_c = [env.Q[0], env.Q[1], env.Q[2]]
        >>> reg_t = [env.Q[3], env.Q[4], env.Q[5], env.Q[6]]
        >>> circ_j_ctrl_multiPauli(reg_t, reg_c, 5, 'X0Y2Z3', False)

        Then we operate such quantum gate on quantum registers ``reg_c`` and ``reg_t``.
    """
    if reg_borrowed is None:
        reg_borrowed = []
    list_str_single_Pauli_term = re.split(r',\s*', str_Pauli_term.upper())
    # we can transform such j-ctrl gate into multictrl gate by conjugating operation with several X gates
    for int_k in range(len(reg_ctrlling)):
        if (int_j >> int_k) % 2 == 0:
            X(reg_ctrlling[-1 - int_k])
    for str_single_Pauli_term in list_str_single_Pauli_term:
        match = re.match(r'([XYZ])([0-9]+)(\w+)', str_single_Pauli_term, flags=re.I)
        while match:
            circ_multictrl_Pauli(reg_target[int(match.group(2))], reg_ctrlling, match.group(1).upper(),
                                 reg_borrowed=reg_target[:int(match.group(2))] + reg_target[int(match.group(2)) + 1:] +
                                 reg_borrowed)
            str_single_Pauli_term = match.group(3)
            match = re.match(r'([XYZ])([0-9]+)(\w+)', str_single_Pauli_term, flags=re.I)
        match = re.match(r'([XYZ])([0-9]+)', str_single_Pauli_term, flags=re.I)
        if match:
            circ_multictrl_Pauli(reg_target[int(match.group(2))], reg_ctrlling, match.group(1).upper(),
                                 reg_borrowed=reg_target[:int(match.group(2))] + reg_target[int(match.group(2)) + 1:] +
                                 reg_borrowed)
    # We need to implement a multictrl-(-ID) gate,
    # which just equals to a multictrl-Z gate operated on those ctrlling qubits.
    if if_minus and len(reg_ctrlling) > 0:
        circ_multictrl_Z(reg_ctrlling[-1], reg_ctrlling[:-1], reg_borrowed=reg_target + reg_borrowed)
    # the other part of the conjugating operation with several X gates
    for int_k in range(len(reg_ctrlling)):
        if (int_j >> int_k) % 2 == 0:
            X(reg_ctrlling[-1 - int_k])


def circ_ctrl_Sel_multiPauli(reg_target: List[QRegStorage], qubit_ctrlling: Optional[QRegStorage],
                             reg_ctrlling: List[QRegStorage], list_str_Pauli_rep: List[Tuple[float, str]],
                             reg_borrowed: Optional[List[QRegStorage]] = None) -> None:
    r"""A quantum circuit implementing :math:`C(\operatorname{Sel})` with classical input :math:`\check H`.

    Since

    .. math::

        \operatorname{Sel}= \sum_j |j\rangle\langle j| \otimes P_j=\prod_jC_j(P_j),

    we have

    .. math::

        C(\operatorname{Sel})=\prod_jC(C_j(P_j)),

    where :math:`C(C_j(P_j))` could be regarded as another :math:`C_{j'}(P_j)`.

    When we call this function, a quantum circuit will be operated implementing :math:`C(\operatorname{Sel})`
    on state :math:`|c\rangle|j\rangle|t\rangle`,
    and the final state :math:`C(\operatorname{Sel})|c\rangle|j\rangle|t\rangle` is obtained
    in the register having restored :math:`|c\rangle|j\rangle|t\rangle`. In math, here we have

    .. math::

        C(\operatorname{Sel})|c\rangle|j\rangle|t\rangle=|c\rangle|j\rangle P_j^c|t\rangle,
        \text{ if } c,j\in\mathbb Z.

    :param reg_target: :math:`|t\rangle`, `List[QRegStorage]`, the target quantum register of :math:`\operatorname{Sel}`
    :param qubit_ctrlling: :math:`|c\rangle`, `Optional[QRegStorage]`,
        the ctrl qubit introduced in the ctrl version :math:`C(\operatorname{Sel})`;
        **qubit_ctrlling** could be **None**, indicating none-ctrl version :math:`\operatorname{Sel}`
    :param reg_ctrlling: :math:`|j\rangle`, the ctrlling quantum register of :math:`\operatorname{Sel}`
    :param list_str_Pauli_rep: :math:`\check H`, `List[Tuple[float, str]]`,
        a list of form :math:`(a_j, P_j)` such as
        `[(-0.09706, \'I\'), (-0.045302, \'X0X1Y2Y3\'), (0.17441, \'Z2Z3\'), (-0.2234, \'Z3\')]`,
        where :math:`a_j` is a float indicating the coefficient of :math:`P_j` in :math:`\check H`,
        and :math:`P_j` is a string indicating which multi-Pauli gate
    :param reg_borrowed: :math:`|b\rangle`, `Optional[List[QRegStorage]]`,
        a quantum register containing several qubits as the borrowed qubits for implementing multictrl gates
    :return: **None**
    """
    if reg_borrowed is None:
        reg_borrowed = []
    # the step length change instead of 1 when regarding such C(C_j(P)) gate as C_j'(P)
    int_deviation = 2 ** len(reg_ctrlling)
    for idx in range(len(list_str_Pauli_rep)):
        # regard two part of ctrlling qubits as a same class
        if qubit_ctrlling is None:
            circ_j_ctrl_multiPauli(reg_target, reg_ctrlling, idx + int_deviation, list_str_Pauli_rep[idx][1],
                                   list_str_Pauli_rep[idx][0] < 0, reg_borrowed=reg_borrowed)
        else:
            circ_j_ctrl_multiPauli(reg_target, [qubit_ctrlling] + reg_ctrlling, idx + int_deviation,
                                   list_str_Pauli_rep[idx][1], list_str_Pauli_rep[idx][0] < 0,
                                   reg_borrowed=reg_borrowed)


def circ_block_encoding(reg_sys: List[QRegStorage], reg_blocking: List[QRegStorage],
                        qubit_ctrlling: Optional[QRegStorage], list_str_Pauli_rep: List[Tuple[float, str]],
                        list_float_target_state: List[float],
                        reg_borrowed: Optional[List[QRegStorage]] = None) -> None:
    r"""A quantum circuit implementing a block-encoding for :math:`C(\check H)` with classical input :math:`\check H`.

    Reader may refer to [CW12]_ for more insights.

    By

    .. math::

        (\langle 0|\operatorname{Pre}^\dagger \otimes I_{2^n}) \cdot \operatorname{Sel} \cdot
        (\operatorname{Pre}|0\rangle \otimes I_{2^n}) = R,

    we have :math:`\operatorname{Pre}^\dagger \otimes I_{2^n})\operatorname{Sel}(\operatorname{Pre}\otimes I_{2^n})` is
    a block-encoding for :math:`\check H`. Similarly, we have
    :math:`(I_2\otimes\operatorname{Pre}^\dagger\otimes I)C(\operatorname{Sel})(I_2\otimes\operatorname{Pre}\otimes I)`
    is a block-encoding for :math:`C(\check H)`.

    When we call this function, a quantum circuit will be operated implementing a block-encoding
    :math:`\operatorname{BE}` of :math:`C(\check H)` (or :math:`\check H` for the case **qubit_ctrlling** is **None**)
    on state :math:`|c\rangle|a\rangle|s\rangle`,
    and the final state :math:`\operatorname{BE}|c\rangle|a\rangle|s\rangle` is obtained
    in the register having restored :math:`|c\rangle|a\rangle|s\rangle`. In math, for the case **qubit_ctrlling** is
    not **None**, we have

    .. math::

        (I\otimes\langle0|\otimes I_{2^n})\operatorname{BE}|c\rangle|a\rangle|s\rangle=C(\check H)|c\rangle|s\rangle,
        \text{ if } |a\rangle=|0\rangle.

    and for the case **qubit_ctrlling** is **None**, we have

    .. math::

        (\langle0|\otimes I_{2^n})\operatorname{BE}|a\rangle|s\rangle=\check H|s\rangle,\text{ if } |a\rangle=|0\rangle.

    :param reg_sys: :math:`|s\rangle`, `List[QRegStorage]`, the system register for block-encoding
    :param reg_blocking: :math:`|a\rangle`, `List[QRegStorage]`, the ancilla register introduced in block-encoding step
    :param qubit_ctrlling: :math:`|c\rangle`, `Optional[QRegStorage]`,
        the ctrl qubit introduced in a block-encoding of the ctrl version :math:`C(\check H)`;
        **qubit_ctrlling** could be **None**, indicating a block-encoding of none-ctrl version :math:`\check H`
    :param list_str_Pauli_rep: :math:`\check H`, `List[Tuple[float, str]]`,
        a list of form :math:`(a_j, P_j)` such as
        `[(-0.09706, \'I\'), (-0.045302, \'X0X1Y2Y3\'), (0.17441, \'Z2Z3\'), (-0.2234, \'Z3\')]`,
        where :math:`a_j` is a float indicating the coefficient of :math:`P_j` in :math:`\check H`,
        and :math:`P_j` is a string indicating which multi-Pauli gate
    :param list_float_target_state: :math:`\vec s`, `List[float]`, corresponding to :math:`\check H` and also be the
        classical information for :math:`|sel\rangle`
    :param reg_borrowed: :math:`|b\rangle`, `Optional[List[QRegStorage]]`,
        a quantum register containing several qubits as the borrowed qubits for implementing multictrl gates
    :return: **None**
    """
    circ_state_pre(reg_blocking, [], list_float_target_state, reg_borrowed=reg_sys + reg_borrowed)
    circ_ctrl_Sel_multiPauli(reg_sys, qubit_ctrlling, reg_blocking, list_str_Pauli_rep,
                             reg_borrowed=reg_sys + reg_borrowed)
    circ_state_pre_inverse(reg_blocking, [], list_float_target_state, reg_borrowed=reg_sys + reg_borrowed)
