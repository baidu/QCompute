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
This script is an implement for an algorithm for state preparation.
The target state should be a real vector who has :math:`2`-norm :math:`1`.

When the vector is not of dimension a power of :math:`2`,
it will be encoded as another vector of dimension a power of :math:`2` by filling zeros at the end of the origin vector.

The main idea of this implement refers to [LS01]_.

Reader may refer to the file **Gate** for more information.

.. [LS01] Long, Gui-Lu, and Yang Sun. "Efficient scheme for initializing a quantum register with an arbitrary superposed
    state." Physical Review A 64.1 (2001): 014303.
"""

from typing import List, Optional

import numpy as np
from QCompute import X
from QCompute.QPlatform.QRegPool import QRegStorage

from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Gate.MultiCtrlGates import circ_multictrl_ry


def circ_state_pre(reg_sys: List[QRegStorage], reg_ctrlling: Optional[List[QRegStorage]],
                   list_float_target_state: List[float], reg_borrowed: Optional[List[QRegStorage]] = None) -> None:
    r"""A quantum circuit (or its ctrl version) to prepare a quantum state with classical input.

    Given a normalized real vector :math:`\vec t`, we will implement a quantum circuit :math:`\operatorname{Pre}` or
    its some ctrl version :math:`C^*(\operatorname{Pre})` satisfying

    .. math:: \operatorname{Pre}|s\rangle=|t\rangle,\text{ if }|s\rangle=|0\rangle,

    where each entry of :math:`|t\rangle` equals to the absolutely value of the corresponding entry in :math:`\vec t`.

    In math, the effect in computing basis:

    .. math::

        C^*(\operatorname{Pre})|c\rangle|s\rangle=
        \begin{cases}
        |c\rangle |t\rangle,&
        \text{ if }|c\rangle=|11\cdots1\rangle
        \text{ and }|s\rangle=|0\rangle;\\
        |c\rangle|s\rangle,&\text{ if }|c\rangle\perp|11\cdots1\rangle.
        \end{cases}

    :param reg_sys: :math:`|s\rangle`, `List[QRegStorage]`, the system register used to store the prepared quantum state
    :param reg_ctrlling: :math:`|c\rangle`, `List[QRegStorage]` of length :math:`n` or **None**,
        a quantum register containing several qubits as the ctrlling qubits
        for such :math:`C^*(\operatorname{Pre})` operator;
        it is noted that :math:`|c\rangle` may be **None** indicating none-ctrl version,
        i.e. such :math:`\operatorname{Pre}` operator
    :param list_float_target_state: :math:`\vec t`, `List[QRegStorage]`,
        a normalized real vector would to be regarded as a quantum state
    :param reg_borrowed: :math:`|b\rangle`, `Optional[List[QRegStorage]]`,
        a quantum register containing several qubits as the borrowed qubits for implementing multictrl gates
    :return: **None**

    **Examples**

        The quantum state :math:`|t\rangle:=0.25|0\rangle+0.25|1\rangle+0.5|2\rangle+0.5|3\rangle
        +0.5|4\rangle+0.25|5\rangle+0.25|6\rangle` could be prepared by

        >>> from QCompute import QEnv
        >>> from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Oracle.StatePreparation import circ_state_pre
        >>> list_t = [0.25, 0.25, 0.5, 0.5, 0.5, 0.25, 0.25]
        >>> env = QEnv()
        >>> reg_s = [env.Q[0], env.Q[1], env.Q[2]]
        >>> circ_state_pre(reg_s, None, list_t)

        Now the state in ``reg_s`` is at :math:`|t\rangle`.
    """
    if reg_borrowed is None:
        reg_borrowed = []
    if reg_ctrlling is None:
        reg_ctrlling = []
    # If t is not a unit vector, we normalize it.
    float_norm = np.sqrt(sum(idx_cor ** 2 for idx_cor in list_float_target_state))
    if float_norm != 1.0 and float_norm != 0.0:
        list_float_target_state = list(idx_cor / float_norm for idx_cor in list_float_target_state)
    # We need to know how many qubits are needed to encode the target state |t> first.
    int_len = len(list_float_target_state)
    num_qubit_used = max(int(np.ceil(np.log2(int_len))), 1)  # the number of qubits needed to encode state |t>
    # Fill the list by 0 at the end to make the dimension of t is a power of 2
    list_float_target_state += [0 for _ in range(2 ** num_qubit_used - int_len)]
    if len(reg_sys) > num_qubit_used:  # If the used qubits is less than those in |s>,
        reg_sys = reg_sys[-num_qubit_used:]  # we only use a part of those in |s>.
    float_norm_half_1 = np.sqrt(sum(idx_cor ** 2 for idx_cor in list_float_target_state[:2 ** (num_qubit_used - 1)]))
    float_norm_half_2 = np.sqrt(sum(idx_cor ** 2 for idx_cor in list_float_target_state[2 ** (num_qubit_used - 1):]))
    float_theta = 2 * np.arccos(float_norm_half_1)  # compute the rotation angle in this recursion
    circ_multictrl_ry(reg_sys[0], reg_ctrlling, float_theta, reg_borrowed=reg_sys[1:] + reg_borrowed)
    if len(reg_sys) > 1:  # if we haven't prepared the target state
        if float_norm_half_1 != 0.0:  # if the first half needs to prepare
            list_float_state_half_1 = list(
                idx_cor / float_norm_half_1 for idx_cor in list_float_target_state[:2 ** (num_qubit_used - 1)])
            X(reg_sys[0])
            # introduce a recursion to prepare the state l_f_s_h_1 by set reg_sys[0] as a ctrlling qubit
            circ_state_pre(reg_sys[1:], reg_ctrlling + [reg_sys[0]], list_float_state_half_1,
                           reg_borrowed=reg_borrowed)
            X(reg_sys[0])
        if float_norm_half_2 != 0.0:  # if the second half needs to prepare
            # here is another recursion to prepare the state l_f_s_h_2
            list_float_state_half_2 = list(
                idx_cor / float_norm_half_2 for idx_cor in list_float_target_state[2 ** (num_qubit_used - 1):])
            circ_state_pre(reg_sys[1:], reg_ctrlling + [reg_sys[0]], list_float_state_half_2,
                           reg_borrowed=reg_borrowed)


def circ_state_pre_inverse(reg_sys: List[QRegStorage], reg_ctrlling: Optional[List[QRegStorage]],
                           list_float_target_state: List[float],
                           reg_borrowed: Optional[List[QRegStorage]] = None) -> None:
    r"""The inverse of such quantum circuit implementing by **circ_state_pre** with the same input.

    :param reg_sys: :math:`|s\rangle`, `List[QRegStorage]`, the system register used to store the prepared quantum state
    :param reg_ctrlling: :math:`|c\rangle`, `List[QRegStorage]` of length :math:`n` or **None**,
        a quantum register containing several qubits as the ctrlling qubits
        for such :math:`C^*(\operatorname{Pre})` operator;
        it is noted that :math:`|c\rangle` may be **None** indicating none-ctrl version,
        i.e. such :math:`\operatorname{Pre}` operator
    :param list_float_target_state: :math:`\vec t`, `List[QRegStorage]`,
        a normalized real vector would to be regarded as a quantum state
    :param reg_borrowed: :math:`|b\rangle`, `Optional[List[QRegStorage]]`,
        a quantum register containing several qubits as the borrowed qubits for implementing multictrl gates
    :return: **None**
    """
    # If t is not a unit vector, we normalize it.
    float_norm = np.sqrt(sum(idx_cor ** 2 for idx_cor in list_float_target_state))
    if float_norm != 1.0 and float_norm != 0.0:
        list_float_target_state = list(idx_cor / float_norm for idx_cor in list_float_target_state)
    # We need to know how many qubits are needed to encode the target state |t> first.
    int_len = len(list_float_target_state)
    num_qubit_used = max(int(np.ceil(np.log2(int_len))), 1)  # the number of qubits needed to encode state |t>
    # Fill the list by 0 at the end to make the dimension of t is a power of 2
    list_float_target_state += [0 for _ in range(2 ** num_qubit_used - int_len)]
    if len(reg_sys) > num_qubit_used:  # If the used qubits is less than those in |s>,
        reg_sys = reg_sys[-num_qubit_used:]  # we only use a part of those in |s>.
    float_norm_half_1 = np.sqrt(sum(idx_cor ** 2 for idx_cor in list_float_target_state[:2 ** (num_qubit_used - 1)]))
    float_norm_half_2 = np.sqrt(sum(idx_cor ** 2 for idx_cor in list_float_target_state[2 ** (num_qubit_used - 1):]))
    float_theta = 2 * np.arccos(float_norm_half_1)
    if len(reg_sys) > 1:
        if float_norm_half_2 != 0.0:
            list_float_state_half_2 = list(idx_cor / float_norm_half_2 for idx_cor in
                                           list_float_target_state[2 ** (num_qubit_used - 1):])
            circ_state_pre_inverse(reg_sys[1:], reg_ctrlling + [reg_sys[0]], list_float_state_half_2,
                                   reg_borrowed=reg_borrowed)
        if float_norm_half_1 != 0.0:
            list_float_state_half_1 = list(idx_cor / float_norm_half_1 for idx_cor in
                                           list_float_target_state[:2 ** (num_qubit_used - 1)])
            X(reg_sys[0])
            circ_state_pre_inverse(reg_sys[1:], reg_ctrlling + [reg_sys[0]], list_float_state_half_1,
                                   reg_borrowed=reg_borrowed)
            X(reg_sys[0])
    # In the inverse circuit, the multictrl Ry gate is operated lastly, and also the rotation angle need to take the
    # opposite number
    circ_multictrl_ry(reg_sys[0], reg_ctrlling, -float_theta, reg_borrowed=reg_sys[1:] + reg_borrowed)
