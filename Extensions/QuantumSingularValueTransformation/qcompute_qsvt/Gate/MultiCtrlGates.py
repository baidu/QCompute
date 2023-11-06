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
This script contains the implement for decompositions of multi-control (abbr as multictrl) X gate, and several other
multictrl gates.
Reader may refer to following references for more insights.

.. [BBC+95] Barenco, Adriano, et al. "Elementary gates for quantum computation." Physical review A 52.5 (1995): 3457.

.. [G05] Craig Gidney. "Constructing Large Controlled Nots."
    https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html
"""

from typing import List, Optional

from QCompute import X, Y, Z, H, S, SDG, RY, RZ, CX, CY, CZ, CRY, CRZ, CCX
from QCompute.QPlatform.QRegPool import QRegStorage


def func_find_borrowable_qubits(reg_work: List[QRegStorage], num_qubits_borrowed: int = 1) -> List[QRegStorage]:
    r"""Find borrowable qubits based on working qubits.

    More details about borrowed qubits refer to [G05]_.

    When we'd like to implement a multictrl gate, a best way to decompose it is to introduce other borrowed qubit.

    This function will return a register containing several borrowable qubits
    after inputting working register and the number of borrowed qubits we need.
    Here borrowed qubits indicates those qubits not in the working register.

    :param reg_work: `List[QRegStorage]`,
        a quantum register which would be disjoint with such register containing borrowable qubits
    :param num_qubits_borrowed: `int`, the number of borrowed qubits we need
    :return: **reg_borrowed** – `List[QRegStorage]` of length **num_qubits_borrowed**,
        a quantum register containing **num_qubits_borrowed** qubits and disjoint with **reg_work**
    """
    reg_borrowed = []
    assert len(reg_work) > 0, "reg_work must be none-empty!"
    env_quantum = reg_work[0].env  # obtain the env from one qubit
    idx_qubit = 0
    # traverse to find qubits not in reg_work
    while len(reg_borrowed) < num_qubits_borrowed:
        if env_quantum.Q[idx_qubit] not in reg_work:
            reg_borrowed.append(env_quantum.Q[idx_qubit])
        idx_qubit += 1
    return reg_borrowed


def circ_multictrl_X(
    qubit_target: QRegStorage,
    reg_ctrlling: Optional[List[QRegStorage]],
    num_qubit_ctrlling: Optional[int] = None,
    reg_borrowed: Optional[List[QRegStorage]] = None,
) -> None:
    r"""An :math:`O(n)` decomposition for multictrl :math:`X` gate (@ :math:`C^n(X)`) with several borrowed qubits.

    More information refers to [BBC+95]_ and [G05]_.

    In math, :math:`C^n(X)` is defined as

    .. math::

        C^n(X):=|11\cdots1\rangle\langle11\cdots1|\otimes(X-I_2) + I_{2^{n+1}}.

    Also, we have the effect in computing basis:

    .. math::

        C^n(X)|c\rangle|t\rangle = \begin{cases}
            |c\rangle X|t\rangle,&\text{if }|c\rangle=|11\cdots1\rangle;\\
            |c\rangle|t\rangle,&\text{if }|c\rangle\perp|11\cdots1\rangle;
        \end{cases}

    It is noted that after calling this circuit, the quantum state in :math:`|b\rangle` would remains,
    like that we have operated :math:`I` gates on those qubit in :math:`|b\rangle`.

    :param qubit_target: :math:`|t\rangle`, `QRegStorage`, the target qubit of such :math:`C^n(X)` gate
    :param reg_ctrlling: :math:`|c\rangle`, `List[QRegStorage]` of length :math:`n` or **None**,
        a quantum register containing several qubits as the ctrlling qubits for such :math:`C^n(X)` gate;
        it is noted that :math:`|c\rangle` may be **None** indicating none-ctrl version, i.e. such :math:`X` gate
    :param num_qubit_ctrlling: :math:`n`, `int` or **None**,
        `int` indicating the length of :math:`|c\rangle`, or **None** indicating unknown length and need to count
    :param reg_borrowed: :math:`|b\rangle`, `List[QRegStorage]` or **None**,
        a quantum register containing several qubits as the borrowed qubits for such :math:`C^n(X)` gate;
        it is noted that :math:`|b\rangle` may also be **None** indicating an empty register
    :return: **None**
    """
    if num_qubit_ctrlling is None:
        num_qubit_ctrlling = len(reg_ctrlling)  # count the number n in reg_ctrlling
    if reg_borrowed is None:
        reg_borrowed = []
    if num_qubit_ctrlling == 0:  # for the case n == 0, it just |t> -> X|t>
        X(qubit_target)
    elif num_qubit_ctrlling == 1:  # for the case n == 1, it just |c>|t> -> CX|c>|t>
        CX(reg_ctrlling[0], qubit_target)
    elif num_qubit_ctrlling == 2:  # for the case n == 2, it just |c0>|c1>|t> -> CCX|c0>|c1>|t>
        CCX(reg_ctrlling[0], reg_ctrlling[1], qubit_target)
    else:  # for the case n >= 3
        if len(reg_borrowed) == 0:  # if there is no borrowable qubit, we need to find one
            # here the reg_work in following function is all qubits used, i.e. [qubit_target] + reg_ctrlling
            reg_borrowed = func_find_borrowable_qubits([qubit_target] + reg_ctrlling)
        if len(reg_borrowed) < num_qubit_ctrlling - 2:  # if there are several but not enough borrowed qubits,
            # here is a decomposition: C^n(X) -> several C^(n/2)(X) gates
            qubit_borrowed = reg_borrowed[0]
            num_qubit_ctrlling_half = (num_qubit_ctrlling + 1) // 2
            reg_ctrlling_half_1 = reg_ctrlling[:num_qubit_ctrlling_half]
            reg_ctrlling_half_2 = reg_ctrlling[num_qubit_ctrlling_half:] + [qubit_borrowed]
            # The following is exactly the decomposition.
            # Here when we implement another CnX gate with less ctrlling qubits,
            # we regard the other half qubits borrowable.
            circ_multictrl_X(qubit_borrowed, reg_ctrlling_half_1, reg_borrowed=reg_ctrlling_half_2)
            circ_multictrl_X(qubit_target, reg_ctrlling_half_2, reg_borrowed=reg_ctrlling_half_1)
            circ_multictrl_X(qubit_borrowed, reg_ctrlling_half_1, reg_borrowed=reg_ctrlling_half_2)
            circ_multictrl_X(qubit_target, reg_ctrlling_half_2, reg_borrowed=reg_ctrlling_half_1)
        else:  # if there are enough borrowed qubits, here is another decomposition: CnX -> several CCX gates
            # It is this decomposition that drives the complexity O(n).
            for _ in range(2):  # this decomposition repeats the following circuit twice
                CCX(reg_ctrlling[0], reg_borrowed[0], qubit_target)
                for idx in range(1, num_qubit_ctrlling - 2):
                    CCX(reg_ctrlling[idx], reg_borrowed[idx], reg_borrowed[idx - 1])
                CCX(reg_ctrlling[-2], reg_ctrlling[-1], reg_borrowed[num_qubit_ctrlling - 3])
                for idx in range(num_qubit_ctrlling - 3, 0, -1):
                    CCX(reg_ctrlling[idx], reg_borrowed[idx], reg_borrowed[idx - 1])


def circ_multictrl_Y(
    qubit_target: QRegStorage,
    reg_ctrlling: Optional[List[QRegStorage]],
    num_qubit_ctrlling: Optional[int] = None,
    reg_borrowed: Optional[List[QRegStorage]] = None,
) -> None:
    r"""An :math:`O(n)` decomposition for multictrl :math:`Y` gate (@ :math:`C^n(Y)`) with several borrowed qubits.

    In math, :math:`C^n(Y)` is defined as

    .. math::

        C^n(Y):=|11\cdots1\rangle\langle11\cdots1|\otimes(Y-I_2) + I_{2^{n+1}}.

    Also, we have the effect in computing basis:

    .. math::

        C^n(Y)|c\rangle|t\rangle = \begin{cases}
            |c\rangle Y|t\rangle,&\text{if }|c\rangle=|11\cdots1\rangle;\\
            |c\rangle|t\rangle,&\text{if }|c\rangle\perp|11\cdots1\rangle;
        \end{cases}

    It is noted that after calling this circuit, the quantum state in :math:`|b\rangle` would remains,
    like that we have operated :math:`I` gates on those qubit in :math:`|b\rangle`.

    :param qubit_target: :math:`|t\rangle`, `QRegStorage`, the target qubit of such :math:`C^n(Y)` gate
    :param reg_ctrlling: :math:`|c\rangle`, `List[QRegStorage]` of length :math:`n` or **None**,
        a quantum register containing several qubits as the ctrlling qubits for such :math:`C^n(Y)` gate;
        it is noted that :math:`|c\rangle` may be **None** indicating none-ctrl version, i.e. such :math:`Y` gate
    :param num_qubit_ctrlling: :math:`n`, `int` or **None**,
        `int` indicating the length of :math:`|c\rangle`, or **None** indicating unknown length and need to count
    :param reg_borrowed: :math:`|b\rangle`, `List[QRegStorage]` or **None**,
        a quantum register containing several qubits as the borrowed qubits for such :math:`C^n(Y)` gate;
        it is noted that :math:`|b\rangle` may also be **None** indicating an empty register
    :return: **None**
    """
    if num_qubit_ctrlling is None:
        num_qubit_ctrlling = len(reg_ctrlling)  # count the number n in reg_ctrlling
    if num_qubit_ctrlling == 0:  # for the case n == 0, it just |t> -> Y|t>
        Y(qubit_target)
    elif num_qubit_ctrlling == 1:  # for the case n == 1, it just |c>|t> -> CY|c>|t>
        CY(reg_ctrlling[0], qubit_target)
    else:  # for the case n >= 2, we implement CnY by CnX, S and SDG gates
        SDG(qubit_target)
        circ_multictrl_X(qubit_target, reg_ctrlling, num_qubit_ctrlling=num_qubit_ctrlling, reg_borrowed=reg_borrowed)
        S(qubit_target)


def circ_multictrl_Z(
    qubit_target: QRegStorage,
    reg_ctrlling: Optional[List[QRegStorage]],
    num_qubit_ctrlling: Optional[int] = None,
    reg_borrowed: Optional[List[QRegStorage]] = None,
) -> None:
    r"""An :math:`O(n)` decomposition for multictrl :math:`Z` gate (@ :math:`C^n(Z)`) with several borrowed qubits.

    In math, :math:`C^n(Z)` is defined as

    .. math::

        C^n(Z):=|11\cdots1\rangle\langle11\cdots1|\otimes(Z-I_2) + I_{2^{n+1}}.

    Also, we have the effect in computing basis:

    .. math::

        C^n(Z)|c\rangle|t\rangle = \begin{cases}
            |c\rangle Z|t\rangle,&\text{if }|c\rangle=|11\cdots1\rangle;\\
            |c\rangle|t\rangle,&\text{if }|c\rangle\perp|11\cdots1\rangle;
        \end{cases}

    It is noted that after calling this circuit, the quantum state in :math:`|b\rangle` would remains,
    like that we have operated :math:`I` gates on those qubit in :math:`|b\rangle`.

    :param qubit_target: :math:`|t\rangle`, `QRegStorage`, the target qubit of such :math:`C^n(Z)` gate
    :param reg_ctrlling: :math:`|c\rangle`, `List[QRegStorage]` of length :math:`n` or **None**,
        a quantum register containing several qubits as the ctrlling qubits for such :math:`C^n(Z)` gate;
        it is noted that :math:`|c\rangle` may be **None** indicating none-ctrl version, i.e. such :math:`Z` gate
    :param num_qubit_ctrlling: :math:`n`, `int` or **None**,
        `int` indicating the length of :math:`|c\rangle`, or **None** indicating unknown length and need to count
    :param reg_borrowed: :math:`|b\rangle`, `List[QRegStorage]` or **None**,
        a quantum register containing several qubits as the borrowed qubits for such :math:`C^n(Z)` gate;
        it is noted that :math:`|b\rangle` may also be **None** indicating an empty register
    :return: **None**
    """
    if num_qubit_ctrlling is None:
        num_qubit_ctrlling = len(reg_ctrlling)  # count the number n in reg_ctrlling
    if num_qubit_ctrlling == 0:  # for the case n == 0, it just |t> -> Z|t>
        Z(qubit_target)
    elif num_qubit_ctrlling == 1:  # for the case n == 1, it just |c>|t> -> CZ|c>|t>
        CZ(reg_ctrlling[0], qubit_target)
    else:  # for the case n >= 2, we implement CnZ by CnX and H gates
        H(qubit_target)
        circ_multictrl_X(qubit_target, reg_ctrlling, num_qubit_ctrlling=num_qubit_ctrlling, reg_borrowed=reg_borrowed)
        H(qubit_target)


def circ_multictrl_Pauli(
    qubit_target: QRegStorage,
    reg_ctrlling: Optional[List[QRegStorage]],
    char_Pauli: str,
    num_qubit_ctrlling: Optional[int] = None,
    reg_borrowed: Optional[List[QRegStorage]] = None,
) -> None:
    r"""An :math:`O(n)` decomposition for multictrl :math:`P=X,Y\text{ or }Z` gate (@ :math:`C^n(P)`)
    with several borrowed qubits.

    In math, :math:`C^n(P)` is defined as

    .. math::

        C^n(P):=|11\cdots1\rangle\langle11\cdots1|\otimes(P-I_2) + I_{2^{n+1}}.

    Also, we have the effect in computing basis:

    .. math::

        C^n(P)|c\rangle|t\rangle = \begin{cases}
            |c\rangle P|t\rangle,&\text{if }|c\rangle=|11\cdots1\rangle;\\
            |c\rangle|t\rangle,&\text{if }|c\rangle\perp|11\cdots1\rangle;
        \end{cases}

    It is noted that after calling this circuit, the quantum state in :math:`|b\rangle` would remains,
    like that we have operated :math:`I` gates on those qubit in :math:`|b\rangle`.

    :param qubit_target: :math:`|t\rangle`, `QRegStorage`, the target qubit of such :math:`C^n(P)` gate
    :param reg_ctrlling: :math:`|c\rangle`, `List[QRegStorage]` of length :math:`n` or **None**,
        a quantum register containing several qubits as the ctrlling qubits for such :math:`C^n(P)` gate;
        it is noted that :math:`|c\rangle` may be **None** indicating none-ctrl version, i.e. such :math:`P` gate
    :param char_Pauli: :math:`P`, `str`,
        should be one of 'X', 'Y' and 'Z' for corresponding Pauli gate, or others for identity gate
    :param num_qubit_ctrlling: :math:`n`, `int` or **None**,
        `int` indicating the length of :math:`|c\rangle`, or **None** indicating unknown length and need to count
    :param reg_borrowed: :math:`|b\rangle`, `List[QRegStorage]` or **None**,
        a quantum register containing several qubits as the borrowed qubits for such :math:`C^n(P)` gate;
        it is noted that :math:`|b\rangle` may also be **None** indicating an empty register
    :return: **None**
    """
    if num_qubit_ctrlling is None:
        num_qubit_ctrlling = len(reg_ctrlling)  # count the number n in reg_ctrlling
    if reg_borrowed is None:
        reg_borrowed = []
    if char_Pauli == "X":  # we will operate a CnX gate
        circ_multictrl_X(qubit_target, reg_ctrlling, num_qubit_ctrlling=num_qubit_ctrlling, reg_borrowed=reg_borrowed)
    elif char_Pauli == "Y":  # we will operate a CnY gate
        circ_multictrl_Y(qubit_target, reg_ctrlling, num_qubit_ctrlling=num_qubit_ctrlling, reg_borrowed=reg_borrowed)
    elif char_Pauli == "Z":  # we will operate a CnZ gate
        circ_multictrl_Z(qubit_target, reg_ctrlling, num_qubit_ctrlling=num_qubit_ctrlling, reg_borrowed=reg_borrowed)


def circ_multictrl_ry(
    qubit_target: QRegStorage,
    reg_ctrlling: Optional[List[QRegStorage]],
    float_rotation_angle: float,
    num_qubit_ctrlling: Optional[int] = None,
    reg_borrowed: Optional[List[QRegStorage]] = None,
) -> None:
    r"""An :math:`O(n)` decomposition for multictrl :math:`Ry` gate (@ :math:`C^n(Ry(a))`)
    with several borrowed qubits.

    In math, :math:`C^n(Ry(a))` is defined as

    .. math::

        C^n(Ry(a)):=|11\cdots1\rangle\langle11\cdots1|\otimes(Ry(a)-I_2) + I_{2^{n+1}}.

    Also, we have the effect in computing basis:

    .. math::

        C^n(Ry(a))|c\rangle|t\rangle = \begin{cases}
            |c\rangle Ry(a)|t\rangle,&\text{if }|c\rangle=|11\cdots1\rangle;\\
            |c\rangle|t\rangle,&\text{if }|c\rangle\perp|11\cdots1\rangle;
        \end{cases}

    :param qubit_target: :math:`|t\rangle`, `QRegStorage`, the target qubit of such :math:`C^n(Ry(a))` gate
    :param reg_ctrlling: :math:`|c\rangle`, `List[QRegStorage]` of length :math:`n` or **None**,
        a quantum register containing several qubits as the ctrlling qubits for such :math:`C^n(Ry(a))` gate;
        it is noted that :math:`|c\rangle` may be **None** indicating none-ctrl version, i.e. such :math:`Ry(a)` gate
    :param float_rotation_angle: :math:`a`, `float`, the rotation angle of the multictrl :math:`C^n(Ry(a))` gate
    :param num_qubit_ctrlling: :math:`n`, `int` or **None**,
        `int` indicating the length of :math:`|c\rangle`, or **None** indicating unknown length and need to count
    :param reg_borrowed: :math:`|b\rangle`, `List[QRegStorage]` or **None**,
        a quantum register containing several qubits as the borrowed qubits for such :math:`C^n(Ry(a))` gate;
        it is noted that :math:`|b\rangle` may also be **None** indicating an empty register
    :return: **None**
    """
    if reg_ctrlling is None:
        reg_ctrlling = []
    if num_qubit_ctrlling is None:
        num_qubit_ctrlling = len(reg_ctrlling)  # count the number n in reg_ctrlling
    if reg_borrowed is None:
        reg_borrowed = []
    if num_qubit_ctrlling == 0:  # for the case n == 0, it just |t> -> Ry(a)|t>
        RY(float_rotation_angle)(qubit_target)
    elif num_qubit_ctrlling == 1:  # for the case n == 1, it just |c>|t> -> Ry(a)|c>|t>
        CRY(float_rotation_angle)(reg_ctrlling[0], qubit_target)
    else:  # for the case n >= 2
        # here is a decomposition: C^n(Ry) -> several CnX gates and Ry gates
        RY(float_rotation_angle / 2)(qubit_target)
        circ_multictrl_X(qubit_target, reg_ctrlling, num_qubit_ctrlling=num_qubit_ctrlling, reg_borrowed=reg_borrowed)
        RY(-float_rotation_angle / 2)(qubit_target)
        circ_multictrl_X(qubit_target, reg_ctrlling, num_qubit_ctrlling=num_qubit_ctrlling, reg_borrowed=reg_borrowed)


def circ_multictrl_rz(
    qubit_target: QRegStorage,
    reg_ctrlling: Optional[List[QRegStorage]],
    float_rotation_angle: float,
    num_qubit_ctrlling: Optional[int] = None,
    reg_borrowed: Optional[List[QRegStorage]] = None,
) -> None:
    r"""An :math:`O(n)` decomposition for multictrl :math:`Rz` gate (@ :math:`C^n(Rz(a))`)
    with several borrowed qubits.

    In math, :math:`C^n(Rz(a))` is defined as

    .. math::

        C^n(Rz(a)):=|11\cdots1\rangle\langle11\cdots1|\otimes(Rz(a)-I_2) + I_{2^{n+1}}.

    Also, we have the effect in computing basis:

    .. math::

        C^n(Rz(a))|c\rangle|t\rangle = \begin{cases}
            |c\rangle Rz(a)|t\rangle,&\text{if }|c\rangle=|11\cdots1\rangle;\\
            |c\rangle|t\rangle,&\text{if }|c\rangle\perp|11\cdots1\rangle;
        \end{cases}

    :param qubit_target: :math:`|t\rangle`, `QRegStorage`, the target qubit of such :math:`C^n(Rz(a))` gate
    :param reg_ctrlling: :math:`|c\rangle`, `List[QRegStorage]` of length :math:`n` or **None**,
        a quantum register containing several qubits as the ctrlling qubits for such :math:`C^n(Rz(a))` gate;
        it is noted that :math:`|c\rangle` may be **None** indicating none-ctrl version, i.e. such :math:`Rz(a)` gate
    :param float_rotation_angle: :math:`a`, `float`, the rotation angle of the multictrl :math:`C^n(Rz(a))` gate
    :param num_qubit_ctrlling: :math:`n`, `int` or **None**,
        `int` indicating the length of :math:`|c\rangle`, or **None** indicating unknown length and need to count
    :param reg_borrowed: :math:`|b\rangle`, `List[QRegStorage]` or **None**,
        a quantum register containing several qubits as the borrowed qubits for such :math:`C^n(Rz(a))` gate;
        it is noted that :math:`|b\rangle` may also be **None** indicating an empty register
    :return: **None**
    """
    if reg_ctrlling is None:
        reg_ctrlling = []
    if reg_borrowed is None:
        reg_borrowed = []
    if len(reg_ctrlling) == 0:  # for the case n == 0, it just |t> -> Ry(a)|t>
        RZ(float_rotation_angle)(qubit_target)
    elif len(reg_ctrlling) == 1:  # for the case n == 1, it just |c>|t> -> Ry(a)|c>|t>
        CRZ(float_rotation_angle)(reg_ctrlling[0], qubit_target)
    else:  # for the case n >= 2
        # here is a decomposition: C^n(Rz) -> several CnX gates and Rz gates
        RZ(float_rotation_angle / 2)(qubit_target)
        circ_multictrl_X(qubit_target, reg_ctrlling, num_qubit_ctrlling=num_qubit_ctrlling, reg_borrowed=reg_borrowed)
        RZ(-float_rotation_angle / 2)(qubit_target)
        circ_multictrl_X(qubit_target, reg_ctrlling, num_qubit_ctrlling=num_qubit_ctrlling, reg_borrowed=reg_borrowed)
