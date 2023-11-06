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
This script is an implement of Quantum Singular Value Transformation (QSVT).

Given a Hermitian :math:`\check H\in\mathbb C^{2^n\times2^n}` block-encoded in
:math:`\operatorname{BE}\in\mathbb C^{2^{m+n}\times2^{m+n}}`:

.. math::
    (\langle0|\otimes I_{2^n})\operatorname{BE}(|0\rangle\otimes I_{2^n})=\check H

satisfying :math:`\operatorname{BE}^2=I_{2^{m+n}}`, denoting

.. math::

    \begin{aligned}
        e^{i\check Z\phi}:=e^{i(2|0\rangle\langle0|-I_{2^m})\phi}=
        (e^{i\phi}-e^{-i\phi})|0\rangle\langle0|+e^{-i\phi}I_{2^m},\\
        W_{\vec\phi}(\operatorname{BE})=
        e^{i\check Z\phi_0}\prod_{j=1}^d
        \left(e^{-i\check Z\pi/4}\cdot\operatorname{BE}\cdot e^{-i\check Z\pi/4} e^{i\check Z\phi_j}\right),
    \end{aligned}

where :math:`\vec\phi=(\phi_0,\phi_1,\cdots,\phi_d)\in\mathbb R^{d+1}`,
then we have :math:`W_{\vec\phi}(\operatorname{BE})` is a block-encoding for :math:`(-i)^dP_{\vec\phi}(\check H)`,
i.e.

.. math::

    (\langle0|\otimes I_{2^n})\cdot W_{\vec\phi}(\operatorname{BE})\cdot
    (|0\rangle\otimes I_{2^n})=(-i)^dP_{\vec\phi}(\check H)

where the real part of :math:`P_{\vec\phi}(x)\in\mathbb C[x]` is just such quantum signal processing function
:math:`A_{\vec\phi}(x)\in\mathbb R[x]`, i.e. :math:`\Re(P_{\vec\phi}(x))=A_{\vec\phi}(x)`.

Moreover, we have

.. math::

    \begin{aligned}
        |+\rangle\langle+|\otimes W_{\vec\phi}(\operatorname{BE})+
        (-1)^d|-\rangle\langle-|\otimes W_{\vec\phi}(\operatorname{BE})^\dagger
    \end{aligned}

is a block-encoding for :math:`(-i)^dA_{\vec\phi}(\check H)`.

Reader may refer to following reference for more insights.

.. [GLS+19] Gilyén, András, et al. "Quantum singular value transformation and beyond: exponential improvements for
    quantum matrix arithmetics." Proceedings of the 51st Annual ACM SIGACT Symposium on Theory of Computing. 2019.
"""

from typing import List, Optional, Tuple

from QCompute import X, Z, RZ, CZ, CRZ
from QCompute.QPlatform.QRegPool import QRegStorage

from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Oracle.BlockEncoding import circ_block_encoding
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Gate.MultiCtrlGates import circ_multictrl_X


def circ_Pi_double_ctrl_rot(
    qubit_target: QRegStorage,
    reg_ctrlling: List[QRegStorage],
    qubit_ctrlling: QRegStorage,
    float_angle_0: float,
    float_angle_1: float,
    reg_borrowed: Optional[List[QRegStorage]] = None,
) -> None:
    r"""A quantum circuit implementing a class of multictrl rotation gate.

    In math, this circuit implement a quantum operation, which could be represented as following in computing basis:

    .. math::

        |r\rangle|q\rangle|t\rangle\mapsto\begin{cases}
            |r\rangle|q\rangle Rz(a_0)|t\rangle,
            \text{ if }|r\rangle=|0\rangle\ \mathsf{XOR}\ |q\rangle=|0\rangle;\\
            |r\rangle|q\rangle Rz(a_1)|t\rangle,
            \text{ if }|r\rangle\perp|0\rangle\ \mathsf{XOR}\ |q\rangle=|1\rangle.
        \end{cases}

    :param qubit_target: :math:`|t\rangle`, `QRegStorage`, the target qubit operated :math:`Rz` gate on
    :param reg_ctrlling: :math:`|r\rangle`, `List[QRegStorage]`,
        the ctrlling register corresponding to the operator :math:`e^{i\check Z\phi}`
    :param qubit_ctrlling: :math:`|q\rangle`, `QRegStorage`,
        another ctrlling qubit deciding which rotation angle will be used
    :param float_angle_0: :math:`a_0`, `float`, the first rotation angle
    :param float_angle_1: :math:`a_1`, `float`, the second rotation angle
    :param reg_borrowed: :math:`|b\rangle`, `Optional[List[QRegStorage]]`,
        a quantum register containing several qubits as the borrowed qubits for implementing multictrl gates
    :return: **None**

    **Examples**

        The quantum gate :math:`C_5(X\otimes I\otimes Y\otimes Z)\in\mathbb C^{2^7\times 2^7}` could be called by

        >>> from numpy import pi
        >>> from QCompute import QEnv
        >>> from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.QSVT.QSVT import circ_Pi_double_ctrl_rot
        >>> env = QEnv()
        >>> qubit_c = env.Q[0]
        >>> reg_c = [env.Q[1], env.Q[2]]
        >>> qubit_t = env.Q[3]
        >>> float_a0 = pi
        >>> float_a1 = pi / 2
        >>> circ_Pi_double_ctrl_rot(qubit_t, reg_c, qubit_c, float_a0, float_a1)

        Since both ``qubit_t`` and ``reg_c`` are in initial state :math:`|0\rangle`,
        :math:`Rz(\pi)` should be opearted on ``qubit_t``. Thus the final state should be
        :math:`|000\rangle Rz(\pi)|0\rangle=-i|0000\rangle`.
    """
    for idx_qubit in reg_ctrlling:
        X(idx_qubit)
    circ_multictrl_X(qubit_target, reg_ctrlling, reg_borrowed=reg_borrowed)

    RZ(2 * float_angle_0)(qubit_target)
    CRZ(2 * (float_angle_1 - float_angle_0))(qubit_ctrlling, qubit_target)

    circ_multictrl_X(qubit_target, reg_ctrlling, reg_borrowed=reg_borrowed)
    for idx_qubit in reg_ctrlling:
        X(idx_qubit)


def circ_QSVT_from_BE(
    reg_sys: List[QRegStorage],
    reg_blocking: List[QRegStorage],
    qubit_ancilla_b: QRegStorage,
    qubit_ancilla_c: QRegStorage,
    list_str_Pauli_rep: List[Tuple[float, str]],
    list_float_target_state: List[float],
    list_re: List[float],
    list_im: List[float],
) -> None:
    r"""A quantum circuit implementing a case of QSVT circuit.

    Given a quantum signal processing function

    .. math::

        f(x)=A_{\vec\phi_{\Re}}(x) + iA_{\vec\phi_{\Im}}(x)\in\mathbb C[x]

    encoded in two groups of processing parameters :math:`\vec\phi_{\Re}` and :math:`\vec\phi_{\Im}`,
    thus we have a quantum circuit implementing a block-encoding for :math:`f(\check H)/2`.

    It is noted that Here we assume that :math:`\vec\phi_{\Im}` has exactly one more entry than :math:`\vec\phi_{\Re}`.

    In math, this circuit :math:`\operatorname{QSVT}` implement a quantum operation,
    which could be represented as following in computing basis:

    .. math::

        \left(\langle 0|\langle 0|\langle 0|\otimes I_{2^n}\right)\cdot\operatorname{QSVT}\cdot
        \left(|c\rangle|b\rangle|a\rangle|s\rangle\right)
        =f(\check H)|s\rangle, \text{ if } |cba\rangle=|0\rangle.

    :param reg_sys: :math:`|s\rangle`, `List[QRegStorage]`, the system register for block-encoding
    :param reg_blocking: :math:`|a\rangle`, `List[QRegStorage]`, the ancilla register introduced in block-encoding step
    :param qubit_ancilla_b: :math:`|b\rangle`, `QRegStorage`,
        an ancilla qubit used to obtain :math:`A_{*}(\check H) = (P_{*}(\check H) + P_{*}(\check H)^\dagger) / 2`,
        should be at state :math:`|0\rangle` before this circuit is operated
    :param qubit_ancilla_c: :math:`|c\rangle`, `QRegStorage`,
        an ancilla qubit used to obtain :math:`(A_{\vec\phi_{\Re}}(\check H) + iA_{\vec\phi_{\Im}}(\check H)) / 2`,
        should be at state :math:`|0\rangle` before this circuit is operated
    :param list_str_Pauli_rep: :math:`\check H`, `List[Tuple[float, str]]`,
        a list of form :math:`(a_j, P_j)` such as
        `[(-0.09706, \'I\'), (-0.045302, \'X0X1Y2Y3\'), (0.17441, \'Z2Z3\'), (-0.2234, \'Z3\')]`,
        where :math:`a_j` is a float indicating the coefficient of :math:`P_j` in :math:`\check H`,
        and :math:`P_j` is a string indicating which multi-Pauli gate
    :param list_float_target_state: :math:`\vec s`, `List[float]`, corresponding to :math:`\check H` and also be the
        classical information for :math:`|sel\rangle`
    :param list_re: :math:`\vec\phi_{\Re}`, `List[float]`,
        as the processing parameter encoding the real part of :math:`f(x)`
    :param list_im: :math:`\vec\phi_{\Im}`, `List[float]`,
        as the processing parameter encoding the image part of :math:`f(x)`,
        should has exactly one more entry than :math:`\vec\phi_{\Re}`
    """
    # in fact, we have two parts: |c> o-ctrl re and |c> ctrl im
    CZ(qubit_ancilla_c, qubit_ancilla_b)
    Z(qubit_ancilla_c)

    circ_Pi_double_ctrl_rot(
        qubit_ancilla_b, reg_blocking, qubit_ancilla_c, list_re[0], list_im[0], reg_borrowed=reg_sys
    )

    for idx in range(1, len(list_re)):
        circ_block_encoding(
            reg_sys,
            reg_blocking,
            None,
            list_str_Pauli_rep,
            list_float_target_state,
            reg_borrowed=[qubit_ancilla_b, qubit_ancilla_c],
        )
        circ_Pi_double_ctrl_rot(
            qubit_ancilla_b, reg_blocking, qubit_ancilla_c, list_re[idx], list_im[idx], reg_borrowed=reg_sys
        )

    circ_block_encoding(
        reg_sys,
        reg_blocking,
        qubit_ancilla_c,
        list_str_Pauli_rep,
        list_float_target_state,
        reg_borrowed=[qubit_ancilla_b],
    )

    circ_Pi_double_ctrl_rot(qubit_ancilla_b, reg_blocking, qubit_ancilla_c, 0, list_im[-1], reg_borrowed=reg_sys)


def circ_QSVT_from_BE_inverse(
    reg_sys: List[QRegStorage],
    reg_blocking: List[QRegStorage],
    qubit_ancilla_b: QRegStorage,
    qubit_ancilla_c: QRegStorage,
    list_str_Pauli_rep: List[Tuple[float, str]],
    list_float_target_state: List[float],
    list_re: List[float],
    list_im: List[float],
) -> None:
    r"""The inverse of such quantum circuit implementing by **circ_QSVT_from_BE** with the same input.

    :param reg_sys: :math:`|s\rangle`, `List[QRegStorage]`, the system register for block-encoding
    :param reg_blocking: :math:`|a\rangle`, `List[QRegStorage]`, the ancilla register introduced in block-encoding step
    :param qubit_ancilla_b: :math:`|b\rangle`, `QRegStorage`,
        an ancilla qubit used to obtain :math:`A_{*}(\check H) = (P_{*}(\check H) + P_{*}(\check H)^\dagger) / 2`,
        should be at state :math:`|0\rangle` before this circuit is operated
    :param qubit_ancilla_c: :math:`|c\rangle`, `QRegStorage`,
        an ancilla qubit used to obtain :math:`(A_{\vec\phi_{\Re}}(\check H) + iA_{\vec\phi_{\Im}}(\check H)) / 2`,
        should be at state :math:`|0\rangle` before this circuit is operated
    :param list_str_Pauli_rep: :math:`\check H`, `List[Tuple[float, str]]`,
        a list of form :math:`(a_j, P_j)` such as
        `[(-0.09706, \'I\'), (-0.045302, \'X0X1Y2Y3\'), (0.17441, \'Z2Z3\'), (-0.2234, \'Z3\')]`,
        where :math:`a_j` is a float indicating the coefficient of :math:`P_j` in :math:`\check H`,
        and :math:`P_j` is a string indicating which multi-Pauli gate
    :param list_float_target_state: :math:`\vec s`, `List[float]`, corresponding to :math:`\check H` and also be the
        classical information for :math:`|sel\rangle`
    :param list_re: :math:`\vec\phi_{\Re}`, `List[float]`,
        as the processing parameter encoding the real part of :math:`f(x)`
    :param list_im: :math:`\vec\phi_{\Im}`, `List[float]`,
        as the processing parameter encoding the image part of :math:`f(x)`,
        should has exactly one more entry than :math:`\vec\phi_{\Re}`
    """
    circ_Pi_double_ctrl_rot(qubit_ancilla_b, reg_blocking, qubit_ancilla_c, 0, -list_im[-1], reg_borrowed=reg_sys)

    circ_block_encoding(
        reg_sys,
        reg_blocking,
        qubit_ancilla_c,
        list_str_Pauli_rep,
        list_float_target_state,
        reg_borrowed=[qubit_ancilla_b],
    )

    for idx in reversed(range(1, len(list_re))):
        circ_Pi_double_ctrl_rot(
            qubit_ancilla_b, reg_blocking, qubit_ancilla_c, -list_re[idx], -list_im[idx], reg_borrowed=reg_sys
        )
        circ_block_encoding(
            reg_sys,
            reg_blocking,
            None,
            list_str_Pauli_rep,
            list_float_target_state,
            reg_borrowed=[qubit_ancilla_b, qubit_ancilla_c],
        )

    circ_Pi_double_ctrl_rot(
        qubit_ancilla_b, reg_blocking, qubit_ancilla_c, -list_re[0], -list_im[0], reg_borrowed=reg_sys
    )
    CZ(qubit_ancilla_c, qubit_ancilla_b)
    Z(qubit_ancilla_c)
