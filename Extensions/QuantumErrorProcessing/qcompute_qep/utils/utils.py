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
Utility functions used in the ``qcompute_qep`` module.
"""
from qcompute_qep.utils.linalg import dagger
from QCompute.QPlatform.QOperation.QProcedure import QProcedure
import math
import functools

import numpy as np
from typing import Tuple


def global_phase(U: np.ndarray) -> float:
    r"""Compute the global phase of a :math:`2\times 2` unitary matrix.

    Each :math:`2\times 2` unitary matrix can be equivalently characterized as:

    .. math::   U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda).

    We aim to compute the global phase :math:`\alpha`.
    See Theorem 4.1 in `Nielsen & Chuang`'s book for details.

    :param U: the matrix representation of a :math:`2\times 2` unitary operator
    :return: the global phase of the unitary matrix
    """
    # Notice that the determinant of the unitary is given by :math:`e^{2i\alpha}`
    coe = np.linalg.det(U) ** (-0.5)
    alpha = - np.angle(coe)
    return alpha


def decompose_yzy(U: np.ndarray) -> Tuple[float, float, float, float]:
    r"""Compute the Euler angles :math:`(\alpha,\theta,\phi,\lambda)` of a :math:`2\times 2` unitary matrix.

    Each :math:`2\times 2` unitary matrix can be equivalently characterized as:

    .. math::

        U = e^{i\alpha} R_z(\phi) R_y(\theta) R_z(\lambda)
          = e^{i(\alpha-\phi/2-\lambda/2)}
            \begin{bmatrix}
                \cos(\theta/2) & -e^{i\lambda}\sin(\theta/2) \\
                e^{i\phi}\sin(\theta/2) & e^{i(\phi+\lambda)}\cos(\theta/2)
            \end{bmatrix}.

    We aim to compute the parameters :math:`(\alpha,\theta,\phi,\lambda)`.
    See Theorem 4.1 in `Nielsen & Chuang`'s book for details.


    :param U: the matrix representation of the qubit unitary
    :return: Tuple[float, float, float, float], the Euler angles
    """
    if U.shape != (2, 2):
        raise ValueError("in decompose_yzy(): input should be a 2x2 matrix!")
    # Remove the global phase
    alpha = global_phase(U)
    U = U * np.exp(- 1j * alpha)
    U = U.round(10)
    # Compute theta
    theta = 2 * math.atan2(abs(U[1, 0]), abs(U[0, 0]))

    # Compute phi and lambda
    phi_lam_sum = 2 * np.angle(U[1, 1])
    phi_lam_diff = 2 * np.angle(U[1, 0])
    phi = (phi_lam_sum + phi_lam_diff) / 2.0
    lam = (phi_lam_sum - phi_lam_diff) / 2.0

    return alpha, theta, phi, lam


def str_to_state(state_str: str, bits: int = None, LSB: bool = True) -> np.ndarray:
    r"""Return the computational basis state in density matrix form.

    Notice that we assume the LSB (the least significant bit) mode, i.e., the right-most bit represents q[0]:

    ::

        "1        0        1"

        q[2]    q[1]      q[0]

    :param state_str: string-format, e.g. '1110', '11', '0', etc.
    :param bits: int, the number of bits of the input string
    :param LSB: the least significant bit (LSB) mode, default is True
    :return: np.ndarray, density matrix in type of `ndarray`

    **Examples**

        >>> str_to_state("1")
        array([[0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j]])
        >>> str_to_state("01")
        array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
        >>> str_to_state("10")
        array([[0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 1.+0.j, 0.+0.j],
               [0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j]])
    """
    # convert 16-base to 2-base
    if state_str[:2].lower() == '0x':
        state_str = bin(int(state_str, 16))[2:]
    if bits is not None:
        if bits < len(state_str):
            raise ValueError('bits can not be less than length of {}'.format(state_str))
        else:
            state_str = state_str.zfill(bits)
    # Map each binary character to qubit value 0 or 1
    qubits = list(map(int, list(state_str)))
    # If not LSB, reverse the qubits order
    if not LSB:
        qubits = reversed(qubits)
    # Compute the tensor product state
    states = [np.diag([1, 0]).astype(complex) if q == 0 else np.diag([0, 1]).astype(complex) for q in qubits]
    state_prod = functools.reduce(np.kron, states)
    return state_prod


def expval_from_counts(A: np.ndarray, counts: dict) -> float:
    r"""Expectation value of the given operator :math:`A` from counts.

    We assume `a priori` that :math:`A` is diagonalized with respect to the measurement basis
    on which the quantum state is measured and counts is obtained.

    :param A: np.ndarray, a Hermitian operator that is diagonalized in the measurement basis
    :param counts: dict, dict-type counts data, means result of shot measurements, e.g. ``{'000000': 481, '111111': 519}``
    :return: float, the estimated expectation value
    """
    expects = []
    if list(counts.keys())[0][:2].lower() == '0x':
        bits = len(bin(max(map(lambda x: int(x, 16), counts.keys())))[2:])
    else:
        bits = None
    for k, v in counts.items():
        state = str_to_state(k, bits=bits)
        if state.shape != A.shape:
            raise ValueError("Shapes of density matrix and operator are not equal!")
        expects.append(np.real(np.trace(state @ A)))
    return np.average(expects, weights=list(counts.values()))


def expval_z_from_counts(counts: dict) -> float:
    r"""Expectation value of the :math:`Z^{\otimes n}` operator from counts.

    :param counts: dict-type counts data, records the measurement outcomes, e.g. ``{'000000': 481, '111111': 519}``
    :return: float, the expectation value of the :math:`Z^{\otimes n}` operator
    """
    # Determine the number of qubits
    n = len(list(counts)[0])
    Z = np.diag([1, -1]).astype(complex)
    A = functools.reduce(np.kron, [Z] * n)

    return expval_from_counts(A, counts)


def limit_angle(theta: float) -> float:
    r"""Limit an angle value into the range :math:`[-\pi, \pi]`.

    :param theta: origin angle value
    :return: a limited angle value in the range :math:`[-\pi, \pi]`
    """
    if theta > 0:
        theta = theta % (2 * np.pi)
        if theta > np.pi:
            return theta - 2 * np.pi
        else:
            return theta
    else:
        theta_abs = abs(theta)
        theta_abs = theta_abs % (2 * np.pi)
        if theta_abs > np.pi:
            return - theta_abs + 2 * np.pi
        else:
            return - theta_abs
