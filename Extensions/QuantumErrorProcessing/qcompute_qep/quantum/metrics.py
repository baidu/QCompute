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
This module aims to implement distance metrics frequently utilized in Quantum Information Theory.
"""
import math

import numpy as np
import scipy.linalg as la
from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.utils.linalg import vec_to_operator


def state_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""Compute the state fidelity of two quantum states.

    Compute the state fidelity of two quantum states :math:`\rho` (@rho) and :math:`\sigma` (@sigma),
    which is mathematically defined as:

    .. math:: F(\rho, \sigma) := \left(\text{Tr}\left[\sqrt{\sqrt{\rho}\sigma\sqrt{\rho}}\right]\right)^2.

    If one quantum state is a pure state, say :math:`\rho = \vert\psi\rangle\!\langle\psi\vert`, then

    .. math:: F(\rho, \sigma) = \langle \psi \vert \sigma \vert \psi \rangle.

    If both :math:`\rho = \vert\psi\rangle\!\langle\psi\vert`
    and :math:`\sigma = \vert\phi\rangle\!\langle\phi\vert` are pure states, then

    .. math:: F(\rho, \sigma) = \vert \langle \psi \vert \phi \rangle\vert^2.

    :param rho: np.ndarray, a quantum state (state vector or density matrix)
    :param sigma: np.ndarray, a quantum state (state vector or density matrix)
    :return: float, the fidelity of the input quantum states
    """
    # If the input is a state vector, convert it to density operator; otherwise, unchanged
    rho = vec_to_operator(rho)
    sigma = vec_to_operator(sigma)

    if rho.shape != sigma.shape:
        raise ArgumentError("The dimensions of the two input density matrices mismatch!")

    fid = np.trace(la.sqrtm(la.sqrtm(rho) @ sigma @ la.sqrtm(rho))) ** 2
    return float(np.real(fid))


def total_variation_distance(A: np.ndarray, B: np.ndarray) -> float:
    r"""Compute the total variation distance between two matrices.

    Compute the total variation distance between two matrices with the same dimension.
    For two matrices A and B, which both are math:`m \times n` dimensional,
    the total variation distance (TVD) is defined as:

    .. math:: \text{TVD}(A,B) := \frac{1}{2}\max_{n=1}^N \sum_{m=1}^M | A_{m,n} - B_{m,n} |,

    where :math:`A_{m,n}` is the element of :math:`A` in the :math:`m`-th row and :math:`n`-th column.

    :param A: np.ndarray, a :math:`M \times N` matrix
    :param B: np.ndarray, a :math:`M \times N` matrix
    :return: float, the total variation distance between A and B
    """
    if A.shape != B.shape:
        raise ArgumentError("The shape of A: {} mismatches the shape of B: {}.".format(A.shape, B.shape))

    return np.linalg.norm(A - B, 1) / 2


# TODO: fit to probabilities inputs
def trace_distance(A: np.ndarray, B: np.ndarray) -> float:
    r"""Compute the trace distance between two matrices.

    Compute the trace distance between two matrices with the same dimension.
    For two matrices A and B, which both are math:`m \times n` dimensional,
    the trace distance is defined as:

    .. math:: D(A,B) := \frac{1}{2}\text{Tr}\vert A - B \vert,

    where :math:`\vert A\vert = \sqrt{A^\dagger A}` is the modulus of :math:`A`.

    :param A: np.ndarray, a :math:`M \times N` matrix
    :param B: np.ndarray, a :math:`M \times N` matrix
    :return: float, the complete distance between A and B
    """
    if A.shape != B.shape:
        raise ArgumentError("The shape of A: {} mismatches the shape of B: {}.".format(A.shape, B.shape))

    return np.sum(abs(A - B))


def frobenius_distance(A: np.ndarray, B: np.ndarray) -> float:
    r"""Compute the Frobenius distance between two matrices.

    Compute the Frobenius distance between two matrices with the same dimension.
    For two matrices A and B, which both are math:`m \times n` dimensional,
    the Frobenius distance is defined as:

    .. math:: \text{Fro}(A,B) := \sqrt{\sum_{n=1}^N \sum_{m=1}^M | A_{m,n} - B_{m,n} |^2},

    where :math:`A_{m,n}` is the element of :math:`A` in the :math:`m`-th row and :math:`n`-th column.

    :param A: np.ndarray, a :math:`M \times N` matrix
    :param B: np.ndarray, a :math:`M \times N` matrix
    :return: float, the complete distance between A and B
    """
    if A.shape != B.shape:
        raise ArgumentError("The shape of A: {} mismatches the shape of B: {}.".format(A.shape, B.shape))

    return np.linalg.norm(A - B, 'fro')


def process_fidelity(proc_M: np.ndarray, proc_N: np.ndarray) -> float:
    r"""Compute the process fidelity of two quantum processes.

    Compute the process fidelity of two quantum processes @proc_M and @proc_N, which is mathematically defined as:

    .. math:: F(\mathcal{M}, \mathcal{N}) := \frac{1}{d^2}\text{Tr}\left[[\mathcal{M}]^\dagger [\mathcal{N}]\right],

    where :math:`[\mathcal{M}]` is the Pauli transfer matrix of the quantum process and
    :math:`d` is the dimension of the input quantum system.

    :param proc_M: np.ndarray, the Pauli transfer matrix of quantum process @M
    :param proc_N: np.ndarray, the Pauli transfer matrix of quantum process @N
    :return: float, the process fidelity of the input quantum maps
    """
    if proc_M.shape != proc_N.shape:
        raise ArgumentError("in process_fidelity(): the dimensions of two quantum processes mismatch.")

    dim = proc_M.shape[0]
    fid = np.trace(np.transpose(proc_M) @ proc_N)
    return float(np.real(fid)) / dim


def entanglement_fidelity(proc_M: np.ndarray, proc_N: np.ndarray) -> float:
    r"""Compute the entanglement fidelity of two quantum processes.

    The entanglement fidelity of two quantum processes @proc_M and @proc_N is mathematically defined as:

    .. math::

            F_{\rm ent}(\mathcal{M}, \mathcal{N})
            := \frac{1}{d^2}\left(1 + \text{Tr}\left[[\mathcal{M}]_u^\dagger [\mathcal{N}]_u\right]\right),

    where :math:`[\mathcal{M}]_u` is the unital part of the Pauli transfer matrix and
    :math:`d` is the dimension of the input quantum system.

    The definition is excerpted from the following reference:

    .. [SHBT] Helsen, Jonas, Francesco Battistel, and Barbara M. Terhal.
            "Spectral quantum tomography."
            npj Quantum Information 5.1 (2019): 1-11.

    :param proc_M: np.ndarray, the Pauli transfer matrix of quantum process @M
    :param proc_N: np.ndarray, the Pauli transfer matrix of quantum process @N
    :return: float, the entanglement fidelity of the input quantum maps
    """
    if proc_M.shape != proc_N.shape:
        raise ArgumentError("in entanglement_fidelity(): the dimensions of two quantum processes mismatch.")

    # dimension of the PTM
    dim = proc_M.shape[0]
    # extract the unital part of the two PTMs
    proc_M_u = proc_M[1:, 1:]
    proc_N_u = proc_N[1:, 1:]
    fid = (1 + np.real(np.trace(np.transpose(proc_M_u) @ proc_N_u))) / dim
    return fid


def average_gate_fidelity(proc_M: np.ndarray, proc_N: np.ndarray) -> float:
    r"""Compute the average gate fidelity of two quantum processes.

    The average gate fidelity of two quantum processes @proc_M and @proc_N is mathematically defined as:

    .. math::

            F_{\rm avg}(\mathcal{M}, \mathcal{N}) := \int_\psi d\eta(\psi)\langle\psi\vert
               \mathcal{M}^\dagger\circ\mathcal{N}(\vert\psi\rangle\!\langle\psi\vert)\vert\psi\rangle,

    where :math:`\eta(\psi)` is the Haar measure. It connects to the above entanglement fidelity via:

    .. math::

            F_{\rm avg}(\mathcal{M}, \mathcal{N}) = \frac{dF_{\rm ent}(\mathcal{M}, \mathcal{N}) + 1}{d+1},

    where :math:`d` is the dimension of the input quantum system.

    The definition is excerpted from the following reference:

    .. [SHBT] Helsen, Jonas, Francesco Battistel, and Barbara M. Terhal.
            "Spectral quantum tomography."
            npj Quantum Information 5.1 (2019): 1-11.

    :param proc_M: np.ndarray, the Pauli transfer matrix of quantum process @M
    :param proc_N: np.ndarray, the Pauli transfer matrix of quantum process @N
    :return: float, the average gate fidelity of the input quantum maps
    """
    if proc_M.shape != proc_N.shape:
        raise ArgumentError("in average_gate_fidelity(): the dimensions of two quantum processes mismatch.")

    # dimension of the input quantum system
    d = math.sqrt(proc_M.shape[0])

    # First compute the entanglement fidelity
    fid = entanglement_fidelity(proc_M, proc_N)

    return (d*fid + 1) / (d + 1)
