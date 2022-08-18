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
This script aims to supply a set of functions for computing various entropies quantities.
"""
import numpy as np
import math
from qcompute_qep.exceptions import ArgumentError


def entropy(rho: np.ndarray, base: float = 2) -> float:
    r"""Compute the von Neumann entropy of the given quantum state.

    The von Neumann entropy of a quantum state :math:`\rho` is mathematically defined as:

    .. math:: S(\rho) := - \text{Tr}[\rho\log\rho],

    where the logarithm is defined in base :math:`2` by default.

    :param rho: np.ndarray, a quantum state in its density matrix representation
    :param base: float, the base with respect to which the entropy is evaluated
    :return: float, the entropy of the input state
    """
    eig_vals = np.linalg.eigvals(rho)
    nz_eig_vals = eig_vals[eig_vals != 0]
    if base == 2:
        lg_eig_vals = np.log2(nz_eig_vals)
    elif base == np.e:
        lg_eig_vals = np.log2(nz_eig_vals)
    elif base == 10:
        lg_eig_vals = np.log10(nz_eig_vals)
    else:
        raise ArgumentError("entropy(): the base must be chosen from the set {2, e, 10}.")

    return float(np.real(-sum(nz_eig_vals * lg_eig_vals)))


def quantum_relative_entropy(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""Compute the quantum relative entropy of two quantum states.

    The quantum relative entropy of two quantum states :math:`\rho` (@\rho) and :math:`\sigma` (@sigma)
    is mathematically defined as:

    .. math:: D(\rho \lVert \sigma) := \text{Tr}[\rho(\log\rho - \log\sigma)],

    where the logarithm is defined in base :math:`2` by default.

    :param rho: np.ndarray, a quantum state in its density matrix representation
    :param sigma: np.ndarray, a quantum state in its density matrix representation
    :return: float, the quantum relative entropy of the two input quantum states
    """
    pass
