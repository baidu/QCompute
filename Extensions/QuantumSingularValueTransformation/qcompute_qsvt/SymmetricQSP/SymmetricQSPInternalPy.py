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
This script supplies core functions used to optimize the loss function in Symmetric Quantum Signal Processing (SQSP).

.. note::

    In module `SymmetricQSP`, Variables with the same meaning will be named uniformly:

        + **vec_phi @**:math:`\vec\phi`: an array of length **int_d @**:math:`d`, whose components are the free
          variables in processing parameters

        + **vec_x @**:math:`\vec x`: an array of length **int_n @**:math:`n`, whose components are distinct signal
          parameters

        + **list_expiZ @**:math:`e^{iZ\vec\phi}`: an :math:`d\times2\times2` array, as a sequence of processing quantum
          gate :math:`2\times2` matrix :math:`e^{iZ\phi}` for :math:`\phi` in :math:`\vec\phi`

        + **list_Wx @**:math:`W(\vec x)`: an :math:`n\times2\times2` array, as a sequence of signal quantum gate
          :math:`2\times2` matrix :math:`e^{iX\arccos x}` for :math:`x` in :math:`\vec x`

        + **bool_parity @**:math:`p`: an `int`, denoting which symmetrization used in such SQSP.

        + **vec_A @**:math:`\vec A`: an array of length :math:`n`, as a sequence of :math:`A_{\vec\phi_p}(x)` for
          :math:`x` in :math:`\vec x`, where :math:`\vec\phi_p` is the :math:`p`-symmetrization of :math:`\vec\phi`,
          also the actual processing parameters in such SQSP regarded as QSP.

        + **mat_gradA @**:math:`\nabla\vec A`: an :math:`n\times d` array, as the Jacobi matrix for :math:`\vec A`
          w.r.t. :math:`\vec\phi`

        + **vec_fx @**:math:`\vec f`: an array of length :math:`n`, as a sequence of :math:`f(x)` for :math:`x` in
          :math:`\vec x`, where :math:`f` is the target signal processing function, our mission is just to find
          :math:`\vec\phi` such that :math:`\vec A\approx\vec f`
"""

from typing import List, Tuple, Union

import numpy as np


def __func_expiZ_map(vec_phi: Union[List[float], np.ndarray]) -> np.ndarray:
    r"""Map processing parameters :math:`\vec\phi` (**@vec_phi**) to processing quantum gates :math:`e^{iZ\vec\phi}`
    (**@list_expiZ**).

    Input a list of float numbers :math:`\vec\phi=(\phi_1,\phi_2,\cdots,\phi_d)\in\mathbb R^d` for some :math:`d`,
    and return the list of matrices :math:`e^{iZ\vec\phi}:=(e^{iZ\phi_1},e^{iZ\phi_2},\cdots,e^{iZ\phi_d})`, where

    .. math:: e^{iZ\phi_k}= \begin{pmatrix} e^{i\phi_k}&0\\ 0&e^{-i\phi_k} \end{pmatrix}\text{ for }k=1,2,\cdots,d

    :param vec_phi: :math:`\vec\phi`, `List[float]` or `np.ndarray`,
        whose components :math:`\phi_k\in\mathbb R` are processing parameters
    :return: **list_expiZ** – :math:`e^{iZ\vec\phi}`, `np.ndarray` of dimension :math:`n\times2\times2`,
        as a sequence of :math:`e^{iZ\phi_k}` for :math:`\phi_k\in\vec\phi`
    """
    list_mat = []
    for float_phi in vec_phi:
        comp_entry = np.exp(1j * float_phi)
        list_mat.append(np.array([[comp_entry, 0], [0, comp_entry.conjugate()]]))
    return np.array(list_mat)


def func_symQSP_A_map(vec_phi: Union[List[float], np.ndarray], list_Wx: np.ndarray, bool_parity: int) -> np.ndarray:
    r"""Compute :math:`\vec A` (**@vec_A**).

    For signal parameter :math:`x\in[-1,1]`, processing parameter
    :math:`\vec\phi=(\phi_1,\phi_2,\cdots,\phi_d)\in\mathbb{R}^d` and parity :math:`p\in\{0,1\}`,
    we define SQSP matrix

    .. math::

        W_{\vec\phi_p}(x):=\begin{cases}
            e^{iZ\phi_1}W(x)e^{iZ\phi_2}W(x)\cdots e^{iZ\phi_{d-1}}W(x)e^{iZ\phi_d}W(x)e^{iZ\phi_{d-1}}\cdots
                e^{iZ\phi_1},&\text{if }p=0;\\
            e^{iZ\phi_1}W(x)e^{iZ\phi_2}W(x)\cdots e^{iZ\phi_{d}}W(x)e^{iZ\phi_d}W(x)e^{iZ\phi_{d-1}}\cdots
                e^{iZ\phi_1},&\text{if }p=1,
        \end{cases}

    and SQSP function

    .. math:: A_{\vec\phi_p}(x) := \operatorname{Tr}(W_{\vec\phi_p}(x))/2.

    Then we need to compute the mapping image

    .. math:: \vec A:= (A_{\vec\phi_p}(x_1),A_{\vec\phi_p}(x_2),\cdots,A_{\vec\phi_p}(x_n))\in\mathbb R^n.

    :param vec_phi: :math:`\vec\phi`, `List[float]` or `np.ndarray`,
        whose components are the free variables in symmetric processing parameters
    :param list_Wx: :math:`W(\vec x)`, `np.ndarray` of dimension :math:`n \times 2 \times 2`,
        as a sequence of :math:`W(x_j)` for :math:`x_j\in\vec x`
    :param bool_parity: :math:`p`, `int`,
        denoting which symmetrization used in SQSP and also the parity of :math:`f`
    :return: **vec_A** – :math:`\vec A`, `np.ndarray` of dimension :math:`n`,
        as a sequence of :math:`A_{\vec\phi_p}(x_j)` for :math:`x_j\in\vec x`
    """
    int_n = len(list_Wx)  # int_n is the length of vec_x and list_Wx
    list_expiZ = __func_expiZ_map(vec_phi)

    # initialized list_symQSP_half as [list_expiZ[0], ... , list_expiZ[0]], list_expiZ[0][:1] is a 1x2 matrix
    list_symQSP_half = [list_expiZ[0][:1].copy() for _ in range(int_n)]

    # The outer loop of the following double loop can be calculated in parallel
    for idx_x in range(int_n):
        for mat_expiZ in list_expiZ[1:]:
            # list_symQSP_half[idx_x] is a 1x2 matrix
            # the following multiplication is (1x2)@(2x2)@(2x2) and obtain a 1x2 matrix
            list_symQSP_half[idx_x] = (list_symQSP_half[idx_x] @ list_Wx[idx_x]) @ mat_expiZ

    # initial vec_A, which will stores the return value [A(vec_phi, xj) for xj in vec_x]
    vec_A = np.zeros(int_n)

    # the following calculation is depend on the parity and simplified.
    if bool_parity == 0:
        for idx_x in range(int_n):
            mat_half0 = list_symQSP_half[idx_x][0]
            vec_A[idx_x] = (mat_half0[0] ** 2 * list_expiZ[-1][1, 1] + mat_half0[1] ** 2 * list_expiZ[-1][0, 0]).real
    else:
        for idx_x in range(int_n):
            mat_half0 = list_symQSP_half[idx_x][0]
            vec_A[idx_x] = ((mat_half0[0] ** 2 + mat_half0[1] ** 2) * list_Wx[idx_x][0, 0] +
                            2 * mat_half0[0] * mat_half0[1] * list_Wx[idx_x][0, 1]).real
    return vec_A


def func_symQSP_gradA_map(vec_phi: Union[List[float], np.ndarray], list_Wx: np.ndarray,
                          bool_parity: int) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute :math:`\nabla\vec A` (**@mat_gradA**) and :math:`\vec A` (**@vec_A**).

    For signal parameter :math:`x\in[-1,1]`, processing parameter
    :math:`\vec\phi=(\phi_1,\phi_2,\cdots,\phi_d)\in\mathbb{R}^d` and parity :math:`p\in\{0,1\}`,
    we define SQSP matrix

    .. math::

        W_{\vec\phi_p}(x):=\begin{cases}
            e^{iZ\phi_1}W(x)e^{iZ\phi_2}W(x)\cdots e^{iZ\phi_{d-1}}W(x)e^{iZ\phi_d}W(x)e^{iZ\phi_{d-1}}\cdots
                e^{iZ\phi_1},&\text{if }p=0;\\
            e^{iZ\phi_1}W(x)e^{iZ\phi_2}W(x)\cdots e^{iZ\phi_{d}}W(x)e^{iZ\phi_d}W(x)e^{iZ\phi_{d-1}}\cdots
                e^{iZ\phi_1},&\text{if }p=1,
        \end{cases}

    and SQSP function

    .. math:: A_{\vec\phi_p}(x) := \operatorname{Tr}(W_{\vec\phi_p}(x))/2.

    Then we need to compute the mapping image

    .. math:: \vec A:= (A_{\vec\phi_p}(x_1),A_{\vec\phi_p}(x_2),\cdots,A_{\vec\phi_p}(x_n))\in\mathbb R^n.

    and its Jacobi matrix

    .. math::

        \nabla\vec A:= \begin{pmatrix}
            \frac{\partial A_{\vec\phi_p}(x_1)}{\partial \phi_1} &\cdots
            &\frac{\partial A_{\vec\phi_p}(x_1)}{\partial \phi_d}\\
            \vdots &\ddots &\vdots \\
            \frac{\partial A_{\vec\phi_p}(x_n)}{\partial \phi_1} &\cdots
            &\frac{\partial A_{\vec\phi_p}(x_n)}{\partial \phi_d}
        \end{pmatrix}\in\mathbb R^{n\times d}.

    Here we introduce :math:`W_{\text{front}}\in\mathbb{C}^{n\times d\times2\times2}` (**@reg_mat_front**) to record
    intermediate results:

    .. math::

        W_{\text{front}}[j,k]=\prod_{l=1}^{k}\left(e^{iZ\phi_l}W(x_{j+1})\right)\in\mathbb C^{2\times 2},
        \text{ for }j=0,1,\cdots,n-1,\text{ and }k=0,1,\cdots,d-1.

    Then, we use :math:`W_{\text{front}}[j,k]` and :math:`p` to compute entries
    :math:`\frac{\partial A_{\vec\phi_p}(x_{j+1})}{\partial \phi_{k+1}}` in :math:`\nabla\vec A`. Meanwhile
    :math:`\vec A[j]` can be naturally computed using :math:`p` and :math:`W_{\text{front}}[j,d-1]`.

    :param vec_phi: :math:`\vec\phi`, `List[float]` or `np.ndarray`,
        whose components are the free variables in symmetric processing parameters
    :param list_Wx: :math:`W(\vec x)`, `np.ndarray` of dimension :math:`n \times 2 \times 2`,
        as a sequence of :math:`W(x_j)` for :math:`x_j\in\vec x`
    :param bool_parity: :math:`p`, `int`,
        denoting which symmetrization used in SQSP and also the parity of :math:`f`


    :return: **(mat_gradA, vec_A)** – `Tuple[np.ndarray, np.ndarray]`, where:

        + **mat_gradA** – :math:`\nabla\vec A`, `np.ndarray` of dimension :math:`n\times d`,
          denoting the Jacobi matrix of :math:`\vec A` w.r.t. :math:`\vec\phi`

        + **vec_A** – :math:`\vec A`, `np.ndarray` of dimension :math:`n`,
          as a sequence of :math:`A_{\vec\phi_p}(x_j)` for :math:`x_j\in\vec x`
    """
    int_d = len(vec_phi)
    int_n = len(list_Wx)
    list_expiZ = __func_expiZ_map(vec_phi)

    # initialize reg_mat_front which will register QSP(vec_phi[:k], xj) for xj and k
    reg_mat_front = np.zeros((int_n, int_d, 2, 2), dtype=complex)
    # initialize reg_mat_QSP which will register QSP(vec_phi_sym, xj) for xj
    reg_mat_QSP = np.zeros((int_n, 2, 2), dtype=complex)
    # initialize vec_A which will register A(vec_phi_sym, xj) for xj
    vec_A = np.zeros(int_n)

    # The following 'if' clause is to compute reg_mat_front, reg_mat_QSP and vec_A.
    # Especially, the 'if' and 'else' branch codes have
    # only difference in the calculation of reg_mat_QSP[idx_x]
    if bool_parity == 0:
        # The outer loop of the following double loop can be calculated in parallel
        for idx_x in range(int_n):
            reg_mat_front[idx_x, 0] = np.eye(2)
            for idx_phi in range(int_d - 1):
                reg_mat_front[idx_x, idx_phi + 1] = reg_mat_front[idx_x, idx_phi] @ (
                        list_expiZ[idx_phi] @ list_Wx[idx_x])

            reg_mat_QSP[idx_x] = np.inner(reg_mat_front[idx_x, -1] @ list_expiZ[-1], reg_mat_front[idx_x, -1])

            vec_A[idx_x] = reg_mat_QSP[idx_x, 0, 0].real
    else:
        # The outer loop of the following double loop can be calculated in parallel
        for idx_x in range(int_n):
            reg_mat_front[idx_x, 0] = np.eye(2)
            for idx_phi in range(int_d - 1):
                reg_mat_front[idx_x, idx_phi + 1] = reg_mat_front[idx_x, idx_phi] @ (
                        list_expiZ[idx_phi] @ list_Wx[idx_x])

            mat_temp = reg_mat_front[idx_x, -1] @ list_expiZ[-1]
            reg_mat_QSP[idx_x] = np.inner(mat_temp @ list_Wx[idx_x], mat_temp)

            vec_A[idx_x] = reg_mat_QSP[idx_x, 0, 0].real

    # initialize Jacobi matrix mat_gradA
    # which will register the derivative of vec_A[j] w.r.t vec_phi[k] for j and k
    mat_gradA = np.zeros((int_n, int_d))

    # The following double loops can be calculated in parallel
    for idx_x in range(int_n):
        for idx_phi in range(int_d):
            mat_temp_front = reg_mat_front[idx_x, idx_phi, :, 0]  # a 2D complex vector
            # (1x2) @ (2x2) @ (2x1) is a scalar
            mat_gradA[idx_x, idx_phi] = -2 * (np.conjugate(mat_temp_front) @ reg_mat_QSP[idx_x] @ mat_temp_front).imag

    # even case modification
    if bool_parity == 0:
        mat_gradA[:, -1] *= 0.5
    return mat_gradA, vec_A
