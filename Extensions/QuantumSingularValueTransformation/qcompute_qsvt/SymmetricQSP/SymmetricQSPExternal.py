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
This script is an implement for an optimization based algorithm for computing the processing parameters in Symmetric
Quantum Signal Processing (SQSP).

Reader may refer to following references for more insights.

.. [LC17] Low, Guang Hao, and Isaac L. Chuang. "Optimal Hamiltonian simulation by quantum signal processing."
    Physical review letters 118.1 (2017): 010501.

.. [DMW+21] Dong, Yulong, et al. "Efficient phase-factor evaluation in quantum signal processing."
    Physical Review A 103.4 (2021): 042419.

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
from scipy import optimize as opt

import Extensions.QuantumSingularValueTransformation.qcompute_qsvt.SymmetricQSP.SymmetricQSPInternalPy as SymmetricQSPInternal


def __func_Wx_map(vec_x: Union[List[float], np.ndarray]) -> np.ndarray:
    r"""Map signal parameters :math:`\vec x` (**@vec_x**) to signal quantum gates :math:`W(\vec x)` (**@list_Wx**).

    Input a list of float numbers :math:`\vec x=(x_1,x_2,\cdots,x_n)\in[-1,1]^n` for some :math:`n`,
    and return the list of matrices :math:`W(\vec x):=(W(x_1),W(x_2),\cdots,W(x_n))`, where

    .. math:: W(x):=e^{iX\arccos x}= \begin{pmatrix} x&i\sqrt{1-x^2}\\ i\sqrt{1-x^2}&x \end{pmatrix}

    :param vec_x: :math:`\vec x`, `List[float]` or `np.ndarray`,
        whose components :math:`x_j\in[-1,1]` are distinct signal parameters
    :return: **list_Wx** – :math:`W(\vec x)`, `np.ndarray` of dimension :math:`n\times2\times2`,
        as a sequence of :math:`W(x_j)` for :math:`x_j\in\vec x`
    """
    list_mat = []
    for float_x in vec_x:
        comp_y = 1j * np.sqrt(1 - float_x**2)
        list_mat.append(np.array([[float_x, comp_y], [comp_y, float_x]]))
    return np.array(list_mat)


def __func_L(
    vec_phi: Union[List[float], np.ndarray],
    list_Wx: np.ndarray,
    vec_fx: Union[List[float], np.ndarray],
    bool_parity: int,
) -> float:
    r"""Compute the value of the loss function :math:`L_{f,\vec x}(\vec \phi)` (**@float_L**).

    The loss function is defined as

    .. math::

        L_{f,\vec x}(\vec \phi):=\frac{1}{2n}\sum_{j=1}^n\left|A_{\vec\phi_p}(x_j)-f(x_j)\right|^2=\frac{1}{2n}
        \left\|\vec A-\vec f\right\|^2.

    We need to compute such value :math:`L_{f,\vec x}(\vec\phi)` when inputting :math:`\vec\phi`, :math:`W(\vec x)`,
    :math:`f` and :math:`p`.

    :param vec_phi: :math:`\vec\phi`, `List[float]` or `np.ndarray`,
        whose components are the free variables in symmetric processing parameters
    :param list_Wx: :math:`W(\vec x)`, `np.ndarray` of dimension :math:`n\times2\times2`,
        as a sequence of :math:`W(x_j)` for :math:`x_j\in\vec x`
    :param vec_fx: :math:`\vec f`, `List[float]` or `np.ndarray`,
        denoting the sequence of function values for the target function :math:`f` at point sequence :math:`\vec x`
    :param bool_parity: :math:`p`, `int`,
        denoting which symmetrization used in SQSP and also the parity of :math:`f`
    :return: **float_L** – :math:`L_{f,\vec x}(\vec \phi)`, `float`, the value of such loss function
    """
    int_n = len(list_Wx)
    vec_A = SymmetricQSPInternal.func_symQSP_A_map(vec_phi, list_Wx, bool_parity)
    vec_dif = vec_A - vec_fx
    float_L = (vec_dif @ vec_dif) * (0.5 / int_n)
    return float_L


def __func_gradL(
    vec_phi: Union[List[float], np.ndarray],
    list_Wx: np.ndarray,
    vec_fx: Union[List[float], np.ndarray],
    bool_parity: int,
) -> Tuple[np.ndarray, float]:
    r"""Compute the gradient :math:`\nabla L_{f,\vec x}(\vec \phi)` (**@vec_gradL**) and value
    :math:`L_{f,\vec x}(\vec \phi)` (**@float_L**) simultaneously.

    The loss function is defined as

    .. math::

        L_{f,\vec x}(\vec \phi):=\frac{1}{2n}\sum_{j=1}^n\left|A_{\vec\phi_p}(x_j)-f(x_j)\right|^2=\frac{1}{2n}
        \left\|\vec A-\vec f\right\|^2,

    and its gradient upon :math:`\vec \phi` is

    .. math::

        \nabla L_{f,\vec x}(\vec \phi)=\frac{1}{2n}\nabla \left\|\vec A-\vec f\right\|^2
        =\frac{1}{n}\nabla\vec A\cdot (\vec A-\vec f).

    We need to compute the gradient :math:`\nabla L_{f,\vec x}(\vec \phi)` and value :math:`L_{f,\vec x}(\vec \phi)`
    when inputting :math:`\vec\phi`, :math:`W(\vec x)`, :math:`f` and :math:`p`.

    :param vec_phi: :math:`\vec\phi`, `List[float]` or `np.ndarray`,
        whose components are the free variables in symmetric processing parameters
    :param list_Wx: :math:`W(\vec x)`, `np.ndarray` of dimension :math:`n\times2\times2`,
        as a sequence of :math:`W(x_j)` for :math:`x_j\in\vec x`
    :param vec_fx: :math:`\vec f`, `List[float]` or `np.ndarray`,
        denoting the sequence of function values for the target function :math:`f` at point sequence :math:`\vec x`
    :param bool_parity: :math:`p`, `int`,
        denoting which symmetrization used in SQSP and also the parity of :math:`f`
    :return: **(vec_gradL, float_L)** – `Tuple[np.ndarray, float]`, where:

        + **vec_gradL** – :math:`\nabla L_{f,\vec x}(\vec \phi)`, `np.ndarray` of dimension :math:`d`,
          denoting the gradient of such loss function w.r.t. :math:`\vec\phi`

        + **float_L** – :math:`L_{f,\vec x}(\vec \phi)`, `float`, the value of such loss function
    """
    int_n = len(list_Wx)
    mat_gradA, vec_A = SymmetricQSPInternal.func_symQSP_gradA_map(vec_phi, list_Wx, bool_parity)
    vec_dif = vec_A - vec_fx
    vec_gradL = vec_dif @ mat_gradA * (1 / int_n)
    float_L = vec_dif @ vec_dif * (0.5 / int_n)
    return vec_gradL, float_L


def __func_LBFGS_QSP_backtracking(
    int_deg: int,
    list_Wx: np.ndarray,
    vec_fx: Union[List[float], np.ndarray],
    bool_parity: int,
    int_m: int = 300,
    float_gamma: float = 0.5,
    float_accuracy_rate: float = 1e-3,
    float_minstep: float = 1e-5,
    float_criteria: float = 1e-14,
    int_maxiter: int = 50000,
) -> np.ndarray:
    r"""Using L-BFGS algorithm to minimize loss function :math:`L_{f,\vec x}(\vec \phi)`, here the linear searching
    method is backtracking with Armijo rule.

    This function will loop for at most **int_maxiter** iterations,
    and return :math:`\vec\phi` once :math:`L_{f,\vec x}(\vec\phi)<\epsilon^2`.

    Parameters for SQSP:

    :param int_deg: :math:`\deg`, `int`,
        denoting the degree in :math:`x` of the polynomial :math:`A_{\vec \phi_p}(x)` to be obtained
    :param list_Wx: :math:`W(\vec x)`, `np.ndarray` of dimension :math:`n\times2\times2`,
        as a sequence of :math:`W(x_j)` for :math:`x_j\in\vec x`
    :param vec_fx: :math:`\vec f`, `List[float]` or `np.ndarray`,
        denoting the sequence of function values for the target function :math:`f` at point sequence :math:`\vec x`
    :param bool_parity: :math:`p`, `int`,
        denoting which symmetrization used in SQSP and also the parity of :math:`f`

    Parameters for L-BFGS:

    :param int_m: `int`, storage length in L-BFGS, 20 is enough by experience
    :param float_gamma: `float`, step reducing rate in linear searching
    :param float_accuracy_rate: `float`, confirm step size parameters in linear search
    :param float_minstep: `float`, minimum step size in linear search
    :param float_criteria: :math:`\epsilon`, `float`, stop criteria for L-BFGS
    :param int_maxiter: `int`, max iteration for L-BFGS

    :return: **vec_phi** – :math:`\vec\phi`, `np.ndarray` of dimension :math:`d` with :math:`A_{\vec\phi_p}\approx f`
    """
    int_d = int_deg // 2 + 1  # length of vec_phi

    # initialize vec_phi = [pi/4, 0, 0, ...]
    vec_phi = np.zeros(int_d)
    vec_phi[0] = np.pi / 4

    idx_t = 0  # the register for iteration number
    int_m_used = 0  # length of used storage in L-BFGS. a queue structure is used.
    int_m_front = -1  # the index for the newest element in the queue, -1 means the queue is empty
    # list_s, list_y, list_rho are synchronous queues
    list_s = np.zeros((int_m, int_d))  # the register for increments for the sequence of vec_phi
    list_y = np.zeros((int_m, int_d))  # the register for increments for the sequence of vec_gradL
    list_rho = np.zeros(int_m)  # the register for intermediate value rho_i, should be distinguished with float_rho

    # the initial value and grad for target function
    vec_g, float_L = __func_gradL(vec_phi, list_Wx, vec_fx, bool_parity)

    #  start L-BFGS algorithm

    while True:
        # we use BFGS iteration to compute the optimizing direction Hess ** (-1).vec_q,
        # vec_q is initialed as vec_g
        vec_q = vec_g.copy()
        list_alpha = np.zeros(int_m_used)
        for idx_i in range(int_m_used):
            idx_pointer = (int_m_front - idx_i) % int_m
            list_alpha[idx_i] = list_rho[idx_pointer] * np.dot(list_s[idx_pointer], vec_q)
            vec_q = vec_q - list_alpha[idx_i] * list_y[idx_pointer]

        # Hess_0 == diag(2,2,...,2,2) if deg is odd
        vec_q *= 0.5
        # and Hess_0 == diag(2,2,...,2,1) if deg is even
        if bool_parity == 0:
            vec_q[-1] *= 2

        for idx_i in reversed(range(int_m_used)):
            idx_pointer = (int_m_front - idx_i) % int_m
            float_beta = list_rho[idx_pointer] * np.dot(list_y[idx_pointer], vec_q)
            vec_q += (list_alpha[idx_i] - float_beta) * list_s[idx_pointer]

        # -vec_q is the searching direction in the next step
        # beginning searching

        float_step = 1.08  # based on experience, chosen between 1~1.3.

        exp_des = np.dot(vec_g, vec_q) * float_accuracy_rate  # comes from Armijo rule
        while True:
            vec_phi_next = vec_phi - float_step * vec_q
            float_L_next = __func_L(vec_phi_next, list_Wx, vec_fx, bool_parity)
            # Armijo rule
            if float_L - float_L_next > exp_des * float_step or float_step < float_minstep:
                break
            float_step *= float_gamma

        # ending linear searching
        # begin cleanup
        # update the register, moving pointer, storage, etc.

        vec_g_next, _ = __func_gradL(vec_phi_next, list_Wx, vec_fx, bool_parity)  # calc the new grad
        int_m_used = min(int_m_used + 1, int_m)  # update the queue length
        int_m_front = (int_m_front + 1) % int_m  # moving queue pointer
        list_s[int_m_front] = -float_step * vec_q  # store the increment
        list_y[int_m_front] = vec_g_next - vec_g  # store the increment
        list_rho[int_m_front] = 1 / np.dot(list_s[int_m_front], list_y[int_m_front])  # store the intermediate var
        # ending the cleanup. prepare for the next iteration
        vec_phi = vec_phi_next.copy()
        vec_g = vec_g_next.copy()
        float_L = float_L_next
        idx_t += 1
        print("The {0[0]} iteration: L={0[1]}".format([idx_t, float_L]))

        # the stop of the iteration
        if idx_t >= int_maxiter:
            print("Max iteration reached.")
            break
        if float_L < float_criteria**2:
            print("Stop criteria satisfied.")
            break

    return vec_phi


def __func_LBFGS_QSP_interpolation(
    int_deg: int,
    list_Wx: np.ndarray,
    vec_fx: Union[List[float], np.ndarray],
    bool_parity: int,
    int_m: int = 300,
    float_criteria: float = 1e-14,
    int_maxiter: int = 50000,
) -> np.ndarray:
    r"""Using L-BFGS algorithm to minimize loss function :math:`L_{f,\vec x}(\vec \phi)`, here the linear searching
    method is quadratic interpolation.

    This function will loop for at most **int_maxiter** iterations,
    and return :math:`\vec\phi` once :math:`L_{f,\vec x}(\vec\phi)<\epsilon^2`.

    Parameters for SQSP:

    :param int_deg: :math:`\deg`, `int`,
        denoting the degree in :math:`x` of the polynomial :math:`A_{\vec \phi_p}(x)` to be obtained
    :param list_Wx: :math:`W(\vec x)`, `np.ndarray` of dimension :math:`n\times2\times2`,
        as a sequence of :math:`W(x_j)` for :math:`x_j\in\vec x`
    :param vec_fx: :math:`\vec f`, `List[float]` or `np.ndarray`,
        denoting the sequence of function values for the target function :math:`f` at point sequence :math:`\vec x`
    :param bool_parity: :math:`p`, `int`,
        denoting which symmetrization used in SQSP and also the parity of :math:`f`

    Parameters for L-BFGS:

    :param int_m: `int`, storage length in L-BFGS, 20 is enough by experience
    :param float_criteria: :math:`\epsilon`, `float`, stop criteria for L-BFGS
    :param int_maxiter: `int`, max iteration for L-BFGS

    :return: **vec_phi** – :math:`\vec\phi`, `np.ndarray` of dimension :math:`d` with :math:`A_{\vec\phi_p}\approx f`
    """
    int_d = int_deg // 2 + 1  # length of vec_phi

    # initialize vec_phi = [pi/4, 0, 0, ...]
    vec_phi = np.zeros(int_d)
    vec_phi[0] = np.pi / 4

    idx_t = 0  # the register for iteration number
    int_m_used = 0  # length of used storage in L-BFGS. a queue structure is used.
    int_m_front = -1  # the index for the newest element in the queue, -1 means the queue is empty
    # list_s, list_y, list_rho are synchronous queues
    list_s = np.zeros((int_m, int_d))  # the register for increments for the sequence of vec_phi
    list_y = np.zeros((int_m, int_d))  # the register for increments for the sequence of vec_gradL
    list_rho = np.zeros(int_m)  # the register for intermediate value rho_i, should be distinguished with float_rho

    # the initial value and grad for target function
    vec_g, float_L = __func_gradL(vec_phi, list_Wx, vec_fx, bool_parity)

    #  start L-BFGS algorithm

    while True:
        # we use BFGS iteration to compute the optimizing direction Hess ** (-1).vec_q,
        # vec_q is initialed as vec_g
        vec_q = vec_g.copy()
        list_alpha = np.zeros(int_m_used)
        for idx_i in range(int_m_used):
            idx_pointer = (int_m_front - idx_i) % int_m
            list_alpha[idx_i] = list_rho[idx_pointer] * np.dot(list_s[idx_pointer], vec_q)
            vec_q = vec_q - list_alpha[idx_i] * list_y[idx_pointer]

        # Hess_0 == diag(2,2,...,2,2) if deg is odd
        vec_q *= 0.5
        # and Hess_0 == diag(2,2,...,2,1) if deg is even
        if bool_parity == 0:
            vec_q[-1] *= 2

        for idx_i in reversed(range(int_m_used)):
            idx_pointer = (int_m_front - idx_i) % int_m
            float_beta = list_rho[idx_pointer] * np.dot(list_y[idx_pointer], vec_q)
            vec_q += (list_alpha[idx_i] - float_beta) * list_s[idx_pointer]

        # -vec_q is the searching direction in the next step
        # beginning searching
        # use quadratic interpolation for calculate the step size based on experience.

        if idx_t == 0:
            float_step = 1.0
        else:
            float_b = -vec_q @ vec_g
            float_a = __func_L(vec_phi - vec_q, list_Wx, vec_fx, bool_parity) - float_L - float_b
            float_step = float_b / (-2 * float_a)

        # ending linear searching
        # begin cleanup
        # update the register, moving pointer, storage, etc.
        vec_phi_next = vec_phi - vec_q * float_step
        vec_g_next, float_L_next = __func_gradL(vec_phi_next, list_Wx, vec_fx, bool_parity)  # calc the new grad

        int_m_used = min(int_m_used + 1, int_m)  # update the queue length
        int_m_front = (int_m_front + 1) % int_m  # moving queue pointer
        list_s[int_m_front] = -float_step * vec_q  # store the increment
        list_y[int_m_front] = vec_g_next - vec_g  # store the increment
        list_rho[int_m_front] = 1 / np.dot(list_s[int_m_front], list_y[int_m_front])  # store the intermediate var
        # ending the cleanup. prepare for the next iteration
        vec_phi = vec_phi_next.copy()
        vec_g = vec_g_next.copy()
        float_L = float_L_next
        idx_t += 1
        # print("The {0[0]} iteration: L={0[1]}".format([idx_t, float_L]))

        # the stop of the iteration
        if idx_t >= int_maxiter:
            print("Max iteration reached.")
            break
        if float_L < float_criteria**2:
            # print("Stop criteria satisfied.")
            break

    return vec_phi


def __func_LBFGS_QSP_scipy(
    int_deg: int,
    list_Wx: np.ndarray,
    vec_fx: Union[List[float], np.ndarray],
    bool_parity: int,
    m: int = 300,
    pgtol: float = 1e-24,
    factr: float = 1e-12,
) -> np.ndarray:
    r"""Using the implement `scipy.optimize.fmin_l_bfgs_b` for L-BFGS-B algorithm to minimize loss function
    :math:`L_{f,\vec x}(\vec \phi)`.

    Parameters for SQSP:

    :param int_deg: :math:`\deg`, `int`,
        denoting the degree in :math:`x` of the polynomial :math:`A_{\vec \phi_p}(x)` to be obtained
    :param list_Wx: :math:`W(\vec x)`, `np.ndarray` of dimension :math:`n\times2\times2`,
        as a sequence of :math:`W(x_j)` for :math:`x_j\in\vec x`
    :param vec_fx: :math:`\vec f`, `List[float]` or `np.ndarray`,
        denoting the sequence of function values for the target function :math:`f` at point sequence :math:`\vec x`
    :param bool_parity: :math:`p`, `int`,
        denoting which symmetrization used in SQSP and also the parity of :math:`f`

    Parameters for `scipy.optimize.fmin_l_bfgs_b`, details may refer to its `API documents
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_:

    :param m: `int`, the maximum number of variable metric corrections used to define the limited memory matrix.
    :param factr: `float`
    :param pgtol: `float`

    :return: **vec_phi** – :math:`\vec\phi`, `np.ndarray` of dimension :math:`d` with :math:`A_{\vec\phi_p}\approx f`
    """
    int_d = int_deg // 2 + 1  # the length of vec_phi

    # initialize vec_phi = [pi/4, 0, 0, ...]
    vec_phi = np.zeros(int_d)
    vec_phi[0] = np.pi / 4

    vec_phi = opt.fmin_l_bfgs_b(
        func=lambda arg: __func_L(arg, list_Wx, vec_fx, bool_parity),
        x0=vec_phi,
        fprime=lambda arg: __func_gradL(arg, list_Wx, vec_fx, bool_parity)[0],
        m=m,
        pgtol=pgtol,
        factr=factr,
        iprint=1,
    )[0]
    return vec_phi


def __func_LBFGS_QSP(
    int_deg: int,
    list_Wx: np.ndarray,
    vec_fx: Union[List[float], np.ndarray],
    bool_parity: int,
    method: str = "interpolation",
) -> np.ndarray:
    r"""Pack up three `__func_LBFGS_QSP` functions together.

    Other parameters used in L-BFGS or L-BFGS-B algorithm are set as default values.

    :param int_deg: :math:`\deg`, `int`,
        denoting the degree in :math:`x` of the polynomial :math:`A_{\vec \phi_p}(x)` to be obtained
    :param list_Wx: :math:`W(\vec x)`, `np.ndarray` of dimension :math:`n\times2\times2`,
        as a sequence of :math:`W(x_j)` for :math:`x_j\in\vec x`
    :param vec_fx: :math:`\vec f`, `List[float]` or `np.ndarray`,
        denoting the sequence of function values for the target function :math:`f` at point sequence :math:`\vec x`
    :param bool_parity: :math:`p`, `int`,
        denoting which symmetrization used in SQSP and also the parity of :math:`f`
    :param method: `str`, should be "interpolation", "backtracking" or "scipy"

    :return: **vec_phi** – :math:`\vec\phi`, `np.ndarray` of dimension :math:`d` with :math:`A_{\vec\phi_p}\approx f`
    """
    if method == "interpolation":
        return __func_LBFGS_QSP_interpolation(int_deg, list_Wx, vec_fx, bool_parity)
    elif method == "backtracking":
        return __func_LBFGS_QSP_backtracking(int_deg, list_Wx, vec_fx, bool_parity)
    elif method == "scipy":
        return __func_LBFGS_QSP_scipy(int_deg, list_Wx, vec_fx, bool_parity)
    else:
        raise Exception('LBFGS method error, should be "interpolation", "backtracking" or "scipy".')
