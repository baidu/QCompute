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
This script computes the processing parameters for the mission Hamiltonian simulation.

Given a Hamiltonian :math:`\check H\in\mathbb C^{2^n\times2^n}` block-encoded by circuit
:math:`\operatorname{BE}\in\mathbb C^{2^{m+n}\times2^{m+n}}` satisfying :math:`\operatorname{BE}^2=I_{2^{m+n}}`,
the :math:`\tau`-time evolution for :math:`\check H` is

.. math::

    e^{-i\tau \check H}=\cos(\tau \check H) - i\sin(\tau \check H).

To implement a approximated block-encoding for :math:`e^{-i\tau \check H}`, we may find :math:`\vec\phi_\Re` and
:math:`\vec\phi_\Im` satisfying

.. math::

    A_{\vec\phi_\Re}(x)\approx\cos(\tau x)/2,\
    A_{\vec\phi_\Im}(x)\approx-\sin(\tau x)/2,

where the scalar :math:`1/2` is introduced to use module **qcompute_qsvt.SymmetricQSP**.
Besides, we assume the degree of :math:`A_{\vec\phi_\Re}` equals to :math:`\deg`,
and that of :math:`A_{\vec\phi_\Im}` :math:`\deg + 1`, and use Jacobi–Anger expansion to decision
:math:`A_{\vec\phi_\Re}` and :math:`A_{\vec\phi_\Im}`.

Then we could use **qcompute_qsvt.SymmetricQSP.SymmetricQSPExternal.__func_LBFGS_QSP** to compute their  processing
parameters.

Reader may refer to following reference for more insights.

.. [DMW+21] Dong, Yulong, et al. "Efficient phase-factor evaluation in quantum signal processing."
    Physical Review A 103.4 (2021): 042419.
"""

from typing import List, Tuple

import numpy as np
from scipy.special import jv as BesselJ

from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.SymmetricQSP.SymmetricQSPExternal import (
    __func_Wx_map,
    __func_LBFGS_QSP,
)


def __HS_approx_data(float_tau: float, int_deg: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Decision :math:`\vec x` and compute :math:`\vec f` in mission Hamiltonian Simulation.

    After inputting the evolution time :math:`\tau` and approximating degree :math:`\deg`, we decide which signal
    parameters :math:`\vec x` will be used.
    Then we can use Jacobi–Anger expansion to compute the real part and image part of :math:`\vec f`, respectively.
    Here :math:`f(x)=e^{-i\tau x}\in\mathbb C[x]` denotes the QSP function, and :math:`\vec f` is defined in
    **qcompute_qsvt.SymmetricQSP.SymmetricQSPExternal.__func_LBFGS_QSP**

    :param float_tau: :math:`\tau`, `float`, the simulation time in Hamiltonian simulation
    :param int_deg: :math:`\deg`, `int`, should be even and non-negative,
        the degree of approximation polynomials for real part :math:`\cos(\tau x)`
    :return: **(vec_fx_re, vec_fx_im, vec_x)** – `Tuple[np.ndarray, np.ndarray, np.ndarray]`, where:

        + **vec_fx_re** – :math:`\vec f_\Re`, `np.ndarray` of dimension :math:`n`,
          approximating :math:`[\cos(\tau x_j) / 2]_{j=1}^n`

        + **vec_fx_im** – :math:`\vec f_\Im`, `np.ndarray` of dimension :math:`n`,
          approximating :math:`[-\sin(\tau x_j) / 2]_{j=1}^n`

        + **vec_x** – :math:`\vec x`, `np.ndarray` of dimension :math:`n`,
          the components of :math:`\vec x=(x_1,x_2,\cdots,x_n)\in[-1,1]^n` are distinct fixed signal parameters
    """
    int_n = int_deg // 2 + 1  # the length of vec_x
    int_period = 4 * (int_deg + 2)
    float_factor = np.pi / (2 * (int_deg + 2))
    reg_cos_quarter = [np.cos(idx * float_factor) for idx in range(int_deg + 3)]
    reg_cos_half = reg_cos_quarter + [-float_cos for float_cos in reversed(reg_cos_quarter[1:-1])]
    reg_cos = np.array(reg_cos_half + [-float_cos for float_cos in reg_cos_half])
    vec_x = np.array(reg_cos_quarter[1::2])
    vec_BesselJ = np.array([BesselJ(idx, float_tau) * (1j ** (idx % 4)) for idx in range(int_deg + 1)], dtype=complex)
    vec_BesselJ[0] *= 0.5
    vec_fx_re = np.zeros(int_n)
    vec_fx_im = np.zeros(int_n)
    for idx_x in range(int_n):
        float_fx = vec_BesselJ.dot([reg_cos[idx_k * (2 * idx_x + 1) % int_period] for idx_k in range(int_deg + 1)])
        vec_fx_re[idx_x] = float_fx.real
        vec_fx_im[idx_x] = -float_fx.imag
    return vec_fx_re, vec_fx_im, vec_x


def func_LBFGS_QSP_HS(
    float_tau: float, float_epsilon: float = 1e-14, method: str = "interpolation"
) -> Tuple[List[float], List[float]]:
    r"""Compute processing parameters :math:`\vec\phi_\Re` and :math:`\vec\phi_\Im` encoding the QSP function in
    Hamiltonian simulation.

    Three method is supported, and we suggest the method "interpolation" as default.

    :param float_tau: :math:`\tau`, `float`, the simulation time in Hamiltonian simulation
    :param float_epsilon: :math:`\epsilon`, `float`, the simulation precision in Hamiltonian simulation
    :param method: `str`, should be one of "interpolation", "backtracking" and "scipy"
    :return: **(vec_phi_sym_re, vec_phi_sym_im)** – `Tuple[np.ndarray, np.ndarray]`, where:

        + **vec_phi_sym_re** – :math:`\vec\phi_\Re`, `np.ndarray` of dimension :math:`\deg + 1`,
          with :math:`A_{\vec\phi_\Re}(x)\approx\cos(\tau x)/2`

        + **vec_phi_sym_im** – :math:`\vec\phi_\Im`, `np.ndarray` of dimension :math:`\deg + 2`,
          with :math:`A_{\vec\phi_\Im}(x)\approx-\sin(\tau x)/2`
    """
    # we need to implement the evolution function exp(-1j * tau * x) = cos(tau x) - 1j * sin(tau x)
    # by implement cos(tau x) / 2 and -sin(tau x) / 2 individually

    # Polynomial degree is determined by tau & epsilon
    int_deg = int(np.ceil(1.4 * abs(float_tau) - np.log(float_epsilon))) // 2 * 2

    # call __HS_approx_data to obtain vec_fx_re, vec_fx_im, vec_x simultaneously.
    vec_fx_re, vec_fx_im, vec_x = __HS_approx_data(float_tau, int_deg)

    # store list_Wx for reducing unnecessary calculation
    list_Wx = __func_Wx_map(vec_x)

    # cos simulation for bool_parity == 0
    # use func_LBFGS_QSP to find vec_phi_re
    vec_phi_re = __func_LBFGS_QSP(int_deg, list_Wx, vec_fx_re, bool_parity=0, method=method)

    # sin simulation for bool_parity == 1
    # use func_LBFGS_QSP to find vec_phi_im
    vec_phi_im = __func_LBFGS_QSP(int_deg, list_Wx, vec_fx_im, bool_parity=1, method=method)

    # return the symmetrization versions, which is easy to use in QSP
    return list(vec_phi_re[:-1]) + list(reversed(vec_phi_re)), list(vec_phi_im) + list(reversed(vec_phi_im))


if __name__ == "__main__":
    print(func_LBFGS_QSP_HS(float_tau=200))
