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
This script implements various optimization methods employed in Quantum Gate Set Tomography (GST) [G15]_.
Currently, the following optimization methods are supported:

+ Linear inversion method (LinearInversionOptimizer), see Section 3.4 in [G15]_;
+ Maximum likelihood estimation method (MLEOptimizer), see Section 3.5 in [G15]_.

We plan to implement the following method in the next version:

+ Linear approximation method (LinearApproximationOptimizer), see [MGS+13]_.

What's more, we offer the ``GSTOptimizer`` abstract class, any optimization class inherits this abstract
class can be used as an optimizer in the ``GateSetTomography`` class.

References:

.. [MGS+13] Merkel, Seth T., et al. "Self-consistent quantum process tomography." Physical Review A 87.6 (2013): 062119.

.. [G15] Greenbaum, Daniel. "Introduction to quantum gate set tomography." arXiv preprint arXiv:1509.02921 (2015).
"""
import itertools

import numpy as np
import abc
from typing import Dict, List
from copy import deepcopy
import scipy.linalg as la
from scipy.optimize import minimize, Bounds

from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.quantum.pauli import ptm_to_operator, operator_to_ptm, complete_pauli_basis
from qcompute_qep.tomography import GateSet
from qcompute_qep.utils.linalg import cholesky_matrix_to_vec, vec_to_cholesky_matrix, \
    tensor, cholesky_decomposition, complex_matrix_to_vec


class GSTOptimizer(abc.ABC):
    r"""The Gate Set Tomography Optimizer abstract class.
    """

    def __init__(self):
        r"""init function of the `GSTOptimizer` abstract class.
        """
        self._gateset: GateSet = None
        self._gateset_exp: Dict[str, np.ndarray] = None
        self._optimized_result = dict()
        pass

    @abc.abstractmethod
    def optimize(self, gateset: GateSet, gateset_exp: Dict[str, np.ndarray], **kwargs) -> dict:
        r"""The optimize method offered by this abstract class.

        This function accepts the target gate set `gateset` and the experimentally accessible
        gate set `gateset_exp` as input, and estimates the best gate set that matches the experimental data.

        .. note::

            It is required that the 'gateset.gateset_opt' property be set when the optimization is done.
            The 'gateset.gateset_opt' is a dictionary and records the optimized gate set.

        :return: dict, a dictionary that records the optimization results
        """
        pass


class LinearInversionOptimizer(GSTOptimizer):
    r"""The linear inversion optimizer used in Gate Set Tomography.

    The linear inversion optimizer is a simple, easy to understand, and closed-form algorithm
    for obtaining self-consistent gate estimates. Basically, it estimates the best gate set
    that matches the experimental data by optimizing the gauge operator.
    For more details on the linear inversion method, see Section 3.4 in [G15]_.
    """

    def _construct_gateset_gram(self) -> dict:
        r"""Construct the Gram matrix transformed gate set :math:`\hat{\mathcal{G}}`.

        In the notation of [G15], the Gram matrix transformed gate set :math:`\hat{\mathcal{G}}` is
        defined in Eqs. (3.21)-(3.23) and has the following form

        .. math::

            \hat{\mathcal{G}} = \{ |\hat{\rho}\rangle\!\rangle, \langle\!\langle\hat{E}\vert,
                                    \hat{G}_1, \cdots, \hat{G}_K\}.

        where the quantities are obtained from the experimentally accessible gate set :math:`\tilde{\mathcal{G}}`
        using the Gram matrix :math:`g` as follows:

        .. math::

            \begin{aligned}
                \begin{align*}
                    \hat{G}_k &= g^{-1}\tilde{G}_k \\
                    \vert\hat{\rho}\rangle\!\rangle &= g^{-1}|\tilde{\rho}\rangle\!\rangle \\
                    \langle\!\langle\hat{E}\vert &= \langle\!\langle\tilde{E}\vert
                \end{align*}
            \end{aligned}

        .. note::

            We emphasize that the Gram matrix transformed gate set does not contain the 'null' gate.

        :return: dict, a dictionary that records the Gram matrix transformed gate set
        """
        # Load the experimentally accessible gate set and the Gram matrix
        gateset_exp = deepcopy(self._gateset_exp)
        g = gateset_exp[GateSet.NULL_GATE_NAME]

        gateset_gram: Dict[str, np.ndarray] = dict()
        # Construct the Gram matrix transformed quantum state and measurement
        gateset_gram.update({'rho': la.pinv(g) @ gateset_exp['rho']})
        gateset_gram.update({'E': gateset_exp['E']})
        gateset_exp.pop('rho', None)
        gateset_exp.pop('E', None)
        # delete the 'null' gate so that the Gram matrix transformed gate set does not contain it
        gateset_exp.pop(GateSet.NULL_GATE_NAME, None)

        # Construct the Gram matrix transformed quantum gates, which does not include the 'null' gate
        for key, val in gateset_exp.items():
            gateset_gram.update({key: la.pinv(g) @ val})

        # Record the Gram matrix transformed gate set
        self._optimized_result.update({'gateset gram': gateset_gram})

        return gateset_gram

    def _optimize_gauge_matrix(self) -> np.ndarray:
        r"""Optimize the gauge matrix :math:`P`.

        This optimization function implements Eq. (3.25) in [G15].

        :return: np.ndarray, the optimized gauge matrix P_opt
        """
        # The Gram matrix transformed gate set
        gateset_gram = deepcopy(self._optimized_result['gateset gram'])
        # Record the ideal gate set described by the PTM representation of the elements, excluding the 'null' gate
        gateset_ideal = deepcopy(self._gateset.gateset_ptm)
        # Construct the optimization arguments
        # Add the outer product of quantum state and measurement as arguments, cf. Eq. (3.25) in [G15]
        args_gram = [gateset_gram['rho'] @ gateset_gram['E']]
        args_ideal = [gateset_ideal['rho'] @ gateset_ideal['E']]
        gateset_gram.pop('rho', None)
        gateset_gram.pop('E', None)

        # Add Gram matrix transformed gates and ideal gates as arguments
        for key in gateset_gram.keys():
            args_gram.append(gateset_gram[key])
            args_ideal.append(gateset_ideal[key])

        # Optimize the unobservable matrix P
        def _optimize_func(x, *args) -> float:
            r"""Calculate the RMS error, cf. Eq. (3.25) in [G15]

            :param x: the variable (the unobservable matrix P) which need to be optimized
            :param args: Gram matrix transformed gate set, ideal gate set, and the number of qubits
            :return: the RMS error
            """
            # Reshape the input to :math:`4^n\times 4^n`
            x = np.reshape(x, (4 ** args[2], -1))

            return sum([la.norm(G_gram - np.asarray(la.pinv(x)) @ G @ x, 'fro')
                        for G_gram, G in zip(args[0], args[1])])

        # Number of qubits
        n = self._gateset.n
        # Obtain the minimization result
        res = minimize(fun=_optimize_func,
                       x0=np.reshape(self._gateset.trans_matrix_prep(full=False), (4 ** n * 4 ** n,)),
                       args=(args_gram, args_ideal, n),
                       bounds=Bounds(np.full(4 ** n * 4 ** n, - 1.0), np.full(4 ** n * 4 ** n, 1.0)),
                       tol=1e-3,  # tolerance for termination
                       method='Powell')

        # Get the optimized gauge operator P_opt
        if res.success is True:
            P_opt = np.reshape(res.x, (4 ** n, -1))
            self._optimized_result.update({'P_opt': P_opt})
        else:
            raise ArgumentError(res.message)

        return P_opt

    def _preprocess_gateset_exp(self, gateset_exp: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        r"""Preprocess experimentally accessible gate set to make all data of size :math:`4^n\times 4^n`.

        If the number of state preparation and measurement circuits are larger than :math:`4^n`,
        then the experimentally accessible gate set cannot be used directly by the linear inversion method,
        since the gauge operator might not be inverted. In this case, we truncate the data
        to construct a *new* gate set such that

        + the experimentally accessible :math:`\vert\rho\rangle\!\rangle` has size :math:`4^n\times 1`,

        + the experimentally accessible :math:`\langle\!\langle E\vert` has size :math:`1\times 4^n`, and

        + each experimentally accessible :math:`\tilde{G}_k` has size :math:`4^n\times 4^n`.

        :param gateset_exp: Dict[str, np.ndarray], the experimentally accessible gate set to be processed
        :return: Dict[str, np.ndarray], a new gate set whose data are of size :math:`4^n\times 4^n`
        """
        new_gateset_exp = dict()
        # Truncate the experimentally accessible :math:`\vert\rho\rangle\!\rangle`
        rho = gateset_exp['rho']
        new_gateset_exp.update({'rho': rho[0:4 ** self._gateset.n, :]})
        gateset_exp.pop('rho', None)
        # Truncate the experimentally accessible :math:`\langle\!\langle E\vert`
        E = gateset_exp['E']
        new_gateset_exp.update({'E': E[:, 0:4 ** self._gateset.n]})
        gateset_exp.pop('E', None)
        # Truncate the experimentally accessible :math:`\tilde{G}_k`
        for key, val in gateset_exp.items():
            new_gateset_exp.update({key: val[0:4 ** self._gateset.n, 0:4 ** self._gateset.n]})

        return new_gateset_exp

    def optimize(self, gateset: GateSet, gateset_exp: Dict[str, np.ndarray], **kwargs) -> dict:
        r"""The linear inverse optimization function.

        .. admonition:: Important!

            In case that the experimental data is overcomplete, we only use data generated by first :math:`4^n`
            preparation and measurement quantum circuits to accomplish the optimization procedure.

        .. note::

            The return value is a dictionary that records the optimized results. It includes the following keys:

            + `gateset gram`, whose value is a dictionary that records the Gram matrix transformed gate set;

            + `gateset opt`, whose value is a dictionary that records the optimized gate set;

            + `g`, whose value is the Gram matrix;

            + `P_opt`, whose value is the optimized gauge matrix.

            Note that neither `gateset gram` nor `gateset opt` contain the 'null' gate information.

        :param gateset: GateSet, the target gate set whose gates are characterized by `CircuitLine` objects
        :param gateset_exp: Dict[str, np.ndarray], the experimentally accessible gate set
        :return: dict, a dictionary that records the optimized results
        """
        if gateset is None or gateset_exp is None:
            raise ArgumentError("in LinearInversionOptimizer.optimize(): either the ideal gate set or "
                                "the experimentally accessible gate set is 'None'!")
        self._gateset = gateset

        # Preprocess the experimentally accessible gate set data.
        # If there are more than `4^n` number of state preparation circuits and/or measurement circuits,
        # we use experimental data collected from the first `4**n` quantum circuits only.
        num_prep, num_meas = len(self._gateset.prep_gates), len(self._gateset.meas_gates)
        if num_prep != 4 ** self._gateset.n or num_meas != 4 ** self._gateset.n:
            # Print warning message using ANSI code
            print("\033[1;31m WARNING: You are using the linear inversion optimizer @LinearInversionOptimizer, but "
                  "\n both the number of state preparation circuits and the number of measurement circuits "
                  "\n are larger than `4**n` (including the 'null' circuit), where `n` is the number of qubits. "
                  "\n In this case, we use experimental data collected from the first `4**n` quantum circuits only!")
            self._gateset_exp = self._preprocess_gateset_exp(gateset_exp)
        else:
            self._gateset_exp = gateset_exp

        # Construct the Gram matrix
        self._construct_gateset_gram()
        # Compute the optimized gauge matrix
        P_opt = self._optimize_gauge_matrix()
        P_opt_inv = la.pinv(P_opt)
        # Compute the optimized quantum state, measurement, and gates
        gateset_gram = deepcopy(self._optimized_result['gateset gram'])
        gateset_opt: Dict[str, np.ndarray] = dict()

        # Optimized quantum state and measurement operator
        gateset_opt.update({'rho': P_opt @ gateset_gram['rho']})
        gateset_opt.update({'E': gateset_gram['E'] @ P_opt_inv})
        gateset_gram.pop('rho', None)
        gateset_gram.pop('E', None)

        # Optimized quantum gates
        for key, val in gateset_gram.items():
            gateset_opt.update({key: P_opt @ val @ P_opt_inv})

        # Record the optimized gate set
        self._optimized_result.update({'gateset opt': gateset_opt})
        self._gateset.gateset_opt = gateset_opt

        return self._optimized_result


class MLEOptimizer(GSTOptimizer):
    r"""The maximum likelihood estimation optimizer used in Gate Set Tomography.

    The maximum likelihood estimation (MLE) optimizer produces estimated gate set that are physical
    and is capable of working with overcomplete data.
    Basically, it estimates the best gate set that matches the experimental data by maximizing the likelihood.
    To deal with statistical error, we assume the probability distribution obeys the normal Gaussian distribution.
    For more details on our implemented MLE optimizer, see Section 3.5 in [G15]_.

    .. note::

        Here we explain how we parameterize the quantum gate set. The following quantities must be parameterized:

        .. math:: \mathcal{G} = \{\rho, E, G_1, \cdots, G_K\}.

        In the optimization, :math:`\rho` and :math:`E` are :math:`2^n\times 2^n` positive semidefinite matrices,
        and each gate :math:`G_k`, when represented by its Choi operator,
        is also a positive semidefinite matrix of dimension :math:`4^n\times 4^n`,
        where :math:`n` is the number of qubits. Now the question is to parameterize positive semidefinite matrices.

        The positive semidefiniteness constraints are imposed by invoking the Cholesky decomposition.
        The Cholesky decomposition states that a :math:`2^n\times 2^n` positive semidefinite matrix :math:`G`
        can be decomposed into the product of a lower triangular matrix :math:`T` with real diagonal elements
        and its complex conjugate transpose:

        .. math:: G = TT^\dagger.

        As so, to parameterize a positive semidefinite matrix :math:`G`, it is equivalent to parameterize its
        corresponding Cholesky matrix :math:`T`. Since :math:`T` is lower triangular with real diagonal elements,
        it has :math:`4^n` free real parameters :math:`\vec{x}` and can be expressed as follows:

        .. math::

                T(\vec{x}) = \begin{bmatrix}
                                x_0         & 0     & \cdots & 0 \\
                                x_1 + ix_2 & x_3 & \cdots & 0 \\
                                \dots & \vdots & \ddots & \vdots \\
                                x_{4^n-2^{n+1}+1} + ix_{4^n-2^{n+1}+2} & x_{4^n-2^{n+1}+3}
                                + ix_{4^n-2^{n+1}+4} & \cdots & x_{4^n-1}
                            \end{bmatrix}.

        To summarize, we need a total number of

        .. math:: (4^n) + (4^n) + (16^n \times K)

        free real parameters to parameterize the gateset :math:`\mathcal{G}`.
    """

    def __init__(self):
        r"""init function of the `GSTOptimizer` abstract class.
        """
        super().__init__()
        self._gateset: GateSet = None
        self._gateset_exp: Dict[str, np.ndarray] = None
        self._shots = None
        self._optimized_result = dict()
        pass

    def _compute_expval(self, rho: np.ndarray, E: np.ndarray, gates: dict,
                        g_k: str, prep_idx: int, meas_idx: int) -> float:
        r"""Compute the expectation value from the parametric gate set.

        The expectation value is computed from the parametric gate set via the equation

        .. math:: p_{ikj} = \langle\!\langle E \vert M_i G_k P_j \vert\rho\rangle\!\rangle

        where

        + :math:`\vert\rho\rangle\!\rangle` is the parametric input quantum state,

        + :math:`G_k` is the parametric target quantum gate,

        + :math:`\langle\!\langle E \vert` is the parametric measurement operator,

        + :math:`P_j` are preparation circuits whose gates are from the gate set :math:`\{G_k\}`,

        +  :math:`M_i` are measurement circuits whose gates are from the gate set :math:`\{G_k\}`.

        :param rho: np.ndarray, the parametric quantum state :math:`\rho` in PTM form
        :param E: np.ndarray, the parametric measurement operator :math:`E` in PTM form
        :param gates: Dict[str, np.ndarray], the parametric quantum gates, each in PTM form
        :param g_k: str, the name of the target gate whose theoretical value we aim to compute
        :param prep_idx: the index of state preparation quantum circuit :math:`P_j`
        :param meas_idx: the index of measurement quantum circuit :math:`M_i`
        :return: float, the theoretical expectation value
        """
        # Multiply by preparation gates
        if prep_idx is not None:
            gate_names = self._gateset.prep_gates[prep_idx]
            # remove the 'null' gate if exists
            gate_names.remove(GateSet.NULL_GATE_NAME) if GateSet.NULL_GATE_NAME in gate_names else None
            for gate_name in gate_names:
                rho = gates[gate_name] @ rho
        # Multiply by target quantum gate
        if g_k is not None and g_k != GateSet.NULL_GATE_NAME:
            rho = gates[g_k] @ rho
        # Multiply by measurement gates
        if meas_idx is not None:
            gate_names = self._gateset.meas_gates[meas_idx]
            # remove the 'null' gate if exists
            gate_names.remove(GateSet.NULL_GATE_NAME) if GateSet.NULL_GATE_NAME in gate_names else None
            for gate_name in gate_names:
                rho = gates[gate_name] @ rho
        # Compute the theoretical value from the parametric measurement operator E
        expval = np.real(np.trace(E @ rho))

        return expval

    def _x_to_rho(self, x) -> np.ndarray:
        r"""Construct the parametric quantum state (in PTM form) from new variables x.

        In our parameterization, the positive semidefinite quantum state :math:`\rho` is parameterized
        by :math:`4^n` free real parameters.

        :param x: np.array, a 1-D array with shape (n,) recording the new variables
        :return: np.ndarray, PTM representation of the quantum state
        """
        num_of_vars_of_rho = 4 ** self._gateset.n
        begin_idx = 0
        # Construct the corresponding Cholesky matrix
        rho_T = vec_to_cholesky_matrix(x[begin_idx:begin_idx + num_of_vars_of_rho])
        # rho in matrix form
        rho = rho_T @ rho_T.T.conj()
        return np.reshape(operator_to_ptm(rho), (4 ** self._gateset.n, 1))

    def _x_to_E(self, x) -> np.ndarray:
        r"""Construct the parametric measurement operator (in PTM form) from new variables x.

        In our parameterization, the positive semidefinite measurement operator :math:`E` is parameterized
        by :math:`4^n` free real parameters.

        :param x: np.array, a 1-D array with shape (n,) recording the new variables
        :return: np.ndarray, PTM representation of the measurement operator
        """
        num_of_vars_of_E = 4 ** self._gateset.n
        begin_idx = 4 ** self._gateset.n
        # Construct the corresponding Cholesky matrix
        E_T = vec_to_cholesky_matrix(x[begin_idx:begin_idx + num_of_vars_of_E])
        # E in matrix form
        E = E_T @ E_T.T.conj()
        return np.reshape(operator_to_ptm(E), (1, 4 ** self._gateset.n))

    def _x_to_gates(self, x) -> Dict[str, np.ndarray]:
        r"""Construct the parametric gates (in PTM form) from new variables x.

        In our parameterization, each quantum gate :math:`G_k` is parameterized by :math:`8^n` free real parameters,
        representing the Cholesky matrix of its Choi matrix.

        :param x: np.array, a 1-D array with shape (n,) recording the new variables
        :return: List[np.ndarray], a list of PTM matrices characterizing the parametric gates
        """
        num_of_vars_per_gate = 16 ** self._gateset.n
        gates = dict()
        for k, g_k in enumerate(self._gateset.gate_names):
            begin_idx = 2 * 4 ** self._gateset.n + num_of_vars_per_gate * k
            # Construct the corresponding Cholesky matrix
            G_T = vec_to_cholesky_matrix(x[begin_idx:begin_idx + num_of_vars_per_gate])
            gates.update({g_k: choi_to_ptm(G_T @ G_T.T.conj())})
        return gates

    def _likelihood_func(self, x: np.array) -> float:
        r"""Calculate the log-likelihood value from parametric quantum circuits.

        This likelihood function is defined in Eq. (3.33) in [G15]_.

        :param x: np.array, the variable (the unobservable matrix P) which need to be optimized
        :return: float, the log-likelihood value
        """
        # Update the parametric gate set from the variables
        gateset_exp = deepcopy(self._gateset_exp)
        num_prep, num_meas = len(self._gateset.prep_gates), len(self._gateset.meas_gates)
        # Read the parametric quantum gate set in PTM form
        rho = self._x_to_rho(x)
        E = self._x_to_E(x)
        gates = self._x_to_gates(x)

        # Compute likelihood value from the new parametric gate set
        likelihood = 0.0

        # Step 1. Compute likelihood value for the experimentally accessible data rho_tilde
        for i in range(num_meas):
            p_i = self._compute_expval(rho, E, gates, g_k=None, prep_idx=None, meas_idx=i)
            m_i = gateset_exp['rho'][i, 0]
            # Update the likelihood value
            sigma_i = m_i * (1 - m_i) / self._shots
            likelihood = likelihood + (m_i - p_i) ** 2 / (sigma_i ** 2)

        # Step 2. Compute likelihood value for the experimentally accessible data E_tilde
        for j in range(num_prep):
            p_j = self._compute_expval(rho, E, gates, g_k=None, prep_idx=j, meas_idx=None)
            m_j = gateset_exp['E'][0, j]
            # Update the likelihood value
            sigma_j = m_j * (1 - m_j) / self._shots
            likelihood = likelihood + (m_j - p_j) ** 2 / (sigma_j ** 2)

        # Delete experimentally accessible data rho_tilde and E_tilde
        gateset_exp.pop('rho', None)
        gateset_exp.pop('E', None)

        # Step 3. Compute likelihood value for the experimentally accessible data G_k_tilde
        # Notice that here @i is the index for measurement and @j is the index for state
        for g_k in gateset_exp.keys():
            for i, j in itertools.product(range(num_meas), range(num_prep)):
                p_ij = self._compute_expval(rho, E, gates, g_k=g_k, prep_idx=j, meas_idx=i)
                m_ij = gateset_exp[g_k][i, j]
                # Update the likelihood value
                sigma_ij = m_ij * (1 - m_ij) / self._shots
                likelihood = likelihood + (m_ij - p_ij) ** 2 / (sigma_ij ** 2)

        return likelihood

    # Add complete positivity constraint on quantum gates
    def _gates_cp_constraint(self, x: np.array) -> List[float]:
        gates = self._x_to_gates(x)
        cons_ineq = []
        for g_k in gates.values():
            d = g_k.shape[0]
            g_k_vec = complex_matrix_to_vec(g_k)
            for i in np.arange(start=d, stop=d**2, step=1, dtype=int):
                cons_ineq.append(g_k_vec[i] + 1)
                cons_ineq.append(- g_k_vec[i] + 1)
        return cons_ineq

    # Add trace-preserving constraint on quantum gates
    def _gates_tp_constraint(self, x: np.array) -> List[float]:
        gates = self._x_to_gates(x)
        cons_eq = []
        for g_k in gates.values():
            g_k_real = g_k.real
            g_k_imag = g_k.imag
            cons_eq.append(g_k_real[0, 0] - 1)  # real part of G_{0,0} is 1
            for j in np.arange(1, g_k_real.shape[0]):
                cons_eq.append(g_k_real[0, j])  # The real part of G_{0,j} is 0
            for i in range(g_k.shape[0]):
                for j in range(g_k.shape[1]):
                    cons_eq.append(g_k_imag[i, j])  # the imaginary part of G_{i, j} must be 0
        return cons_eq

    # Add trace unit constraint on the quantum state rho
    def _rho_trace_constraint(self, x: np.array):
        rho = self._x_to_rho(x)
        rho = ptm_to_operator(rho)
        val = sum([rho[i, i] for i in range(rho.shape[0])])
        return [val.real - 1, val.imag]

    # Add trace positivity constraint on the quantum state rho
    def _rho_positivity_constraint(self, x: np.array):
        rho = self._x_to_rho(x)  # in PTM form
        # Since rho is required to be positive semidefinite, each PTM element must be in the range [-1, 1]
        cons_ineq = []
        for i in range(rho.size):
            rho_real = rho.real
            cons_ineq.append(rho_real[i, 0] + 1)
            cons_ineq.append(- rho_real[i, 0] + 1)
        return cons_ineq

    def _constraints(self) -> List[Dict]:
        """Generates the constraints for the MLE optimization.

        The constraints are listed in the optimization problem in Page 30 of [G15]_ and
        are briefly summarized as follows:

        + Complete positivity constraint on quantum gates (revealed by the Choi representation),

        + Trace preservation constraint on quantum gates (revealed by the PTM representation),

        + Trace unity constraint on the quantum state (revealed by its matrix representation).

        The positivity constraint on the quantum state and measurement operator is guaranteed
        by the Cholesky decomposition.
        We do not impose the constraint on the measurement operator that :math:`E \leq I`,
        where :math:`I` is the identity matrix.

        :return: List[Dict], a list of constraints, each constraint is a dictionary type
        """
        cons = [{'type': 'ineq', 'fun': self._gates_cp_constraint}]

        # {'type': 'ineq', 'fun': self._gates_cp_constraint}
        # {'type': 'eq', 'fun': self._gates_tp_constraint}
        # {'type': 'ineq', 'fun': self._rho_positivity_constraint}
        # {'type': 'eq', 'fun': self._rho_trace_constraint}

        return cons

    def _initialize_x(self) -> np.array:
        r"""Sets initial values for MLE optimization.

        .. admonition:: Important!

            We use the Linear Inversion Optimization generated optimized gate set as the starting point
            for MLE optimization. In case that the experimental data is overcomplete, we only use data
            generated by first :math:`4^n` preparation and measurement quantum circuits.

        :return: np.array, a 1-D array with shape (n,), recording the initial variables
        """
        # Perform Linear Inversion Optimization and use the optimized gate set as the starting point
        opt_res = LinearInversionOptimizer().optimize(gateset=self._gateset, gateset_exp=self._gateset_exp)
        gateset_init = opt_res['gateset opt']

        x_init = np.array([], dtype=float)
        # Add the initial parameters for rho
        rho_T = cholesky_decomposition(ptm_to_operator(gateset_init['rho']))
        x_init = np.concatenate((x_init, cholesky_matrix_to_vec(rho_T)), axis=0)
        # Add the initial parameters for E
        E_T = cholesky_decomposition(ptm_to_operator(gateset_init['E']))
        x_init = np.concatenate((x_init, cholesky_matrix_to_vec(E_T)), axis=0)

        # Delete experimentally accessible data rho_tilde and E_tilde
        gateset_init.pop('rho', None)
        gateset_init.pop('E', None)
        # Add the initial parameters for G_k. Remove the 'null' gate if it exists
        gateset_init.pop(GateSet.NULL_GATE_NAME, None)
        for g_k in gateset_init.keys():
            G_Choi = ptm_to_choi(gateset_init[g_k])
            G_T = cholesky_decomposition(G_Choi)
            x_init = np.concatenate((x_init, cholesky_matrix_to_vec(G_T)), axis=0)

        return x_init.flatten()

    def optimize(self, gateset: GateSet, gateset_exp: Dict[str, np.ndarray], shots: int = None, **kwargs) -> dict:
        r"""The maximum likelihood optimization function.

        .. note::

            The return value is a dictionary that records the optimized results. It includes the following keys:

            + `gateset opt`, whose value is a dictionary that records the optimized gate set.

            Note that `gateset opt` does not contain the 'null' gate information.

        :param gateset: GateSet, the target gate set whose gates are characterized by `CircuitLine` objects
        :param gateset_exp: Dict[str, np.ndarray], the experimentally accessible gate set
        :param shots: int, the number of shots of each measurement when collecting the experimental data.
                        This parameter is required and is used in constructing the object function
        :return: dict, a dictionary that records the optimized results
        """
        # Step 1. Set the parameters
        self._gateset = gateset
        self._gateset_exp = deepcopy(gateset_exp)
        if self._gateset is None or self._gateset_exp is None:
            raise ArgumentError("in MLEOptimizer.optimize(): either the ideal gate set or "
                                "the experimentally accessible gate set is 'None'!")
        self._shots = shots
        if self._shots is None:
            raise ArgumentError("in MLEOptimizer.optimize(): to use the MLE optimizer, parameter 'shots' must be set!")

        np.set_printoptions(precision=3, threshold=5, edgeitems=4, suppress=True)
        # Step 2. Call the MLE optimization
        res = minimize(fun=self._likelihood_func,
                       x0=self._initialize_x(),
                       method='SLSQP',
                       constraints=self._constraints(),
                       options={'disp': True})

        # Get the optimized variable x_opt
        if res.success is True:
            x_opt = res.x
        else:
            print(res)
            raise ArgumentError(res.message)

        # Step 3. obtain the optimized gate set from the optimal variables
        gateset_opt: Dict[str, np.ndarray] = dict()
        # Obtain optimized quantum state and measurement operator (in PTM form)
        gateset_opt.update({'rho': self._x_to_rho(x_opt)})
        gateset_opt.update({'E': self._x_to_E(x_opt)})

        # Obtain optimized quantum gates (in PTM form) by concatenating dictionary
        gateset_opt.update(self._x_to_gates(x_opt))

        # Record the optimized gate set
        self._optimized_result.update({'gateset opt': gateset_opt})
        self._gateset.gateset_opt = gateset_opt

        return self._optimized_result


class LinearApproximationOptimizer(GSTOptimizer):
    r"""The linear approximation optimizer used in Gate Set Tomography.

    This optimizer is originally introduced in [MGS+13]_.
    """

    def optimize(self, gateset: GateSet, gateset_exp: Dict[str, np.ndarray], **kwargs) -> dict:
        raise NotImplementedError


def choi_to_ptm(choi: np.ndarray) -> np.ndarray:
    r"""Convert a quantum ptm in Choi representation to PTM representation.

    We assume the Choi representation :math:`J_{\mathcal{N}}` of a quantum map :math:`\mathcal{N}` is
    unnormalized (i.e., use the unnormalized maximally entangled state as input)
    and the Pauli matrices are normalized in the PTM representation.
    Under this assumption, the Choi representation can be converted to the PTM representation via

    .. math::

        [\mathcal{N}]_{ij} := \textrm{Tr}[J_{\mathcal{N}}(P_i^T\otimes P_i)].

    :param choi: np.ndarray, the Choi matrix of the quantum channel
    :return: np.ndarray, the PTM matrix of the quantum channel
    """
    # The dimension of the input and output quantum system
    n = int(np.log2(np.sqrt(choi.shape[0])))
    cpb = complete_pauli_basis(n)
    ptm = np.zeros((len(cpb), len(cpb)), dtype=complex)
    for i, P_i in enumerate(cpb):
        for j, P_j in enumerate(cpb):
            ptm[i, j] = np.trace(choi @ tensor(P_j.matrix.T, P_i.matrix))

    return ptm


def ptm_to_choi(ptm: np.ndarray) -> np.ndarray:
    r"""Convert a quantum channel in PTM representation to Choi representation.

    We assume the Choi representation :math:`J_{\mathcal{N}}` of a quantum map :math:`\mathcal{N}` is
    unnormalized (i.e., use the unnormalized maximally entangled state as input)
    and the Pauli matrices are normalized in the PTM representation.
    Under this assumption, the PTM representation can be converted to the Choi representation via

    .. math::

        J_{\mathcal{N}} := \sum_{i,j} [\mathcal{N}]_{ij} P_j^T\otimes P_i.

    :param ptm: np.ndarray, the PTM matrix of the quantum channel
    :return: np.ndarray, the Choi matrix of the quantum channel
    """
    # The dimension of the input and output quantum system
    n = int(np.log2(np.sqrt(ptm.shape[0])))
    cpb = complete_pauli_basis(n)
    choi = np.zeros((len(cpb), len(cpb)), dtype=complex)
    for i, P_i in enumerate(cpb):
        for j, P_j in enumerate(cpb):
            choi += ptm[i, j] * tensor(P_j.matrix.T, P_i.matrix)

    return choi