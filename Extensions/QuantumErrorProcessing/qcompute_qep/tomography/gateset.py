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
r"""
This script collects classes and functions implementing the Quantum Gate Set [G15]_,
which is the essential building block of Quantum Gate Set Tomography (GST). Mathematically, a gate set consists of

.. math:: \mathcal{G} = \{ \rho, E, G_1, \cdots, G_K\},

where

+ :math:`\rho` is the initial quantum state,
+ :math:`E` is the quantum measurement operator, and
+ :math:`\{G_1,\cdots,G_K\}` is the set of quantum gates we aim to implement.

Also, the gate set must contain two lists of quantum gates specifying how
other quantum states and measurement operators must be constructed.
They are commonly called the fiducial gates and should be chosen from :math:`\{G_1,\cdots,G_K\}`.

References:

.. [G15] Greenbaum, Daniel. "Introduction to quantum gate set tomography." arXiv preprint arXiv:1509.02921 (2015).
"""

import numpy as np
import abc
from typing import Dict, List, Union
from copy import deepcopy
import itertools

import QCompute
from qcompute_qep.utils.circuit import circuit_to_unitary, circuit_to_state
from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.quantum.pauli import ptm_to_operator, operator_to_ptm
from qcompute_qep.quantum.channel import unitary_to_ptm
from qcompute_qep.utils.linalg import basis, vec_to_operator
from QCompute.QPlatform.QOperation import CircuitLine
import qcompute_qep.quantum.metrics as metrics


class GateSet(abc.ABC):
    r"""The Gate Set class.

    It specifies the list of gates we want to estimate.

    .. note::

        Unlike the definition of gate set in [G15]_, in our implementation of gate set,
        we restrict the initial quantum state to :math:`\vert 0\cdots 0\rangle`
        and the initial measurement operator to :math:`\vert 0\cdots 0\rangle\!\langle 0\cdots 0\vert`.
        That is, we do not have freedom to choose arbitrary initial states and measurement operators.
        This convention makes it much easier to implement GST.

    .. note::

        When describing quantum gates in the gate set, we assume that qubits are indexed from :math:`0`.
        That is, the abstract gate set itself does not contain *qubit position information*.
        Such information is specified by a particular GateSetTomography instance.
        In GateSetTomography, we will use the ``map_qubits`` function to map the tomographic
        quantum circuits to the desired target quantum qubits.
        For example, the following is a valid single-qubit GateSet instance:

            >>> gateset = GateSet(gates={'G_rx90': CircuitLine(QCompute.RX(np.pi / 2), [0]),
            >>>                          'G_ry90': CircuitLine(QCompute.RY(np.pi / 2), [0])},
            >>>                   prep_gates=[['G_rx90', 'G_rx90'], ['G_rx90'], ['G_ry90']],
            >>>                   meas_gates=[['G_rx90', 'G_rx90'], ['G_rx90'], ['G_ry90']])

        The quantum gates are operated on qubit :math:`0`, you do not need to give the true qubit position.

    .. note::

        When creating a GateSet instance, you do not indeed to add the ``null`` gate explicitly.
        We will add such a gate by default. The ``null`` gate is specified by the static variable ``NULL_GATE_NAME``.

    .. note::

        When creating a GateSet instance, the first :math:`2^n-1` quantum circuits in ``meas_gates``,
        together with the measurement operator, must form an orthonormal basis of the :math:`n`-qubit quantum system.
        For example, in the qubit case, if the measurement operator is :math:`E=\vert 0\rangle\!\langle 0\vert`,
        then the first quantum circuit :math:`F_1` in ``meas_gates`` must satisfy that
        :math:`F_1\vert 0\rangle = \vert 1\rangle`.
    """
    # Declare the null gate name
    NULL_GATE_NAME = 'G_null'

    def __init__(self, gates: Dict[str, CircuitLine],
                 prep_gates: List[List[str]],
                 meas_gates: List[List[str]],
                 **kwargs):
        r"""The init function of the Gate Set class.

        In our implementation of gate set,
        we restrict the initial quantum state :math:`\rho` to :math:`\vert 0\cdots 0\rangle`
        and the initial measurement operator :math:`E` to :math:`\vert 0\cdots 0\rangle\!\langle 0\cdots 0\vert`.
        That is, we do not have freedom to choose arbitrary initial states and measurement operators.

        Optional keywords list are:

            + `name`: default to ``GateSet``, name the gate set instance

        :param gates: Dict[str, CircuitLine], the list of quantum gates we want to estimate
        :param prep_gates: List[List[str]], the list of preparation circuits :math:`\{P_i\}`
                                            which evolves :math:`\rho` to different states
        :param meas_gates: List[List[str]], the list of measurement circuits :math:`\{M_j\}`
                                            which evolves :math:`E` to different measurement operators

        Usage:

        .. code-block:: python
            :linenos:

            import numpy
            gate_set = GateSet(gates={'G_rx90': CircuitLine(QCompute.RX(numpy.pi / 2), [0]),
                                      'G_ry90': CircuitLine(QCompute.RY(numpy.pi / 2), [0])},
                               prep_gates=[['G_rx90', 'G_rx90'], ['G_rx90'], ['G_ry90']],
                               meas_gates=[['G_rx90', 'G_rx90'], ['G_rx90'], ['G_ry90']],
                               name='GateSet_1Q_RxRy')

        **Examples**

            >>> import numpy
            >>> from qcompute_qep.tomography import GateSet
            >>> from QCompute.QPlatform.QOperation import CircuitLine
            >>> gate_set = GateSet(gates={'G_rx90': CircuitLine(QCompute.RX(numpy.pi / 2), [0]),
            >>>                           'G_ry90': CircuitLine(QCompute.RY(numpy.pi / 2), [0])},
            >>>                    prep_gates=[['G_rx90', 'G_rx90'], ['G_rx90'], ['G_ry90']],
            >>>                    meas_gates=[['G_rx90', 'G_rx90'], ['G_rx90'], ['G_ry90']],
            >>>                    name='GateSet_1Q_RxRy')
        """
        # Get the number of qubits of the gate set
        self._n = max([max(gates[g].qRegList) for g in gates.keys()]) + 1

        # Store the ideal version
        self._gates = {self.NULL_GATE_NAME: None}  # add the 'null' gate in the beginning
        self._gates.update(gates)
        self._prep_gates = [[self.NULL_GATE_NAME]] + prep_gates  # add the 'null' gate in the beginning
        self._meas_gates = [[self.NULL_GATE_NAME]] + meas_gates  # add the 'null' gate in the beginning
        self._name = kwargs.get('name', 'GateSet')
        self._gateset_ptm: Dict[str, np.ndarray] = dict()
        # Verify if the input state preparation and measurement circuits are complete
        self.verify_completeness()
        # Verify if the first :math:`2^n` measurement circuits (including the 'null' gate) form an orthogonal basis
        self.verify_orthogonality_of_meas()

        # Store the optimized version of the gateset
        self._gateset_opt: Dict[str, np.ndarray] = dict()

    @property
    def gate_names(self) -> List[str]:
        r"""Return the full list of gate names, excluding the 'null' gate.
        """
        names = list(self._gates.keys())
        # Remove the 'null' gate if exists
        names.remove(self.NULL_GATE_NAME) if self.NULL_GATE_NAME in names else None
        return names

    @property
    def n(self) -> int:
        r"""Number of qubits in the gate set.
        """
        return self._n

    @property
    def name(self) -> str:
        r"""Name of the gate set.
        """
        return self._name

    @name.setter
    def name(self, val):
        r"""Set the gate set's name.
        """
        self._name = val

    @property
    def gates(self) -> Dict[str, CircuitLine]:
        r"""The list of quantum gates characterized by CircuitLine.
        """
        return self._gates

    @property
    def prep_gates(self):
        r"""The list of state preparation quantum circuits.
        """
        return self._prep_gates

    @property
    def meas_gates(self):
        r"""The list of measurement quantum circuits.
        """
        return self._meas_gates

    @property
    def gateset_ptm(self) -> dict:
        r"""Construct PTM representation of the gate set :math:`\mathcal{G}` from the circuit description.

        .. note::

            We emphasize that the PTM representation of gate set does not contain the 'null' gate.

        :return: dict, a dictionary that records the ideal gate set (in PTM representation)
        """
        if not self._gateset_ptm:
            gateset_ptm: Dict[str, np.ndarray] = dict()
            for key in self.gate_names:  # gate_names does not contain the 'null' gate
                gateset_ptm.update({key: self.get_ideal_matrix(key)})
            gateset_ptm.update({'rho': self.get_ideal_matrix('rho')})
            gateset_ptm.update({'E': self.get_ideal_matrix('E')})
            self._gateset_ptm = gateset_ptm
        return self._gateset_ptm

    @property
    def gateset_opt(self) -> dict:
        r"""Return the optimized gate set.

        :return: dict, a dictionary that records the ideal gate set (in PTM representation)
        """
        if self._gateset_opt is None:
            raise ArgumentError("in GateSet(): the gate set is not optimized yet!")
        return self._gateset_opt

    @gateset_opt.setter
    def gateset_opt(self, val: dict):
        r"""Set the optimized gate set.
        """
        self._gateset_opt = val

    @property
    def is_overcomplete(self) -> bool:
        r"""Check if the gate set is overcomplete.

        A gate set is overcomplete if the following two conditions are satisfied:

        + It is complete, as verified by the `verify_completeness()` function;
        + There are more than :math:`4^n` state preparation and measurement circuits.

        :return: bool, True if the gate set is overcomplete
        """
        if not self.verify_completeness():
            return False
        if len(self.prep_gates) > 4 ** self._n or len(self.meas_gates) > 4 ** self._n:
            return False
        return True

    def trans_matrix_prep(self, full: bool = True) -> np.ndarray:
        r"""Construct the transformation matrix induced by preparation circuits.

        The transformation matrix induced by preparation circuits is defined in Eq. (3.14) of [G15]_,
        which has the form

        .. math:: P = \sum_j P_j\vert\rho\rangle\!\rangle\langle\!\langle j\vert,

        where :math:`\{P_j\}_j` is the set of preparation circuits and :math:`\rho` is the initial state.

        :param full: bool, default to True. If True, we construct the preparation transformation matrix from the
                    full preparation gate set; If False, we construct the preparation transformation matrix
                    from the first :math:`4^n` preparation circuits.
        :return: np.ndarray, the transformation matrix induced by preparation circuits
        """
        if full:
            num_prep = len(self.prep_gates)
        else:
            num_prep = 4 ** self._n

        P = np.zeros((4 ** self._n, num_prep), dtype=float)
        for j in range(num_prep):
            qp = QCompute.QEnv()
            qp.Q.createList(self._n)
            qp.circuit += self.create_prep_circuit(gate_idx=j)
            P[:, j] = np.asarray(operator_to_ptm(circuit_to_state(qp))).reshape((4 ** self._n, )).real

        return P

    def trans_matrix_meas(self, full: bool = True) -> np.ndarray:
        r"""Construct the transformation matrix induced by measurement circuits.

        The transformation matrix induced by measurement circuits is defined in Eq. (3.13) of [G15]_,
        which has the form

        .. math:: M = \sum_i\vert i\rangle\!\rangle\langle\!\langle E\vert M_i,

        where :math:`\{M_i\}_i` is the set of measurement circuits and :math:`E` is the measurement operator.

        :param full: bool, default to True. If True, we construct the measurement transformation matrix from the
                    full measurement gate set; If False, we construct the measurement transformation matrix
                    from the first :math:`4^n` measurement circuits.
        :return: np.ndarray, the transformation matrix induced by measurement circuits
        """
        if full:
            num_meas = len(self.meas_gates)
        else:
            num_meas = 4 ** self._n

        M = np.zeros((num_meas, 4 ** self._n), dtype=float)
        for i in range(num_meas):
            qp = QCompute.QEnv()
            qp.Q.createList(self._n)
            qp.circuit += self.create_meas_circuit(gate_idx=i)
            u = unitary_to_ptm(circuit_to_unitary(qp)).data
            state = self.gateset_ptm['E'] @ u
            M[i, :] = np.asarray(state).reshape((4 ** self._n, )).real

        return M

    def verify_completeness(self) -> bool:
        r"""Check if the input state preparation and measurement circuits are complete.

        Given a list of state preparation (measurement) circuits :math:`\{F_1,\cdots, F_K\}`,
        we say it is complete, if the states :math:`\{F_1\rho F_1^\dagger,\cdots, F_K\rho F_K^\dagger\}`
        span the whole operator space. That is to say,
        For an :math:`n`-qubit quantum system, the list must at least of size :math:`4^n`.

        :return: bool, True if both state preparation and measurement circuits are complete
        """
        num_prep, num_meas = len(self.prep_gates), len(self.meas_gates)
        trans_matrix_prep = self.trans_matrix_prep()
        trans_matrix_meas = self.trans_matrix_meas()

        if num_prep < 4 ** self._n or np.linalg.matrix_rank(trans_matrix_prep) < 4 ** self._n:
            raise ArgumentError("in GateSet(): The given list of state preparation quantum circuits "
                                "does not satisfy the completeness requirement!")

        if np.linalg.matrix_rank(trans_matrix_prep[:, 0:4 ** self._n]) < 4 ** self._n < num_prep:
            raise ArgumentError("in GateSet(): The number of state preparation circuits is larger than `4**n`."
                                "\n It is required that the first `4**n` quantum circuits "
                                "(including the 'null' gate) must satisfy the completeness requirement!")

        if num_meas < 4 ** self._n or np.linalg.matrix_rank(trans_matrix_meas) < 4 ** self._n:
            raise ArgumentError("in GateSet(): The given list of measurement quantum circuits "
                                "does not satisfy the completeness requirement!")

        if np.linalg.matrix_rank(trans_matrix_meas[:, 0:4 ** self._n]) < 4 ** self._n < num_prep:
            raise ArgumentError("in GateSet(): The number of measurement quantum circuits is larger than `4**n`. "
                                "\n It is required that the first `4**n` quantum circuits "
                                "(including the 'null' gate) must satisfy the completeness requirement!")

        return True

    def verify_orthogonality_of_meas(self) -> bool:
        r"""Verify if the states generated by the first :math:`2^n` measurement circuits form an orthogonal basis.

        Note that when creating a GateSet instance, we require that
        the first :math:`2^n-1` quantum circuits in ``meas_gates``,
        together with the measurement operator,
        must form an orthonormal basis of the :math:`n`-qubit quantum system.
        For example, in the qubit case, if the measurement operator is :math:`E=\vert 0\rangle\!\langle 0\vert`,
        then the first quantum circuit :math:`F_1` in ``meas_gates`` must satisfy that
        :math:`F_1\vert 0\rangle = \vert 1\rangle` (global phase is allowed).
        Note that the :math:`0`-th circuit in ``meas_gates`` defaults to the 'null' gate.

        This requirement is important, since this orthogonal basis will be used to estimate
        the measurement fidelity, cf. function ``GateSet.fidelity()`` for more details.

        :return: bool, True if the :math:`2^n` measurement circuits form an orthogonal basis
        """
        orthogonal_basis = []
        for j in range(2 ** self._n):
            qp = QCompute.QEnv()
            qp.Q.createList(self._n)
            qp.circuit += self.create_meas_circuit(gate_idx=j)
            orthogonal_basis.append(circuit_to_state(qp, vector=True).reshape(2 ** self._n, ))

        # Compute the inner products
        indices = range(2 ** self._n)
        mutual_indices = itertools.combinations_with_replacement(indices, 2)
        mutual_indices = [pair for pair in mutual_indices if list(pair)[0] != list(pair)[1]]

        for pair in mutual_indices:
            if not np.isclose(abs(np.dot(orthogonal_basis[pair[0]], orthogonal_basis[pair[1]])), 0.0):
                raise ArgumentError("in GateSet(): the given 'meas_gates' is invalid! "
                                    "we require that the states generated by the first {} gates in 'meas_gates' "
                                    "together with the measurement operator |0><0|, "
                                    "must form the computational basis.".format(2 ** self._n - 1))
        return True

    def get_gate(self, gate_name: str = None, ideal: bool = True) -> Union[np.ndarray, CircuitLine]:
        r"""Get the ideal or noisy quantum gate description given gate name.

        Given a gate name, the corresponding ideal quantum gate is described by a CircuitLine object, and
        the corresponding noisy quantum gate is described by its PTM matrix.

        :param gate_name: str, the name of gate, default to return dict of all gates
        :param ideal: bool, default to True, indicates whether the ideal or noisy quantum gate description
                    should be returned. If ideal==True, return the ideal quantum gate description;
                    Else, return the noisy quantum gate description.
        :return: Union[np.ndarray, CircuitLine], the ideal or noisy target quantum gate description
        """
        if ideal:
            if gate_name in self._gates.keys():
                return self._gates[gate_name]
            else:
                raise ArgumentError()
        else:
            if gate_name in self.gateset_opt.keys():
                return self.gateset_opt[gate_name]
            else:
                raise ArgumentError()

    def create_prep_circuit(self, gate_idx: int) -> List[CircuitLine]:
        r"""Create the state preparation quantum circuit from the preparation gate index.

        :param gate_idx: the index of state preparation quantum circuit :math:`P_j`
        :return: List[CircuitLine], the state preparation quantum circuit
        """
        gate_names = self.prep_gates[gate_idx]
        # Remove the 'null' gate if exists
        gate_names.remove(self.NULL_GATE_NAME) if self.NULL_GATE_NAME in gate_names else None
        circuit = []
        for gate_name in gate_names:
            g_k = deepcopy(self.gates[gate_name])  # a CircuitLine object
            circuit.append(g_k)

        return circuit

    def create_meas_circuit(self, gate_idx: int) -> List[CircuitLine]:
        r"""Create the measurement quantum circuit from the measurement gate index.

        :param gate_idx: the index of measurement quantum circuit :math:`M_i`
        :return: List[CircuitLine], the measurement quantum circuit
        """
        gate_names = self.meas_gates[gate_idx]
        # Remove the 'null' gate if exists
        gate_names.remove(self.NULL_GATE_NAME) if self.NULL_GATE_NAME in gate_names else None
        circuit = []
        for gate_name in gate_names:
            g_k = deepcopy(self.gates[gate_name])  # a CircuitLine object
            circuit.append(g_k)

        return circuit

    def get_ideal_matrix(self, name: str = None) -> np.ndarray:
        r"""Get the PTM representation of a given gate/state/measurement in the gate set.

        :param name: str, the gate name or 'rho' (state) or 'E' (measurement operator)
        :return: np.ndarray, the PTM representation of the target quantity
        """
        if name == 'rho':
            return np.asarray(operator_to_ptm(vec_to_operator(basis(self._n, 0))), dtype=float).reshape(
                (4 ** self._n, 1))
        elif name == 'E':
            return np.asarray(operator_to_ptm(vec_to_operator(basis(self._n, 0))), dtype=float).reshape(
                (1, 4 ** self._n))
        elif name == self.NULL_GATE_NAME:
            # NOTICE! For the simplicity of analysis,
            # we assume that the unitary representation of the 'null' gate is the identity.
            return np.asarray(unitary_to_ptm(np.identity(2 ** self._n)).data, dtype=float)
        elif name in self.gate_names:
            qp = QCompute.QEnv()
            qp.Q.createList(self._n)
            qp.circuit.append(self._gates[name])
            return np.asarray(unitary_to_ptm(circuit_to_unitary(qp)).data).real
        else:
            raise ArgumentError("in get_ideal_matrix(): undefined object name {}!".format(name))

    def fidelity(self, name: str) -> float:
        r"""Calculate the fidelity of state / measurement / gates in the gate set.

        When calculating the quantum fidelities, you can

        + To access the quantum state fidelity, set name to 'rho';
        + To access the quantum measurement fidelity, set name to 'meas';
        + To access a quantum gate fidelity, set name to the gate name.

        .. note ::

            The measurement fidelity of a noisy measurement :math:`\mathcal{M}=\{E_i\}_i`
            with respect to the :math:`Z`-basis measurement is defined as:

            .. math::

                    F(\mathcal{M}) := \frac{1}{2^n} \sum_{i=0}^{2^n - 1}\langle i\vert E_i \vert i \rangle.

        :param name: str, the name of the target quantum instance, can be 'rho', 'meas', or a particular gate name
        :return: float, a dictionary containing the fidelities between noisy and ideal state, measurement,
                and quantum gates
        """
        if self.gateset_opt is None:
            raise ArgumentError("in GateSet.fidelity: the optimized gate set is not assigned yet!")

        if name == 'rho':
            fid = metrics.state_fidelity(ptm_to_operator(self.gateset_opt['rho']),
                                         ptm_to_operator(self.get_ideal_matrix(name='rho')))
        elif name == 'meas':   # Compute the fidelity of the computational basis measurement.
            fid = 0.0
            for i in range(2 ** self._n):
                # Compute the noisy and ideal version of the i-th measurement operator
                M_i_noisy = np.identity(4 ** self._n, dtype=float)  # the i-th noisy measurement circuit
                M_i_ideal = np.identity(4 ** self._n, dtype=float)  # the i-th ideal measurement circuit
                gate_names = self.meas_gates[i]
                # remove the 'null' gate if exists
                gate_names.remove(self.NULL_GATE_NAME) if self.NULL_GATE_NAME in gate_names else None
                for gate_name in gate_names:
                    M_i_noisy = M_i_noisy @ self.gateset_opt[gate_name]
                    M_i_ideal = M_i_ideal @ self.get_ideal_matrix(name=gate_name)

                # Compute the i-th noisy and ideal measurement operator :math:`E_i`
                E_i_noisy = self.gateset_opt['E'] @ M_i_noisy
                E_i_ideal = self.get_ideal_matrix(name='E') @ M_i_ideal
                # Compute the fidelity of the i-th measurement operator
                fid += metrics.state_fidelity(ptm_to_operator(E_i_noisy), ptm_to_operator(E_i_ideal))
            fid = fid / 2 ** self._n  # normalization
        elif name in self.gates.keys():
            fid = metrics.average_gate_fidelity(self.gateset_opt[name], self.get_ideal_matrix(name=name))
        else:
            raise ArgumentError("in GateSet.fidelity(): undefined object name {}!".format(name))

        return fid if fid < 1.0 else 1.0


#######################################################################################################################
# Standard single- and two-qubit gate sets.
#######################################################################################################################
# The following single-qubit standard gate set is from Section 3.4.1 of [G15]_
STD1Q_GATESET_RXRY = GateSet(gates={'G_rx90': CircuitLine(QCompute.RX(np.pi / 2), [0]),
                                    'G_ry90': CircuitLine(QCompute.RY(np.pi / 2), [0])},
                             prep_gates=[['G_rx90', 'G_rx90'], ['G_rx90'], ['G_ry90']],
                             meas_gates=[['G_rx90', 'G_rx90'], ['G_rx90'], ['G_ry90']],
                             name='STD1Q_GATESET_RXRY')

# The following single-qubit standard gate set is from Section 3.4.1 of [G15]_
STD1Q_GATESET_RXRYRX = GateSet(gates={'G_rx90': CircuitLine(QCompute.RX(np.pi / 2), [0]),
                                      'G_ry90': CircuitLine(QCompute.RY(np.pi / 2), [0]),
                                      'G_rx180': CircuitLine(QCompute.RX(np.pi), [0])},
                               prep_gates=[['G_rx180'], ['G_rx90'], ['G_ry90']],
                               meas_gates=[['G_rx180'], ['G_rx90'], ['G_ry90']],
                               name='STD1Q_GATESET_RXRYRX')

# The following single-qubit standard gate set is from Figure 6 of [NGR+21]_
STD1Q_GATESET_RXRYID = GateSet(gates={'G_rx90': CircuitLine(QCompute.RX(np.pi / 2), [0]),
                                      'G_ry90': CircuitLine(QCompute.RY(np.pi / 2), [0]),
                                      'id': CircuitLine(QCompute.ID, [0])},
                               prep_gates=[['G_rx90', 'G_rx90'],
                                           ['G_rx90'], ['G_ry90'],
                                           ['G_rx90', 'G_rx90', 'G_rx90'],
                                           ['G_ry90', 'G_ry90', 'G_ry90']],
                               meas_gates=[['G_rx90', 'G_rx90'],
                                           ['G_rx90'], ['G_ry90'],
                                           ['G_rx90', 'G_rx90', 'G_rx90'],
                                           ['G_ry90', 'G_ry90', 'G_ry90']],
                               name='STD1Q_GATESET_RXRYID')

# The following two-qubit standard gate set is from http://www.pygsti.info/tutorials/13_GST_on_2_qubits.html
STD2Q_GATESET_RXRYCX = GateSet(gates={'G_id_rx90': CircuitLine(data=QCompute.RX(np.pi / 2), qRegList=[0]),
                                      'G_rx90_id': CircuitLine(data=QCompute.RX(np.pi / 2), qRegList=[1]),
                                      'G_id_ry90': CircuitLine(data=QCompute.RY(np.pi / 2), qRegList=[0]),
                                      'G_ry90_id': CircuitLine(data=QCompute.RY(np.pi / 2), qRegList=[1]),
                                      'G_cx': CircuitLine(data=QCompute.CX, qRegList=[0, 1])},
                               prep_gates=[['G_id_rx90', 'G_id_rx90'], ['G_rx90_id', 'G_rx90_id'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_id_rx90'], ['G_id_ry90'], ['G_rx90_id'], ['G_rx90_id', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_id_ry90'], ['G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_ry90_id'], ['G_ry90_id', 'G_id_rx90'], ['G_ry90_id', 'G_id_ry90'],
                                           ['G_ry90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_ry90']],
                               meas_gates=[['G_id_rx90', 'G_id_rx90'], ['G_rx90_id', 'G_rx90_id'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_id_rx90'], ['G_id_ry90'], ['G_rx90_id'], ['G_rx90_id', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_id_ry90'], ['G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_ry90_id'], ['G_ry90_id', 'G_id_rx90'], ['G_ry90_id', 'G_id_ry90'],
                                           ['G_ry90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_ry90']],
                               name='STD2Q_GATESET_RXRYCX')

# two-qubit standard gate set using CZ gate
STD2Q_GATESET_RXRYCZ = GateSet(gates={'G_id_rx90': CircuitLine(data=QCompute.RX(np.pi / 2), qRegList=[0]),
                                      'G_rx90_id': CircuitLine(data=QCompute.RX(np.pi / 2), qRegList=[1]),
                                      'G_id_ry90': CircuitLine(data=QCompute.RY(np.pi / 2), qRegList=[0]),
                                      'G_ry90_id': CircuitLine(data=QCompute.RY(np.pi / 2), qRegList=[1]),
                                      'G_cz': CircuitLine(data=QCompute.CZ, qRegList=[0, 1])},
                               prep_gates=[['G_id_rx90', 'G_id_rx90'], ['G_rx90_id', 'G_rx90_id'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_id_rx90'], ['G_id_ry90'], ['G_rx90_id'], ['G_rx90_id', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_id_ry90'], ['G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_ry90_id'], ['G_ry90_id', 'G_id_rx90'], ['G_ry90_id', 'G_id_ry90'],
                                           ['G_ry90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_ry90']],
                               meas_gates=[['G_id_rx90', 'G_id_rx90'], ['G_rx90_id', 'G_rx90_id'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_id_rx90'], ['G_id_ry90'], ['G_rx90_id'], ['G_rx90_id', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_id_ry90'], ['G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_ry90_id'], ['G_ry90_id', 'G_id_rx90'], ['G_ry90_id', 'G_id_ry90'],
                                           ['G_ry90_id', 'G_id_rx90', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_rx90'],
                                           ['G_rx90_id', 'G_rx90_id', 'G_id_ry90']],
                               name='STD2Q_GATESET_RXRYCZ')

# two-qubit standard gate set using SWAP gate
STD2Q_GATESET_RXRYSWAP = GateSet(gates={'G_id_rx90': CircuitLine(data=QCompute.RX(np.pi / 2), qRegList=[0]),
                                        'G_rx90_id': CircuitLine(data=QCompute.RX(np.pi / 2), qRegList=[1]),
                                        'G_id_ry90': CircuitLine(data=QCompute.RY(np.pi / 2), qRegList=[0]),
                                        'G_ry90_id': CircuitLine(data=QCompute.RY(np.pi / 2), qRegList=[1]),
                                        'G_swap': CircuitLine(data=QCompute.SWAP, qRegList=[0, 1])},
                                 prep_gates=[['G_id_rx90', 'G_id_rx90'], ['G_rx90_id', 'G_rx90_id'],
                                             ['G_rx90_id', 'G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                             ['G_id_rx90'], ['G_id_ry90'], ['G_rx90_id'], ['G_rx90_id', 'G_id_rx90'],
                                             ['G_rx90_id', 'G_id_ry90'], ['G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                             ['G_ry90_id'], ['G_ry90_id', 'G_id_rx90'], ['G_ry90_id', 'G_id_ry90'],
                                             ['G_ry90_id', 'G_id_rx90', 'G_id_rx90'],
                                             ['G_rx90_id', 'G_rx90_id', 'G_id_rx90'],
                                             ['G_rx90_id', 'G_rx90_id', 'G_id_ry90']],
                                 meas_gates=[['G_id_rx90', 'G_id_rx90'], ['G_rx90_id', 'G_rx90_id'],
                                             ['G_rx90_id', 'G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                             ['G_id_rx90'], ['G_id_ry90'], ['G_rx90_id'], ['G_rx90_id', 'G_id_rx90'],
                                             ['G_rx90_id', 'G_id_ry90'], ['G_rx90_id', 'G_id_rx90', 'G_id_rx90'],
                                             ['G_ry90_id'], ['G_ry90_id', 'G_id_rx90'], ['G_ry90_id', 'G_id_ry90'],
                                             ['G_ry90_id', 'G_id_rx90', 'G_id_rx90'],
                                             ['G_rx90_id', 'G_rx90_id', 'G_id_rx90'],
                                             ['G_rx90_id', 'G_rx90_id', 'G_id_ry90']],
                                 name='STD2Q_GATESET_RXRYSWAP')
