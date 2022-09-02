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
This file aims to collect functions related to the Quantum Gate Set Tomography.
"""
import scipy.linalg as la
import numpy as np
import abc
from typing import Dict, List, Union, Tuple
from copy import deepcopy
from scipy.sparse import bsr_matrix
from scipy.optimize import minimize, Bounds
from tqdm import tqdm

from QCompute import *
from qcompute_qep.utils.circuit import execute, circuit_to_unitary, circuit_to_state
from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.tomography import Tomography
from qcompute_qep.utils.types import QComputer, QProgram, number_of_qubits
from qcompute_qep.quantum.pauli import ptm_to_operator, operator_to_ptm, unitary_to_ptm
from qcompute_qep.utils.linalg import basis, vec_to_operator, dagger, expand
from QCompute.QPlatform.QOperation import CircuitLine
import qcompute_qep.quantum.metrics as metrics

import warnings
warnings.filterwarnings("ignore")


class GateSet(abc.ABC):
    r"""The Gate Set class.

    It contains unknown quantum initial states, measurement POVM, and a list of gates we are interested in.
    """
    def __init__(self, prep_circuit: List[CircuitLine], meas_circuit: List[CircuitLine], gates: Dict[str, CircuitLine],
                 prep_gates: List[List[str]], meas_gates: List[List[str]], meas_basis: int = 0):
        r"""The init function of the Gate Set class.

        :param prep_circuit: List[CircuitLine], the circuit prepare unknown quantum state :math:`\rho`
        :param meas_circuit: List[CircuitLine], the measurement circuit, corresponding to 2-outcome POVM :math:`E`
        :param gates: Dict[str, CircuitLine], the gates we are interested in
        :param prep_gates: List[List[str]], used to generate state preparation circuit
        :param meas_gates: List[List[str]], used to generate measurement circuit
        :param meas_basis: int, chose the basis of measurement device, default to :math:`0`

        Usage:

        .. code-block:: python
            :linenos:

            gate_set = GateSet(prep_circuit, meas_circuit,
                               gates={'rx_90': rx_90, 'ry_90': ry_90},
                               prep_gates=[['rx_90'], ['ry_90'], ['rx_90', 'rx_90']],
                               meas_gates=[['rx_90'], ['ry_90'], ['rx_90', 'rx_90']])

        **Examples**

            >>> import QCompute
            >>> import numpy as np
            >>> from qcompute_qep.tomography import GateSet
            >>> from QCompute.QPlatform.QOperation import CircuitLine
            >>> prep_circuit = []
            >>> meas_circuit = []
            >>> qc = QCompute.BackendName.LocalBaiduSim2
            >>> rx_90 = CircuitLine(QCompute.RX(np.pi / 2), [0])
            >>> ry_90 = CircuitLine(QCompute.RY(np.pi / 2), [0])
            >>> rx_180 = CircuitLine(QCompute.RX(np.pi), [0])
            >>> prep_gates = [['rx_90'], ['ry_90'], ['rx_90', 'rx_90']]
            >>> meas_gates = [['rx_90'], ['ry_90'], ['rx_90', 'rx_90']]
            >>> gate_set = GateSet(prep_circuit, meas_circuit,
            >>>                    gates={'rx_90': rx_90, 'ry_90': ry_90},
            >>>                    prep_gates=prep_gates, meas_gates=meas_gates, meas_basis=1)

        """
        self._prep_circuit = prep_circuit
        self._meas_circuit = meas_circuit
        # Get the qubit number of gateset
        self._n = max([0] +
                      [max(n.qRegList) for n in prep_circuit] +
                      [max(n.qRegList) for n in meas_circuit] +
                      [max(gates[g].qRegList) for g in gates.keys()]) + 1
        # TODO: change the value of 'null'
        self._meas_basis = meas_basis
        if self._meas_basis >= 2**self._n:
            raise ArgumentError("in GateSet Class: the meas_basis is illegal!")
        self._gates = {'null': None}
        self._gates.update(gates)
        self._prep_gates = [['null']] + prep_gates
        self._meas_gates = [['null']] + meas_gates
        self._noisy_gates = {k: None for k, v in gates.items()}
        self._noisy_state: np.ndarray = None
        self._noisy_meas: np.ndarray = None
        self._complete: bool = self._verify()

    @property
    def n(self):
        return self._n

    @property
    def meas_basis(self):
        return self._meas_basis

    @property
    def complete(self):
        return self._complete

    @property
    def prep_circuit(self):
        return self._prep_circuit

    @property
    def meas_circuit(self):
        return self._meas_circuit

    @property
    def prep_gates(self):
        return self._prep_gates

    @property
    def meas_gates(self):
        return self._meas_gates

    @property
    def noisy_gate(self):
        return self._noisy_gates

    @property
    def noisy_state(self):
        if self._noisy_state is None:
            raise ArgumentError("Run quantum gate set tomography first to obtain the noisy state!")
        else:
            return self._noisy_state

    @property
    def noisy_meas(self):
        if self._noisy_meas is None:
            raise ArgumentError("Run quantum gate set tomography first to obtain the noisy measurement!")
        else:
            return self._noisy_meas

    def _verify(self) -> bool:
        r"""Check if the input state preparation and measurement circuits are legal.

        :return: True if state prepare circuit and meas_circuit is legal.
        """
        n = self._n
        num_meas, num_prep = len(self.meas_gates), len(self.prep_gates)

        rho_span = np.zeros((4**self._n, num_prep), dtype=float)
        for i in range(num_prep):
            qp = self.get_circuit(circuit=i, type='prep')
            qp.circuit = self.prep_circuit + qp.circuit
            rho_span[:, [i]] = np.asarray(operator_to_ptm(circuit_to_state(qp)),
                                          dtype=float).reshape((4**n, 1))

        meas_span = np.zeros((4**n, num_meas), dtype=float)
        for j in range(num_meas):
            qp = self.get_circuit(circuit=j, type='meas')
            qp.circuit = self.meas_circuit + qp.circuit
            meas_span[:, [j]] = np.asarray(operator_to_ptm(circuit_to_state(qp)),
                                           dtype=float).reshape((4**n, 1))

        if np.linalg.matrix_rank(rho_span) < 4**n and np.linalg.matrix_rank(meas_span) < 4**n:
            raise ArgumentError("The preparation and measurement circuit are illegal!")
        elif np.linalg.matrix_rank(meas_span) < 4**n:
            raise ArgumentError("The measurement circuit is illegal!")
        elif np.linalg.matrix_rank(rho_span) < 4**n:
            raise ArgumentError("The preparation circuit is illegal!")

        return True

    def get_gate(self, gate: str = None, type: str = 'ideal') -> Union[np.ndarray, CircuitLine, Dict]:
        r"""Get ideal CircuitLine and PTM of noisy gates.

        :param gate: str, the name of gate, default to return dict of all gates
        :param type: str, ``ideal`` or ``noisy``, corresponding to ideal and noisy gate, default to return ideal CircuitLine(s)
        :return: depends on input parameters

        """
        if type == 'ideal':
            if gate is None:
                return self._gates
            if gate in self._gates.keys():
                return self._gates[gate]
            else:
                raise ArgumentError()

        elif type == 'noisy':
            if gate is None:
                return self._noisy_gates
            elif gate in self._noisy_gates.keys():
                return self._noisy_gates[gate]
            else:
                raise ArgumentError()

    def get_circuit(self, circuit: int, type: str = 'meas', qubits: List[int] = None) -> QProgram:
        r"""Construct prepare and measurement circuit.

        :param circuit: the index of state-prepare or measurement circuit :math:`F_i`
        :param type: str, ``prep`` or ``meas``, corresponding to state-prepare and measurement
        :param qubits: List[int], the qubit(s) we want to tomography
        :return: QProgram, the SPAM quantum program
        """
        qp = QEnv()
        qp.Q.createList(self._n)

        if type == 'prep':
            gates = self.prep_gates[circuit]
        elif type == 'meas':
            gates = self.meas_gates[circuit]
        else:
            raise ArgumentError()

        for gate in gates:
            if gate != 'null':
                g_k = deepcopy(self.get_gate(gate=gate, type='ideal'))
                if qubits is not None:
                    g_k.qRegList = [qubits[q] for q in g_k.qRegList]
                qp.circuit.append(g_k)
            else:
                pass

        return qp

    def get_ideal_matrix(self, name: str = None) -> np.ndarray:
        r"""Get the ideal information of gateset.

        :param name: ``rho`` / ``meas`` / the name of gate
        :return: the ideal matrix of input
        """
        n = self._n
        qp = QEnv()
        qp.Q.createList(n)
        if name == 'rho':
            qp.circuit += self._prep_circuit
            return np.asarray(operator_to_ptm(circuit_to_state(qp)),
                              dtype=float).reshape((4**n, 1))
        elif name == 'meas':
            qp.circuit += self._meas_circuit
            return np.asarray(operator_to_ptm(vec_to_operator(circuit_to_unitary(qp)
                                                              @ basis(n, self._meas_basis))),
                              dtype=float).reshape((1, 4**n))
        elif name in self.noisy_gate.keys():
            qp.circuit.append(self._gates[name])
            return np.asarray(unitary_to_ptm(circuit_to_unitary(qp)).data, dtype=float)
        else:
            raise ArgumentError("")

    @property
    def fidelity(self):
        r"""Calculate the fidelity of state / measurement /gates.

        :param name: ``rho`` / ``meas`` / the name of gate
        :return: the fidelity between noisy and ideal state / measurement /gates
        """
        fidelity = {}
        fid = metrics.state_fidelity(ptm_to_operator(self.noisy_state),
                                     ptm_to_operator(self.get_ideal_matrix('rho')))
        fidelity.update({'rho': fid if fid < 1.0 else 1.0})

        fid = metrics.state_fidelity(ptm_to_operator(self.noisy_meas.T),
                                     ptm_to_operator(self.get_ideal_matrix('meas').T))
        fidelity.update({'meas': fid if fid < 1.0 else 1.0})

        for name in self.noisy_gate.keys():
            fid = metrics.average_gate_fidelity(self.noisy_gate[name],
                                                self.get_ideal_matrix(name))
            fidelity.update({name: fid if fid < 1.0 else 1.0})

        return fidelity


class GateSetTomography(Tomography):
    r"""The Quantum GateSet Tomography class.

    """
    def __init__(self, qc: QComputer = None, gate_set: GateSet = None, **kwargs):
        r"""init function of the Quantum GateSet Tomography class.

        Optional keywords list are:

            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out
            + `qubits`: default to None, the index of target qubit(s) we want to tomography

        :param qc: QComputer, the quantum computer
        :param gate_set: GateSet, the gate set we interested in

        """
        super().__init__(qc=qc, **kwargs)
        self._qc: QComputer = qc
        self._gate_set = gate_set
        self._shots: int = kwargs.get('shots', 4096)
        self._qubits: List[int] = kwargs.get('qubits', None)

        # Store the gate set tomography results. Initialize to an empty dictionary
        self._result = dict()

        self._n = max(self._qubits)+1 if self._qubits is not None else None

    @property
    def result(self):
        return self._result

    def fit(self, qc: QComputer = None, gate_set: GateSet = None, **kwargs) -> GateSet:
        r"""Execute the quantum gate set tomography procedure on the quantum computer @qc.

        Optional keywords list are:

            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out
            + `qubits`: default to None, the index of target qubit(s) we want to tomography

        :param qc: QComputer, the quantum computer
        :param gate_set: GateSet, the gate set we interested in
        :return: GateSet, contain the information of noisy gates

        Usage:

        .. code-block:: python
            :linenos:

            gate_set = GateSetTomography.fit(qc, gate_set=gate_set)
            gate_set = GateSetTomography.fit(qc, gate_set=gate_set, shots=2 ** 15)

        **Examples**

            >>> import QCompute
            >>> import numpy as np
            >>> from qcompute_qep.tomography import GateSetTomography, GateSet
            >>> from QCompute.QPlatform.QOperation import CircuitLine
            >>> prep_circuit = []
            >>> meas_circuit = []
            >>> qc = QCompute.BackendName.LocalBaiduSim2
            >>> rx_90 = CircuitLine(QCompute.RX(np.pi / 2), [0])
            >>> ry_90 = CircuitLine(QCompute.RY(np.pi / 2), [0])
            >>> rx_180 = CircuitLine(QCompute.RX(np.pi), [0])
            >>> prep_gates = [['rx_90'], ['ry_90'], ['rx_90', 'rx_90']]
            >>> meas_gates = [['rx_90'], ['ry_90'], ['rx_90', 'rx_90']]
            >>> gate_set = GateSet(prep_circuit, meas_circuit, gates={'rx_90': rx_90, 'ry_90': ry_90},
            >>>                    prep_gates=prep_gates, meas_gates=meas_gates, qubits=[2])
            >>> tomo = GateSetTomography()
            >>> gate_set = tomo.fit(qc, gate_set=gate_set)

        """
        # Init parameters
        self._gate_set = gate_set if gate_set is not None else self._gate_set
        self._qc = qc if qc is not None else self._qc
        self._shots = kwargs.get('shots', self._shots)
        self._qubits = kwargs.get('qubits', self._qubits)
        self._n = max(self._qubits)+1 if self._qubits is not None else self._gate_set.n

        if self._gate_set is None:
            raise ArgumentError("in GateSetTomography.fit(): the GateSet is not set!")
        if self._gate_set.complete is False:
            raise ArgumentError("in GateSetTomography.fit(): the GateSet is illegal!")
        if self._qc is None:
            raise ArgumentError("in GateSetTomography.fit(): the quantum computer is not set!")
        if self._qubits is None:
            self._qubits = list(range(self._n))
        else:
            qubits_set = set(self._qubits)
            if len(qubits_set) != len(self._qubits):
                raise ArgumentError("in GateSetTomography.fit(): the input qubits are not repeatable!")
        # Step 1. Construct the circuit with gate set to get p_ikj, correspond to Eq. (3.12)
        pbar = tqdm(total=100, desc='GST Step 1/4 : Running quantum circuits...', ncols=80)
        p_ikj: List[np.ndarray] = []
        for g_k in self._gate_set.get_gate(type='ideal').values():
            pbar.update(100/4/int(len(self._gate_set.get_gate(type='ideal').values())))
            p_ikj.append(self._expvals_of_gate(g_k, **kwargs))

        self._result.update({'p_ikj': p_ikj})
        self._result.update({'g': p_ikj[0]})

        # Step 2. Construct ideal and new gate set with g, correspond to Eq. (3.19)-(3.24)
        pbar.desc = "GST Step 2/4 : Constructing new gate set..."
        self._result.update({'new gateset': self._construct_new_gateset(**kwargs)})
        self._result.update({'ideal gateset': self._construct_ideal_gateset()})
        pbar.update(100 / 4)

        # Step 3. Use optimizer to estimate B, correspond to Eq. (3.25)
        pbar.desc = "GST Step 3/4 : Optimizing matrix B..."
        B = self._optimize()
        pbar.update(100 / 4)

        # Step 4. Estimate the original GateSet, correspond to Eq. (3.26)-(3.28)
        pbar.desc = "GST Step 4/4 : Estimating origin gate set..."
        for i, gate in enumerate(self._gate_set.noisy_gate):
            self._gate_set.noisy_gate[gate] = B @ self._result['new gateset'][i] @ la.pinv(B)
        self._gate_set._noisy_state = B @ self._result['new rho']
        self._gate_set._noisy_meas = self._result['new meas'] @ la.pinv(B)

        pbar.update(100-pbar.n)
        pbar.desc = "Successfully finished GST!"

        return self._gate_set

    def _optimize(self) -> np.ndarray:
        r"""Optimize the gauge operator :math:`B`.

        :return: the estimation of B
        """
        n = self._gate_set.n
        num_prep = len(self._gate_set.prep_gates)
        # Set starting point for optimization
        B_0 = np.zeros((4**n, num_prep), dtype=float)
        for i in range(4**n):
            B_0[i, 0] = (bsr_matrix((np.ones(1, dtype=int), (np.zeros(1), np.asarray([i], dtype=int))),
                                    shape=(1, 4**n)).toarray()
                         @ unitary_to_ptm(np.identity(2**n)).data
                         @ self._result['ideal rho'])[0, 0]
            for j in range(num_prep)[1:]:
                B_0[i, j] = (bsr_matrix((np.ones(1, dtype=int), (np.zeros(1), np.asarray([i], dtype=int))),
                                        shape=(1, 4**n)).toarray()
                             @ unitary_to_ptm(circuit_to_unitary(self._gate_set.get_circuit(j, type='prep'))).data
                             @ self._result['ideal rho'])[0, 0]

        # Start optimize matrix B
        B_est = minimize(_optimize_problem, np.reshape(B_0, (4**n * num_prep, )), method='Powell',
                         args=(self._result['new gateset'], self._result['ideal gateset'], n),
                         bounds=Bounds(np.full(4**n * num_prep, -1.0), np.full(4**n * num_prep, 1.0)))

        # Get the estimation of B
        if B_est.success is True:
            B = np.reshape(B_est.x, (4**n, -1))
            self._result.update({'B': B})
        else:
            raise ArgumentError(B_est.message)
        return B

    def _expvals_of_gate(self, gate_k: CircuitLine = None, **kwargs) -> np.ndarray:
        r"""Construct circuit to estimate :math:`p_{ikj}` given gate :math:`k`.

        :param gate_k: QOperation, the gate we are interested in, default to None (corresponding to 'null' gate)
        :return: np.ndarray, the matrix records output for all cases
        """
        n = self._n
        qp = QEnv()
        qp.Q.createList(n)
        num_meas, num_prep = len(self._gate_set.meas_gates), len(self._gate_set.prep_gates)

        # Use null gate to get matrix g
        p_ij = np.zeros((num_meas, num_prep), dtype=float)
        for i in range(num_meas):
            meas_qp = self._gate_set.get_circuit(circuit=i, type='meas', qubits=self._qubits)
            for j in range(num_prep):
                qp_ij = deepcopy(qp)
                # state preparation and meas circuit
                prep_circuit, meas_circuit = self._change_spam_qubits()
                qp_ij.circuit += prep_circuit
                qp_ij.circuit += self._gate_set.get_circuit(circuit=j, type='prep', qubits=self._qubits).circuit
                if gate_k is not None:
                    qp_ij.circuit.append(gate_k) if len(self._qubits) == n \
                        else qp_ij.circuit.append(CircuitLine(gate_k.data, [self._qubits[i] for i in gate_k.qRegList]))
                qp_ij.circuit += meas_qp.circuit + meas_circuit
                # Partial measurement
                qreglist, indexlist = qp_ij.Q.toListPair()
                MeasureZ(qRegList=[qreglist[x] for x in self._qubits],
                         cRegList=[indexlist[x] for x in self._qubits])
                counts = execute(qp=qp_ij, qc=self._qc, **kwargs)
                p_ij[i, j] = counts[bin(self._gate_set.meas_basis)[2:].zfill(self._gate_set.n)] / self._shots \
                    if bin(self._gate_set.meas_basis)[2:].zfill(self._gate_set.n) in counts.keys() else 0
        return p_ij

    def _construct_new_gateset(self, **kwargs) -> List[np.ndarray]:
        r"""Construct new gate set from the expectation values.

        :return: np.ndarray, the information of new gate set
        """
        n = self._n
        num_meas, num_prep = len(self._gate_set.meas_gates), len(self._gate_set.prep_gates)
        rho_new = np.zeros((num_prep, 1), dtype=float)
        meas_new = np.zeros((1, num_meas), dtype=float)

        # Run prep_circuit and record measurement result. The data is used to estimate rho tilde Eq. (3.19).
        for i in range(num_prep):
            qp_rho = QEnv()
            qp_rho.Q.createList(n)
            prep_circuit, meas_circuit = self._change_spam_qubits()
            qp_rho.circuit += prep_circuit
            qp_rho.circuit += self._gate_set.get_circuit(circuit=i, type='prep', qubits=self._qubits).circuit
            qp_rho.circuit += meas_circuit
            # Partial measurement
            qreglist, indexlist = qp_rho.Q.toListPair()
            MeasureZ(qRegList=[qreglist[x] for x in self._qubits],
                     cRegList=[indexlist[x] for x in self._qubits])

            counts = execute(qp=qp_rho, qc=self._qc, **kwargs)
            rho_new[i, 0] = counts[bin(self._gate_set.meas_basis)[2:].zfill(self._gate_set.n)]/self._shots \
                if bin(self._gate_set.meas_basis)[2:].zfill(self._gate_set.n) in counts.keys() else 0

        # run meas_circuit and record measurement result. The data is used to estimate rho tilde Eq. (3.20).
        for j in range(num_meas):
            qp_meas = QEnv()
            qp_meas.Q.createList(n)
            prep_circuit, meas_circuit = self._change_spam_qubits()
            qp_meas.circuit += prep_circuit
            qp_meas.circuit += self._gate_set.get_circuit(circuit=j, type='meas', qubits=self._qubits).circuit
            qp_meas.circuit += meas_circuit
            # Partial measurement
            qreglist, indexlist = qp_meas.Q.toListPair()
            MeasureZ(qRegList=[qreglist[x] for x in self._qubits],
                     cRegList=[indexlist[x] for x in self._qubits])
            counts = execute(qp=qp_meas, qc=self._qc, **kwargs)
            meas_new[0, j] = counts[bin(self._gate_set.meas_basis)[2:].zfill(self._gate_set.n)]/self._shots \
                if bin(self._gate_set.meas_basis)[2:].zfill(self._gate_set.n) in counts.keys() else 0

        # Construct new gate set with matrix g
        g = self._result['g']

        # Construct the new gateset with gauge freedom. More precisely, Eqs. (3.21)-(3.23)
        gateset_new = []

        for p_ikj in self._result['p_ikj'][1:]:
            gateset_new.append(la.inv(g) @ p_ikj)
        gateset_new.append(la.inv(g) @ rho_new @ meas_new)
        self._result.update({'new rho': la.inv(g) @ rho_new})
        self._result.update({'new meas': meas_new})

        return gateset_new

    def _construct_ideal_gateset(self) -> List[np.ndarray]:
        r"""Get the ideal information of the gate set, then optimize matrix :math:`B`.

        :return: np.ndarray, the information of ideal gate set
        """
        # Get the target gate set first
        gates_ideal = []
        for key, val in self._gate_set.get_gate(type='ideal').items():
            if key != 'null':
                gates_ideal.append(self._gate_set.get_ideal_matrix(key))

        # Get ideal rho @ meas
        rho_ideal = self._gate_set.get_ideal_matrix('rho')
        meas_ideal = self._gate_set.get_ideal_matrix('meas')
        gates_ideal.append(rho_ideal @ meas_ideal)
        self._result.update({'ideal rho': rho_ideal})
        self._result.update({'ideal meas': meas_ideal})

        return gates_ideal

    def _change_spam_qubits(self) -> Tuple[List[CircuitLine], List[CircuitLine]]:
        r"""Reset the qubit mapping when @self._qubits is set.

        :return: new preparation and measurement list of CircuitLine
        """
        if self._qubits is None:
            return self._gate_set.prep_circuit, self._gate_set.meas_circuit
        else:
            return [CircuitLine(c.data, [self._qubits[i] for i in c.qRegList])
                    for c in self._gate_set.prep_circuit], \
                   [CircuitLine(c.data, [self._qubits[i] for i in c.qRegList])
                    for c in self._gate_set.meas_circuit]


def _optimize_problem(x, *args) -> float:
    r"""Calculate the RMS error.

    :param x: the variable which need to be optimized
    :param args: new gate set, ideal gate set, number of qubits
    :return: the RMS error
    """
    gateset_new = args[0]
    gateset_ideal = args[1]

    # Reshape the input first
    x = np.reshape(x, (4**args[2], -1))

    return sum([la.norm(new - la.pinv(x) @ ideal @ x, 'nuc')
                for new, ideal in zip(gateset_new, gateset_ideal)])
