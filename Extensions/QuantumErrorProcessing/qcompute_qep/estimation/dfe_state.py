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
In this script, we implement the state fidelity estimation method described in
``Direct Fidelity Estimation of Quantum States`` method [FL11]_,
which aims at determining the overlap between the actual quantum state
implemented in a given set-up and the ideal one.

References:

.. [FL11] Flammia, Steven T., and Yi-Kai Liu.
        "Direct fidelity estimation from few Pauli measurements."
        Physical Review Letters 106.23 (2011): 230501.
"""
import numpy as np
from typing import List
import math
import statistics
from tqdm import tqdm
import collections

from Extensions.QuantumErrorProcessing.qcompute_qep.exceptions.QEPError import ArgumentError
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.types import QComputer, QProgram, number_of_qubits
from Extensions.QuantumErrorProcessing.qcompute_qep.quantum.pauli import complete_pauli_basis, from_name_to_matrix
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.circuit import execute, circuit_to_state, map_qubits
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.utils import expval_from_counts
import Extensions.QuantumErrorProcessing.qcompute_qep.estimation as estimation


class DFEState(estimation.Estimation):
    r"""The Direct Fidelity Estimation of Quantum States class.

    `Direct Fidelity Estimation (DFE) of Quantum States` is the procedure that
    determines the overlap between the actual quantum state implemented in a given set-up and the ideal one.
    Let :math:`\hat{Y}` be the estimated fidelity via DFE and :math:`Y` is the true fidelity, DFE guarantees that

    .. math::

        \textrm{Pr}[\vert \hat{Y} - Y \vert \geq 2 \epsilon] \leq 2 \delta,

    where :math:`\epsilon` is the estimation error and :math:`\delta` is the failure probability.

    """
    def __init__(self, qp: QProgram = None, qc: QComputer = None, **kwargs):
        r"""The init function of the DFEState class.

        Optional keywords list are:

            + `epsilon`: float, default to :math:`0.05`, the estimation error :math:`\epsilon`
            + `delta`: float, default to :math:`0.05`, the failure probability :math:`\delta`
            + `qubits`: default to None, the index of target qubit(s) we want to estimate

        :param qp: QProgram, a quantum program for creating the target quantum process
        :param qc: QComputer, the quantum computer
        """
        super().__init__(qp, qc, **kwargs)
        self._epsilon: float = kwargs.get('epsilon', 0.05)
        self._delta: float = kwargs.get('delta', 0.05)
        self._qubits: List[int] = kwargs.get('qubits', None)
        self._fidelity: float = -1.0
        self._std: float = -1.0

    def estimate(self, qp: QProgram = None, qc: QComputer = None, **kwargs) -> float:
        r"""The kernel estimation method of the DFEState class.

        Optional keywords list are:

            + `epsilon`: float, default to :math:`0.05`, the estimation error :math:`\epsilon`
            + `delta`: float, default to :math:`0.05`, the failure probability :math:`\delta`
            + `qubits`: default to None, the index of target qubit(s) we want to estimate

        .. math::

            \textrm{Pr}[\vert \hat{Y} - Y \vert \geq 2 \epsilon] \leq 2 \delta,

        where :math:`\epsilon` is the estimation error and :math:`\delta` is the failure probability.

        :param qp: QProgram, a quantum program for creating the target quantum process
        :param qc: QComputer, the quantum computer
        :return: float, the estimated fidelity between the ideal pure state and the noisy state
        """
        self._qp = qp if qp is not None else self._qp
        self._qc = qc if qc is not None else self._qc
        self._epsilon = kwargs.get('epsilon', self._epsilon)
        self._delta = kwargs.get('delta', self._delta)
        self._qubits = kwargs.get('qubits', self._qubits)

        # If the quantum program or the quantum computer is not set, then DFE cannot be executed
        if self._qp is None:
            raise ArgumentError("DFEState.estimate(): the quantum program is not set!")
        if self._qc is None:
            raise ArgumentError("DFEState.estimate(): the quantum computer is not set!")
        if self._qubits is None:
            self._qubits = list(range(number_of_qubits(qp)))
        # Check if the indices in self._qubits is unique
        if len(set(self._qubits)) != len(self._qubits):
            raise ArgumentError("DFEState.estimate(): the input qubits are not repeatable!")
        # Check if the number of qubits in @qp and @qubits are equal
        if len(self._qubits) != number_of_qubits(qp):
            raise ArgumentError("DFEState.estimate(): the number of qubits in '@qp' "
                                "must be equal to the number of qubits in '@qubits'!")

        # Number of qubits under consideration
        n = len(self._qubits)

        # Step 1. Compute the ideal quantum state and its expectation values of Pauli operators
        pbar = tqdm(total=100, desc='DFEState Step 1/3 : Sampling Pauli operators ...', ncols=80)
        complete_pauli = complete_pauli_basis(n)
        ptm_ideal = [np.real(np.trace(from_name_to_matrix(p.name[::-1]) @ self.ideal_state)) for p in complete_pauli]

        # Step 2. Sampling Pauli operators according to the probability distribution
        sample_times = math.ceil(1.0 / (self._epsilon ** 2 * self._delta))
        pauli_list = np.random.choice(complete_pauli, sample_times, p=[coe ** 2 for coe in ptm_ideal]).tolist()
        pauli_dic = dict(collections.Counter(pauli_list))

        # Step 3. Estimate the expectation values of the sampled Pauli operators w.r.t. the noisy state
        fids = []
        pbar.update(100 / 3)
        pbar.desc = "DFEState Step 2/3 : Running quantum circuits ..."

        for p, num in pauli_dic.items():
            # Obtain the expectation value (the corresponding coefficient) of the Pauli operator
            pbar.update(100/3/len(pauli_list))
            coe = ptm_ideal[complete_pauli.index(p)]

            # Compute the number of shots must be carried out
            shots = math.ceil(2 * np.log2(2 / self._delta)
                              / (2 ** n * coe ** 2 * sample_times * self._epsilon ** 2))

            # Modify the circuit to implement the Pauli measurement and run it on the noisy quantum computer
            p_qp, p_ob = p.meas_circuit(self._qp)

            # Map to target quantum qubits we actually want to estimate
            p_qp = map_qubits(qp=p_qp, qubits=self._qubits)

            # Execute the quantum circuits
            counts = execute(qp=p_qp, qc=self._qc, shots=num*shots, **kwargs)
            fids.append(expval_from_counts(A=p_ob, counts=counts) / coe / np.sqrt(2 ** n))

        # Step 4. Obtain the estimated fidelity by computing the mean
        pbar.desc = "DFEState Step 3/3 : Processing experimental data ..."
        avg_fid = statistics.mean(fids)
        self._std = statistics.stdev(fids)
        self._fidelity = avg_fid if avg_fid < 1.0 else 1.0
        pbar.update(100 - pbar.n)
        pbar.desc = "Successfully finished DFEState!"
        return self._fidelity

    @property
    def ideal_state(self):
        r"""Ideal state in matrix form of the given quantum program.
        """
        if self._qp is None:
            raise ArgumentError("The quantum program characterizing the quantum state is not given!")

        return circuit_to_state(self._qp)

    @property
    def fidelity(self):
        if self._fidelity == -1.0:
            self.estimate(self._qp, self._qc)

        return self._fidelity

    @property
    def delta(self):
        return self._delta

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def qubits(self):
        return self._qubits

    @property
    def std(self):
        if self._std == -1.0:
            self.estimate(self._qp, self._qc)
        return self._std
