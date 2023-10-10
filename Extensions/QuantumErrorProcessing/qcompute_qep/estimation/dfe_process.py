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
In this script, we implement the process fidelity estimation method described in
``Direct Fidelity Estimation of Quantum Processes`` method [FL11]_,
which aims at determining the overlap between the actual quantum process
implemented in a given set-up and the ideal one.

References:

.. [FL11] Flammia, Steven T., and Yi-Kai Liu.
        "Direct fidelity estimation from few Pauli measurements."
        Physical Review Letters 106.23 (2011): 230501.
"""
import collections

import numpy as np
from typing import List
import math
import statistics
from tqdm import tqdm
import itertools


import Extensions.QuantumErrorProcessing.qcompute_qep.estimation as estimation
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.types import QComputer, QProgram, number_of_qubits
from Extensions.QuantumErrorProcessing.qcompute_qep.quantum.pauli import complete_pauli_basis
from Extensions.QuantumErrorProcessing.qcompute_qep.quantum.channel import unitary_to_ptm
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.circuit import execute, circuit_to_unitary, map_qubits
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.utils import expval_from_counts
from Extensions.QuantumErrorProcessing.qcompute_qep.exceptions.QEPError import ArgumentError
import QCompute
from QCompute.QPlatform.Utilities import nKron
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.linalg import expand, permute_systems, dagger, basis


class DFEProcess(estimation.Estimation):
    r"""The Direct Fidelity Estimation of Quantum Processes class.

    `Direct Fidelity Estimation of Quantum Processes` is the procedure that
    determines the fidelity between the actual quantum process implemented in a given set-up and the ideal one.

    Let :math:`\hat{F_e}` be the estimated fidelity via DFE and :math:`F_e` is the true fidelity, DFE guarantees that

    .. math::

        \textrm{Pr}[\vert \hat{F_e} - F_e \vert \geq 2 \epsilon] \leq 2 \delta,

    where :math:`\epsilon` is the estimation error and :math:`\delta` is the failure probability.

    The entanglement fidelity is considered, i.e.,

    .. math::

        F_e = \frac{1}{4^n} {\rm Tr}[T_{U}^{\dagger} T_{N}]

    where :math:'n' is the number of qubits, :math:`T_{U}` and :math:`T_{N}` are the Pauli transfer matrix of the ideal
    unitary channel :math:`U` and the actual quantum process :math:`N`, respectively.
    """
    def __init__(self, qp: QProgram = None, qc: QComputer = None, **kwargs):
        r"""The init function of the DFEProcess class.

        Optional keywords list are:

            + `epsilon`: float, default to :math:`0.05`, the estimation error :math:`\epsilon`
            + `delta`: float, default to :math:`0.05`, the failure probability :math:`\delta`
            + `qubits`: default to None, the index of target qubit(s) we want to estimate

        :param qp: QProgram, a quantum program for creating the target quantum process
        :param qc: QComputer, the quantum computer
        """
        super().__init__(qp, qc, **kwargs)
        self._epsilon: float = kwargs.get('epsilon', 0.2)
        self._delta: float = kwargs.get('delta', 0.4)
        self._qubits: List[int] = kwargs.get('qubits', None)
        self._fidelity: float = -1.0
        self._std: float = -1.0

    def estimate(self, qp: QProgram = None, qc: QComputer = None, **kwargs) -> float:
        r"""The kernel estimation method of the DFEProcess class.

        Optional keywords list are:

            + `epsilon`: float, default to :math:`0.05`, the estimation error :math:`\epsilon`
            + `delta`: float, default to :math:`0.05`, the failure probability :math:`\delta`
            + `qubits`: default to None, the index of target qubit(s) we want to estimate

        .. math::

            \textrm{Pr}[\vert \hat{F_e} - F_e \vert \geq 2 \epsilon] \leq 2 \delta,

        where :math:`\epsilon` is the estimation error and :math:`\delta` is the failure probability.

        :param qp: QProgram, a quantum program for creating the target quantum process
        :param qc: QComputer, the quantum computer
        :return: float, the estimated entanglement fidelity between the ideal unitary channel
        and the actual quantum process.
        """
        super().__init__(qp, qc, **kwargs)
        self._qp = qp if qp is not None else self._qp
        self._qc = qc if qc is not None else self._qc
        self._epsilon = kwargs.get('epsilon', self._epsilon)
        self._delta = kwargs.get('delta', self._delta)
        self._qubits = kwargs.get('qubits', self._qubits)

        if self._qp is None:
            raise ArgumentError("In DFEProcess.estimate(): the quantum program is not set!")
        if self._qc is None:
            raise ArgumentError("In DFEProcess.estimate(): the quantum computer is not set!")
        if self._qubits is None:
            self._qubits = list(range(number_of_qubits(qp)))
        # Check if the indices in self._qubits is unique
        if len(set(self._qubits)) != len(self._qubits):
            raise ArgumentError("In DFEProcess.estimate(): the input qubits are not repeatable!")
        # Check if the number of qubits in @qp and @qubits are equal
        if len(self._qubits) != number_of_qubits(qp):
            raise ArgumentError("In DFEProcess.estimate(): the number of qubits in '@qp' "
                                "must be equal to the number of qubits in '@qubits'!")

        # Number of qubits in the quantum program
        n = len(self._qubits)

        # Step 1.1. Compute the ideal unitary channel and its PTM matrix elements
        pbar = tqdm(total=100, ncols=80)
        pbar.desc = 'DFEProcess Step 1/3 : Sampling Pauli operators ...'
        ideal_unitary = permute_systems(self.ideal_unitary, perm=list(reversed(range(n))))
        ptm_matrix_ideal_unitary = np.real(unitary_to_ptm(ideal_unitary).data)
        ptm_list_ideal_unitary = [coe for i in ptm_matrix_ideal_unitary for coe in i]
        # Step 1.2. Sampling Pauli operator pairs by the probability distribution in terms of PTM matrix elements
        complete_pauli = complete_pauli_basis(n)
        complete_pauli_pair = list(itertools.product(complete_pauli, repeat=2))
        sample_times = math.ceil(1.0/(self._epsilon ** 2 * self._delta))
        pauli_pair_list_index = np.random.choice(4 ** (2 * n), sample_times, p=[coe ** 2 / (2 ** n) ** 2 for coe in
                                                                                ptm_list_ideal_unitary])
        pauli_pair_index_dic = dict(collections.Counter(pauli_pair_list_index))
        pbar.update(100 / 3)
        pbar.desc = "DFEProcess Step 2/3 : Running quantum circuits ..."
        # Step 2. Estimate the expectation values of the first Pauli operator of a sampled Pauli pair and the output
        # of the noisy unitary channel with the second Pauli operator of the sampled Pauli pair as the input
        fids = []
        for pair_index, num in pauli_pair_index_dic.items():
            # Get the matrix elements of the sampled Pauli pair
            pbar.update(100 / 3 / len(pauli_pair_list_index))
            coe = ptm_list_ideal_unitary[pair_index]
            pair = complete_pauli_pair[pair_index]
            shots = math.ceil(4 * np.log(4 / self._delta) / (coe ** 2 * sample_times * self._epsilon ** 2))
            pair1_qp_eigenstates, pair1_eigenvalues = pair[1].preps_circuits(self._qp)
            temp_fids = []
            for i in range(2 ** n):
                eigenvalue = pair1_eigenvalues[i]
                circuit_eigenstate = pair1_qp_eigenstates[i]
                # Modify the circuit to implement the Pauli measurement of the first Pauli of
                # the pair and run it on the noisy quantum computer
                pair0_qp, pair0_ob = pair[0].meas_circuit(circuit_eigenstate)

                # Map to target quantum qubits we actually want to estimate
                pair0_qp = map_qubits(qp=pair0_qp, qubits=self._qubits)

                # Execute the quantum circuits
                counts = execute(qp=pair0_qp, qc=self._qc, shots=math.ceil(num * shots/(2 ** n)), **kwargs)
                temp_fids.append(expval_from_counts(A=pair0_ob, counts=counts) * eigenvalue/2 ** n)
            fids.append(sum(temp_fids)/coe)
        # Step 3. Calculate the estimated fidelity by computing the mean
        pbar.desc = "DFEProcess Step 3/3 : Processing experimental data ..."
        avg_fid = statistics.mean(fids)
        self._std = fids
        self._fidelity = avg_fid
        pbar.update(100 - pbar.n)
        pbar.desc = "Successfully finished DFEProcess!"
        pbar.close()
        return self._fidelity

    @property
    def ideal_unitary(self):
        r"""Ideal unitary channel in matrix form of the given quantum program.
        """
        if self._qp is None:
            raise ArgumentError("In DFEProcess.estimate(): the quantum program is not set!")
        return circuit_to_unitary(self._qp)

    @property
    def epsilon(self):
        return self._epsilon

    @property
    def delta(self):
        return self._delta

    @property
    def qubits(self):
        return self._qubits

    @property
    def fidelity(self):
        if self._fidelity == -1.0:
            self.estimate(self._qp, self._qc)
        return self._fidelity

    @property
    def std(self):
        if self._std == -1.0:
            self.estimate(self._qp, self._qc)
        return self._std
