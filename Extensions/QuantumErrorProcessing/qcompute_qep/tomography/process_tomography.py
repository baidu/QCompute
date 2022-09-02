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
This file aims to collect functions related to the Quantum Process Tomography.

"""
import scipy.linalg as la
import numpy as np
from typing import List
from copy import deepcopy
from tqdm import tqdm

from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.tomography import Tomography, StateTomography, MeasurementBasis, PreparationBasis, \
                            init_measurement_basis, init_preparation_basis
from qcompute_qep.utils.types import QComputer, QProgram, number_of_qubits
from qcompute_qep.utils.linalg import dagger
from qcompute_qep.utils.circuit import circuit_to_unitary
from qcompute_qep.quantum.pauli import ptm_to_process, unitary_to_ptm
from qcompute_qep.quantum.channel import QuantumChannel, PTM
from qcompute_qep.quantum.metrics import average_gate_fidelity


class ProcessTomography(Tomography):
    """The Quantum Process Tomography class.

    Quantum Process Tomography deals with identifying an unknown quantum dynamical process.
    It requires the use of quantum state tomography to reconstruct the process.
    """
    def __init__(self, qp: QProgram = None, qc: QComputer = None, **kwargs):
        r"""
        The init function of the Quantum Process Tomography class.

        Optional keywords list are:

            + `method`: default to ``inverse``, specify the process tomography method
            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out
            + `ptm`: default to ``True``, if the quantum process should be returned to the Pauli transfer matrix form
            + `qubits`: default to None, the index of target qubit(s) we want to tomography, now only support full tomography
            + `prep_basis`: default to ``PauliMeasBasis``, the preparation (state) basis
            + `meas_basis`: default to ``PauliMeasBasis``, the measurement basis

        :param qp: QProgram, a quantum program for creating the target quantum process
        :param qc: QComputer, the quantum computer


        """
        super().__init__(qp, qc, **kwargs)
        self._qp: QProgram = qp
        self._qc: QComputer = qc
        self._method: str = kwargs.get('method', 'inverse')
        self._shots: int = kwargs.get('shots', 4096)
        self._ptm: bool = kwargs.get('ptm', True)
        self._qubits: List[int] = kwargs.get('qubits', None)
        # Setup the preparation and measurement bases for quantum process tomography
        self._prep_basis: PreparationBasis = init_preparation_basis(kwargs.get('prep_basis', None))
        self._meas_basis: MeasurementBasis = init_measurement_basis(kwargs.get('meas_basis', None))
        self._noisy_ptm: np.ndarray = None

    def fit(self, qp: QProgram = None, qc: QComputer = None, **kwargs) -> QuantumChannel:
        r"""
        Execute the quantum process procedure for the quantum process specified by @qp on the quantum computer @qc.

        Optional keywords list are:

        + `method`: default to ``inverse``, specify the process tomography method. Current support:

            + ``inverse``: the inverse method;
            + ``lstsq``: the least square method;
            + ``mle``: the maximum likelihood estimation method.

        + `shots`: default to :math:`4096`, the number of shots each measurement should carry out
        + `ptm`: default to ``False``, if the quantum process should be returned to the Pauli transfer matrix form
        + `prep_basis`: default to ``PauliMeasBasis``, the preparation (state) basis
        + `meas_basis`: default to ``PauliMeasBasis``, the measurement basis

        :param qp: QProgram, quantum program for creating the target quantum process
        :param qc: QComputer, the quantum computer instance
        :return: QuantumChannel, the estimated quantum process in the Pauli transfer matrix form


        Usage:

        .. code-block:: python
            :linenos:

            rho = ProcessTomography.fit(qp=qp, qc=qc)
            rho = ProcessTomography.fit(qp=qp, qc=qc, method='inverse')
            rho = ProcessTomography.fit(qp=qp, qc=qc, method='lstsq', shots=shots)

        **Examples**

            >>> import QCompute
            >>> import qcompute_qep.tomography as tomography
            >>> from qcompute_qep.utils.circuit import circuit_to_unitary
            >>> from qcompute_qep.quantum.pauli import unitary_to_ptm
            >>> import qcompute_qep.utils.types as typing
            >>> qp = QCompute.QEnv()
            >>> qp.Q.createList(2)
            >>> QCompute.CZ(qp.Q[1], qp.Q[0])
            >>> ideal_cnot = circuit_to_unitary(qp)
            >>> ideal_ptm = unitary_to_ptm(ideal_cnot).data
            >>> qc = QCompute.BackendName.LocalBaiduSim2
            >>> qc_name = typing.get_qc_name(qc)
            >>> st = tomography.ProcessTomography()
            >>> noisy_ptm = st.fit(qp, qc, method='inverse', shots=4096, ptm=True)
            >>> diff_ptm = ideal_ptm - noisy_ptm.data
            >>> tomography.compare_process_ptm(ptms=[ideal_ptm, noisy_ptm.data, diff_ptm])

        """
        # Parse the arguments. If not set, use the default arguments set by the init function
        self._qp = qp if qp is not None else self._qp
        self._qc = qc if qc is not None else self._qc
        self._method = kwargs.get('method', self._method)
        self._shots = kwargs.get('shots', self._shots)
        self._ptm = kwargs.get('ptm', self._ptm)
        self._qubits = kwargs.get('qubits', self._qubits)
        # Setup the the preparation and measurement bases for quantum process tomography
        self._prep_basis = init_preparation_basis(kwargs.get('prep_basis', self._prep_basis))
        self._meas_basis = init_measurement_basis(kwargs.get('meas_basis', self._meas_basis))

        # If the quantum program or the quantum computer is not set, the process tomography cannot be executed
        if self._qp is None:
            raise ArgumentError("in ProcessTomography.fit(): the quantum program is not set!")
        if self._qc is None:
            raise ArgumentError("in ProcessTomography.fit(): the quantum computer is not set!")
        if self._qubits is None:
            self._qubits = list(range(number_of_qubits(qp)))
        else:
            qubits_set = set(self._qubits)
            if len(qubits_set) != len(self._qubits):
                raise ArgumentError("in ProcessTomography.fit(): the input qubits are not repeatable!")
        # Number of qubits in the quantum program
        n = len(self._qubits)

        # Initialize the probability matrix
        prep_size = self._prep_basis.size(n)
        meas_size = self._meas_basis.size(n)

        ptm = np.zeros((meas_size, prep_size), dtype=complex)

        # Step 1. construct a list of preparation quantum circuits from the quantum program
        pbar = tqdm(total=100, desc='QPT Step 1/3 : Constructing quantum circuits...', ncols=80)
        prep_qps = self._prep_basis.prep_circuits(self._qp, self._qubits)
        pbar.update(100 / 3)
        # Step 2. for each preparation quantum circuit, carry out the quantum state tomography and obtain the estimates
        pbar.desc = "QPT Step 2/3 : Running quantum circuits..."
        # Initialize a StateTomography instance
        st = StateTomography(basis=self._meas_basis, method=self._method,
                             shots=self._shots, ptm=True, qubits=self._qubits)
        for i, prep_qp in enumerate(prep_qps):
            pbar.update(100/3/int(len(prep_qps)))
            val = st.fit(qp=prep_qp, qc=self._qc, **kwargs)
            ptm[:, i] = np.asarray(val)

        P = self._prep_basis.transition_matrix(n)
        M = self._meas_basis.transition_matrix(n)
        # Step 3. perform the fitting procedure to estimate the quantum process
        pbar.desc = "QPT Step 3/3 : Processing experimental data..."
        # Obtain the transition matrices for the preparation and measurement basis
        if self._method.lower() == 'inverse':  # The naive inverse method
            process_ptm = la.pinv(M) @ ptm @ la.pinv(np.transpose(P))
        elif self._method.lower() == 'lstsq':  # the ordinary least square method
            process_ptm = la.pinv(dagger(M) @ M) @ dagger(M) \
                          @ ptm \
                          @ np.conjugate(P) @ la.pinv(np.transpose(P) @ np.conjugate(P))
        elif self._method.lower() == 'mle':  # the maximum likelihood estimation
            process_ptm = None
            pass
        else:
            raise ArgumentError("In ProcessTomography.fit(), unsupported tomography method '{}'".format(self._method))
        pbar.update(100-pbar.n)
        pbar.desc = "Successfully finished QPT!"
        self._noisy_ptm = PTM(process_ptm)
        # TODO: implement the `ptm_to_process` function
        if self._ptm:
            return self._noisy_ptm
        else:
            return ptm_to_process(self._noisy_ptm, type='kraus')

    @property
    def ideal_ptm(self):
        r"""Compute the ideal PTM from the given quantum program.
        """
        if self._qp is None:
            raise ArgumentError("The quantum program characterizing the quantum process is not given!")

        if self._qubits is None:
            self._qubits = [i for i in range(len(self._qp.Q.registerMap.keys()))]

        U = circuit_to_unitary(self._qp, self._qubits)

        return unitary_to_ptm(U).data

    @property
    def noisy_ptm(self):
        r"""Noisy state in matrix form obtained via state tomography.
        """
        if self._noisy_ptm is None:
            raise ArgumentError("Run quantum state tomography first to obtain the noisy state!")
        else:
            return self._noisy_ptm

    @property
    def fidelity(self):
        r"""Compute the average gate fidelity between ideal and noisy process's PTM.

        The average gate fidelity of two quantum processes @proc_M and @proc_N is mathematically defined as:

        .. math::

                F_{\rm avg}(\mathcal{M}, \mathcal{N}) := \int_\psi d\eta(\psi)\langle\psi\vert
                   \mathcal{M}^\dagger\circ\mathcal{N}(\vert\psi\rangle\!\langle\psi\vert)\vert\psi\rangle
                   = \frac{\left(1 + \text{Tr}\left[[\mathcal{M}]_u^\dagger [\mathcal{N}]_u\right]\right)/d + 1}{d+1},

        where :math:`\eta(\psi)` is the Haar measure,
        :math:`d` is the dimension of the input quantum system,
        and :math:`[\mathcal{M}]_u` is the unital part of the Pauli transfer matrix.
        The definition is excerpted from the following reference:

        .. [SHBT] Helsen, Jonas, Francesco Battistel, and Barbara M. Terhal.
            "Spectral quantum tomography."
            npj Quantum Information 5.1 (2019): 1-11.

        :param noisy_ptm: np.ndarray, the Pauli transfer matrix of real quantum process
        :return: float, the average gate fidelity between ideal and noisy process's PTM
        """
        fid = average_gate_fidelity(self.ideal_ptm, self.noisy_ptm.data)

        return fid if fid < 1.0 else 1.0
