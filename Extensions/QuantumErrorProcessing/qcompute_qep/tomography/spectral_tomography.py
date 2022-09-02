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

"""This file aims to collect functions related to the Quantum Spectral
Tomography."""
import scipy.linalg as la
import numpy as np
from copy import deepcopy
from tqdm import tqdm

from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.tomography import Tomography, ProcessTomography
from qcompute_qep.utils.types import QComputer, QProgram, number_of_qubits
from qcompute_qep.utils.circuit import execute
from qcompute_qep.quantum.pauli import complete_pauli_basis, unitary_to_ptm
from typing import List, Union, Tuple
from qcompute_qep.utils.utils import expval_from_counts
from qcompute_qep.utils.linalg import permute_systems, expand


class SpectralTomography(Tomography):
    """The Quantum Spectral Tomography class.

    Quantum Spectral Tomography deals with identifying the eigenvalues
    of an unknown quantum dynamical process in PTM form.
    """
    def __init__(self, qp: QProgram = None, qc: QComputer = None, **kwargs):
        """The init function of the Quantum Spectral Tomography class.

        The init function of the Quantum Spectral Tomography class. Optional keywords list are:

            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out
            + `k`: default to None, number of channel reuse
            + `l`: default to None, pencil parameter, determine the shape of matrix :math:`Y`
            + `N`: default to None, the number of eigenvalues
            + `a`: default to False, decide whether to calculate amplitude
            + `qubits`: default to None, the index of target qubit(s) we want to tomography, now only support full tomography

        :param qp: QProgram, quantum program for creating the target quantum process
        :param qc: QComputer, the quantum computer
        """
        super().__init__(qp, qc, **kwargs)
        self._qp: QProgram = qp
        self._qc: QComputer = qc
        # self._method: str = kwargs.get('method', 'inverse')
        self._shots: int = kwargs.get('shots', 4096)
        self._K: int = kwargs.get('k', None)
        self._L: int = kwargs.get('l', None)
        self._N: int = kwargs.get('N', None)
        self._amp = kwargs.get('a', False)
        self._qubits: List[int] = kwargs.get('qubits', None)
        self._ideal_eigenvalues: np.ndarray = None

    def _repeat_channel(self) -> List[QProgram]:
        """
        :return: a list of quantum circuit repeating k times quantum channel
        """
        # Construct g(0)
        start_qp = deepcopy(self._qp)
        start_qp.circuit.clear()

        k_qps: List[QProgram] = [start_qp]
        # Construct g(1) .... g(K)
        for _ in range(self._K):
            k_qp = deepcopy(k_qps[-1])
            k_qp.circuit = k_qp.circuit + self._qp.circuit
            k_qps.append(k_qp)

        return k_qps

    def fit(self, qp: QProgram = None, qc: QComputer = None, **kwargs) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """Execute the quantum spectral procedure for the quantum process
        specified by @qp on the quantum computer @qc.

        Optional keywords list are:

            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out
            + `k`: default to :math:`2 N-2`, number of channel reuse
            + `l`: default to :math:`K/2`, pencil parameter, determine the shape of matrix :math:`Y`
            + `N`: default to :math:`4^n - 1`, a variable to fit the signal (consider degenerate)
            + `a`: default to False, decide whether to calculate amplitude
            + `qubits`: default to None, the index of target qubit(s) we want to tomography, now only support full tomography

        :param qp: QProgram, quantum program for creating the target quantum process
        :param qc: QComputer, the quantum computer instance
        :return: the estimated quantum process in the Pauli transfer matrix form

        Usage:

        .. code-block:: python
            :linenos:

            rho = SpectralTomography.fit(qp=qp, qc=qc)
            rho = SpectralTomography.fit(qp=qp, qc=qc, method='inverse')
            rho = SpectralTomography.fit(qp=qp, qc=qc, method='lstsq', shots=shots)

        **Examples**

            >>> import QCompute
            >>> import qcompute_qep.tomography as tomography
            >>> qp = QCompute.QEnv()
            >>> qp.Q.createList(2)
            >>> QCompute.H(qp.Q[0])
            >>> QCompute.CZ(qp.Q[1], qp.Q[0])
            >>> QCompute.H(qp.Q[0])
            >>> qc = QCompute.BackendName.LocalBaiduSim2
            >>> st = tomography.SpectralTomography()
            >>> noisy_ptm = st.fit(qp, qc, k=50, l=30, N=2)
        """
        # Parse the arguments. If not set, use the default arguments set by the init function
        self._qp = qp if qp is not None else self._qp
        self._qc = qc if qc is not None else self._qc
        self._shots = kwargs.get('shots', self._shots)
        self._K = kwargs.get('k', self._K)
        self._L = kwargs.get('l', self._L)
        self._amp = kwargs.get('a', self._amp)
        self._qubits = kwargs.get('qubits', self._qubits)

        # If the quantum program or the quantum computer is not set, the process tomography cannot be executed
        if self._qp is None:
            raise ArgumentError("in SpectralTomography.fit(): the quantum program is not set!")
        if self._qc is None:
            raise ArgumentError("in SpectralTomography.fit(): the quantum computer is not set!")
        if self._qubits is None:
            self._qubits = list(range(number_of_qubits(qp)))
        else:
            qubits_set = set(self._qubits)
            if len(qubits_set) != len(self._qubits):
                raise ArgumentError("in SpectralTomography.fit(): the input qubits are not repeatable!")

        # Number of qubits in the quantum program
        n = len(self._qubits)

        # Consider the :math:`N` as a variable, default to :math:`4^n-1`
        N = kwargs.get('N', 4**n-1) if self._N is None else kwargs.get('N', self._N)
        # Set the default parameter K and L
        self._K = 2 * N - 2 if self._K is None else self._K
        self._L = int(self._K / 2) if self._L is None else self._L
        meas_paulis = complete_pauli_basis(n)[1:]  # a list of string, Pauli basis
        # Step 1. construct a list of tomographic quantum circuits from the quantum program
        pbar = tqdm(total=100, desc='SQT Step 1/3 : Constructing quantum circuits...', ncols=80)
        k_qps = self._repeat_channel()
        g = np.zeros(self._K+1, dtype=float)
        pbar.update(100 / 3)

        # Step 2. run the tomographic quantum circuits on the quantum computer and estimate the expectation values
        pbar.desc = "SQT Step 2/3 : Running quantum circuits..."
        for k, k_qp in enumerate(k_qps):
            # Construct the measurement circuit for k-th circuit
            pbar.update(100/3/int(len(k_qps)))
            for i, meas_pauli in enumerate(meas_paulis):
                meas_qp, meas_ob = meas_pauli.meas_circuit(k_qp, qubits=self._qubits)
                qps, eig_vals = meas_pauli.preps_circuits(meas_qp, qubits=self._qubits)
                for j, qp in enumerate(qps):
                    counts = execute(qp=qp, qc=self._qc, **kwargs)
                    expval = expval_from_counts(A=meas_ob, counts=counts)
                    g[k] = g[k] + expval * eig_vals[j]

            g[k] = g[k] / (2**n)

        # Step 3. perform the fitting procedure to estimate the quantum state
        pbar.desc = "SQT Step 3/3 : Processing experimental data..."
        # Construct a (K-L+1) \times (L+1) dimensional data matrix Y
        Y = np.zeros((self._K-self._L+1, self._L+1), dtype=float)
        for i in range(self._L+1):
            Y[:, i] = np.asarray(g[i:i+self._K-self._L+1].T)

        # Construct a singular-value decomposition of the matrix Y
        _, sigma, vt = np.linalg.svd(Y)
        if len(sigma) > N:
            sigma = sigma[:N]
            vt = vt[:N, :]
        vt0 = vt[:, :-1]
        vt1 = vt[:, 1:]
        self._ideal_eigenvalues, _ = np.linalg.eig(vt0 @ la.pinv(vt1))

        pbar.update(100-pbar.n)
        pbar.desc = "Successfully finished SQT!"
        # Only calculate eigenvalues we estimate
        if self._amp is False:
            return self._ideal_eigenvalues

        # TODO: use least-squares minimization to calculate the amplitude of noisy signal
        amplitudes = np.zeros((N, 1), dtype=complex)

        return self._ideal_eigenvalues, amplitudes

    @property
    def ideal_eigenvalues(self):
        """Compute the ideal spectral of given quantum process specified by a list of qubits.

        Usage:


        .. code-block:: python
            :linenos:

            ideal_values = tomography.ideal_eigenvalues(qp)
            ideal_values = tomography.ideal_eigenvalues(qp, qubits=[1])

        :param qp: QProgram, quantum program
        :param qubits: List[int], the target qubits
        :return: np.ndarray, the ideal spectral of given quantum process
        """
        _TWO_QUBIT_GATESET = {'CX', 'CY', 'CZ', 'CH', 'SWAP'}
        _THREE_QUBIT_GATESET = {'CCX', 'CSWAP'}

        if self._qp is None:
            raise ArgumentError("The quantum program characterizing the quantum spectral is not given!")

        if self._qubits is None:
            self._qubits = [i for i in range(len(self._qp.Q.registerMap.keys()))]

        # Number of qubits in the quantum program
        n = len(self._qubits)

        U = np.identity(2 ** n, dtype='complex')
        # Process the circuit layer by layer, each layer corresponds to a :math:`2^n\times 2^n` unitary matrix
        for circuit in self._qp.circuit:
            indices = [self._qubits.index(i) for i in circuit.qRegList]
            local_u = circuit.data.getMatrix()
            # In QCompute, all multi-qubit gates' matrix is fixed as qubit i controls qubit i-1,
            # we thus must analyze the control qubit and permute the system.
            if (circuit.data.name in _TWO_QUBIT_GATESET) and (indices[0] < indices[1]):
                local_u = permute_systems(local_u, [1, 0])
            elif (circuit.data.name in _THREE_QUBIT_GATESET) and (indices[0] < indices[2]):
                local_u = permute_systems(local_u, [2, 1, 0])
            V_i = expand(local_u, indices, n)
            V_i = permute_systems(V_i, perm=list(reversed(range(n))))
            U = V_i @ U

        self._ideal_eigenvalues, _ = np.linalg.eig(unitary_to_ptm(U).data[1:, 1:])

        return self._ideal_eigenvalues

