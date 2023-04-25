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
This file aims to collect functions related to quantum state tomography.
"""

from qcompute_qep.utils.circuit import execute
from typing import List, Union
import scipy.linalg as la
import numpy as np
from tqdm import tqdm

from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.tomography import Tomography, MeasurementBasis, init_measurement_basis
from qcompute_qep.utils.types import QComputer, QProgram, number_of_qubits
from qcompute_qep.utils.linalg import dagger
from qcompute_qep.quantum.pauli import ptm_to_operator
from qcompute_qep.quantum.metrics import state_fidelity
from qcompute_qep.utils.utils import expval_from_counts
from qcompute_qep.utils.circuit import circuit_to_state, map_qubits


class StateTomography(Tomography):
    """The Quantum State Tomography class.

    Quantum state tomography is the process by which a quantum state is reconstructed
    using measurements on an ensemble of identical quantum states.
    """
    def __init__(self, qp: QProgram = None, qc: QComputer = None, **kwargs):
        r"""
        The init function of the Quantum State Tomography class.

        Optional keywords list are:

            + `method`: default to ``inverse``, specify the state tomography method

            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out

            + `basis`: default to ``PauliMeasBasis``, the measurement basis

            + `ptm`: default to ``False``, if the quantum state should be returned to the Pauli transfer matrix form

            + `qubits`: default to ``None``, the indices of qubits we actually want to tomography.

        :param qp: QProgram, quantum program for creating the target quantum state
        :param qc: QComputer, the quantum computer

        """
        super().__init__(qp, qc, **kwargs)
        self._qp: QProgram = qp
        self._qc: QComputer = qc
        self._method: str = kwargs.get('method', 'inverse')
        self._shots: int = kwargs.get('shots', 4096)
        self._ptm: bool = kwargs.get('ptm', False)
        self._qubits: List[int] = kwargs.get('qubits', None)
        # Set up the measurement basis for quantum state tomography
        self._basis: Union[str, MeasurementBasis] = init_measurement_basis(kwargs.get('basis', None))
        self._noisy_state: np.ndarray = None

    def fit(self, qp: QProgram = None, qc: QComputer = None, **kwargs) -> np.ndarray:
        r"""Execute quantum state procedure for the quantum state specified by @qp on the quantum computer @qc.

        Optional keywords list are:

            + `method`: default to ``inverse``, specify the state tomography method. Currently support:

                + ``inverse``: the inverse method;
                + ``lstsq``: the least square method; and
                + ``mle``: the maximum likelihood estimation method.

            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out

            + `qubits`: default to ``None``, the indices of target qubits we actually want to tomography

            + `basis`: default to ``PauliMeasBasis``, the measurement basis

            + `ptm`: default to ``False``, if the quantum state should be returned to the Pauli transfer matrix form

        :param qp: QProgram, quantum program for creating the target quantum state
        :param qc: QComputer, the quantum computer instance
        :return: np.ndarray, the estimated quantum state

        Usage:

        .. code-block:: python
            :linenos:

            rho = StateTomography.fit(qp=qp, qc=qc)
            rho = StateTomography.fit(qp=qp, qc=qc, method='inverse')
            rho = StateTomography.fit(qp=qp, qc=qc, method='lstsq', shots=4096)

        **Examples**

            >>> import QCompute
            >>> from qcompute_qep.quantum.pauli import operator_to_ptm, complete_pauli_basis
            >>> from qcompute_qep.utils.circuit import circuit_to_state
            >>> from qcompute_qep.quantum.metrics import state_fidelity
            >>> qp = QCompute.QEnv()
            >>> qp.Q.createList(2)
            >>> QCompute.H(qp.Q[0])
            >>> QCompute.CX(qp.Q[0], qp.Q[1])
            >>> qc = QCompute.BackendName.LocalBaiduSim2
            >>> st = StateTomography()
            >>> st.fit(qp, qc, method='inverse', shots=4096, ptm='False')
            >>> print('Fidelity the Bell state: F = {:.5f}'.format(st.fidelity))
            Fidelity the Bell state: F = 1.00000
            >>> # You can also perform quantum state tomography on other qubits, for example [q1, q2]
            >>> qubits = [1, 2]
            >>> st.fit(qp, qc, qubits=[1, 2], method='lstsq', shots=4096, ptm=False)
            >>> print('Fidelity of the Bell state on qubits {}: F = {:.5f}'.format(qubits, st.fidelity))
            Fidelity of the Bell state in qubits [1, 2]: F = 1.00000
        """
        # Parse the arguments. If not set, use the default arguments set by the init function
        self._qp = qp if qp is not None else self._qp
        self._qc = qc if qc is not None else self._qc
        self._method = kwargs.get('method', self._method)
        self._shots = kwargs.get('shots', self._shots)
        self._ptm = kwargs.get('ptm', self._ptm)
        self._qubits = kwargs.get('qubits', self._qubits)
        # Set up the measurement basis for state tomography
        self._basis = init_measurement_basis(kwargs.get('basis', self._basis))

        # If the quantum program or the quantum computer is not set, the state tomography cannot be executed
        if self._qp is None:
            raise ArgumentError("StateTomography.fit(): the quantum program is not set!")
        if self._qc is None:
            raise ArgumentError("StateTomography.fit(): the quantum computer is not set!")
        if self._qubits is None:
            self._qubits = list(range(number_of_qubits(qp)))
        # Check if the indices in self._qubits is unique
        if len(set(self._qubits)) != len(self._qubits):
            raise ArgumentError("StateTomography.fit(): the input qubits are not repeatable!")
        # Check if the number of qubits in @qp and @qubits are equal
        if len(self._qubits) != number_of_qubits(qp):
            raise ArgumentError("StateTomography.fit(): the number of qubits in '@qp' "
                                "must be equal to the number of qubits in '@qubits'!")

        # Number of qubits in the quantum program
        n = len(self._qubits)

        # Step 1. Construct a list of tomographic quantum circuits from the quantum program
        pbar = tqdm(total=100, desc='QST Step 1/3: Constructing quantum circuits ...')
        pbar.update(0)
        tomo_qps, tomo_obs = self._basis.meas_circuits(self._qp)

        # Map to target quantum qubits we actually want to tomography
        tomo_qps = [map_qubits(qp=tomo_qp, qubits=self._qubits) for tomo_qp in tomo_qps]

        # Step 2. run the tomographic quantum circuits on the quantum computer and estimate the expectation values
        ptm: List[float] = []
        pbar.desc = "QST Step 2/3: Running quantum circuits which is time consuming ..."
        pbar.update(100 / 3)

        # Run the quantum circuits in batch and fetch back the results
        counts_list = execute(qp=tomo_qps, qc=self._qc, **kwargs)

        for i, counts in enumerate(counts_list):
            expval = expval_from_counts(A=tomo_obs[i], counts=counts)
            ptm.append(expval)

        # Step 3. perform the fitting procedure to estimate the quantum state
        pbar.desc = "QST Step 3/3: Processing experimental data ..."
        pbar.update(100 / 3)
        # Obtain the transition matrix for the measurement basis
        tm = self._basis.transition_matrix(n)
        if self._method.lower() == 'inverse':  # The naive inverse method
            # Compute the pseudo-inverse of the transition matrix
            tm_inv = la.pinv(tm)
            rho_ptm = np.dot(tm_inv, np.asarray(ptm))
        elif self._method.lower() == 'lstsq':  # the ordinary least square method
            rho_ptm = la.pinv(dagger(tm) @ tm) @ dagger(tm) @ np.asarray(ptm)
        elif self._method.lower() == 'mle':  # the maximum likelihood estimation
            raise ArgumentError("StateTomography.fit(), unsupported tomography method '{}'".format(self._method))
        else:
            raise ArgumentError("StateTomography.fit(), unsupported tomography method '{}'".format(self._method))

        pbar.desc = "QST successfully finished!"
        pbar.update(100 - pbar.n)
        pbar.close()

        # Record noisy quantum state
        self._noisy_state = ptm_to_operator(rho_ptm)

        # Return noisy quantum state
        if self._ptm is False:
            return self._noisy_state
        else:
            return rho_ptm

    @property
    def ideal_state(self):
        r"""Ideal state in matrix form from the given quantum program.
        """
        if self._qp is None:
            raise ArgumentError("The quantum program characterizing the quantum state is not given!")

        return circuit_to_state(self._qp)

    @property
    def noisy_state(self):
        r"""Noisy state in matrix form obtained via state tomography.
        """
        if self._noisy_state is None:
            raise ArgumentError("Run quantum state tomography first to obtain the noisy state!")
        else:
            return self._noisy_state

    @property
    def fidelity(self):
        r"""Compute the fidelity between ideal and noisy states.
        """
        fid = state_fidelity(self.ideal_state, self.noisy_state)
        return fid if fid < 1.0 else 1.0
