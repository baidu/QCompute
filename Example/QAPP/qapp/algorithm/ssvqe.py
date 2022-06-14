# -*- coding: UTF-8 -*-
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
Subspace-Search Variational Quantum Eigensolver
"""

from typing import List, Union
import numpy as np
from qapp.circuit import BasisEncodingCircuit
from qapp.circuit import ParameterizedCircuit
from qapp.circuit import PauliMeasurementCircuit
from qapp.circuit import PauliMeasurementCircuitWithAncilla
from qapp.circuit import SimultaneousPauliMeasurementCircuit
from qapp.optimizer import BasicOptimizer


class SSVQE:
    """Subspace-Search Variational Quantum Eigensolver class

    Please see https://journals.aps.org/prresearch/abstract/10.1103/PhysRevResearch.1.033062 for details on this algorithm.
    """

    def __init__(self, num: int, ex_num: int, hamiltonian: List, ansatz: ParameterizedCircuit,
                 optimizer: BasicOptimizer, backend: str, measurement: str = 'default'):
        """The constructor of the SSVQE class

        :param num: Number of qubits
        :param ex_num: Number of extra eignevalues to be solved. When ex_num = 0, only compute the minimum eigenvalue
        :param hamiltonian: Hamiltonian whose eigenvalues are to be solved
        :param ansatz: Ansatz used to search for the eigenstates of the Hamiltonian
        :param optimizer: Optimizer used to optimize the parameters in the ansatz
        :param backend: Backend to be used in this task. Please refer to https://quantum-hub.baidu.com/quickGuide for details
        :param measurement: Method chosen from 'default', 'ancilla', and 'SimMeasure' for measuring the expectation value, defaults to 'default'
        """
        self._num = num
        self._subspace_basis = [bin(index)[2:].zfill(self._num) for index in range(ex_num + 1)]
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._optimizer = optimizer
        self._backend = backend
        self._measurement = measurement
        if measurement == 'default':
            self._measurement_circuit = PauliMeasurementCircuit
        elif measurement == 'ancilla':
            self._measurement_circuit = PauliMeasurementCircuitWithAncilla
        elif measurement == 'SimMeasure':
            self._measurement_circuit = SimultaneousPauliMeasurementCircuit
        else:
            raise ValueError('Invalid measurement method!')
        self._minimum_eigenvalues = "Run SSVQE.run() first"

    def _pauli_expectation(self, position_string: str, shots: int) -> float:
        """Returns the expectation value of the Hamiltonian

        :param shots: Number of measurement shots
        :return: Expectation value of the Hamiltonian
        """
        state_prep = BasisEncodingCircuit(self._num, position_string)
        measurement_circuit = self._measurement_circuit(self._num, self._hamiltonian)
        expectation = measurement_circuit.get_expectation([state_prep, self._ansatz], shots, self._backend)

        return expectation

    def _compute_gradient(self, parameters: np.ndarray, shots: int) -> np.ndarray:
        """Computes gradient by the parameter shift rule

        :param parameters: Current parameters of the ansatz
        :param shots: Number of measurement shots
        """
        gradient = np.zeros_like(parameters)
        for i in range(len(parameters)):
            param_plus = parameters.copy()
            param_minus = parameters.copy()
            param_plus[i] += np.pi / 2
            param_minus[i] -= np.pi / 2
            loss_plus = self._compute_loss(param_plus, shots)
            loss_minus = self._compute_loss(param_minus, shots)
            gradient[i] = ((loss_plus - loss_minus) / 2)
        self._ansatz.set_parameters(parameters)

        return gradient

    def _compute_loss(self, parameters: np.ndarray, shots: int) -> float:
        """Computes loss

        :param parameters: Current parameters of the ansatz
        :param shots: Number of measurement shots
        """
        self._ansatz.set_parameters(parameters)
        loss = sum([(i + 1) * self._pauli_expectation(init_state, shots=shots)
                    for i, init_state in enumerate(self._subspace_basis)])

        return loss

    def get_gradient(self, shots: int = 1024) -> np.ndarray:
        """Calculates the gradient with respect to current parameters in circuit

        :param shots: Number of measurement shots, defaults to 1024
        :return: Gradient with respect to current parameters
        """
        curr_param = self._ansatz.parameters
        gradient = self._compute_gradient(curr_param, shots)

        return gradient

    def get_loss(self, shots: int = 1024) -> float:
        """Calculates the loss with respect to current parameters in circuit

        :param shots: Number of measurement shots, defaults to 1024
        :return: Loss with respect to current parameters
        """
        loss = 0
        loss = sum([(i + 1) * self._pauli_expectation(init_state, shots=shots)
                    for i, init_state in enumerate(self._subspace_basis)])

        return loss

    def run(self, shots: int = 1024):
        """Searches for the minimum eigenvalue of the input Hamiltonian with the given ansatz and optimizer

        :param shots: Number of measurement shots, defaults to 1024
        """
        self._optimizer.minimize(shots, self._compute_loss, self._compute_gradient)
        self._minimum_eigenvalues = [self._pauli_expectation(init_state, shots=shots) for init_state in self._subspace_basis]

    @property
    def minimum_eigenvalues(self) -> Union[str, List]:
        """The optimized minimum eigenvalue from last run

        :return: Optimized minimum eigenvalues from last run
        """

        return self._minimum_eigenvalues

    def set_backend(self, backend: str):
        """Sets the backend to be used

        :param backend: Backend to be used
        """
        self._backend = backend
