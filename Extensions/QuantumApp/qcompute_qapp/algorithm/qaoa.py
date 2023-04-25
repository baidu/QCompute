# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2021 Baidu, Inc. All Rights Reserved.
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
Quantum Approximate Optimization Algorithm
"""

from typing import Union
import numpy as np
from QCompute.QPlatform.QEnv import QEnv
from QCompute.QPlatform.QOperation.Measure import MeasureZ
from ..circuit import QAOAAnsatz
from ..circuit import PauliMeasurementCircuit
from ..circuit import PauliMeasurementCircuitWithAncilla
from ..circuit import SimultaneousPauliMeasurementCircuit
from ..optimizer import BasicOptimizer


class QAOA:
    r"""
    Quantum Approximate Optimization Algorithm class
    """
    def __init__(
            self, num: int, hamiltonian: list, ansatz: QAOAAnsatz, optimizer: BasicOptimizer,
            backend: str, measurement: str = 'default', delta: float = 0.1
    ):
        r"""The constructor of the QAOA class

        Args:
            num (int): Number of qubits
            hamiltonian (list): Hamiltonian used to construct the QAOA ansatz
            ansatz (QAOAAnsatz): QAOA ansatz used to search for the maximum eigenstate of the Hamiltonian
                optimizer (BasicOptimizer): Optimizer used to optimize the parameters in the ansatz
            backend (str): Backend to be used in this task. Please refer to https://quantum-hub.baidu.com/quickGuide
                for details
            measurement (str): Method chosen from 'default', 'ancilla', and 'SimMeasure' for measuring the expectation
                value, defaults to 'default'
            delta (float): Parameter used to calculate gradients, defaults to 0.1

        """
        self._num = num
        self._hamiltonian = hamiltonian
        self._ansatz = ansatz
        self._optimizer = optimizer
        self._backend = backend
        self._measurement = measurement
        self._delta = delta
        if measurement == 'default':
            self._measurement_circuit = PauliMeasurementCircuit
        elif measurement == 'ancilla':
            self._measurement_circuit = PauliMeasurementCircuitWithAncilla
        elif measurement == 'SimMeasure':
            self._measurement_circuit = SimultaneousPauliMeasurementCircuit
        else:
            raise ValueError('Error EA01003(QAPP): Invalid measurement method!')
        self._maximum_eigenvalue = "Run QAOA.run() first"

    def _pauli_expectation(self, shots: int) -> float:
        r"""Returns the expectation value of the Hamiltonian

        Args:
            shots (int): Number of measurement shots

        Returns:
            float: Expectation value of the Hamiltonian

        """
        measurement_circuit = self._measurement_circuit(self._num, self._hamiltonian)
        expectation = measurement_circuit.get_expectation([self._ansatz], shots, self._backend)

        return expectation

    def get_measure(self, shots: int = 1024) -> dict:
        r"""Returns the measurement results

        Args:
            shots (int): Number of measurement shots, defaults to 1024

        Returns:
            float : Measurement results in bitstrings with the number of counts

        """
        env = QEnv()
        env.backend(self._backend)
        q = env.Q.createList(self._num)
        # Add circuit
        self._ansatz.add_circuit(q)
        # Measurement
        MeasureZ(q, range(self._num))
        counts = env.commit(shots, fetchMeasure=True)['counts']

        return counts

    def _compute_gradient(self, parameters: np.ndarray, shots: int) -> np.ndarray:
        r"""Computes gradient by the finite-difference method

        Args:
            parameters (np.ndarray): Current parameters of the ansatz
            shots (int): Number of measurement shots

        Returns:
            np.ndarray: gradient of parameters

        """
        gradient = np.zeros_like(parameters)
        for i in range(len(parameters)):
            param_plus = parameters.copy()
            param_minus = parameters.copy()
            param_plus[i] += self._delta / 2
            param_minus[i] -= self._delta / 2
            loss_plus = self._compute_loss(param_plus, shots)
            loss_minus = self._compute_loss(param_minus, shots)
            gradient[i] = ((loss_plus - loss_minus) / self._delta)
        self._ansatz.set_parameters(parameters)

        return gradient

    def _compute_loss(self, parameters: np.ndarray, shots: int) -> float:
        r"""Computes loss

        Args:
            parameters (np.ndarray): Current parameters of the ansatz
            shots (int): Number of measurement shots

        Returns:
            float: loss

        """
        self._ansatz.set_parameters(parameters)
        loss = -self._pauli_expectation(shots=shots)

        return loss

    def get_gradient(self, shots: int = 1024) -> np.ndarray:
        r"""Calculates the gradient with respect to current parameters in circuit

        Args:
            shots (int): Number of measurement shots, defaults to 1024

        Returns:
            np.ndarray: Gradient with respect to current parameters

        """
        curr_param = self._ansatz.parameters
        gradient = self._compute_gradient(curr_param, shots)

        return gradient

    def get_loss(self, shots: int = 1024) -> float:
        r"""Calculates the loss with respect to current parameters in circuit

        Args:
            shots (int): Number of measurement shots, defaults to 1024

        Returns:
            float: Loss with respect to current parameters

        """
        loss = -self._pauli_expectation(shots=shots)

        return loss

    def run(self, shots: int = 1024) -> None:
        r"""Searches for the maximum eigenvalue of the input Hamiltonian with the given ansatz and optimizer

        Args:
            shots (int): Number of measurement shots, defaults to 1024

        """
        self._optimizer.minimize(shots, self._compute_loss, self._compute_gradient)
        self._maximum_eigenvalue = -self._optimizer._loss_history[-1]

    @property
    def maximum_eigenvalue(self) -> Union[str, float]:
        r"""The optimized maximum eigenvalue from last run

        Returns:
            Union[str, float]: Optimized maximum eigenvalue from last run

        """

        return self._maximum_eigenvalue

    def set_backend(self, backend: str) -> None:
        r"""Sets the backend to be used

        Args:
            backend: Backend to be used

        """
        self._backend = backend
