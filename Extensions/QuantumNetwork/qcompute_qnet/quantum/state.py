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
Module for quantum states.
"""

from argparse import ArgumentTypeError
from typing import List, Any
import numpy
from numpy import conj, random, trace, sqrt, reshape, transpose, real, square, abs
from qcompute_qnet import EPSILON
from qcompute_qnet.quantum.gate import Gate

__all__ = [
    "QuantumState",
    "PureState",
    "MixedState",
    "Zero",
    "One",
    "Plus",
    "Minus"
]


class QuantumState:
    r"""Class for creating a quantum state.

    Attributes:
        state (numpy.ndarray): matrix representation of the quantum state
        systems (list): system labels of the quantum state
    """

    def __init__(self, state=None, systems=None):
        r"""Constructor for QuantumState class.

        Args:
            state (numpy.ndarray): initial quantum state
            systems (list): system labels of the quantum state
        """
        self.state = numpy.array([[1]]) if state is None else state
        self.systems = [] if systems is None else systems
        if len(self.systems) != self.size:
            raise ArgumentTypeError("The size of the quantum state and its systems do not match.")

    @property
    def length(self) -> int:
        r"""The length of the matrix of the quantum state.

        Returns:
            int: length of the matrix
        """
        return max(self.state.shape)

    @property
    def size(self) -> int:
        r"""The size of the quantum state.

        Returns:
            int: size of the quantum state
        """
        return int(numpy.log2(self.length))

    def init(self, state: numpy.ndarray) -> None:
        r"""Quantum state initialization.

        Args:
            state (numpy.ndarray): initial quantum state
        """
        self.state = state

    def reset(self) -> None:
        r"""Reset the quantum state.
        """
        self.state = numpy.array([[1]])
        self.systems = []


class PureState(QuantumState):
    r"""Class for pure quantum states.

    Note:
        The matrix representation of a pure state is a column vector.
    """

    def __init__(self, state=None, systems=None):
        r"""Constructor for PureState class.

        Args:
            state (numpy.ndarray): initial quantum state
            systems (list): system labels of the quantum state
        """
        super().__init__(state, systems)
        if len(self.state.shape) != 2:
            raise ArgumentTypeError("The matrix representation of the state is not of two dimensional.")
        elif self.state.shape[1] > 1:
            raise ArgumentTypeError("The matrix shape is not correct!")

    @property
    def ket(self) -> numpy.ndarray:
        r"""Return the ket form of the pure quantum state.

        Returns:
            numpy.ndarray: the ket form of the pure quantum state
        """
        return self.state

    @property
    def bra(self) -> numpy.ndarray:
        r"""Return the bra form of the pure quantum state.

        Returns:
            numpy.ndarray: the bra form of the pure quantum state
        """
        return conj(self.state).T

    @property
    def norm(self):
        r"""Return the norm of the pure quantum state.

        Returns:
            numpy.ndarray: the norm of the pure quantum state
        """
        return sqrt(self.bra @ self.ket)

    @property
    def projector(self) -> numpy.ndarray:
        r"""Transform the state vector to a projector.

        Returns:
            numpy.ndarray: projector of the state vector
        """
        return self.ket @ self.bra

    def evolve(self, which_qubit: list, operator: numpy.ndarray) -> None:
        r"""Evolve a quantum state on a given qubit by the given operator.

        Warning:
            The operator to act should be a two-dimensional square matrix such as a unitary gate matrix.
            The systems to act on should match the dimension of the operator.

        Args:
            which_qubit (list): a list of qubit systems to act on
            operator (numpy.ndarray): operator to act on
        """
        assert isinstance(which_qubit, list), f"Input {which_qubit} should a list."
        assert set(which_qubit).issubset(self.systems), \
            f"Input {which_qubit} does not match the quantum system {self.systems}."
        assert len(operator.shape) == 2, f"Input {operator} should be a two-dimensional matrix."
        assert operator.shape[0] == operator.shape[1], f"Input {operator} is not a square matrix."
        assert 2 ** len(which_qubit) == operator.shape[0], \
            f"Input {which_qubit} does not match the dimension of {operator}."

        self.permute_systems(which_qubit)  # permute the target systems to the front
        shape = [2 ** len(which_qubit), 2 ** (self.size - len(which_qubit))]
        self.state = reshape(operator @ reshape(self.state, shape), [self.length, 1])

    def evolve_by_gates(self, gate_history: list) -> None:
        r"""Evolve the quantum state by a list of gates.

        Note:
            This aims to get the quantum output. So the measurement in the list is skipped.

        Args:
            gate_history (list): a list of quantum gates
        """
        to_matrix = {
            's': Gate.S, 't': Gate.T, 'h': Gate.H, 'x': Gate.X, 'y': Gate.Y, 'z': Gate.Z,
            'u3': Gate.U3, 'rx': Gate.Rx, 'ry': Gate.Ry, 'rz': Gate.Rz,
            'cnot': Gate.CNOT, 'cz': Gate.CZ, 'swap': Gate.SWAP
        }

        for gate in gate_history:
            if gate["name"] in ['s', 't', 'h', 'x', 'y', 'z']:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]]())
            elif gate["name"] in ['rx', 'ry', 'rz']:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]](gate["params"]))
            elif gate["name"] in ['u3']:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]](*gate["params"]))
            elif gate["name"] in ['cnot', 'cz', 'swap']:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]]())
            elif gate["name"] in ['m']:
                continue

    def sample(self, shots: int) -> dict:
        r"""Sample from the quantum state with a given number of shots.

        This is equivalent to performing quantum measurement in the computational basis.

        Warning:
            Please be aware of the system order and permute the quantum state first if necessary.

        Args:
            shots: the number of samples

        Returns:
            dict: sampling results
        """
        samples = list(random.choice(list(range(self.length)), shots, p=square(abs(self.state)).T[0]))
        counts = {f'{key:0{self.size}b}': samples.count(key) for key in set(samples)}

        return counts

    def measure(self, basis: numpy.ndarray) -> int:
        r"""Measure the quantum state with a given basis.

        Args:
            basis (numpy.ndarray): basis vectors

        Returns:
            int: measurement outcome
        """
        raise NotImplementedError

    def permute_to_front(self, system: Any) -> None:
        r"""Permute a system of the quantum state to the front.

        Args:
            system (any): system to permute
        """
        if system not in self.systems:
            raise ArgumentTypeError(f"The system to permute must be one of the state systems.")

        system_idx = self.systems.index(system)

        if system_idx == 0:
            pass
        elif system_idx == self.size - 1:  # last system
            new_shape = [2 ** (self.size - 1), 2]
            new_axis = [1, 0]
            new_systems = [system] + self.systems[: system_idx]
            self.state = reshape(transpose(reshape(self.state, new_shape), new_axis), [self.length, 1])
            self.systems = new_systems
        else:  # middle system
            new_shape = [2 ** system_idx, 2, 2 ** (self.size - system_idx - 1)]
            new_axis = [1, 0, 2]
            new_systems = [system] + self.systems[: system_idx] + self.systems[system_idx + 1:]
            self.state = reshape(transpose(reshape(self.state, new_shape), new_axis), [self.length, 1])
            self.systems = new_systems

    def permute_systems(self, systems: list):
        r"""Permute the system of quantum state to a given system order.

        Args:
            systems (list): target system order
        """
        for system in reversed(systems):
            self.permute_to_front(system)

    def compare_by_vector(self, other: "PureState") -> float:
        r"""Compare two pure quantum states by their state vectors.

        Args:
            other (PureState): the other pure quantum state to compare

        Returns:
            float: norm difference
        """
        if self.size != other.size:
            raise ArgumentTypeError(f"The two states to compare must have the same size.")

        if not (set(self.systems).issubset(other.systems) and set(other.systems).issubset(self.systems)):
            raise ArgumentTypeError(f"The two states to compare have different system labels,\n"
                                    f"{self.systems}\n"
                                    f"{other.systems}.")

        if abs(self.norm - 1) >= EPSILON:
            raise ArgumentTypeError(f"{self} is not normalized.")
        elif abs(other.norm - 1) >= EPSILON:
            raise ArgumentTypeError(f"{other} is not normalized.")
        else:
            self.permute_systems(other.systems)  # match the system order
            self_state_list = list(self.state)
            # Find an index with the largest absolute value
            idx = self_state_list.index(max(self_state_list, key=abs))
            if abs(other.state[idx]) <= EPSILON:
                error = 1
            else:
                # Calculate the relative phase and erase it
                phase = self.state[idx] / other.state[idx]
                self_phase = self.state / phase
                error = numpy.linalg.norm(self_phase - other.state)
        return error

    def compare_by_density(self, other: "PureState") -> float:
        r"""Compare two pure quantum states by their density matrices.

        Args:
            other (PureState): the second quantum state
        """
        if self.size != other.size:
            raise ArgumentTypeError(f"The two states to compare must have the same size.")

        if not (set(self.systems).issubset(other.systems) and set(other.systems).issubset(self.systems)):
            raise ArgumentTypeError(f"The two states to compare have different system labels,\n"
                                    f"{self.systems}\n"
                                    f"{other.systems}.")

        self.permute_systems(other.systems)  # match the system order
        error = numpy.linalg.norm(self.projector - other.projector)

        return error

    @classmethod
    def random_state_vector(cls, qubit_number: int, is_real=False) -> numpy.ndarray:
        r"""Generate a random pure quantum state vector with given number of qubits.

        Args:
            qubit_number (int): number of qubits
            is_real (int, optional): whether to generate a real matrix

        Returns:
            numpy.ndarray: state vector
        """
        if not isinstance(qubit_number, int):
            raise ArgumentTypeError(f"Input {qubit_number} should be an int value.")

        if not isinstance(is_real, bool):
            raise ArgumentTypeError(f"Input {is_real} should be a bool value.")

        if is_real:
            psi = random.randn(2 ** qubit_number, 1)
            inner_prod = conj(psi).T @ psi
        else:
            psi = random.randn(2 ** qubit_number, 1) + 1j * random.randn(2 ** qubit_number, 1)
            inner_prod = real(conj(psi).T @ psi)

        psi = psi / sqrt(inner_prod)  # normalize the vector
        return psi


class MixedState(QuantumState):
    r"""Class for mixed quantum states.
    """

    def __init__(self, state=None, systems=None):
        r"""Constructor for MixedState class.

        Args:
            state (numpy.ndarray): initial quantum state
            systems (list): system labels of the quantum state
        """
        super().__init__(state, systems)

    def evolve(self, kraus_list: List[numpy.ndarray]) -> None:
        r"""Evolution of the quantum state.

        Args:
            kraus_list (List[numpy.ndarray]): kraus operators acting on the quantum state
        """
        raise NotImplementedError

    def measure(self, basis: numpy.ndarray) -> int:
        r"""Measure the quantum state with a given basis.

        Warning:
            We only consider single-qubit state measurement in this version.

        Args:
            basis (numpy.ndarray): measurement basis

        Returns:
            int: measurement outcome
        """
        prob = [trace(item @ conj(item).T @ self.state).real for item in basis]

        if random.random_sample() < prob[0]:
            self.state = basis[0] @ conj(basis[0]).T
            return 0
        else:
            self.state = basis[1] @ conj(basis[1]).T
            return 1


class Zero:
    r"""Class to obtain a zero state.
    """

    SV = numpy.array([[1], [0]], dtype=complex)
    DM = numpy.array([[1, 0], [0, 0]], dtype=complex)


class One:
    r"""Class to obtain a one state.
    """

    SV = numpy.array([[0], [1]], dtype=complex)
    DM = numpy.array([[0, 0], [0, 1]], dtype=complex)


class Plus:
    r"""Class to obtain a plus state.
    """

    SV = numpy.array([[1], [1]], dtype=complex) / sqrt(2.0)
    DM = numpy.array([[1, 1], [1, 1]], dtype=complex) / 2


class Minus:
    r"""Class to obtain a minus state.
    """

    SV = numpy.array([[1], [-1]], dtype=complex) / sqrt(2.0)
    DM = numpy.array([[1, -1], [-1, 1]], dtype=complex) / 2
