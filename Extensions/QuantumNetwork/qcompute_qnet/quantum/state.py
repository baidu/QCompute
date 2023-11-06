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
from math import pi
from typing import List, Any, Tuple
import numpy
from numpy import conj, random, trace, sqrt, reshape, transpose, real, square, abs

from Extensions.QuantumNetwork.qcompute_qnet import EPSILON
from Extensions.QuantumNetwork.qcompute_qnet.quantum.gate import Gate
from Extensions.QuantumNetwork.qcompute_qnet.quantum.noise import Noise
from Extensions.QuantumNetwork.qcompute_qnet.quantum.utils import kron, dagger, to_projector, to_superoperator

__all__ = ["QuantumState", "PureState", "MixedState", "Zero", "One", "Plus", "Minus"]


class QuantumState:
    r"""Class for creating a quantum state.

    Attributes:
        substates (List[QuantumState.SubState]): a tensor product decomposition of the quantum state
        outcome (dict): a dict that records all measurement outcomes

    Note:
        The ``substates`` attribute is a tensor product decomposition of a quantum state. It will be used in
        the actual computational steps to save resources. The ``outcome`` is a dict that records all measurement
        outcomes, whose keys are measurement IDs and values are the outcomes.
    """

    def __init__(self, matrix=None, systems=None, substates=None):
        r"""Constructor for QuantumState class.

        Args:
            matrix (numpy.ndarray, optional): matrix representation of the quantum state
            systems (list, optional): system labels of the quantum state
            substates (List[QuantumState.SubState], optional): a tensor product decomposition of the quantum state
        """
        self.substates = None
        self.init(matrix, systems, substates)
        self.outcome = {}

    def init(self, matrix=None, systems=None, substates=None) -> None:
        r"""Quantum state initialization.

        Args:
            matrix (numpy.ndarray, optional): matrix representation of the quantum state
            systems (list, optional): system labels of the quantum state
            substates (List[QuantumState.SubState], optional): a tensor product decomposition of the quantum state
        """
        if substates is not None:
            if not all(isinstance(substate, self.SubState) for substate in substates):
                raise ArgumentTypeError("All the substates should be of 'SubState' type!")
            # Check if substates are on different systems
            all_systems = sum([substate.systems for substate in substates], [])
            if len(set(all_systems)) != len(all_systems):
                raise ArgumentTypeError("The substates have overlapped systems!")
            self.substates = substates
        else:
            matrix = numpy.array([[1]]) if matrix is None else matrix
            systems = [] if systems is None else systems
            self.substates = [self.SubState(matrix=matrix, systems=systems)]
        # Check matrix shape
        if len(self.matrix.shape) != 2:
            raise ArgumentTypeError("The matrix representation of the state is not of two dimensional.")

    @property
    def matrix(self) -> numpy.ndarray:
        r"""The matrix representation of the quantum state.

        Returns:
            numpy.ndarray: matrix representation of the quantum state
        """
        return kron([substate.matrix for substate in self.substates])

    @property
    def systems(self) -> list:
        r"""The system labels of the quantum state.

        Returns:
            list: system labels of the quantum state
        """
        return sum([substate.systems for substate in self.substates], [])

    @property
    def length(self) -> int:
        r"""The length of the matrix of the quantum state.

        Returns:
            int: length of the matrix
        """
        return max(self.matrix.shape)

    @property
    def size(self) -> int:
        r"""The size of the quantum state.

        Returns:
            int: size of the quantum state
        """
        return int(numpy.log2(self.length))

    def merge_substates(self, substates: list) -> "QuantumState.SubState":
        r"""Merge the matrices and systems of the given substates to a new substate.

        Args:
            substates (list): substates to merge

        Returns:
            QuantumState.SubState: merged substate
        """
        matrix = kron([substate.matrix for substate in substates])
        systems = sum([substate.systems for substate in substates], [])
        return self.SubState(matrix=matrix, systems=systems)

    def sum_outcomes(self, mids: list, add_number=None) -> int:
        r"""Sum the measurement outcome with given IDs.

        Args:
            mids (list): a list of measurement IDs
            add_number (int, optional): extra number to add to the summation

        Returns:
            int: summation result
        """
        if add_number is None:
            add_number = 0
        else:
            if not isinstance(add_number, int):
                raise ArgumentTypeError(f"Input {add_number} should be an int value.")

        return add_number if mids == [] else sum([self.outcome[mid] for mid in mids], add_number)

    class SubState:
        r"""Class for creating a substate.

        Attributes:
            matrix (numpy.ndarray): matrix representation of the substate
            systems (list): system labels of the substate

        Note:
            The ``SubState`` class can be regarded as a subcomponent of ``QuantumState``. It stores part of the tensor
            product decomposition of a quantum state and is used in the actual computational steps to save resources.
        """

        def __init__(self, matrix=None, systems=None):
            r"""Constructor for SubState class.

            Args:
                matrix (numpy.ndarray, optional): matrix representation of the substate
                systems (list, optional): system labels of the substate
            """
            self.matrix = numpy.array([[1]]) if matrix is None else matrix
            self.systems = [] if systems is None else systems
            if len(self.systems) != self.size:
                raise ArgumentTypeError("The size of the matrix and the systems do not match.")

        @property
        def length(self) -> int:
            r"""The length of the matrix of the substate.

            Returns:
                int: length of the matrix
            """
            return max(self.matrix.shape)

        @property
        def size(self) -> int:
            r"""The size of the substate.

            Returns:
                int: size of the substate
            """
            return int(numpy.log2(self.length))

        def permute_to_front(self, system: Any) -> None:
            r"""Permute a system of the substate to the front.

            Args:
                system (Any): system to permute
            """
            pass

        def permute_systems(self, systems: list) -> None:
            r"""Permute the given systems to the front.

            Args:
                systems (list): target system order
            """
            for system in reversed(systems):
                self.permute_to_front(system)

        def evolve(self, which_qubit: list, operator: Any) -> None:
            r"""Evolve the substate by the given operator.

            Args:
                which_qubit (list): a list of qubit systems to act on
                operator (Any): operator to act on
            """
            pass

        def measure(self, which_qubit: Any, basis: numpy.ndarray) -> Tuple[int, list]:
            r"""Measure the substate with a given basis.

            Args:
                which_qubit (Any): qubit to measure
                basis (numpy.ndarray): basis vectors

            Returns:
                Tuple[int, list]: measurement outcome and the list of substate(s) after the measurement
            """
            pass


class PureState(QuantumState):
    r"""Class for pure quantum states.

    Note:
        The matrix representation of a pure state should be a column vector.
    """

    def __init__(self, matrix=None, systems=None, substates=None):
        r"""Constructor for PureState class.

        Args:
            matrix (numpy.ndarray, optional): matrix representation of the quantum state
            systems (list, optional): system labels of the quantum state
            substates (List[PureState.Substate], optional): a tensor product decomposition of the quantum state
        """
        super().__init__(matrix, systems, substates)
        # Check matrix shape
        if self.matrix.shape[1] > 1:
            raise ArgumentTypeError("The matrix shape is not correct!")

    @property
    def ket(self) -> numpy.ndarray:
        r"""Return the ket form of the pure quantum substate.

        Returns:
            numpy.ndarray: the ket form of the pure quantum substate
        """
        return self.matrix

    @property
    def bra(self) -> numpy.ndarray:
        r"""Return the bra form of the pure quantum substate.

        Returns:
            numpy.ndarray: the bra form of the pure quantum substate
        """
        return dagger(self.matrix)

    @property
    def norm(self) -> numpy.ndarray:
        r"""Return the norm of the pure quantum substate.

        Returns:
            numpy.ndarray: norm of the pure quantum substate
        """
        return sqrt(self.bra @ self.ket)

    @property
    def projector(self) -> numpy.ndarray:
        r"""Transform the state vector to a projector.

        Returns:
            numpy.ndarray: projector of the state vector
        """
        return to_projector(self.matrix)

    def is_normalized(self) -> bool:
        r"""Check if the pure quantum state is normalized.

        Returns:
            bool: whether the pure quantum state is normalized
        """
        return abs(self.norm - 1) < EPSILON

    def check_operator(self, which_qubit: list, operator: numpy.ndarray) -> None:
        r"""Check the validity of the operator.

        Args:
            which_qubit (list): a list of qubit systems to act on
            operator (numpy.ndarray): operator to act on the pure quantum state
        """
        assert isinstance(which_qubit, list), f"Input {which_qubit} should be a list."
        assert set(which_qubit).issubset(
            self.systems
        ), f"Input {which_qubit} does not match the quantum system {self.systems}."
        assert len(operator.shape) == 2, f"Input {operator} should be a two-dimensional matrix."
        assert operator.shape[0] == operator.shape[1], f"Input {operator} is not a square matrix."
        assert (
            2 ** len(which_qubit) == operator.shape[0]
        ), f"Input {which_qubit} does not match the dimension of {operator}."

    def evolve(self, which_qubit: list, operator: numpy.ndarray) -> None:
        r"""Evolve the pure quantum state by the given operator.

        Args:
            which_qubit (list): a list of qubit systems to act on
            operator (numpy.ndarray): operator to act on

        Warning:
            The operator to act should be a two-dimensional square matrix such as a unitary matrix.
            The systems to act on should match the dimension of the operator.
        """
        # Check validity of the operator
        self.check_operator(which_qubit, operator)

        # Find the relevant substates involved in the computation
        relevant_substates = []
        other_substates = []
        for substate in self.substates:
            if set(substate.systems) & set(which_qubit):  # if there is an overlap
                relevant_substates.append(substate)
            else:
                other_substates.append(substate)

        # Merge the substates and compute
        merged_state = self.merge_substates(relevant_substates)
        merged_state.evolve(which_qubit, operator)
        other_substates.append(merged_state)
        self.substates = other_substates

    def measure(self, which_qubit: Any, basis: numpy.ndarray, mid: Any) -> int:
        r"""Measure the pure quantum state with a given basis.

        Args:
            which_qubit (Any): qubit to measure
            basis (numpy.ndarray): basis vectors
            mid (Any): measurement ID that is used to fetch the outcome

        Returns:
            int: measurement outcome
        """
        # Pick relevant substate
        relevant_substate = []
        for substate in self.substates:
            if which_qubit in substate.systems:
                relevant_substate = substate
                self.substates.remove(substate)
                break

        result, post_substate = relevant_substate.measure(which_qubit, basis)
        self.outcome[mid] = result  # save the measurement outcome
        self.substates = sum([self.substates, post_substate], [])

        return result

    def reset(self, system: Any, matrix=None) -> None:
        r"""Reset a system to a given matrix.

        Args:
            system (Any): qubit to reset state
            matrix (numpy.ndarray, optional): matrix to reset (zero state by default)
        """
        for substate in self.substates:
            if system in substate.systems:
                if len(substate.systems) > 1:  # non-isolated systems
                    raise ArgumentTypeError(f"The system to reset seems to be correlated with others!")
                elif len(substate.systems) == 1:  # isolated systems
                    substate.matrix = Zero.SV if matrix is None else matrix  # reset the quantum state to zero state
                    break

    def evolve_by_gates(self, gate_history: list) -> None:
        r"""Evolve the pure quantum state by a list of gates.

        Args:
            gate_history (list): a list of quantum gates
        """
        to_matrix = {
            "id": Gate.I,
            "s": Gate.S,
            "t": Gate.T,
            "h": Gate.H,
            "x": Gate.X,
            "y": Gate.Y,
            "z": Gate.Z,
            "u": Gate.U,
            "u3": Gate.U3,
            "rx": Gate.Rx,
            "ry": Gate.Ry,
            "rz": Gate.Rz,
            "cx": Gate.CNOT,
            "cz": Gate.CZ,
            "swap": Gate.SWAP,
            "bit_flip": Noise.BitFlip,
            "phase_flip": Noise.PhaseFlip,
            "bit_phase_flip": Noise.BitPhaseFlip,
            "amplitude_damping": Noise.AmplitudeDamping,
            "phase_damping": Noise.PhaseDamping,
            "depolarizing": Noise.Depolarizing,
        }

        for gate in gate_history:
            if gate["name"] in ["id", "s", "t", "h", "x", "y", "z"]:
                if gate.get("condition") is None:  # without classical control
                    self.evolve(gate["which_qubit"], to_matrix[gate["name"]]())
                else:  # classically conditioned gate
                    if self.outcome[gate["condition"]] == 0:
                        pass
                    elif self.outcome[gate["condition"]] == 1:
                        self.evolve(gate["which_qubit"], to_matrix[gate["name"]]())
            elif gate["name"] in ["rx", "ry", "rz"]:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]](gate["angle"]))
            elif gate["name"] in ["u", "u3"]:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]](*gate["angles"]))
            elif gate["name"] in ["cx", "cz", "swap"]:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]]())
            elif gate["name"] == "m":
                from Extensions.QuantumNetwork.qcompute_qnet.quantum.basis import Basis

                angle, plane = gate["basis"]["angle"], gate["basis"]["plane"]
                signal_s = self.sum_outcomes(gate["basis"]["domain_s"])
                signal_t = self.sum_outcomes(gate["basis"]["domain_t"])
                # The adaptive angle is (-1)^{signal_s} * angle + {signal_t} * pi
                adaptive_angle = (-1) ** signal_s * angle + signal_t * pi
                basis = Basis.Plane(plane, adaptive_angle)
                self.measure(gate["which_qubit"][0], basis, gate["mid"])
            elif gate["name"] == "r":
                self.reset(gate["which_qubit"][0], gate["matrix"])
            elif gate["name"] in [
                "bit_flip",
                "phase_flip",
                "bit_phase_flip",
                "phase_damping",
                "amplitude_damping",
                "depolarizing",
            ]:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]](gate["prob"]))
            else:
                raise NotImplementedError

    def sample(self, shots: int) -> dict:
        r"""Sample from the pure quantum state with a given number of shots.

        This is equivalent to performing quantum measurement in the computational basis.

        Args:
            shots (int): the number of samples

        Returns:
            dict: sampling results

        Warning:
            Please be aware of the system order and permute the quantum state first if necessary.
        """
        samples = list(random.choice(list(range(self.length)), shots, p=square(abs(self.matrix)).T[0]))
        counts = {f"{key:0{self.size}b}": samples.count(key) for key in set(samples)}

        return counts

    def compare_by_vector(self, other: "PureState") -> float:
        r"""Compare two pure quantum states by their state vectors.

        Args:
            other (PureState): the other pure quantum state to compare

        Returns:
            float: norm difference
        """
        self_merged_state = self.merge_substates(self.substates)
        other_merged_state = other.merge_substates(other.substates)
        error = self_merged_state.compare_by_vector(other_merged_state)
        return error

    def compare_by_density(self, other: "PureState") -> float:
        r"""Compare two pure quantum substates by their density matrices.

        Args:
            other (PureState): the other pure quantum substate to compare

        Returns:
            float: norm difference
        """
        self_merged_state = self.merge_substates(self.substates)
        other_merged_state = other.merge_substates(other.substates)
        error = self_merged_state.compare_by_density(other_merged_state)
        return error

    def to_mixed_state(self) -> "MixedState":
        r"""Convert the current pure state to a mixed state.

        Returns:
            MixedState: equivalent mixed state
        """
        return MixedState(substates=[substate.to_mixed_substate() for substate in self.substates])

    @classmethod
    def random_state_vector(cls, qubit_number: int, is_real=False) -> numpy.ndarray:
        r"""Generate a random pure quantum state vector with given number of qubits.

        Args:
            qubit_number (int): number of qubits
            is_real (bool, optional): whether to generate a real matrix

        Returns:
            numpy.ndarray: state vector
        """
        if not isinstance(qubit_number, int):
            raise ArgumentTypeError(f"Input {qubit_number} should be an int value.")

        if not isinstance(is_real, bool):
            raise ArgumentTypeError(f"Input {is_real} should be a bool value.")

        if is_real:
            psi = random.randn(2**qubit_number, 1)
            inner_prod = conj(psi).T @ psi
        else:
            psi = random.randn(2**qubit_number, 1) + 1j * random.randn(2**qubit_number, 1)
            inner_prod = real(conj(psi).T @ psi)

        psi = psi / sqrt(inner_prod)  # normalize the vector
        return psi

    class SubState(QuantumState.SubState):
        r"""Class for creating a pure quantum substate."""

        def __init__(self, matrix=None, systems=None):
            r"""Constructor for SubState class.

            Args:
                matrix (numpy.ndarray, optional): matrix representation of the pure quantum substate
                systems (list, optional): system labels of the pure quantum substate
            """
            super().__init__(matrix, systems)

        @property
        def ket(self) -> numpy.ndarray:
            r"""Return the ket form of the pure quantum substate.

            Returns:
                numpy.ndarray: the ket form of the pure quantum substate
            """
            return self.matrix

        @property
        def bra(self) -> numpy.ndarray:
            r"""Return the bra form of the pure quantum substate.

            Returns:
                numpy.ndarray: the bra form of the pure quantum substate
            """
            return dagger(self.matrix)

        @property
        def norm(self) -> numpy.ndarray:
            r"""Return the norm of the pure quantum substate.

            Returns:
                numpy.ndarray: norm of the pure quantum substate
            """
            return sqrt(self.bra @ self.ket)

        @property
        def projector(self) -> numpy.ndarray:
            r"""Transform the state vector to a projector.

            Returns:
                numpy.ndarray: projector of the state vector
            """
            return to_projector(self.matrix)

        def is_normalized(self) -> bool:
            r"""Check if the pure quantum state is normalized.

            Returns:
                bool: whether the pure quantum state is normalized
            """
            return abs(self.norm - 1) < EPSILON

        def permute_to_front(self, system: Any) -> None:
            r"""Permute a system of the pure quantum substate to the front.

            Args:
                system (Any): system to permute
            """
            if system not in self.systems:
                raise ArgumentTypeError(f"The system to permute must be one of the state systems.")

            system_idx = self.systems.index(system)

            if system_idx == 0:
                pass
            elif system_idx == self.size - 1:  # last system
                new_shape = [2 ** (self.size - 1), 2]
                new_axis = [1, 0]
                new_systems = [system] + self.systems[:system_idx]
                self.matrix = reshape(transpose(reshape(self.matrix, new_shape), new_axis), [self.length, 1])
                self.systems = new_systems
            else:  # middle system
                new_shape = [2**system_idx, 2, 2 ** (self.size - system_idx - 1)]
                new_axis = [1, 0, 2]
                new_systems = [system] + self.systems[:system_idx] + self.systems[system_idx + 1 :]
                self.matrix = reshape(transpose(reshape(self.matrix, new_shape), new_axis), [self.length, 1])
                self.systems = new_systems

        def evolve(self, which_qubit: list, operator: numpy.ndarray) -> None:
            r"""Evolve the pure quantum substate by the given operator.

            Args:
                which_qubit (list): a list of qubit systems to act on
                operator (numpy.ndarray): operator to act on
            """
            self.permute_systems(which_qubit)
            shape = [2 ** len(which_qubit), 2 ** (self.size - len(which_qubit))]
            self.matrix = reshape(operator @ reshape(self.matrix, shape), [self.length, 1])

        def measure(self, which_qubit: Any, basis: numpy.ndarray) -> Tuple[int, list]:
            r"""Measure the pure quantum substate with a given basis.

            Args:
                which_qubit (Any): qubit to measure
                basis (numpy.ndarray): basis vectors

            Returns:
                Tuple[int, list]: measurement outcome and the list of substate(s) after the measurement
            """
            self.permute_systems([which_qubit])

            prob = [0, 0]
            state_unnorm = [None, None]

            # Calculate the probability and post-measurement states
            for result in [0, 1]:
                basis_dag = dagger(basis[result])
                half_length = int(self.length / 2)
                state_unnorm[result] = reshape(basis_dag @ reshape(self.matrix, [2, half_length]), [half_length, 1])
                probability = conj(state_unnorm[result]).T @ state_unnorm[result]
                prob[result] = real(probability) if probability.dtype.name == "COMPLEX128" else probability

            # Randomly choose a result and its corresponding post-measurement state
            prob_zero, prob_one = real(prob[0].item()), real(prob[1].item())

            if prob_zero < EPSILON:
                result = 1
                post_state_vector = state_unnorm[1]
            elif prob_one < EPSILON:
                result = 0
                post_state_vector = state_unnorm[0]
            else:
                result = random.choice(2, 1, p=[prob_zero, prob_one]).item()
                # Normalize the state after measurement
                post_state_vector = state_unnorm[result] / sqrt(prob[result])

            # Update the quantum state
            if len(self.systems) == 1:  # single-qubit systems
                self.matrix = basis[result]
                return result, [self]
            else:  # multi-qubit systems
                state1 = PureState.SubState(matrix=basis[result], systems=[which_qubit])  # system being measured
                post_systems = self.systems[:]
                post_systems.remove(which_qubit)
                state2 = PureState.SubState(matrix=post_state_vector, systems=post_systems)  # systems unmeasured
                return result, [state1, state2]

        def compare_size_systems(self, other: "PureState.SubState") -> None:
            r"""Compare two pure quantum substates by their sizes and systems.

            Args:
                other (PureState.SubState): the other pure quantum substate to compare
            """
            if set(self.systems) != set(other.systems):
                raise ArgumentTypeError(
                    f"The two substates to compare have different system labels,\n"
                    f"{self.systems}\n"
                    f"{other.systems}."
                )

        def compare_by_vector(self, other: "PureState.SubState") -> float:
            r"""Compare two pure quantum substates by their state vectors.

            Args:
                other (PureState.SubState): the other pure quantum substate to compare

            Returns:
                float: norm difference
            """
            self.compare_size_systems(other)

            if not self.is_normalized():
                raise ArgumentTypeError(f"{self} is not normalized.")
            elif not other.is_normalized():
                raise ArgumentTypeError(f"{other} is not normalized.")

            self.permute_systems(other.systems)  # match the system order
            self_state_list = list(self.matrix)
            # Find an index with the largest absolute value
            idx = self_state_list.index(max(self_state_list, key=abs))
            if abs(other.matrix[idx]) <= EPSILON:
                error = 1
            else:
                # Calculate the relative phase and erase it
                phase = self.matrix[idx] / other.matrix[idx]
                self_phase = self.matrix / phase
                error = numpy.linalg.norm(self_phase - other.matrix)

            return error

        def compare_by_density(self, other: "PureState.SubState") -> float:
            r"""Compare two pure quantum substates by their density matrices.

            Args:
                other (PureState): the other pure quantum substate to compare

            Returns:
                float: norm difference
            """
            self.compare_size_systems(other)
            self.permute_systems(other.systems)  # match the system order

            error = numpy.linalg.norm(self.projector - other.projector)
            return error

        def to_mixed_substate(self) -> "MixedState.SubState":
            r"""Convert the current pure substate to a mixed substate.

            Returns:
                MixedState: equivalent mixed state
            """
            return MixedState.SubState(matrix=to_projector(self.matrix), systems=self.systems)


class MixedState(QuantumState):
    r"""Class for mixed quantum states."""

    def __init__(self, matrix=None, systems=None, substates=None):
        r"""Constructor for MixedState class.

        Args:
            matrix (numpy.ndarray, optional): matrix representation of the quantum state
            systems (list, optional): system labels of the quantum state
            substates (List[MixedState.SubState], optional): a tensor product decomposition of the quantum state
        """
        super().__init__(matrix, systems, substates)
        # Check matrix shape
        if self.matrix.shape[0] != self.matrix.shape[1]:
            raise ArgumentTypeError("The matrix is not a square matrix!")

    def evolve(self, which_qubit: list, kraus_list: List[numpy.ndarray]) -> None:
        r"""Evolution of the quantum state.

        The evolution is calculated by using the relation that

        .. math::
            |\mathcal E(\rho)>> = S|\rho>>

        where S is the superoperator of the operation \mathcal E
        and the double ket notation is the vectorization of a matrix.

        Args:
            which_qubit (list): a list of qubit systems to act on
            kraus_list (List[numpy.ndarray]): a list of kraus operators
        """
        if not isinstance(kraus_list, list):  # covert a single kraus operator to a list
            kraus_list = [kraus_list]

        # Find the relevant substates involved in the computation
        relevant_substates = []
        other_substates = []
        for substate in self.substates:
            if set(substate.systems) & set(which_qubit):  # if there is an overlap
                relevant_substates.append(substate)
            else:
                other_substates.append(substate)

        # Merge the substates and compute
        merged_state = self.merge_substates(relevant_substates)
        merged_state.evolve(which_qubit, kraus_list)
        other_substates.append(merged_state)
        self.substates = other_substates

    def measure(self, which_qubit: Any, basis: numpy.ndarray, mid: Any) -> int:
        r"""Measure the mixed quantum state with a given basis.

        Note:
            We consider rank-one projective measurements in this version. These include the commonly used
            'XY', 'YZ', 'XZ' plane measurements. The input ``basis`` should be given by measurement vectors.

        Args:
            which_qubit (Any): qubit to measure
            basis (numpy.ndarray): basis vectors
            mid (Any): measurement ID that is used to fetch the outcome

        Returns:
            int: measurement outcome
        """
        # Pick relevant substate
        relevant_substate = []
        for substate in self.substates:
            if which_qubit in substate.systems:
                relevant_substate = substate
                self.substates.remove(substate)
                break

        result, post_substate = relevant_substate.measure(which_qubit, basis)
        self.outcome[mid] = result  # save the measurement outcome
        self.substates = sum([self.substates, post_substate], [])

        return result

    def reset(self, system: Any, matrix=None) -> None:
        r"""Reset a system to a given matrix.

        Args:
            system (Any): qubit to reset state
            matrix (numpy.ndarray, optional): matrix to reset (zero state by default)
        """
        for substate in self.substates:
            if system in substate.systems:
                if len(substate.systems) > 1:  # non-isolated systems
                    raise ArgumentTypeError(f"The system to reset seems to be correlated with others!")
                elif len(substate.systems) == 1:  # isolated systems
                    substate.matrix = Zero.DM if matrix is None else matrix  # reset the quantum state to zero state
                    break

    def evolve_by_gates(self, gate_history: list) -> None:
        r"""Evolve the mixed quantum state by a list of gates.

        Args:
            gate_history (list): a list of quantum gates
        """
        to_matrix = {
            "id": Gate.I,
            "s": Gate.S,
            "t": Gate.T,
            "h": Gate.H,
            "x": Gate.X,
            "y": Gate.Y,
            "z": Gate.Z,
            "u": Gate.U,
            "u3": Gate.U3,
            "rx": Gate.Rx,
            "ry": Gate.Ry,
            "rz": Gate.Rz,
            "cx": Gate.CNOT,
            "cz": Gate.CZ,
            "swap": Gate.SWAP,
            "bit_flip": Noise.BitFlip,
            "phase_flip": Noise.PhaseFlip,
            "bit_phase_flip": Noise.BitPhaseFlip,
            "amplitude_damping": Noise.AmplitudeDamping,
            "phase_damping": Noise.PhaseDamping,
            "depolarizing": Noise.Depolarizing,
        }

        for gate in gate_history:
            if gate["name"] in ["id", "s", "t", "h", "x", "y", "z"]:
                if gate.get("condition") is None:  # without classical control
                    self.evolve(gate["which_qubit"], to_matrix[gate["name"]]())
                else:  # classically conditioned gate
                    if self.outcome[gate["condition"]] == 0:
                        pass
                    elif self.outcome[gate["condition"]] == 1:
                        self.evolve(gate["which_qubit"], to_matrix[gate["name"]]())
            elif gate["name"] in ["rx", "ry", "rz"]:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]](gate["angle"]))
            elif gate["name"] in ["u", "u3"]:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]](*gate["angles"]))
            elif gate["name"] in ["cx", "cz", "swap"]:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]]())
            elif gate["name"] == "m":
                from Extensions.QuantumNetwork.qcompute_qnet.quantum.basis import Basis

                angle, plane = gate["basis"]["angle"], gate["basis"]["plane"]
                signal_s = self.sum_outcomes(gate["basis"]["domain_s"])
                signal_t = self.sum_outcomes(gate["basis"]["domain_t"])
                # The adaptive angle is (-1)^{signal_s} * angle + {signal_t} * pi
                adaptive_angle = (-1) ** signal_s * angle + signal_t * pi
                basis = Basis.Plane(plane, adaptive_angle)
                self.measure(gate["which_qubit"][0], basis, gate["mid"])
            elif gate["name"] == "r":
                self.reset(gate["which_qubit"][0], gate["matrix"])
            elif gate["name"] in [
                "bit_flip",
                "phase_flip",
                "bit_phase_flip",
                "phase_damping",
                "amplitude_damping",
                "depolarizing",
            ]:
                self.evolve(gate["which_qubit"], to_matrix[gate["name"]](gate["prob"]))
            else:
                raise NotImplementedError

    def compare_by_density(self, other: "MixedState") -> float:
        r"""Compare two mixed quantum states by their density matrices.

        Args:
            other (MixedState): the other mixed quantum state to compare

        Returns:
            float: norm difference
        """
        self_merged_substate = self.merge_substates(self.substates)
        other_merged_substate = other.merge_substates(other.substates)
        error = self_merged_substate.compare_by_density(other_merged_substate)
        return error

    @classmethod
    def random_density_matrix(cls, qubit_number: int, is_real=False) -> numpy.ndarray:
        r"""Generate a random density matrix with given number of qubits.

        Args:
            qubit_number (int): number of qubits
            is_real (bool, optional): whether to generate a real matrix

        Returns:
            numpy.ndarray: state vector
        """
        if not isinstance(qubit_number, int):
            raise ArgumentTypeError(f"Input {qubit_number} should be an int value.")

        if not isinstance(is_real, bool):
            raise ArgumentTypeError(f"Input {is_real} should be a bool value.")

        if is_real:
            mat = random.random((2**qubit_number, 2**qubit_number))
        else:
            mat = random.random((2**qubit_number, 2**qubit_number)) + 1j * random.random(
                (2**qubit_number, 2**qubit_number)
            )

        mat_pos = transpose(conj(mat)) @ mat
        return mat_pos / trace(mat_pos)

    class SubState(QuantumState.SubState):
        r"""Class for creating a mixed quantum substate."""

        def __init__(self, matrix=None, systems=None):
            r"""Constructor for SubState class.

            Args:
                matrix (numpy.ndarray, optional): matrix representation of the mixed quantum substate
                systems (list, optional): system labels of the mixed quantum substate
            """
            super().__init__(matrix, systems)

        def permute_to_front(self, system: Any) -> None:
            r"""Permute a system of the mixed quantum substate to the front.

            Args:
                system (Any): system to permute
            """
            if system not in self.systems:
                raise ArgumentTypeError(f"The system to permute must be one of the state systems.")

            systems_copy = self.systems[:]
            systems_copy.remove(system)
            systems_to_front = [system] + systems_copy
            double_systems = self.systems + ["ancilla_" + str(system) for system in self.systems]
            double_systems_to_front = systems_to_front + ["ancilla_" + str(system) for system in systems_to_front]

            n = self.size
            vec_matrix = reshape(self.matrix, [2 ** (2 * n), 1], order="F")  # |rho>>
            vec_state = PureState.SubState(matrix=vec_matrix, systems=double_systems)
            vec_state.permute_systems(double_systems_to_front)
            self.matrix = reshape(vec_state.matrix, [2**n, 2**n], order="F")
            self.systems = systems_to_front

        def evolve(self, which_qubit: list, kraus_list: List[numpy.ndarray]) -> None:
            r"""Evolve the mixed quantum substate by the given list of kraus operators.

            The evolution is calculated by using the relation that

            .. math::
                |\mathcal E(\rho)\rangle\rangle = S|\rho\rangle\rangle

            where S is the superoperator of the operation \mathcal E
            and the double ket notation is the vectorization of a matrix.

            Args:
                which_qubit (list): a list of qubit systems to act on
                kraus_list (List[numpy.ndarray]): a list of kraus operators
            """
            # Introduce ancillary systems for purification
            double_systems = self.systems + ["ancilla_" + str(system) for system in self.systems]
            double_acting = which_qubit + ["ancilla_" + str(qubit) for qubit in which_qubit]

            # Computation with double ket notation
            vec_merged_state_matrix = reshape(self.matrix, [2 ** (2 * self.size), 1], order="F")  # |rho>>
            vec_merged_state = PureState.SubState(matrix=vec_merged_state_matrix, systems=double_systems)
            vec_merged_state.evolve(double_acting, to_superoperator(kraus_list))  # evolve by the superoperator
            vec_merged_state.permute_systems(double_systems)
            self.matrix = reshape(vec_merged_state.matrix, [2**self.size, 2**self.size], order="F")

        def measure(self, which_qubit: Any, basis: numpy.ndarray) -> Tuple[int, list]:
            r"""Measure the mixed quantum substate with a given basis.

            Args:
                which_qubit (Any): qubit to measure
                basis (numpy.ndarray): basis vectors

            Returns:
                Tuple[int, list]: measurement outcome and the list of substate(s) after the measurement
            """
            # Compute <phi|rho|phi>
            double_systems = self.systems + ["ancilla_" + str(system) for system in self.systems]
            double_acting = [which_qubit] + ["ancilla_" + str(which_qubit)]
            post_systems = self.systems[:]
            post_systems.remove(which_qubit)

            double_systems_copy = double_systems[:]
            for sys in double_acting:
                double_systems_copy.remove(sys)

            double_acting_to_front = double_acting + double_systems_copy

            n = self.size
            vec_matrix = reshape(self.matrix, [2 ** (2 * n), 1], order="F")  # |rho>>
            vec_state = PureState.SubState(matrix=vec_matrix, systems=double_systems)
            vec_state.permute_systems(double_acting_to_front)

            prob = [0, 0]
            post_state_unnorm = [None, None]

            for result in [0, 1]:
                super_operator = to_superoperator([dagger(basis[result])])
                measured_matrix = reshape(
                    super_operator @ reshape(vec_state.matrix, [4, 2 ** (2 * n - 2)]), [2 ** (2 * n - 2), 1]
                )

                # <phi|rho|phi>
                post_state_unnorm[result] = reshape(measured_matrix, [2 ** (n - 1), 2 ** (n - 1)], order="F")
                prob[result] = trace(post_state_unnorm[result])

            # Randomly choose a result and its corresponding post-measurement state
            prob_zero, prob_one = real(prob[0].item()), real(prob[1].item())

            if prob_zero < EPSILON:
                result = 1
                post_state = post_state_unnorm[1]
            elif prob_one < EPSILON:
                result = 0
                post_state = post_state_unnorm[0]
            else:
                result = random.choice(2, 1, p=[prob_zero, prob_one]).item()
                # Normalize the state after measurement
                post_state = post_state_unnorm[result] / prob[result]

            # Update the quantum state
            if len(self.systems) == 1:  # single-qubit systems
                self.matrix = to_projector(basis[result])
                return result, [self]
            else:  # multi-qubit systems
                state1 = MixedState.SubState(
                    matrix=to_projector(basis[result]), systems=[which_qubit]
                )  # system being measured
                state2 = MixedState.SubState(matrix=post_state, systems=post_systems)  # systems unmeasured
                return result, [state1, state2]

        def compare_size_systems(self, other: "MixedState.SubState") -> None:
            r"""Compare two mixed quantum substates by their sizes and systems.

            Args:
                other (MixedState.SubState): the other mixed quantum substate to compare
            """
            if set(self.systems) != set(other.systems):
                raise ArgumentTypeError(
                    f"The two substates to compare have different system labels,\n"
                    f"{self.systems}\n"
                    f"{other.systems}."
                )

        def compare_by_density(self, other: "MixedState.SubState") -> float:
            r"""Compare two mixed quantum substates by their density matrices.

            Args:
                other (MixedState.SubState): the other mixed quantum substate to compare

            Returns:
                float: norm difference
            """
            self.compare_size_systems(other)
            self.permute_systems(other.systems)  # match the system order
            error = numpy.linalg.norm(self.matrix - other.matrix)

            return error


class Zero:
    r"""Class to obtain a zero state."""

    SV = numpy.array([[1], [0]], dtype=complex)
    DM = numpy.array([[1, 0], [0, 0]], dtype=complex)


class One:
    r"""Class to obtain a one state."""

    SV = numpy.array([[0], [1]], dtype=complex)
    DM = numpy.array([[0, 0], [0, 1]], dtype=complex)


class Plus:
    r"""Class to obtain a plus state."""

    SV = numpy.array([[1], [1]], dtype=complex) / sqrt(2.0)
    DM = numpy.array([[1, 1], [1, 1]], dtype=complex) / 2


class Minus:
    r"""Class to obtain a minus state."""

    SV = numpy.array([[1], [-1]], dtype=complex) / sqrt(2.0)
    DM = numpy.array([[1, -1], [-1, 1]], dtype=complex) / 2
