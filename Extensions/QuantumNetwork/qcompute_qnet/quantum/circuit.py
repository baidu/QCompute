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
Module for quantum circuits.
"""

import copy
from argparse import ArgumentTypeError
from enum import Enum
from typing import List, Optional, Union
import numpy as np
import pandas as pd
from qcompute_qnet.quantum.backends import qcompute, Backend, mbqc

__all__ = [
    "Circuit"
]


class Circuit:
    r"""Class for creating a quantum circuit.

    Warning:
        The current version only supports gates in [I, H, X, Y, Z, S, T, Rx, Ry, Rz, U3,
        CH, CX / CNOT, CY, CZ, CRx, CRy, CRz, CU3, SWAP].

    Attributes:
        name (str): name of the circuit
        agenda (list): agenda of the circuit
        _history (List[dict]): history of the quantum gates
    """

    def __init__(self, name=None):
        r"""Constructor for Circuit class.
        """
        self.name = 'Circuit' if name is None else name
        self.agenda = []
        self.__num_qreg_unit = -1
        self._history = []

    def init_new_qreg_unit(self) -> int:
        r"""Initialize a new quantum register unit.

        Returns:
            int: address of the new quantum register unit
        """
        self.__num_qreg_unit += 1
        return self.__num_qreg_unit

    @property
    def occupied_indices(self) -> List[int]:
        r"""Get the occupied register indices in the current circuit.

        Returns:
            List[int]: occupied register indices
        """
        occupied_indices = []
        for gate in self._history:
            occupied_indices += gate["which_qubit"]

        return list(set(occupied_indices))

    @property
    def width(self) -> int:
        r"""Return the quantum circuit width.

        Returns:
            int: circuit width
        """
        return len(self.occupied_indices)

    @property
    def measured_qubits(self) -> List[int]:
        r"""Get the measured qubits in the current circuit.

        Returns:
            List[int]: measured qubits
        """
        measured_qubits = []
        for gate in self._history:
            if gate["name"] == 'm':
                measured_qubits += gate["which_qubit"]

        return list(set(measured_qubits))

    @property
    def gate_history(self) -> List[dict]:
        r"""Return the gate history of the quantum circuit.

        Returns:
            List[dict]: a list of quantum gates
        """
        return self._history

    @staticmethod
    def __check_rotation_angle(angle: Union[float, int]) -> None:
        r"""Check format of the rotation angle.

        Args:
            angle (Union[float, int]): rotation angle to check
        """
        assert isinstance(angle, float) or isinstance(angle, int), \
            f"Invalid rotation angle {angle.__repr__()} with {type(angle)} type! " \
            "Only 'float' and 'int' are supported as the type of rotation angle."

    def __check_qubit_validity(self, qubit: int) -> None:
        r"""Check validity of the qubit.

        Args:
            qubit (int): qubit to check validity
        """
        assert isinstance(qubit, int), f"Invalid qubit index {qubit.__repr__()} with {type(qubit)} type! " \
                                       f"Only 'int' is supported as the type of qubit index."
        assert qubit not in self.measured_qubits, f"Invalid qubit index: {qubit}! " \
                                                  "This qubit has already been measured."

    def __add_single_qubit_gate(self, name: str, which_qubit: int, signature=None, **params) -> None:
        r"""Add a single qubit gate to the circuit list.

        Args:
            name (str): single qubit gate name
            which_qubit (int): qubit index
            signature (Any): signature of the operation
            **params (Any): gate parameters
        """
        self.__check_qubit_validity(which_qubit)

        gate = {"name": name, "which_qubit": [which_qubit], "signature": signature, **params}
        self._history.append(gate)

    def __add_double_qubit_gate(self, name: str, which_qubit: List[int], signature=None, **params) -> None:
        r"""Add a double qubit gate to the circuit list.

        Args:
            name (str): double qubit gate name
            which_qubit (list): qubit indices in the order of [control, target]
            signature (Any): signature of the operation
            **params (Any): gate parameters
        """
        ctrl, targ = which_qubit

        self.__check_qubit_validity(ctrl)
        self.__check_qubit_validity(targ)
        if ctrl == targ:
            raise TypeError(f"Invalid qubit indices: {ctrl} and {targ}!\n"
                            "Control qubit must not be the same as target qubit.")

        gate = {"name": name, "which_qubit": which_qubit, "signature": signature, **params}
        self._history.append(gate)

    def id(self, which_qubit: int, signature=None) -> None:
        r"""Add an identity gate.

        The matrix form is:

        .. math::

            I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node): node that implements the identity gate
        """
        self.__add_single_qubit_gate('id', which_qubit, signature)

    def h(self, which_qubit: int, signature=None, **condition) -> None:
        r"""Add a Hadamard gate.

        The matrix form is:

        .. math::

            H = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node): node that implements the Hadamard gate
            **condition (int): condition of the operation
        """
        self.__add_single_qubit_gate('h', which_qubit, signature, **condition)

    def x(self, which_qubit: int, signature=None, **condition) -> None:
        r"""Add a Pauli-X gate.

        The matrix form is:

        .. math::

            X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node): node that implements the Pauli-X gate
            **condition (int): condition of the operation
        """
        self.__add_single_qubit_gate('x', which_qubit, signature, **condition)

    def y(self, which_qubit: int, signature=None, **condition) -> None:
        r"""Add a Pauli-Y gate.

        The matrix form is:

        .. math::

            Y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node): node that implements the Pauli-Y gate
            **condition (int): condition of the operation
        """
        self.__add_single_qubit_gate('y', which_qubit, signature, **condition)

    def z(self, which_qubit: int, signature=None, **condition) -> None:
        r"""Add a Pauli-Z gate.

        The matrix form is:

        .. math::

            Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node): node that implements the Pauli-Z gate
            **condition (int): condition of the operation
        """
        self.__add_single_qubit_gate('z', which_qubit, signature, **condition)

    def s(self, which_qubit: int, signature=None) -> None:
        r"""Add a S gate.

        The matrix form is:

        .. math::

            S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node): node that implements the S gate
        """
        self.__add_single_qubit_gate('s', which_qubit, signature)

    def t(self, which_qubit: int, signature=None) -> None:
        r"""Add a T gate.

        The matrix form is:

        .. math::

            T = \begin{bmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node): node that implements the T gate
        """
        self.__add_single_qubit_gate('t', which_qubit, signature)

    def rx(self, which_qubit: int, theta: Union[float, int], signature=None, **condition) -> None:
        r"""Add a rotation gate around x-axis.

        The matrix form is:

        .. math::

            R_x(\theta) =
            \begin{bmatrix}
            \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
            -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            theta (Union[float, int]): rotation angle
            signature (Node): node that implements the rotation gate
            **condition (int): condition of the operation
        """
        self.__check_rotation_angle(theta)
        self.__add_single_qubit_gate('rx', which_qubit, signature, angle=theta, **condition)

    def ry(self, which_qubit: int, theta: Union[float, int], signature=None, **condition) -> None:
        r"""Add a rotation gate around y-axis.

        The matrix form is:

        .. math::

            R_y(\theta) =
            \begin{bmatrix}
            \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
            \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            theta (Union[float, int]): rotation angle
            signature (Node): node that implements the rotation gate
            **condition (int): condition of the operation
        """
        self.__check_rotation_angle(theta)
        self.__add_single_qubit_gate('ry', which_qubit, signature, angle=theta, **condition)

    def rz(self, which_qubit: int, theta: Union[float, int], signature=None, **condition) -> None:
        r"""Add a rotation gate around z-axis.

        The matrix form is:

        .. math::

            R_z(\theta) =
            \begin{bmatrix}
            e^{-i\frac{\theta}{2}} & 0 \\
            0 & e^{i\frac{\theta}{2}}
            \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            theta (Union[float, int]): rotation angle
            signature (Node): node that implements the rotation gate
            **condition (int): condition of the operation
        """
        self.__check_rotation_angle(theta)
        self.__add_single_qubit_gate('rz', which_qubit, signature, angle=theta, **condition)

    def u3(self, which_qubit: int, theta: Union[float, int], phi: Union[float, int], gamma: Union[float, int],
           signature=None, **condition) -> None:
        r"""Add a single qubit unitary gate.

        It has a decomposition form:

        .. math::

            \begin{align}
            U_3(\theta, \varphi, \gamma) = R_z(\varphi) R_x(\theta) R_z(\gamma) =
                \begin{bmatrix}
                    \cos\frac\theta2 & -e^{i\gamma}\sin\frac\theta2\\
                    e^{i\varphi}\sin\frac\theta2 & e^{i(\varphi+\gamma)}\cos\frac\theta2
                \end{bmatrix}
            \end{align}

        Warnings:
            Please be aware of the order of the rotation angles.

        Args:
            which_qubit (int): qubit index
            theta (Union[float, int]): the rotation angle of the Rx gate
            phi (Union[float, int]): the rotation angle of the left Rz gate
            gamma (Union[float, int]): the rotation angle of the right Rz gate
            signature (Node): node that implements the unitary gate
            **condition (int): condition of the operation
       """
        angles = [theta, phi, gamma]
        for angle in angles:
            self.__check_rotation_angle(angle)

        self.__add_single_qubit_gate('u3', which_qubit, signature, angles=angles, **condition)

    def ch(self, which_qubit: List[int], signature=None) -> None:
        r"""Add a Controlled-Hadamard gate.

        Let ``which_qubit`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
                CH =
                \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \frac{1}{\sqrt{2}} & \frac{1}{\sqrt{2}} \\
                0 & 0 & \frac{1}{\sqrt{2}} & -\frac{1}{\sqrt{2}}
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (list): a list of qubit indices in the order of [control, target]
            signature (Node): node that implements the Controlled-Hadamard gate
        """
        self.__add_double_qubit_gate('ch', which_qubit, signature)

    def cx(self, which_qubit: List[int], signature=None) -> None:
        r"""Add a Controlled-X gate.

        Let ``which_qubit`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
                CX =
                \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (list): a list of qubit indices in the order of [control, target]
            signature (Node): node that implements the Controlled-X gate
        """
        self.__add_double_qubit_gate('cx', which_qubit, signature)

    def cnot(self, which_qubit: List[int], signature=None) -> None:
        r"""Add a Controlled-NOT gate.

        Let ``which_qubit`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
                CNOT =
                \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (list): a list of qubit indices in the order of [control, target]
            signature (Node): node that implements the Controlled-NOT gate
        """
        self.__add_double_qubit_gate('cx', which_qubit, signature)

    def cnot15(self, which_qubit: List[int], signature=None) -> None:
        r"""Add a Controlled-NOT gate whose measurement pattern has 15 qubits.

        Let ``which_qubit`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
                CNOT =
                \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & 1 \\
                0 & 0 & 1 & 0
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (list): a list of qubit indices in the order of [control, target]
            signature (Node): node that implements the Controlled-NOT gate
        """
        self.__add_double_qubit_gate('cnot15', which_qubit, signature)

    def cy(self, which_qubit: List[int], signature=None) -> None:
        r"""Add a Controlled-Y gate.

        Let ``which_qubit`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
                CY =
                \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 0 & -i \\
                0 & 0 & i & 0
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (list): a list of qubit indices in the order of [control, target]
            signature (Node): node that implements the Controlled-Y gate
        """
        self.__add_double_qubit_gate('cy', which_qubit, signature)

    def cz(self, which_qubit: List[int], signature=None) -> None:
        r"""Add a Controlled-Z gate.

        Let ``which_qubit`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
                CZ =
                \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & 1 & 0 \\
                0 & 0 & 0 & -1
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (list): a list of qubit indices in the order of [control, target]
            signature (Node): node that implements the Controlled-Z gate
        """
        self.__add_double_qubit_gate('cz', which_qubit, signature)

    def crx(self, which_qubit: List[int], theta: Union[float, int], signature=None) -> None:
        r"""Add a Controlled-rotation gate around x-axis.

        Let ``which_qubit`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
                CR_x(\theta) =
                \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
                0 & 0 & -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (List[int]): a list of qubit indices in the order of [control, target]
            theta (Union[float, int]): rotation angle
            signature (Node): node that implements the Controlled-rotation gate
        """
        self.__check_rotation_angle(theta)
        self.__add_double_qubit_gate('crx', which_qubit, signature, angle=theta)

    def cry(self, which_qubit: List[int], theta: Union[float, int], signature=None) -> None:
        r"""Add a Controlled-rotation gate around y-axis.

        Let ``which_qubit`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
                CR_y(\theta) =
                \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
                0 & 0 & \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (List[int]): a list of qubit indices in the order of [control, target]
            theta (Union[float, int]): rotation angle
            signature (Node): node that implements the Controlled-rotation gate
        """
        self.__check_rotation_angle(theta)
        self.__add_double_qubit_gate('cry', which_qubit, signature, angle=theta)

    def crz(self, which_qubit: List[int], theta: Union[float, int], signature=None) -> None:
        r"""Add a Controlled-rotation gate around z-axis.

        Let ``which_qubit`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
                CR_z(\theta) =
                \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & e^{-i\frac{\theta}{2}} & 0 \\
                0 & 0 & 0 & e^{i\frac{\theta}{2}}
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (List[int]): a list of qubit indices in the order of [control, target]
            theta (Union[float, int]): rotation angle
            signature (Node): node that implements the Controlled-rotation gate
        """
        self.__check_rotation_angle(theta)
        self.__add_double_qubit_gate('crz', which_qubit, signature, angle=theta)

    def cu3(self, which_qubit: List[int],
            theta: Union[float, int], phi: Union[float, int], gamma: Union[float, int], signature=None) -> None:
        r"""Add a Controlled-rotation gate.

        Let ``which_qubit`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
                CU_3(\theta, \varphi, \gamma) =
                \begin{bmatrix}
                1 & 0 & 0 & 0 \\
                0 & 1 & 0 & 0 \\
                0 & 0 & \cos\frac\theta2 & -e^{i\gamma}\sin\frac\theta2 \\
                0 & 0 & e^{i\varphi}\sin\frac\theta2 & e^{i(\varphi+\gamma)}\cos\frac\theta2
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (List[int]): a list of qubit indices in the order of [control, target]
            theta (Union[float, int]): the rotation angle of the Rx gate
            phi (Union[float, int]): the rotation angle of the left Rz gate
            gamma (Union[float, int]): the rotation angle of the right Rz gate
            signature (Node): node that implements the Controlled-rotation gate
        """
        angles = [theta, phi, gamma]
        for angle in angles:
            self.__check_rotation_angle(angle)

        self.__add_double_qubit_gate('cu3', which_qubit, signature, angles=angles)

    def swap(self, which_qubit: List[int], signature=None) -> None:
        r"""Add a SWAP gate.

        Let ``which_qubit`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
                SWAP =
                \begin{bmatrix}
                    1 & 0 & 0 & 0 \\
                    0 & 0 & 1 & 0 \\
                    0 & 1 & 0 & 0 \\
                    0 & 0 & 0 & 1
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (list): qubits to swap
            signature (Node): node that implements the SWAP gate
        """
        self.__add_double_qubit_gate('cx', which_qubit, signature)
        self.__add_double_qubit_gate('cx', [which_qubit[1], which_qubit[0]], signature)
        self.__add_double_qubit_gate('cx', which_qubit, signature)

    def measure(self, which_qubit=None, signature=None) -> None:
        r"""Measure a quantum state in the computational basis.

        Measure all the qubits if no specific qubit index is given.

        Args:
            which_qubit (int): qubit index
            signature (Node): node that implements the measurement
        """
        assert which_qubit is None or isinstance(which_qubit, int), \
            f"Input {which_qubit} should be an int value."

        if which_qubit is None:  # measure all qubits in the circuit
            occupied_indices = copy.deepcopy(self.occupied_indices)
            for idx in occupied_indices:
                self.__add_single_qubit_gate('m', idx, signature, basis=[0, 'YZ', [], []])  # Z measurement

        else:
            self.__add_single_qubit_gate('m', which_qubit, signature, basis=[0, 'YZ', [], []])  # Z measurement

    def defer_measurement(self) -> None:
        r"""Defer all measurements to the end.

        Note:
            Classically conditioned gates will be transformed to controlled gates.
        """
        measure_gates = []
        for i, gate in enumerate(self._history):
            if "condition" in gate.keys():
                # Fixed single-qubit gate
                if gate['name'] in ['h', 'x', 'y', 'z']:
                    self._history[i] = {"name": 'c' + gate['name'],
                                        "which_qubit": [gate['condition'], gate['which_qubit'][0]],
                                        "signature": gate['signature']}
                # Parameterized single-qubit gate
                elif gate['name'] in ['rx', 'ry', 'rz']:
                    self._history[i] = {"name": 'c' + gate['name'],
                                        "which_qubit": [gate['condition'], gate['which_qubit'][0]],
                                        "signature": gate['signature'],
                                        "angle": gate['angle']}
                elif gate['name'] == 'u3':
                    self._history[i] = {"name": 'cu3',
                                        "which_qubit": [gate['condition'], gate['which_qubit'][0]],
                                        "signature": gate['signature'],
                                        "angles": gate['angles']}
            elif gate['name'] == 'm':
                measure_gates.append(self._history[i])

        self._history = list(filter(lambda x: x not in measure_gates, self._history))
        self._history.extend(measure_gates)

    def copy(self) -> "Circuit":
        r"""Make a copy of the current circuit.

        Returns:
            Circuit: a copy of the current circuit
        """
        new_circuit = Circuit()
        new_circuit._history = self.gate_history
        return new_circuit

    def is_equal(self, other: "Circuit") -> bool:
        r"""Check if the current circuit is equal to another circuit.

        Returns:
            bool: whether the two circuits equals
        """
        if self.width != other.width:  # check if the circuits have the same size
            return False

        for i in range(self.width):
            gates_1, gates_2 = [], []

            for gate_1, gate_2 in zip(self.gate_history, other.gate_history):
                if i in gate_1['which_qubit']:
                    gates_1.append(gate_1)
                if i in gate_2['which_qubit']:
                    gates_2.append(gate_2)

            for gate_1, gate_2 in zip(gates_1, gates_2):
                if gate_1 != gate_2:
                    if not (gate_1['name'] == gate_2['name'] == "cz" or gate_1['name'] == gate_2['name'] == "swap" and
                            set(gate_1['which_qubit']) == set(gate_2['which_qubit'])):
                        return False
        return True

    def remap_indices(self, remap=None) -> None:
        r"""Remap the indices of quantum register.

        Args:
            remap (dict, optional): remap of the indices

        Examples:
            Both the keys and the values of the remap dictionary should be of int type. Here is an example:

            >>> cir = Circuit()
            >>> cir.h(1)
            >>> cir.cnot([1, 3])
            >>> cir.cnot([3, 4])
            >>> cir.remap_indices(remap={1: 0, 3: 1, 4: 2})

            Note that the new indices should cover all indices in the current circuit, and the new indices should be
            a set of sequential integers starting from zero.

        Warnings:
            Some functionalities of MBQC module require that quantum register has sequential indices.
            We need to remap the indices before using the MBQC model and after deferring all measurements.

        Important:
            This method will directly update the current circuit. Please make a new copy of the circuit if necessary.
        """
        if remap is not None:
            assert set(remap.keys()) == set(self.occupied_indices), \
                f"The remap should contain all indices ({', '.join(str(i) for i in self.occupied_indices)}) " \
                f"of the quantum register."
            assert set(remap.values()) == set(range(self.width)), \
                f"The new indices should be a set of sequential integers starting from zero."

            # Sort by the new indices in ascending order
            remap = {value: key for key, value in zip(remap.keys(), remap.values())}
            remap = {remap[key]: key for key in sorted(remap.keys())}
        else:
            remap = {self.occupied_indices[i]: i for i in range(len(self.occupied_indices))}

        # Update the gate history
        for gate in self._history:
            new_indices = [remap[which_qubit] for which_qubit in gate["which_qubit"]]
            gate["which_qubit"] = new_indices

        print(f"\nThe quantum register indices have been remapped to (old: new): {remap}.")

    def run(self, shots: int, backend: Enum, token: Optional[str] = None) -> dict:
        r"""Run the quantum circuit with a given backend.

        Args:
            shots (int): the number of sampling
            backend (Enum): backend to run the quantum circuit
            token (str, optional): your token for QCompute backend

        Returns:
            dict: circuit results, including the circuit's name, sampling shots and sampling results
        """
        if not hasattr(backend, 'name'):
            raise ArgumentTypeError(f"{backend} has no attribute 'name'. "
                                    f"Please assign a specific backend for {backend}.")

        if backend.name in Backend.QCompute.__members__:
            results = qcompute.run_circuit(self, shots=shots, backend=backend, token=token)
        elif backend.name in Backend.MBQC.__members__:
            results = mbqc.run_circuit(self, shots=shots)
        else:
            raise ArgumentTypeError(f"Cannot find the backend {backend}.")

        cir_results = {'circuit_name': self.name, 'shots': shots, 'counts': self.sort_results(results)}
        return cir_results

    @staticmethod
    def reduce_results(results: dict, indices: List[int]) -> dict:
        r"""Reduce the circuit sampling results with specific indices.

        Args:
            results (dict): circuit sampling results to reduce
            indices (List[int]): global indices for reducing circuit results

        Returns:
            dict: reduced results
        """
        assert all(indices[i] < indices[i + 1] for i in range(len(indices) - 1)), \
            f"The indices should be in ascending order."

        reduced_results = {}
        for key, value in results.items():
            res = ""
            for i in indices:
                res += key[i]
            if res not in reduced_results.keys():
                reduced_results[res] = value
            else:
                reduced_results[res] += value

        return Circuit.sort_results(reduced_results)

    @staticmethod
    def sort_results(results: dict) -> dict:
        r"""Sort the circuit results in ascending order.

        Args:
            results (dict): circuit sampling results to sort

        Returns:
            dict: sorted results
        """
        sorted_results = {key: results[key] for key in sorted(results.keys())}
        return sorted_results

    def print_agenda(self) -> None:
        r"""Print the events scheduled for the circuit.
        """
        from qcompute_qnet.core.des import Event

        df = Event.events_to_dataframe(self.agenda)
        print(f"\nAgenda of {self.name} (unsorted):\n{df.to_string()}")

    def print_list(self) -> None:
        r"""Print the quantum circuit list.
        """
        df = pd.DataFrame(columns=["name", "which_qubit", "signature", "params"])

        for i, gate in enumerate(self.gate_history):
            gate_params = {key: gate[key]
                           for key in list(filter(lambda x: x not in ['name', 'which_qubit', 'signature'], gate))}
            circuit_info = pd.DataFrame({"name": gate['name'],
                                         "which_qubit": str(gate['which_qubit']),
                                         "signature": gate['signature'].name if gate['signature'] is not None else None,
                                         "params": str(gate_params)}, index=[f"Gate {i + 1}"])
            df = pd.concat([df, circuit_info])

        print(f"\nCircuit details:\n{df.to_string()}")

    def print_circuit(self, color: Optional[bool] = False, colors: Optional[dict] = None) -> None:
        r"""Print the circuit on the terminal.

        Args:
            color (bool, optional): whether to print a colored circuit
            colors (dict, optional): specified colors for different nodes

        Examples:

            >>> from qcompute_qnet.quantum.circuit import Circuit
            >>> cir = Circuit()
            >>> cir.x(0)
            >>> cir.y(1)
            >>> cir.h(2)
            >>> cir.z(3)
            >>> cir.s(4)
            >>> cir.rx(0, 0.5)
            >>> cir.ry(1, 0.6)
            >>> cir.rz(1, 0.4)
            >>> cir.cnot([0, 1])
            >>> cir.cz([4, 3])
            >>> cir.measure()
            >>> cir.print_circuit()
        """
        # Optional colors for gates
        gate_colors = {"red": "\033[0;31m",
                       "blue": "\033[0;34m",
                       "green": "\033[0;32m",
                       "yellow": "\033[1;33m",
                       "purple": "\033[0;35m",
                       "cyan": "\033[0;36m",
                       "brown": "\033[0;33m",
                       "dark_gray": "\033[1;30m",
                       "light_red": "\033[1;31m",
                       "light_blue": "\033[1;34m",
                       "light_green": "\033[1;32m",
                       "light_purple": "\033[1;35m",
                       "light_cyan": "\033[1;36m",
                       "light_gray": "\033[0;37m",
                       "none": "\033[0m"}

        # Count the blocks needed for the print of the whole circuit
        width = self.width
        nums_block = np.zeros(width, dtype=int)

        if color is True:
            for i, gate in enumerate(self.gate_history):
                if gate['signature'] is None:
                    print("\n\nNot all gates are assigned signatures. A monochrome circuit will be printed.")
                    color = False
                    break

            if colors is not None:
                # Sanity check for input colors
                assert isinstance(colors, dict), "'colors' should be a dictionary mapping nodes to different colors."
                for node_color in colors.values():
                    assert node_color in gate_colors, f"The {node_color} color is not supported in this version."

            else:  # assign colors for different signatures
                signatures = {gate['signature'] for gate in self.gate_history}
                assigned_colors = list(gate_colors)[:len(signatures)]
                colors = {signature: assigned_color for signature, assigned_color in zip(signatures, assigned_colors)}

        for i, gate in enumerate(self.gate_history):
            if gate['name'] in {'cx', 'cy', 'cz', 'swap', 'crx', 'cry', 'crz', 'cu3', 'ch'}:
                max_c = 0

                for p in range(min(gate['which_qubit']), max(gate['which_qubit']) + 1):
                    max_c = nums_block[p] if max_c < nums_block[p] else max_c

                for p in range(min(gate['which_qubit']), max(gate['which_qubit']) + 1):
                    nums_block[p] = max_c
                    nums_block[p] += 1

            elif 'condition' in gate.keys():
                max_c = 0

                for p in range(min(gate['which_qubit'][-1], gate['condition']),
                               max(gate['which_qubit'][-1], gate['condition'])):
                    max_c = nums_block[p] if max_c < nums_block[p] else max_c

                for p in range(min(gate['which_qubit'][-1], gate['condition']),
                               max(gate['which_qubit'][-1], gate['condition']) + 1):
                    nums_block[p] = max_c
                    nums_block[p] += 1
            else:
                for qubit in gate['which_qubit']:
                    nums_block[qubit] += 1  # count each number's frequency

        # System parameters about printing
        total_length = max(nums_block)
        total_length = total_length * 6 + 3  # the number of blocks needed
        print_list = [['-' if i % 2 == 0 else ' '] * total_length for i in range(width * 2)]  # main string for printing
        print_list_idx = np.ones(width, dtype=int) * 2  # x-axis of the current gate, begins with 2

        # System parameters about coloring
        color_loc = np.zeros(width * 2, dtype=int)  # record the location to be painted
        line_color = ['\033[0m'] * width * 2  # record the color of each line

        for i, gate in enumerate(self.gate_history):
            gate_loc = gate['which_qubit']  # operating qubit(s)

            if 'condition' in gate.keys():  # condition gates
                # Find the largest location between gate location and condition
                max_idx = 0
                condition = gate['condition']

                for p in range(min(gate_loc[0], condition), max(gate_loc[0], condition) + 1):
                    if max_idx < print_list_idx[p]:
                        max_idx = print_list_idx[p]

                for p in range(min(gate_loc[0], condition), max(gate_loc[0], condition) + 1):
                    print_list_idx[p] = max_idx

                # Print the gate
                if len(gate['name']) == 1:
                    print_list[gate_loc[0] * 2][print_list_idx[gate_loc[0]]] = gate['name'].upper()

                else:  # len(gate['name']) = 2
                    print_list[gate_loc[0] * 2][print_list_idx[gate_loc[0]]] = gate['name'][0].upper()
                    print_list[gate_loc[0] * 2][print_list_idx[gate_loc[0]] + 1] = gate['name'][1]

                # Print 'o' and '='
                print_list[condition * 2][print_list_idx[gate_loc[0]]] = 'o'

                for idx in range(total_length):
                    if print_list[condition * 2].index('M') < idx < print_list[condition * 2].index('o'):
                        if print_list[condition * 2][idx][0] == '-':
                            print_list[condition * 2][idx] = '=' + print_list[condition * 2][idx][1:]
                        if print_list[condition * 2][idx][-1] == '-':
                            print_list[condition * 2][idx] = print_list[condition * 2][idx][0:-1] + '='

                # Print '|'
                for p in range(min(gate_loc[0], condition) * 2 + 1, max(gate_loc[0], condition) * 2):
                    print_list[p][print_list_idx[gate_loc[0]]] = '|'

                # Hide the line after 'o'
                for q in range(print_list[condition * 2].index('o') + 1, total_length):
                    print_list[condition * 2][q] = ' '

                # Update the idx
                for p in range(min(gate_loc[0], condition) + 1, max(gate_loc[0], condition)):
                    print_list_idx[p] += 5

            elif len(gate_loc) == 2:  # controlled gates and 'swap'
                # Find the largest location between the control and target qubit
                max_idx = 0
                for p in range(min(gate_loc), max(gate_loc) + 1):
                    if max_idx < print_list_idx[p]:
                        max_idx = print_list_idx[p]

                for p in range(min(gate_loc), max(gate_loc) + 1):
                    print_list_idx[p] = max_idx

                # Get what to be printed and print the gate
                if gate['name'] == 'swap':
                    gate_name = 'swap'
                else:
                    gate_name = gate['name'][1:]  # remove 'c', now len(gate_name) in {1, 2}

                if len(gate_name) == 1:
                    if gate_name != 'swap':
                        print_list[gate_loc[1] * 2][print_list_idx[gate_loc[1]]] = gate_name[0].upper()
                    else:  # swap gate
                        print_list[gate_loc[1] * 2][print_list_idx[gate_loc[1]]] = 'x'

                else:  # len(gate_name) = 2
                    print_list[gate_loc[1] * 2][print_list_idx[gate_loc[1]]] = gate_name[0].upper()
                    print_list[gate_loc[1] * 2][print_list_idx[gate_loc[1]] + 1] = gate_name[1]

                # Print the control and '|'s
                print_list[gate_loc[0] * 2][print_list_idx[gate_loc[0]]] = '*'

                for p in range(min(gate_loc) * 2 + 1, max(gate_loc) * 2):
                    print_list[p][print_list_idx[gate_loc[1]]] = '|'

                # Update the idx
                for p in range(min(gate_loc) + 1, max(gate_loc)):
                    print_list_idx[p] += 5

            else:
                # Gates with name length = 1, contains{'x', 'y', 'z', 'h', 's', 't', 'i', 'm'}
                if len(gate['name']) == 1:
                    print_list[gate_loc[0] * 2][print_list_idx[gate_loc[0]]] = gate['name'].upper()

                else:
                    print_list[gate_loc[0] * 2][print_list_idx[gate_loc[0]]] = gate['name'][0].upper()
                    print_list[gate_loc[0] * 2][print_list_idx[gate_loc[0]] + 1] = gate['name'][1]

            if color:
                # Paint the gate

                # If the paint loc is in front of '|', step to its behind
                if print_list[gate_loc[-1] * 2][color_loc[gate_loc[-1] * 2] + 1] == '|':
                    print_list[gate_loc[-1] * 2][color_loc[gate_loc[-1] * 2] + 2] = \
                        gate_colors[colors[gate['signature']]] + '-'
                else:
                    print_list[gate_loc[-1] * 2][color_loc[gate_loc[-1] * 2]] = \
                        gate_colors[colors[gate['signature']]] + '-'

                # Change the line color
                line_color[gate_loc[-1] * 2] = gate_colors[colors[gate['signature']]]
                # Ensure the color of the line is same after '|'
                for loc in range(color_loc[gate_loc[-1] * 2] + 1, total_length):  # scan the element before '|'
                    if len(print_list[gate_loc[-1] * 2][loc]) > 1:
                        if print_list[gate_loc[-1] * 2][loc][-1] in {'-', '='}:
                            print_list[gate_loc[-1] * 2][loc] = line_color[gate_loc[-1] * 2] + \
                                                                print_list[gate_loc[-1] * 2][loc][-1]

                if len(gate_loc) == 2:
                    # Change the color near the control qubit
                    # ('--*--' if len(gate['name'] == 2), '-*---' if len(gate['name']) == 3)
                    lg = 1 if len(gate['name']) == 3 else 2
                    color_loc[gate_loc[0] * 2] = print_list_idx[gate_loc[0]] - lg
                    print_list[gate_loc[0] * 2][color_loc[gate_loc[0] * 2]] = \
                        gate_colors[colors[gate['signature']]] + '-'  # paint the head of the '--*--'

                    rg = len(gate['name']) + 3  # the range of blocks in the gate's color
                    print_list[gate_loc[0] * 2][color_loc[gate_loc[0] * 2] + rg] = \
                        print_list[gate_loc[0] * 2][color_loc[gate_loc[0] * 2] + rg] + line_color[gate_loc[0] * 2]

                    # Color the '|'
                    for p in range(min(gate_loc) * 2 + 1, max(gate_loc) * 2):
                        block_here = '-' if p % 2 == 0 else ' '
                        print_list[p][print_list_idx[gate_loc[-1]] - 1] = \
                            block_here + gate_colors[colors[gate['signature']]]
                        print_list[p][print_list_idx[gate_loc[-1]] + 1] = line_color[p] + block_here

                    color_loc[gate_loc[0] * 2] = print_list_idx[gate_loc[-1]] + len(gate['name']) + 2 - (
                        1 if 'c' in gate['name'] else 0
                    )

                # Update color location
                color_loc[gate_loc[-1] * 2] = print_list_idx[gate_loc[-1]] + len(gate['name']) + 2 - (
                    1 if 'c' in gate['name'] else 0
                )

                if 'condition' in gate.keys():
                    condition = gate['condition']
                    line_color[condition * 2] = gate_colors[colors[gate['signature']]]
                    # Paint the '=' and 'o'
                    q = print_list[condition * 2].index('M') + 1
                    print_list[condition * 2][q + 2] = gate_colors[colors[gate['signature']]] + '='

                    # Paint the '|'
                    for p in range(min(gate_loc[0], condition) * 2, max(gate_loc[0], condition) * 2):
                        # Change the color before '|'
                        print_list[p][print_list_idx[gate_loc[0]] - 1] = \
                            print_list[p][print_list_idx[gate_loc[0]] - 1][-1] + gate_colors[colors[gate['signature']]]

                        for r in range(q, total_length):
                            if print_list[p][r] == '|' and p % 2 == 0:
                                print_list[p][r + 1] = line_color[p] + print_list[p][r + 1][-1]

            # Update the index for printing of next gate
            for p in gate_loc:
                print_list_idx[p] += 5

        # Shorten the remaining line elements
        line_end = max(print_list_idx)
        for line in range(width * 2):
            print_list[line][line_end] = gate_colors["none"]

            for pos in range(line_end + 1, total_length):
                print_list[line][pos] = ''

        print_list = list(map(''.join, print_list))
        circuit_str = '\n'.join(print_list)

        print(f"\n{self.name}: \n\n" + circuit_str)
        if color is True:
            print("Colors:", {node.name: color for node, color in colors.items()})
