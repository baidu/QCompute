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

from argparse import ArgumentTypeError
from enum import Enum
from typing import List, Tuple, Optional, Union, Any
import copy
import itertools
from functools import reduce, partial

import networkx
import numpy
import random
from numpy import matmul, linalg
from matplotlib import pyplot as plt
import networkx as nx
import pandas as pd
from networkx import DiGraph, MultiDiGraph

from Extensions.QuantumNetwork.qcompute_qnet import EPSILON
from Extensions.QuantumNetwork.qcompute_qnet.quantum.gate import Gate
from Extensions.QuantumNetwork.qcompute_qnet.quantum.state import PureState, MixedState, Zero
from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends import Backend, mbqc, qcompute
from Extensions.QuantumNetwork.qcompute_qnet.quantum.utils import (
    COLOR_TABLE,
    decompose_to_u_gate,
    find_keys_by_value,
    print_progress,
)

__all__ = ["Circuit"]


class Circuit:
    r"""Class for creating a quantum circuit.

    Warning:
        The current version only supports gates in [I, H, X, Y, Z, S, T, Rx, Ry, Rz, U, U3,
        CH, CX / CNOT, CY, CZ, CRx, CRy, CRz, CU, CU3, SWAP].

    Hint:
        Quantum noise can also be applied to a quantum circuit, including:

        1. Bit flip noise
        2. Phase flip noise
        3. Bit-phase flip noise
        4. Amplitude damping noise
        5. Phase damping noise
        6. Depolarizing noise

        Noisy circuit simulation can only be implemented by QNET ``DensityMatrix`` backend.

    Important:
        We support dynamic quantum circuit. That is, we can perform measurement in the middle of the
        computation and reset the qubit. We can also perform classically conditioned operations. For example,
        a circuit for quantum teleportation can be constructed as:

        >>> cir = Circuit()
        >>> cir.rx(0, 0.1)
        >>> cir.h(1)
        >>> cir.cnot([1, 2])
        >>> cir.cnot([0, 1])
        >>> cir.h(0)
        >>> cir.measure(0, mid='a')
        >>> cir.measure(1, mid='b')
        >>> cir.z(2, condition='a')
        >>> cir.x(2, condition='b')
        >>> cir.measure(2, mid='c')
        >>> cir.output_ids = ['a', 'b', 'c']
        >>> results = cir.run(shots=1024, backend=Backend.QNET.StateVector)
        >>> print(results)

    Attributes:
        name (str): name of the circuit
        agenda (list): agenda of the circuit
        output_ids (list): measurement IDs that will be used to obtain the sampling result

    Note:
         In a dynamic quantum circuit, a quantum system will be measured multiple times. Some measurements are used
         to steer the computation while others are used to obtain the final results.
         The attribute ``output_ids`` indicates which measurements will be used as the final results.
         If ``output_ids`` is not specified, the last measurement on each qubit will be used.
    """

    def __init__(self, name=None):
        r"""Constructor for Circuit class.

        Args:
            name (str, optional): name of the circuit
        """
        self.name = "Circuit" if name is None else name
        self.agenda = []
        self.output_ids = None
        self._history = []
        self.__num_qreg_unit = -1

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
        measured_qubits = [gate["which_qubit"][0] for gate in self._history if gate["name"] == "m"]
        return list(set(measured_qubits))

    @property
    def gate_history(self) -> List[dict]:
        r"""Return the gate history of the quantum circuit.

        Returns:
            List[dict]: a list of quantum gates
        """
        return self._history

    def measurement_counter(self, which_qubit: Any) -> int:
        r"""Get the number of rounds that the given qubit has been measured.

        Args:
            which_qubit (Any): the system to check

        Note:
            In a dynamic quantum circuit, a quantum system can be measured multiple times.

        Returns:
            int: the number of times that the given qubit has been measured
        """
        counter = 0
        for gate in self._history:
            if gate["name"] == "m" and gate["which_qubit"] == [which_qubit]:
                counter += 1

        return counter

    def get_qubit_by_mid(self, mid: Any) -> int:
        r"""Get the qubit with a given measurement ID.

        Args:
            mid (Any): measurement ID of the qubit

        Returns:
            int: qubit related to the measurement ID
        """
        if isinstance(mid, int):  # label of the qubit used as measurement ID
            return mid

        for gate in self.gate_history:
            if gate["name"] == "m" and gate.get("mid") == mid:
                return gate["which_qubit"][0]

        raise ArgumentTypeError("Invalid measurement ID!")

    @staticmethod
    def __check_rotation_angle(angle: Union[float, int]) -> None:
        r"""Check format of the rotation angle.

        Args:
            angle (Union[float, int]): rotation angle to check
        """
        assert isinstance(angle, float) or isinstance(angle, int), (
            f"Invalid rotation angle {angle.__repr__()} with {type(angle)} type! "
            "Only 'float' and 'int' are supported as the type of rotation angle."
        )

    @staticmethod
    def __check_qubit_validity(qubit: int) -> None:
        r"""Check validity of the qubit.

        Args:
            qubit (int): qubit to check validity
        """
        assert isinstance(qubit, int), (
            f"Invalid qubit index {qubit.__repr__()} with {type(qubit)} type! "
            f"Only 'int' is supported as the type of qubit index."
        )

    def __add_single_qubit_gate(self, name: str, which_qubit: int, signature=None, **params) -> None:
        r"""Add a single qubit gate to the circuit list.

        Args:
            name (str): single qubit gate name
            which_qubit (int): qubit index
            signature (Any, optional): signature of the operation
            **params (Any): gate parameters
        """
        self.__check_qubit_validity(which_qubit)

        if name in ["rx", "ry", "rz"]:
            self.__check_rotation_angle(params["angle"])
        elif name in ["u", "u3"]:
            for angle in params["angles"]:
                self.__check_rotation_angle(angle)
        elif name in ["bit_flip", "phase_flip", "bit_phase_flip", "amplitude_damping", "phase_damping", "depolarizing"]:
            self.__check_rotation_angle(params["prob"])

        if params.get("condition") is not None:
            assert self.get_qubit_by_mid(params["condition"]) != which_qubit, f"Invalid condition operation!"

        gate = {"name": name, "which_qubit": [which_qubit], "signature": signature, **params}
        self._history.append(gate)

    def __add_double_qubit_gate(self, name: str, which_qubit: List[int], signature=None, **params) -> None:
        r"""Add a double qubit gate to the circuit list.

        Args:
            name (str): double qubit gate name
            which_qubit (list): qubit indices in the order of [control, target]
            signature (Any, optional): signature of the operation
            **params (Any): gate parameters
        """
        ctrl, targ = which_qubit

        self.__check_qubit_validity(ctrl)
        self.__check_qubit_validity(targ)
        if ctrl == targ:
            raise TypeError(
                f"Invalid qubit indices: {ctrl} and {targ}!\n"
                "The control qubit must not be the same as the target qubit."
            )

        if name in ["crx", "cry", "crz"]:
            self.__check_rotation_angle(params["angle"])
        elif name in ["cu", "cu3"]:
            for angle in params["angles"]:
                self.__check_rotation_angle(angle)

        gate = {"name": name, "which_qubit": which_qubit, "signature": signature, **params}
        self._history.append(gate)

    def __add_triple_qubit_gate(self, name: str, which_qubit: List[int], signature=None, **params) -> None:
        r"""Add a triple qubit gate to the circuit list.

        Note:
            We assume that in the list of ``which_qubit``, the first two qubits are control qubits and
            the last qubit is the target qubit.

        Args:
            name (str): triple qubit gate name
            which_qubit (list): qubit indices in the order of [control_1, control_2, target]
            signature (Any, optional): signature of the operation
            **params (Any): gate parameters
        """
        ctrl1, ctrl2, targ = which_qubit

        self.__check_qubit_validity(ctrl1)
        self.__check_qubit_validity(ctrl2)
        self.__check_qubit_validity(targ)
        if ctrl1 == ctrl2:
            raise TypeError(
                f"Invalid control qubit indices: {ctrl1} and {ctrl2}!\n" "The control qubits must not be the same."
            )
        elif ctrl1 == targ or ctrl2 == targ:
            raise TypeError(
                f"Invalid qubit indices: {ctrl1 if ctrl1 == targ else ctrl2} and {targ}!\n"
                "The control qubit must not be the same as the target qubit."
            )

        gate = {"name": name, "which_qubit": which_qubit, "signature": signature, **params}
        self._history.append(gate)

    def id(self, which_qubit: int, signature=None) -> None:
        r"""Add an identity gate.

        The matrix form is:

        .. math::

            I = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node, optional): node that implements the identity gate
        """
        self.__add_single_qubit_gate("id", which_qubit, signature)

    def h(self, which_qubit: int, signature=None, **condition) -> None:
        r"""Add a Hadamard gate.

        The matrix form is:

        .. math::

            H = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node, optional): node that implements the Hadamard gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("h", which_qubit, signature, **condition)

    def x(self, which_qubit: int, signature=None, **condition) -> None:
        r"""Add a Pauli-X gate.

        The matrix form is:

        .. math::

            X = \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node, optional): node that implements the Pauli-X gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("x", which_qubit, signature, **condition)

    def y(self, which_qubit: int, signature=None, **condition) -> None:
        r"""Add a Pauli-Y gate.

        The matrix form is:

        .. math::

            Y = \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node, optional): node that implements the Pauli-Y gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("y", which_qubit, signature, **condition)

    def z(self, which_qubit: int, signature=None, **condition) -> None:
        r"""Add a Pauli-Z gate.

        The matrix form is:

        .. math::

            Z = \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node, optional): node that implements the Pauli-Z gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("z", which_qubit, signature, **condition)

    def s(self, which_qubit: int, signature=None) -> None:
        r"""Add a S gate.

        The matrix form is:

        .. math::

            S = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node, optional): node that implements the S gate
        """
        self.__add_single_qubit_gate("s", which_qubit, signature)

    def t(self, which_qubit: int, signature=None) -> None:
        r"""Add a T gate.

        The matrix form is:

        .. math::

            T = \begin{bmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{bmatrix}

        Args:
            which_qubit (int): qubit index
            signature (Node, optional): node that implements the T gate
        """
        self.__add_single_qubit_gate("t", which_qubit, signature)

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
            signature (Node, optional): node that implements the rotation gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("rx", which_qubit, signature, angle=theta, **condition)

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
            signature (Node, optional): node that implements the rotation gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("ry", which_qubit, signature, angle=theta, **condition)

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
            signature (Node, optional): node that implements the rotation gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("rz", which_qubit, signature, angle=theta, **condition)

    def u(
        self,
        which_qubit: int,
        theta: Union[float, int],
        phi: Union[float, int],
        gamma: Union[float, int],
        signature=None,
        **condition,
    ) -> None:
        r"""Add a single qubit unitary gate.

        It has a decomposition form:

        .. math::

            \begin{align}
            U(\theta, \varphi, \gamma) = R_z(\varphi) R_x(\theta) R_z(\gamma) =
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
            signature (Node, optional): node that implements the unitary gate
            **condition (Any): condition of the operation
       """
        angles = [theta, phi, gamma]
        self.__add_single_qubit_gate("u", which_qubit, signature, angles=angles, **condition)

    def u3(
        self,
        which_qubit: int,
        theta: Union[float, int],
        phi: Union[float, int],
        gamma: Union[float, int],
        signature=None,
        **condition,
    ) -> None:
        r"""Add a single qubit unitary gate.

        It has a decomposition form:

        .. math::

            \begin{align}
            U_3(\theta, \varphi, \gamma) = R_z(\varphi) R_y(\theta) R_z(\gamma)
            \end{align}

        Warnings:
            Please be aware of the order of the rotation angles.

        Args:
            which_qubit (int): qubit index
            theta (Union[float, int]): the rotation angle of the Ry gate
            phi (Union[float, int]): the rotation angle of the left Rz gate
            gamma (Union[float, int]): the rotation angle of the right Rz gate
            signature (Node, optional): node that implements the unitary gate
            **condition (Any): condition of the operation
        """
        angles = [theta, phi, gamma]
        self.__add_single_qubit_gate("u3", which_qubit, signature, angles=angles, **condition)

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
            which_qubit (List[int]): a list of qubit indices in the order of [control, target]
            signature (Node, optional): node that implements the Controlled-Hadamard gate
        """
        self.__add_double_qubit_gate("ch", which_qubit, signature)

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
            which_qubit (List[int]): a list of qubit indices in the order of [control, target]
            signature (Node): node that implements the Controlled-X gate
        """
        self.__add_double_qubit_gate("cx", which_qubit, signature)

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
            which_qubit (List[int]): a list of qubit indices in the order of [control, target]
            signature (Node, optional): node that implements the Controlled-NOT gate
        """
        self.__add_double_qubit_gate("cx", which_qubit, signature)

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
            which_qubit (List[int]): a list of qubit indices in the order of [control, target]
            signature (Node, optional): node that implements the Controlled-Y gate
        """
        self.__add_double_qubit_gate("cy", which_qubit, signature)

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
            which_qubit (List[int]): a list of qubit indices in the order of [control, target]
            signature (Node, optional): node that implements the Controlled-Z gate
        """
        self.__add_double_qubit_gate("cz", which_qubit, signature)

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
            signature (Node, optional): node that implements the Controlled-rotation gate
        """
        self.__add_double_qubit_gate("crx", which_qubit, signature, angle=theta)

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
            signature (Node, optional): node that implements the Controlled-rotation gate
        """
        self.__add_double_qubit_gate("cry", which_qubit, signature, angle=theta)

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
            signature (Node, optional): node that implements the Controlled-rotation gate
        """
        self.__add_double_qubit_gate("crz", which_qubit, signature, angle=theta)

    def cu(
        self,
        which_qubit: List[int],
        theta: Union[float, int],
        phi: Union[float, int],
        gamma: Union[float, int],
        signature=None,
    ) -> None:
        r"""Add a Controlled-rotation gate.

        Args:
            which_qubit (List[int]): a list of qubit indices in the order of [control, target]
            theta (Union[float, int]): the rotation angle of the Rx gate
            phi (Union[float, int]): the rotation angle of the left Rz gate
            gamma (Union[float, int]): the rotation angle of the right Rz gate
            signature (Node, optional): node that implements the Controlled-rotation gate
        """
        angles = [theta, phi, gamma]
        self.__add_double_qubit_gate("cu", which_qubit, signature, angles=angles)

    def cu3(
        self,
        which_qubit: List[int],
        theta: Union[float, int],
        phi: Union[float, int],
        gamma: Union[float, int],
        signature=None,
    ) -> None:
        r"""Add a Controlled-rotation gate.

        Args:
            which_qubit (List[int]): a list of qubit indices in the order of [control, target]
            theta (Union[float, int]): the rotation angle of the Ry gate
            phi (Union[float, int]): the rotation angle of the left Rz gate
            gamma (Union[float, int]): the rotation angle of the right Rz gate
            signature (Node, optional): node that implements the Controlled-rotation gate
        """
        angles = [theta, phi, gamma]
        self.__add_double_qubit_gate("cu3", which_qubit, signature, angles=angles)

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
            which_qubit (List[int]): qubits to swap
            signature (Node, optional): node that implements the SWAP gate
        """
        self.__add_double_qubit_gate("cx", which_qubit, signature)
        self.__add_double_qubit_gate("cx", [which_qubit[1], which_qubit[0]], signature)
        self.__add_double_qubit_gate("cx", which_qubit, signature)

    def bsg(self, which_qubit: List[int], signature=None) -> None:
        r"""Add a H gate and CNOT gate to generate a Bell state.

        Args:
            which_qubit (List[int]): qubit to act on
            signature (Node, optional): node that implements the operation
        """
        self.h(which_qubit[0], signature=signature)
        self.cnot(which_qubit, signature=signature)

    def ccx(self, which_qubit: List[int], signature=None) -> None:
        r"""Add a Toffoli (CCNOT) gate.

        Let ``which_qubit`` be ``[0, 1, 2]``, the matrix form is:

        .. math::

            \begin{align}
                CCX =
                \begin{bmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
                \end{bmatrix}
            \end{align}

        Args:
            which_qubit (List[int]): a list of qubit indices in the order of [control_1, control_2, target]
            signature (Node): node that implements the Toffoli gate
        """
        self.__add_triple_qubit_gate("ccx", which_qubit, signature)

    def bit_flip(self, which_qubit: int, prob: Union[float, int], signature=None, **condition) -> None:
        r"""Add a bit flip noise.

        Args:
            which_qubit (int): qubit index
            prob (Union[float, int]): rotation angle
            signature (Node, optional): node that implements the rotation gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("bit_flip", which_qubit, signature, prob=prob, **condition)

    def phase_flip(self, which_qubit: int, prob: Union[float, int], signature=None, **condition) -> None:
        r"""Add a phase flip noise.

        Args:
            which_qubit (int): qubit index
            prob (Union[float, int]): rotation angle
            signature (Node, optional): node that implements the rotation gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("phase_flip", which_qubit, signature, prob=prob, **condition)

    def bit_phase_flip(self, which_qubit: int, prob: Union[float, int], signature=None, **condition) -> None:
        r"""Add a bit-phase flip noise.

        Args:
            which_qubit (int): qubit index
            prob (Union[float, int]): rotation angle
            signature (Node, optional): node that implements the rotation gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("bit_phase_flip", which_qubit, signature, prob=prob, **condition)

    def amplitude_damping(self, which_qubit: int, prob: Union[float, int], signature=None, **condition) -> None:
        r"""Add an amplitude damping noise.

        Args:
            which_qubit (int): qubit index
            prob (Union[float, int]): rotation angle
            signature (Node, optional): node that implements the rotation gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("amplitude_damping", which_qubit, signature, prob=prob, **condition)

    def phase_damping(self, which_qubit: int, prob: Union[float, int], signature=None, **condition) -> None:
        r"""Add a phase damping noise.

        Args:
            which_qubit (int): qubit index
            prob (Union[float, int]): rotation angle
            signature (Node, optional): node that implements the rotation gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("phase_damping", which_qubit, signature, prob=prob, **condition)

    def depolarizing(self, which_qubit: int, prob: Union[float, int], signature=None, **condition) -> None:
        r"""Add a depolarizing noise.

        Args:
            which_qubit (int): qubit index
            prob (Union[float, int]): rotation angle
            signature (Node, optional): node that implements the rotation gate
            **condition (Any): condition of the operation
        """
        self.__add_single_qubit_gate("depolarizing", which_qubit, signature, prob=prob, **condition)

    def reset(self, which_qubit: int, signature=None, matrix=None) -> None:
        r"""Reset a specific qubit for dynamic quantum circuit.

        Args:
            which_qubit (int): qubit index
            signature (Node, optional): node that implements the reset operation
            matrix (numpy.ndarray, optional): matrix representation of the quantum state to reset
        """
        self.__add_single_qubit_gate("r", which_qubit, signature, matrix=matrix)

    def measure(self, which_qubit=None, signature=None, mid=None) -> None:
        r"""Measure a quantum state in the computational basis.

        Measure all the qubits if no specific qubit index is given.

        Note:
            Each measurement is identified by a **unique** measurement ID which can be specified with ``mid``.
            It is recommended to set ``mid`` to a string.
            If no specific measurement ID is given, ``mid`` is set to a tuple
            containing the index of the measured qubit and the number of times it is measured.

        Args:
            which_qubit (int, optional): qubit index
            signature (Node, optional): node that implements the measurement
            mid (Any, optional): measurement ID
        """
        assert which_qubit is None or isinstance(which_qubit, int), f"Input {which_qubit} should be an int value."

        if which_qubit is None:  # measure all qubits in the circuit
            occupied_indices = copy.deepcopy(self.occupied_indices)
            for idx in occupied_indices:  # Z measurement
                self.__add_single_qubit_gate(
                    "m",
                    idx,
                    signature,
                    basis={"angle": 0, "plane": "YZ", "domain_s": [], "domain_t": []},
                    mid=(idx, self.measurement_counter(idx) + 1),
                )

        else:  # Z measurement
            if mid is not None:  # sanity check: 'mid' should be a unique identifier
                assert all(mid != gate.get("mid") for gate in self.gate_history)
            else:
                mid = (which_qubit, self.measurement_counter(which_qubit) + 1)
            self.__add_single_qubit_gate(
                "m", which_qubit, signature, basis={"angle": 0, "plane": "YZ", "domain_s": [], "domain_t": []}, mid=mid
            )

    def bsm(self, which_qubit: List[int], signature=None, mid=None) -> None:
        r"""Bell state measurement.

        Args:
            which_qubit (List[int]): qubit to act on
            signature (Node, optional): node that implements the operation
            mid (list, optional): a list of measurement IDs
        """
        assert mid is None or isinstance(mid, list), (
            f"Invalid measurement ID {mid.__repr__()} with {type(mid)} type!" f"Please provide a list input."
        )
        self.cnot([which_qubit[0], which_qubit[1]], signature=signature)
        self.h(which_qubit[0], signature=signature)
        self.measure(which_qubit=which_qubit[0], signature=signature, mid=None if mid is None else mid[0])
        self.measure(which_qubit=which_qubit[1], signature=signature, mid=None if mid is None else mid[1])

    def is_equal(self, other: "Circuit") -> bool:
        r"""Check if the current circuit is equal to another circuit.

        Args:
            other (Circuit): the other circuit to compare

        Returns:
            bool: whether the two quantum circuits equal
        """
        if self.width != other.width:  # check if the circuits have the same size
            return False

        for i in range(self.width):
            gates_1, gates_2 = [], []

            for gate_1, gate_2 in zip(self.gate_history, other.gate_history):
                if i in gate_1["which_qubit"]:
                    gates_1.append(gate_1)
                if i in gate_2["which_qubit"]:
                    gates_2.append(gate_2)

            for gate_1, gate_2 in zip(gates_1, gates_2):
                if gate_1 != gate_2:
                    if not (
                        gate_1["name"] == gate_2["name"] == "cz"
                        or gate_1["name"] == gate_2["name"] == "swap"
                        and set(gate_1["which_qubit"]) == set(gate_2["which_qubit"])
                    ):
                        return False
        return True

    def copy(self) -> "Circuit":
        r"""Make a copy of the current circuit.

        Returns:
            Circuit: a copy of the current circuit
        """
        new_circuit = Circuit()
        new_circuit._history = self.gate_history
        new_circuit.output_ids = self.output_ids
        return new_circuit

    def remap_indices(self, remap: Optional[dict] = None, print_index=False) -> None:
        r"""Remap the indices of quantum register.

        Args:
            remap (dict, optional): remap of the indices
            print_index (bool, optional): whether to print the remapped qubit indices

        Examples:
            Both the keys and the values of the remap dictionary should be of int type. Here is an example:

            >>> cir = Circuit()
            >>> cir.h(0)
            >>> cir.h(1)
            >>> cir.cnot([1, 3])
            >>> cir.cnot([2, 3])
            >>> cir.remap_indices(remap={0: 3, 1: 1, 2: 0, 3: 2})

            Note that the new indices should cover all indices in the current circuit, and the new indices should be
            a set of sequential integers starting from zero.

        Warnings:
            Some functionalities of the MBQC module require that the quantum register has sequential indices.
            We need to remap the indices before using the MBQC model and after deferring all measurements.

        Important:
            This method will directly update the current circuit. Please make a new copy of the circuit if necessary.
        """
        if remap is not None:
            assert set(remap.keys()) == set(self.occupied_indices), (
                f"The remap should contain all indices ({', '.join(str(i) for i in self.occupied_indices)}) "
                f"of the quantum register."
            )
            assert set(remap.values()) == set(
                range(self.width)
            ), f"The new indices should be a set of sequential integers starting from zero."

            # Sort by the new indices in ascending order
            remap = {value: key for key, value in zip(remap.keys(), remap.values())}
            remap = {remap[key]: key for key in sorted(remap.keys())}
        else:
            remap = {self.occupied_indices[i]: i for i in range(len(self.occupied_indices))}

        # Update the gate history
        for gate in self._history:
            new_indices = [remap[which_qubit] for which_qubit in gate["which_qubit"]]
            gate["which_qubit"] = new_indices

        if print_index and not all(key == value for key, value in remap.items()):
            print(f"\nThe quantum register indices have been remapped to (old: new): {remap}.")

    def defer_measurement(self) -> None:
        r"""Defer all quantum measurements to the end of the circuit.

        Note:
            In this method, classically conditioned gates will be transformed to quantum controlled gates.
        """
        # Count the measurement on each qubit
        measurement_counts = {str(i): 0 for i in self.occupied_indices}

        for gate in self.gate_history:
            if gate["name"] == "m":
                measurement_counts[str(gate["which_qubit"][0])] += 1

        # Check if any qubit is measured more than once
        if any(count > 1 for count in measurement_counts.values()):
            print(
                f"In 'defer_measurement': Dynamic circuits with multiple measurements "
                f"on a single qubit are not considered."
            )
            return

        measure_gates = []
        for i, gate in enumerate(self._history):
            if gate.get("condition") is not None:
                # Fixed single-qubit gate
                self._history[i] = {
                    "name": "c" + gate["name"],
                    "which_qubit": [self.get_qubit_by_mid(gate["condition"]), gate["which_qubit"][0]],
                    "signature": gate["signature"],
                }
                # Parameterized single-qubit gate
                if gate["name"] in ["rx", "ry", "rz"]:
                    self._history[i]["angle"] = gate["angle"]
                elif gate["name"] in ["u3", "u"]:
                    self._history[i]["angles"] = gate["angles"]

            elif gate["name"] == "m":
                measure_gates.append(self._history[i])

        self._history = list(filter(lambda x: x not in measure_gates, self._history))
        self._history.extend(measure_gates)

    def is_static(self) -> bool:
        r"""Check if a quantum circuit is a static quantum circuit.

        Note:
            A quantum circuit is static if it does not contain mid-circuit reset operations
            and all measurements are performed at the end of the circuit.

            In this context, we presume that at the end of a quantum circuit,
            all qubits within the circuit are measured.

        Returns:
            bool: if the circuit is a static quantum circuit
        """
        # Count the measurement on each qubit
        measurement_counts = {str(i): 0 for i in self.occupied_indices}

        for gate in self.gate_history:
            if gate["name"] == "m":
                measurement_counts[str(gate["which_qubit"][0])] += 1
            # Static circuit does not contain classically controlled quantum gates
            elif gate.get("condition") is not None:
                return False

        # In a static circuit, each qubit is measured at most once
        return not any(count > 1 for count in measurement_counts.values())

    def to_static(self) -> None:
        r"""Compile a dynamic quantum circuit into an equivalent static quantum circuit by adding ancillary qubits.

        Note:
            After a qubit is reset, all the operations on this qubit after the reset operation
            will be moved to a new quantum register where a new qubit with zero state is assigned,
            and the reset operation is removed from the circuit.

        Examples:
            Here is an example of compiling a dynamic quantum circuit into a static one:

            >>> from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit
            >>> cir = Circuit()
            >>> cir.h(0)
            >>> cir.h(1)
            >>> cir.h(2)
            >>> cir.cnot([0, 1])
            >>> cir.measure(1, mid='a')
            >>> cir.x(2, condition='a')
            >>> cir.reset(1)
            >>> cir.h(1)
            >>> cir.measure(2, mid='b')
            >>> cir.reset(2)
            >>> cir.h(2)
            >>> cir.cnot([0, 2])
            >>> cir.measure(1, mid='c')
            >>> cir.z(0, condition='c')
            >>> cir.measure(0, mid='d')
            >>> cir.measure(2, mid='e')
            >>> cir.print_circuit()
            >>> cir.to_static()
            >>> cir.print_circuit()
        """
        if self.is_static():
            print("\nIn 'to_static': The circuit is already a static quantum circuit.")
            return

        max_occupied_index = self.width - 1  # get the maximum occupied qubit index
        reset_gates = []  # record all reset gates in the circuit
        static_circuit = copy.deepcopy(self.gate_history)  # make a copy of gate_history list for processing

        for i, gate in enumerate(static_circuit):
            if gate["name"] == "r":
                reset_qubit = gate["which_qubit"][0]  # record the reset qubit
                max_occupied_index += 1  # initialize a new quantum register
                meas_on_reset = False  # no measurement on the reset qubit
                reset_gates.append(gate)  # record the reset gate
                post_reset_gates = static_circuit[i + 1 :]
                del static_circuit[i + 1 :]

                # Update qubit indices of subsequent gates on the reset qubit
                for post_gate in post_reset_gates:
                    # Subsequent single qubit operations acting on the reset qubit
                    if post_gate["which_qubit"] == [reset_qubit]:
                        post_gate["which_qubit"] = [max_occupied_index]
                        # Subsequent measurement on the reset qubit whose 'mid' is a tuple (default value)
                        if post_gate["name"] == "m" and isinstance(post_gate["mid"], tuple):
                            meas_on_reset = post_gate["mid"]
                            post_gate["mid"] = (max_occupied_index, 1)
                    # Classically controlled gate conditioned on the measurement after the reset qubit
                    elif post_gate.get("condition") == meas_on_reset:
                        post_gate["condition"] = (max_occupied_index, 1)
                    # Subsequent double qubit gate acting on the reset qubit
                    elif reset_qubit in post_gate["which_qubit"]:
                        post_gate["which_qubit"] = [
                            max_occupied_index if qubit == reset_qubit else qubit for qubit in post_gate["which_qubit"]
                        ]
                static_circuit.extend(post_reset_gates)

        # Update the processed list to gate_history
        self._history = static_circuit
        # Delete all reset gates in the circuit
        self._history = list(filter(lambda x: x not in reset_gates, self._history))
        # Defer measurement
        self.defer_measurement()

    def to_brickwork_circuit(self) -> "BrickworkCircuit":
        r"""Transpile a static quantum circuit into a brickwork circuit.

        Returns:
            BrickworkCircuit: an equivalent brickwork circuit
        """
        if not self.is_static():
            raise ArgumentTypeError(f"Please input a static quantum circuit.")

        from Extensions.QuantumNetwork.qcompute_qnet.models.bqc.brickwork import BrickworkCircuit

        bw_cir = BrickworkCircuit(self)
        bw_cir.to_brickwork_circuit()
        return bw_cir

    def to_brickwork_pattern(self, optimize=None, track=False) -> "Pattern":
        r"""Transpile a static quantum circuit into its equivalent brickwork pattern.

        Args:
            optimize (str, optional): whether to optimize the measurement order
            track (bool, optional): whether to track the transpiling progress

        Returns:
            Pattern: an equivalent brickwork pattern
        """
        if not self.is_static():
            raise ArgumentTypeError(f"Please input a static quantum circuit.")

        from Extensions.QuantumNetwork.qcompute_qnet.models.bqc.brickwork import BrickworkCalculus

        bw_cir = self.to_brickwork_circuit()
        mc = BrickworkCalculus()
        mc.track_progress(track)
        mc.set_circuit(bw_cir)
        mc.to_brickwork_pattern()
        mc.standardize()
        if optimize == "by_row":
            mc.optimize_by_row()
        elif optimize == "by_column":
            mc.optimize_by_column()

        pattern = mc.get_pattern()
        return pattern

    def to_pattern(self, shift_signal=False, optimize=None, track=False) -> "Pattern":
        r"""Transpile a static quantum circuit into its equivalent measurement pattern.

        Args:
            shift_signal (bool, optional): whether to perform signal sifting after standardization
            optimize (str, optional): whether to optimize the measurement order
            track (bool, optional): whether to track the transpiling progress

        Returns:
            Pattern: an equivalent measurement pattern

        Warnings:
            We should check if the circuit has sequential registers first.
            If not, we need to perform remapping before transpiling the circuit.
        """
        if not self.is_static():
            raise ArgumentTypeError(f"Please input a static quantum circuit.")

        circuit = self
        # Check if circuit has sequential registers
        if circuit.width != max(circuit.occupied_indices) + 1:
            new_circuit = circuit.copy()
            new_circuit.remap_indices()
            circuit = new_circuit

        circuit.simplify()  # simplify the circuit before transpiling it to a pattern
        from Extensions.QuantumNetwork.qcompute_qnet.quantum.mcalculus import MCalculus

        mc = MCalculus()
        mc.track_progress(track)
        mc.set_circuit(circuit)
        mc.to_pattern()
        mc.standardize()
        if shift_signal:
            mc.shift_signals()

        if optimize == "by_row":
            mc.optimize_by_row()
        elif optimize == "by_column":
            mc.optimize_by_column()

        pattern = mc.get_pattern()
        return pattern

    def __align_to_left(self) -> None:
        r"""Align all gates in the circuit by swapping.

        Note:
            This is an intrinsic method. No need to call it externally.
        """

        def _swap_two_gates(counter: int, two_gates: list) -> Tuple[int, list]:
            r"""Swap two sequential gates in the circuit list.

            In this method, the quantum gates in the circuit list are reordered.
            The qubit indices are arranged in an order from small to large.
            So the corresponding gates operated to these qubits are arranged from left to right in the list.
            But those quantum gates with the same row remain unchanged.

            Note:
                This is an intrinsic method. No need to call it externally.

            Args:
                counter (int): a counter to check whether the gate order is standard
                two_gates (list): a list of two gates to be swapped

            Returns:
                tuple: a counter to check whether the gate order is standard and a list of two new gates after swapping
            """
            gate_1 = two_gates[0]
            gate_2 = two_gates[1]
            vertex_list_1 = gate_1["which_qubit"]
            vertex_list_2 = gate_2["which_qubit"]

            # If there is at least one qubit in 'vertex_list_2' and 'vertex_list_1', do not swap them
            if [bit2 for bit2 in vertex_list_2 if bit2 in vertex_list_1]:
                swap_gates = two_gates

            else:
                # We want an order from small to large
                bit1_min = min(vertex_list_1)
                bit2_min = min(vertex_list_2)
                if bit2_min < bit1_min:
                    counter += 1
                    swap_gates = [gate_2, gate_1]
                else:
                    swap_gates = two_gates
            return counter, swap_gates

        # The idea of this method is the same as the idea of swapping commands in ``MCalculus``
        counter = 1
        while counter > 0:
            counter = 0
            for i in range(len(self._history) - 1):
                before_gates = self._history[:i]
                after_gates = self._history[i + 2 :]
                __swap_gates = self._history[i : i + 2]

                counter, new_gates = _swap_two_gates(counter, __swap_gates)
                self._history = before_gates + new_gates + after_gates

    def __merge_single_qubit_gates(self, gates: list) -> list:
        r"""Merge those sequential single qubit gates to form a new unitary gate.

        In this method, those sequential single qubit gates which operate on the same qubit are merged together.
        So that a new single qubit unitary gate is formed. In this way, the quantum circuit is simplified.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            gates (list): a list of single qubit gates to be merged

        Returns:
            list: a new single qubit unitary gate
        """

        def _to_matrix(gate: list) -> numpy.ndarray:
            r"""Convert a single qubit gate in the circuit list to its matrix form.

            Note:
                This is an intrinsic method. No need to call it externally.

            Args:
                gate (list): a single qubit gate

            Returns:
                numpy.ndarray: the matrix form of the gate
            """
            to_gate = {
                "h": Gate.H,
                "x": Gate.X,
                "y": Gate.Y,
                "z": Gate.Z,
                "s": Gate.S,
                "t": Gate.T,
                "rx": Gate.Rx,
                "ry": Gate.Ry,
                "rz": Gate.Rz,
                "u": Gate.U,
                "u3": Gate.U3,
            }

            if gate["name"] in ["h", "x", "y", "z", "s", "t"]:
                return to_gate[gate["name"]]()
            elif gate["name"] in ["rx", "ry", "rz"]:
                return to_gate[gate["name"]](gate["angle"])
            elif gate["name"] in ["u", "u3"]:
                theta, phi, gamma = gate["angles"]
                return to_gate[gate["name"]](theta, phi, gamma)
            elif gate["name"] in ["id", "m"]:
                return Gate.I()
            else:
                raise ArgumentTypeError(f"Invalid gate")

        if not gates:
            return gates

        u = []
        merged_gates = [[] for _ in range(self.width)]
        for gate in gates:
            which_qubit = gate["which_qubit"][0]
            matrix = _to_matrix(gate)
            merged_gates[which_qubit].append(matrix)

        for i in range(len(merged_gates)):
            row = merged_gates[i]
            if len(row) == 0:
                continue

            # Gate list should be inverse in consistence with the operation order
            row_inv = row[::-1]
            u_mat = reduce(lambda mat1, mat2: mat1 if len(row_inv) == 1 else matmul(mat1, mat2), row_inv)

            # Drop the gates if the 'u_mat' is an identity matrix
            if linalg.norm(u_mat - Gate.I()) < EPSILON:
                continue
            else:
                theta, phi, gamma = decompose_to_u_gate(u_mat)
                u.append({"name": "u", "which_qubit": [i], "angles": [theta, phi, gamma]})

        return u

    @staticmethod
    def __merge_double_qubit_gates(gates: list) -> list:
        r"""Merge those sequential double qubit gates to form a new one.

        In this method, those sequential double qubit gates which operate on the same qubit are merged together.
        So that a new double gate is formed. In this way, the quantum circuit is simplified.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            gates (list): a list of double qubit gates to be merged

        Returns:
            list: a new double qubit gate
        """
        merged_gates = []

        if not gates or len(gates) == 1:
            return gates

        for i in range(len(gates)):
            if not merged_gates:
                merged_gates.append(gates[i])
            else:
                gate_1 = merged_gates[-1]
                name_1 = gate_1["name"]
                which_qubits_1 = gate_1["which_qubit"]

                gate_2 = gates[i]
                name_2 = gate_2["name"]
                which_qubits_2 = gate_2["which_qubit"]

                # Merge CNOT gates if they operate on the same control and target qubits
                # Merge CZ gates if they operate on the same control and target qubits
                # Indeed for the CZ gate, it is no matter which qubit is control qubit and which qubit is target qubit
                if (
                    (name_1 == name_2 and which_qubits_1 == which_qubits_2)
                    or name_1 == name_2 == "cz"
                    and set(which_qubits_1) == set(which_qubits_2)
                ):
                    merged_gates.remove(gate_1)
                else:
                    merged_gates.append(gate_2)

        return merged_gates

    def __merge_sequential_gates(self) -> None:
        r"""Merge the sequential gates to simplify the quantum circuit.

        All sequential single qubit gates are merged by ``__merge_single_qubit_gates``.
        All sequential double qubit gates are merged by ``__merge_double_qubit_gates``.
        In this way, the quantum circuit is simplified.

        Note:
            This is an intrinsic method. No need to call it externally.
        """
        simple_circuit = []

        sgl_idx = 0  # index of single qubit gate
        dbl_idx = 0  # index of double qubit gate

        for i in range(len(self._history)):
            gate = self._history[i]
            name = gate["name"]

            # Find double qubit gates and merge those single qubit gates before them
            if name == "cx" or name == "cz":
                # Merge single qubit gates before them
                # The distance between two 'count_single' is the number of single qubit gates
                gates = self._history[sgl_idx:i]
                simple_circuit += self.__merge_single_qubit_gates(gates)
                sgl_idx = i + 1

            # Find single qubit gates and merge those double qubit gates before them
            else:
                # Merge double qubit gates before them
                # The distance between two 'count_double' is the number of double qubit gates
                gates = self._history[dbl_idx:i]
                simple_circuit += self.__merge_double_qubit_gates(gates)
                dbl_idx = i + 1

        # There are also some sequential single qubit gates in the end, so we need to merge them
        sgl_end = self._history[sgl_idx:]
        simple_circuit += self.__merge_single_qubit_gates(sgl_end)

        # There are also some sequential double qubit gates in the end, so we need to merge them
        dbl_end = self._history[dbl_idx:]
        simple_circuit += self.__merge_double_qubit_gates(dbl_end)

        # Add measurements to the circuit list
        meas_gates = [gate for gate in self._history if gate["name"] == "m"]

        self._history = simple_circuit + meas_gates

    def simplify(self) -> None:
        r"""Simplify a static quantum circuit by merging all sequential single or double qubit gates.

        In this method, the quantum gates are arranged in a standard order.
        All sequential single or double qubit gates are merged together.
        So that the quantum circuit is simplified.

        Note:
            This method can be applied to any static quantum circuit.
            The circuit depth can be reduced through this method.
        """
        self.__align_to_left()
        self.__merge_sequential_gates()

    def to_dag(self, reset=True, draw=False) -> Tuple[DiGraph, list, list]:
        r"""Convert a quantum circuit to its directed acyclic graph representation.

        Args:
            reset (bool, optional): whether to add a reset operation to the beginning of each qubit
            draw (bool, optional): whether to draw the directed acyclic graph

        Returns:
            tuple: a tuple containing the directed acyclic graph representation of the quantum circuit,
            a list of root vertices (vertices with zero indegree) in the graph
            and a list of terminal vertices (vertices with zero outdegree) in the graph
        """
        # Add a reset operation for each qubit as a state initialization
        occupied_indices = self.occupied_indices
        if reset is True:
            for i in reversed(sorted(occupied_indices)):
                self._history = [{"name": "r", "which_qubit": [i], "matrix": None}] + self._history

        # Initialize an empty directed graph
        graph = DiGraph()
        causal_list = [[] for i in range(self.width)]  # list to record the causal relation between vertices in DAG

        # Find gate position
        counter = {i: 0 for i in occupied_indices}
        for gate in self.gate_history:
            for idx in gate["which_qubit"]:
                counter[idx] += 1
            gate["position"] = (counter[gate["which_qubit"][0]], -gate["which_qubit"][0])

        # Construct the DAG representation of the static circuit
        for i, gate in enumerate(self.gate_history):
            # Add an ID for each gate element
            gate["gid"] = str(i)
            # Vertices in the graph correspond to gates in the quantum circuit
            node = gate["gid"]
            graph.add_node(node, pos=gate["position"])
            for qubit in gate["which_qubit"]:
                if len(causal_list[qubit]) > 0:
                    pre_node = causal_list[qubit][-1]
                    graph.add_edge(pre_node, node)
                    causal_list[qubit].append(node)
                else:
                    causal_list[qubit].append(node)

        # Identify root vertices and terminal vertices in the graph
        roots = [qubit[0] for qubit in causal_list]  # list of roots in DAG
        terminals = [qubit[-1] for qubit in causal_list]  # list of terminals in DAG

        # Draw the directed acyclic graph representation of the circuit
        if draw is True:
            self._draw_dag(graph, roots, terminals)

        return graph, roots, terminals

    @staticmethod
    def _draw_dag(graph: DiGraph, roots: List[Any], terminals: List[Any], added_edges=None) -> None:
        r"""Draw the directed acyclic graph representation of a quantum circuit.

        Args:
            graph (networkx.DiGraph): directed acyclic graph representation of a quantum circuit
            roots (List[Any]): a list of root vertices in the graph
            terminals (List[Any]): a list of terminal vertices in the graph
            added_edges (optional, List[Any]): a list of edges added to the graph
        """
        pos = nx.get_node_attributes(graph, "pos")
        nx.draw_networkx(graph, pos)
        # All vertices except roots and terminals are internal vertices
        internals = set(graph.nodes).difference(set(roots).union(set(terminals)))
        # Draw root vertices, internal vertices and terminal vertices with different color
        nx.draw_networkx_nodes(graph, pos, nodelist=roots, node_color="#F7A099", edgecolors="k")
        nx.draw_networkx_nodes(graph, pos, nodelist=terminals, node_color="#9CBCD5", edgecolors="k")
        nx.draw_networkx_nodes(graph, pos, nodelist=internals, node_color="w", edgecolors="k")
        if added_edges is not None:  # draw the added edges
            nx.draw_networkx_edges(graph, pos, edgelist=added_edges, edge_color="#259C59")
            plt.title("Added edges:" + str(added_edges))
        ax = plt.gca()
        ax.margins(0.2)
        plt.axis("off")
        plt.show()

    def _reorder_by_dag(self, graph: DiGraph, new_edges: List[Any]) -> None:
        r"""After adding some edges to the directed acyclic graph representation of a quantum circuit,
        reorder the gate history by the modified graph and a list of edges added to the graph.

        Args:
            graph (networkx.DiGraph): directed acyclic graph after adding some edges
            new_edges (List[Any]): a list of edges added to the graph
        """
        new_gate_history = []  # reordered gate history list
        # Apply topological sort to the modified graph to get a feasible execution order
        topological_sorting = nx.topological_sort(graph)

        # Gate-vertex mapping between quantum circuit and its DAG representation
        node_to_gate = {gate["gid"]: gate for gate in self.gate_history}
        for node in topological_sorting:
            new_gate_history.append(node_to_gate[node])

        # Update qubit indices according to the new edges
        for edge in new_edges:
            old_idx = node_to_gate[edge[1]]["which_qubit"][0]
            new_idx = node_to_gate[edge[0]]["which_qubit"][0]
            for gate in new_gate_history:
                gate["which_qubit"] = [new_idx if idx == old_idx else idx for idx in gate["which_qubit"]]

        self._history = new_gate_history

    @staticmethod
    def _get_candidate_matrix_by_graph_search(graph: DiGraph, roots: List[Any], terminals: List[Any]) -> numpy.ndarray:
        r"""Get the candidate matrix of the simplified bipartite graph corresponding to a quantum circuit
        by performing path search on the directed acyclic graph representation of the quantum circuit.

        Args:
            graph (networkx.Digraph): directed acyclic graph representation of a quantum circuit
            roots (List[Any]): a list of root vertices in the graph
            terminals (List[Any]): a list of terminal vertices in the graph

        Returns:
            numpy.ndarray: the candidate matrix of the simplified bipartite graph corresponding to a quantum circuit
        """
        # Initialize the biadjacency matrix of an empty quantum circuit
        biadjacency_matrix = numpy.zeros((len(roots), len(terminals)), dtype=int)
        # For each root-terminal pair, if there is a path from root to terminal,
        # then the entry corresponding to this root-terminal pair will be one otherwise zero.
        for i, root in enumerate(roots):
            for j, terminal in enumerate(terminals):
                if nx.has_path(graph, root, terminal):
                    biadjacency_matrix[i][j] += 1

        # Calculate the candidate matrix from the biadjacency matrix
        candidate_matrix = numpy.ones((len(terminals), len(roots)), dtype=int) - biadjacency_matrix.transpose()

        return candidate_matrix

    def _get_candidate_matrix_by_boolean_matrix(self) -> numpy.ndarray:
        r"""Get the candidate matrix of the simplified bipartite graph corresponding to a quantum circuit
        by boolean matrix multiplication.

        Returns:
            numpy.ndarray: the candidate matrix of the simplified bipartite graph corresponding to a quantum circuit
        """
        # Initialize the biadjacency matrix of an empty circuit
        biadjacency_matrix = numpy.identity(self.width, dtype=int)
        for i, gate in enumerate(self.gate_history):
            if len(gate["which_qubit"]) > 1:  # multi-qubit gate
                # Update the biadjacency matrix of the circuit after the multi-qubit gate
                columns_to_or = gate["which_qubit"]
                result = numpy.logical_or.reduce(biadjacency_matrix[:, columns_to_or], axis=1)
                biadjacency_matrix[:, columns_to_or] = result[:, numpy.newaxis]

        # Calculate the candidate matrix from the biadjacency matrix
        candidate_matrix = numpy.ones((self.width, self.width), dtype=int) - biadjacency_matrix.transpose()

        return candidate_matrix

    def is_reducible_graph(self) -> bool:
        r"""Determine if a static circuit is reducible from its graph representation.

        Note:
            A quantum circuit is reducible if it can be written as an equivalent dynamic quantum circuit
            with fewer qubits. Otherwise, it is irreducible.

        Returns:
            bool: whether the circuit is reducible
        """
        # Get the directed acyclic graph representation of the quantum circuit
        graph, roots, terminals = self.to_dag(reset=False)
        # Get the candidate matrix by searching paths in the graph
        candidate_matrix = self._get_candidate_matrix_by_graph_search(graph, roots, terminals)

        # The circuit is reducible if the candidate matrix has at least one non-zero entry
        # Otherwise it is irreducible
        return numpy.any(candidate_matrix)

    def is_reducible_reachability(self) -> bool:
        r"""Determine if the circuit is reducible from reachability between qubits.

        Note:
            A quantum circuit is reducible if it can be written as an equivalent dynamic quantum circuit
            with fewer qubits. Otherwise, it is irreducible.

        Returns:
            bool: whether the circuit is reducible
        """
        # Initialize a reachability list to store reachability information for each qubit
        # The i-th element is a reachability set for qubit i, containing all the qubits that that can reach qubit i
        reachability_list = [{i} for i in range(self.width)]  # each qubit is reachable for itself

        for gate in self.gate_history:
            if len(gate["which_qubit"]) == 2:  # only double qubit gate can affect the reachability
                ctrl, targ = gate["which_qubit"]
                union = reachability_list[ctrl].union(reachability_list[targ])
                # Any qubit in union set can reach both ctrl and targ qubit after the double qubit gate
                reachability_list[ctrl] = union
                reachability_list[targ] = union

        # The circuit is irreducible if every reachability set contains all qubits in the circuit
        # Otherwise it is reducible
        return not all(len(reachability_set) == self.width for reachability_set in reachability_list)

    def is_reducible_matrix(self) -> bool:
        r"""Determine if the circuit is reducible by boolean matrix multiplication.

        Note:
            A quantum circuit is reducible if it can be written as an equivalent dynamic quantum circuit
            with fewer qubits. Otherwise, it is irreducible.

        Returns:
            bool: whether the circuit is reducible
        """
        # Get the candidate matrix by boolean matrix multiplication
        candidate_matrix = self._get_candidate_matrix_by_boolean_matrix()

        # The circuit is reducible if the candidate matrix has at least one non-zero entry
        # Otherwise it is irreducible
        return numpy.any(candidate_matrix)

    def is_reducible(self, method=None) -> bool:
        r"""Determine if a static circuit is reducible by the specified method.

        Args:
            method(str, optional): method used to determine if a static circuit is reducible

        Note:
            A quantum circuit is reducible if it can be written as an equivalent dynamic quantum circuit
            with fewer qubits. Otherwise, it is irreducible.

            Three methods are supported for determining whether a static circuit is reducible, including:

            1. Paths searching on the graph representation of a static circuit ("graph")
            2. Reachability between qubits in a static circuit ("reachability")
            3. Boolean matrix multiplication ("matrix")

            If no method is specified, the reducibility is determined by boolean matrix multiplication by default.

        Returns:
            bool: whether the circuit is reducible

        Warning:
            Current method only support static quantum circuit as input.
        """
        if not self.is_static():
            print("\nIn is_reducible: Current method only support static quantum circuit as input.")
            raise NotImplementedError

        if method is None or method == "matrix":
            reducibility = self.is_reducible_matrix()
        elif method == "reachability":
            reducibility = self.is_reducible_reachability()
        elif method == "graph":
            reducibility = self.is_reducible_graph()
        else:
            raise NotImplementedError

        return reducibility

    def get_biadjacency_and_candidate_matrices(self) -> Tuple[numpy.ndarray, numpy.ndarray]:
        r"""Calculate the biadjacency matrix and the candidate matrix of the corresponding simplified graph
        of the quantum circuit by boolean matrix multiplication.

        Returns:
            tuple: a tuple containing the biadjacency matrix and the candidate matrix
            of the corresponding simplified graph of the quantum circuit
        """
        # Get the candidate matrix by boolean matrix multiplication
        candidate_matrix = self._get_candidate_matrix_by_boolean_matrix()
        # Calculate the candidate matrix from the biadjacency matrix
        biadjacency_matrix = numpy.ones((self.width, self.width), dtype=int) - candidate_matrix.transpose()

        return biadjacency_matrix, candidate_matrix

    @staticmethod
    def __update_candidate_matrix(candidate_matrix: numpy.ndarray, t: int, r: int) -> numpy.ndarray:
        r"""Update the candidate matrix after adding an edge from terminal indexed by t to root indexed by r.

        Args:
            candidate_matrix (numpy.ndarray): the candidate matrix to update
            t (int): index of the terminal vertex
            r (int): index of the root vertex

        Returns:
            numpy.ndarray: the updated candidate matrix after adding an edge
            from terminal indexed by t and root indexed by r
        """
        row, col = candidate_matrix.shape
        # Identify all roots capable of reaching terminal indexed by t
        root_set = {i for i in range(col) if candidate_matrix[t][i] == 0}
        # Identify all terminals that are reachable from root indexed by r
        terminal_set = {j for j in range(row) if candidate_matrix[j][r] == 0}
        # After adding this edge, any terminal in terminal_set is reachable from any root in root_set
        for pair in itertools.product(terminal_set, root_set):
            candidate_matrix[pair[0]][pair[1]] = 0
        # The added edges should not share common vertices
        candidate_matrix[t, :] = 0
        candidate_matrix[:, r] = 0

        return candidate_matrix

    def reduce_by_row_order_first(self) -> None:
        r"""Compile a static quantum circuit into an equivalent dynamic quantum circuit
        by the row-order-first algorithm.

        Note:
            The main idea of this algorithm is to rearrange the execution order of all operations by
            the row-order-first principle while keeping the final sampling results unaffected.

            That is, all operations on qubit with smaller index (row order) are executed first,
            then this qubit is measured and reset.
            After reset, the qubit can be reused to execute operations on another qubit that have not been executed yet.
        """

        def _update_qreg(qreg: dict, which_qubit: int) -> Tuple[dict, int]:
            r"""Update the unit status in the given quantum register and assign an available unit to the input qubit.

            Args:
                qreg (dict): quantum register to update
                which_qubit (int): the qubit to assign available unit

            Returns:
                tuple: updated quantum register and unit address assigned to the input qubit

            Note:
                We use a dict to implement quantum register where
                the key is a string that indicates the address of a register unit and
                the value is an int number that represents the qubit to which the register unit is assigned.
                A register unit whose value is None is an available unit.
            """
            if which_qubit in qreg.values():  # the qubit has already occupied a unit
                # Get unit address corresponding to the qubit
                address = int(find_keys_by_value(qreg, which_qubit)[0])

            else:  # no unit assigned for the qubit yet
                available_units = find_keys_by_value(qreg, None)  # find available units for the qubit
                # If all units are occupied, create a new unit for the qubit;
                # else, assign the available unit with the minimum index to the qubit.
                address = int(min(available_units)) if available_units else len(qreg)
                qreg[str(address)] = which_qubit  # update the quantum register

            return qreg, address

        # Convert classically conditioned gates to quantum controlled gates and defer all measurements to the end
        self.defer_measurement()

        # 1. Label 'cmd_index' for each operation
        # For single qubit operation, 'cmd_index' = [which_qubit, index_by_row],
        # where 'which_qubit' is the qubit that the gate acts on, 'index_by_row' is the order of the gate on this qubit.
        # For double qubit gate,
        # 'cmd_index' = [[control_qubit, control_index_by_row], [target_qubit, target_index_by_row]],
        # where 'control_qubit' and 'target_qubit' are qubits that the double qubit gate acts on,
        # 'control_index_by_row' and 'target_index_by_row'
        # are orders of the double qubit gate on each qubit respectively.
        for idx, gate in enumerate(self.gate_history):
            if len(gate["which_qubit"]) == 1:  # single qubit gate
                which_qubit = gate["which_qubit"][0]  # qubit index of the gate
                index_by_row = 1  # row index of the gate

                for i in range(idx - 1, -1, -1):  # traverse the list in reverse order, from current gate to the top
                    if which_qubit in self.gate_history[i]["which_qubit"]:
                        index_by_row += 1

                # Add 'cmd_index' to the single qubit gate
                gate["cmd_index"] = [which_qubit, index_by_row]

            elif len(gate["which_qubit"]) > 1:  # double qubit gate
                ctrl, targ = gate["which_qubit"]
                # Row indices of the double qubit gate on control qubit and target qubit
                ctrl_index_by_row, targ_index_by_row = 1, 1

                for i in range(idx - 1, -1, -1):
                    if ctrl in self.gate_history[i]["which_qubit"]:
                        ctrl_index_by_row += 1
                    if targ in self.gate_history[i]["which_qubit"]:
                        targ_index_by_row += 1

                # Add 'cmd_index' to the double qubit gate
                gate["cmd_index"] = [[ctrl, ctrl_index_by_row], [targ, targ_index_by_row]]

        # 2. Find domain for each gate
        # Domain of a gate is the 'cmd_index' of quantum gates must be executed before this gate.
        # Only quantum gate closest to the current gate on the same qubit is considered.
        for idx, gate in enumerate(self.gate_history):
            gate["domain"] = []  # initialize an empty list to store the domain for each gate

            if len(gate["which_qubit"]) == 1:  # single qubit gate
                which_qubit = gate["which_qubit"][0]  # qubit index of the gate
                for i in range(idx - 1, -1, -1):  # traverse the list in reverse order, starting from the current gate
                    if which_qubit in self.gate_history[i]["which_qubit"]:
                        gate["domain"].append(self.gate_history[i]["cmd_index"])
                        break  # if the closest gate is found, jump out of the for loop

            elif len(gate["which_qubit"]) > 1:  # double qubit gate
                ctrl, targ = gate["which_qubit"]
                ctrl_done, targ_done = False, False

                for i in range(idx - 1, -1, -1):
                    if not ctrl_done and ctrl in self.gate_history[i]["which_qubit"]:
                        gate["domain"].append(self.gate_history[i]["cmd_index"])
                        ctrl_done = True  # the closest gate on control qubit is found
                    if not targ_done and targ in self.gate_history[i]["which_qubit"]:
                        gate["domain"].append(self.gate_history[i]["cmd_index"])
                        targ_done = True  # the closest gate on target qubit is found
                    if ctrl_done and targ_done:
                        break  # if the closest gates on both qubits are found, jump out of the loop

        # 3. Rearrange the gates order in the circuit list
        # 3.1 Get the default order of the circuit list
        # Default ordering rule: the gate with smaller qubit index has a higher priority
        # For double qubit gates, take the smaller value in their 'which_qubit' list as their qubit index
        # For gates with the same qubit index, take their relative order in the original circuit list
        self._history = sorted(self.gate_history, key=lambda gate: min(gate["which_qubit"]))

        # 3.2 Get the optimal order of the circuit list according to the domain value of each gate
        for idx in range(len(self.gate_history)):
            optimal = False
            while not optimal:  # if the circuit list is not optimal, slice the circuit list into three parts
                executed = self.gate_history[:idx]
                executing = self.gate_history[idx]
                to_execute = self.gate_history[idx + 1 :]

                # List of 'cmd_index' of all operations prior to the current one
                executed_indices = [gate["cmd_index"] for gate in executed]
                # Find the 'cmd_index' of operations in domain but not in front of the current operation
                push_indices = [index for index in executing["domain"] if index not in executed_indices]

                # Find operations corresponding to 'cmd_index' in 'push_indices' list,
                # then remove them from 'to_execute' list and push them to the front of the current operation
                if push_indices:
                    # Operations in push list are sored by the default ordering rule
                    push = sorted(
                        [gate for gate in to_execute if gate["cmd_index"] in push_indices],
                        key=lambda gate: min(gate["which_qubit"]),
                    )
                    to_execute = [gate for gate in to_execute if gate["cmd_index"] not in push_indices]
                    self._history = self.gate_history[:idx] + push + [executing] + to_execute
                else:  # if no push operations, jump out of the while loop
                    optimal = True

        # 3.3 Push all measurement gates forward as far as possible
        for idx in range(len(self.gate_history)):
            if self.gate_history[idx]["name"] == "m":
                measured_qubit = self.gate_history[idx]["which_qubit"][0]  # measured qubit
                m_gate = self.gate_history[idx]
                del self._history[idx]  # delete the measurement gate

                # Traverse the list in reverse order, starting from the measurement gate
                for i in range(idx - 1, -1, -1):
                    if measured_qubit in self.gate_history[i]["which_qubit"]:
                        # Insert the measurement gate after the previous gate on the measured qubit then break the loop
                        self._history = self.gate_history[: i + 1] + [m_gate] + self.gate_history[i + 1 :]
                        break

        # 4. Compile the rearranged circuit list to a dynamic circuit list
        qreg = {}  # quantum register used to manage qubits on dynamic circuit
        dynamic_circuit = []  # dynamic circuit list

        for gate in self.gate_history:
            name = gate["name"]
            signature = gate["signature"]

            if len(gate["which_qubit"]) == 1:  # single qubit operation
                # Update status of register units
                qreg, which_qubit = _update_qreg(qreg, gate["which_qubit"][0])

                if name == "m":  # measurement operation
                    basis = gate["basis"]
                    mid = gate["mid"]

                    # Add a measurement operation on updated qubit to the dynamic circuit list
                    m_gate = {
                        "name": "m",
                        "which_qubit": [which_qubit],
                        "signature": signature,
                        "basis": basis,
                        "mid": mid,
                    }
                    dynamic_circuit.append(m_gate)

                    # Once a qubit is measured, recycle the register unit and reset the qubit
                    qreg[str(which_qubit)] = None
                    reset_gate = {"name": "r", "which_qubit": [which_qubit], "signature": None, "matrix": None}
                    dynamic_circuit.append(reset_gate)

                else:
                    # Add a single qubit gate on updated qubit to the dynamic circuit list
                    single_gate = {"name": name, "which_qubit": [which_qubit], "signature": signature}
                    if gate.get("angle") is not None:
                        single_gate["angle"] = gate["angle"]
                    elif gate.get("angles") is not None:
                        single_gate["angles"] = gate["angles"]

                    dynamic_circuit.append(single_gate)

            elif len(gate["which_qubit"]) > 1:  # double qubit operation
                ctrl, targ = gate["which_qubit"]

                # Update status of both register units
                qreg, ctrl = _update_qreg(qreg, ctrl)
                qreg, targ = _update_qreg(qreg, targ)

                # Add a double qubit gate on updated qubits to the dynamic circuit list
                double_gate = {"name": name, "which_qubit": [ctrl, targ], "signature": signature}
                if gate.get("angle") is not None:
                    double_gate["angle"] = gate["angle"]
                elif gate.get("angles") is not None:
                    double_gate["angles"] = gate["angles"]

                dynamic_circuit.append(double_gate)

        self._history = dynamic_circuit

    def _minimum_remaining_values_heuristic(
        self, candidate_matrix: numpy.ndarray, roots: List[Any], terminals: List[Any], method: str
    ) -> list:
        r"""Apply minimum remaining values heuristic algorithm to the candidate matrix of
        the simplified bipartite graph corresponding to a quantum circuit to obtain
        a list of edges can be added to the graph without introducing any directed cycles.

        Args:
            candidate_matrix (numpy.ndarray): the candidate matrix of the simplified bipartite graph
            roots (List[Any]): a list of root vertices in the graph
            terminals (List[Any]): a list of terminal vertices in the graph
            method (str): specific minimum remaining values heuristic method to use

        Returns:
            list: a list of edges can be added to the simplified bipartite graph
            without introducing any directed cycles
        """

        def _add_one_edge_by_mrv_root(candidate):
            r"""Identify a candidate edge by the minimum remaining values heuristic algorithm
            with respect to root vertices and update the candidate matrix after adding the edge.

            Args:
                candidate (numpy.ndarray): the candidate matrix of the graph

            Returns:
                tuple: a tuple containing a candidate edge added to the graph
                and the updated candidate matrix after adding this edge to the graph
            """
            # If there is no more candidate edge, then return an empty edge
            if numpy.count_nonzero(candidate) == 0:
                return tuple(), candidate

            # Find the root with the least number (non-zero) of terminals that can be connected
            candidate_num_r = numpy.sum(candidate, axis=0)
            target_root = numpy.where(candidate_num_r == numpy.min(candidate_num_r[numpy.nonzero(candidate_num_r)]))[0][
                0
            ]
            # Find all candidate terminals that can be connected to the target root
            candidate_terminals = numpy.where(candidate[:, target_root] == 1)[0]
            # Find the terminal with the fewest roots that can be connected
            candidate_num_t = numpy.sum(candidate, axis=1)
            target_terminal = candidate_terminals[numpy.argmin(candidate_num_t[candidate_terminals])]
            # Record the added edge
            edge = (target_terminal, target_root)

            # Update the candidate matrix after adding a candidate edge
            candidate = self.__update_candidate_matrix(candidate, target_terminal, target_root)

            return edge, candidate

        def _add_one_edge_by_mrv_terminal(candidate):
            r"""Identify a candidate edge by the minimum remaining values heuristic algorithm
            with respect to terminal vertices and update the candidate matrix after adding the edge.

            Args:
                candidate (numpy.ndarray): the candidate matrix of the graph

            Returns:
                tuple: a tuple containing a candidate edge added to the graph
                and the updated candidate matrix after adding this edge to the graph
            """
            # If there is no more candidate edge, then return an empty edge
            if numpy.count_nonzero(candidate) == 0:
                return tuple(), candidate

            # Find the terminal with the least number (non-zero) of roots that can be connected
            candidate_num_t = numpy.sum(candidate, axis=1)
            target_terminal = numpy.where(
                candidate_num_t == numpy.min(candidate_num_t[numpy.nonzero(candidate_num_t)])
            )[0][0]
            # Find all candidate roots that can be connected to the target terminal
            candidate_roots = numpy.where(candidate[target_terminal] == 1)[0]
            # Find the root with the fewest terminals that can be connected
            candidate_num_r = numpy.sum(candidate, axis=0)
            target_root = candidate_roots[numpy.argmin(candidate_num_r[candidate_roots])]
            # Record the added edge
            edge = (target_terminal, target_root)

            # Update the candidate matrix after adding a candidate edge
            candidate = self.__update_candidate_matrix(candidate, target_terminal, target_root)

            return edge, candidate

        # Identify the specified heuristic method
        if method == "root":
            _heuristic_method = _add_one_edge_by_mrv_root
        elif method == "terminal":
            _heuristic_method = _add_one_edge_by_mrv_terminal
        else:
            raise NotImplementedError

        added_edges = []
        # Iteratively add candidate edges to the graph until the candidate matrix becomes a zero matrix
        flag = 1
        while flag > 0:
            new_edge, candidate_matrix = _heuristic_method(candidate_matrix)
            if len(new_edge) != 0:
                added_edge = (terminals[new_edge[0]], roots[new_edge[1]])
                added_edges.append(added_edge)
            flag = len(new_edge)

        return added_edges

    def reduce_by_minimum_remaining_values(self, method=None, draw=False) -> None:
        r"""Compile a quantum circuit into an equivalent dynamic circuit with fewer qubits
        by the specified method based on the minimum remaining values (MRV) heuristic algorithm.

        Args:
            method (optional, str): specific method based on the minimum remaining values heuristic
            draw: (optional, bool): whether to draw the modified graph with added edges

        Note:
            There are two different methods based on the minimum remaining values heuristic:

            1. MRV with respect to root vertices ("root")
            2. MRV with respect to terminal vertices ("terminal")

            If no method is specified, then by default the circuit will be compiled by two methods respectively
            and return the dynamic circuit with smaller circuit width.
        """
        # Get the graph representation of the circuit
        graph, roots, terminals = self.to_dag()
        # Obtain the initial candidate matrix of the graph
        candidate_matrix = self._get_candidate_matrix_by_boolean_matrix()

        if method is None:
            # If no method is specified, run both methods respectively
            candidate = copy.deepcopy(candidate_matrix)
            new_edges_root = self._minimum_remaining_values_heuristic(candidate, roots, terminals, "root")
            candidate = copy.deepcopy(candidate_matrix)
            new_edges_terminal = self._minimum_remaining_values_heuristic(candidate, roots, terminals, "terminal")
            # Identify the compilation strategy with fewer qubits
            added_edges = max((new_edges_root, new_edges_terminal), key=len)
        else:
            added_edges = self._minimum_remaining_values_heuristic(candidate_matrix, roots, terminals, method)

        # Compile the input static circuit into dynamic circuit with the modified graph and added edges
        new_graph = graph.copy()
        new_graph.add_edges_from(added_edges)
        if draw is True:  # draw the modified graph with added edges
            self._draw_dag(new_graph, roots, terminals, added_edges)
        self._reorder_by_dag(new_graph, added_edges)
        self.remap_indices(print_index=False)

    def _greedy_heuristic(
        self, candidate_matrix: numpy.ndarray, roots: List[Any], terminals: List[Any], method: str
    ) -> list:
        r"""Apply greedy heuristic algorithm to the candidate matrix of the simplified bipartite graph
        corresponding to a quantum circuit to obtain a list of edges can be added to the graph
        without introducing any directed cycles.

        Args:
            candidate_matrix (numpy.ndarray): the candidate matrix of the simplified bipartite graph
            roots (List[Any]): a list of root vertices in the graph
            terminals (List[Any]): a list of terminal vertices in the graph
            method (str): specific method to identify the local optimum during each iteration of the greedy algorithm

        Returns:
            list: a list of edges can be added to the simplified bipartite graph
            without introducing any directed cycles
        """

        def _add_one_edge_by_greedy(candidate: numpy.ndarray, method=method) -> Tuple[tuple, numpy.ndarray]:
            r"""Identify a candidate edge by greedy heuristic algorithm and update the candidate matrix after
            adding this candidate edge.

            Args:
                candidate (numpy.ndarray): the candidate matrix of the graph
                method (str): specific method to identify the candidate edge during each iteration

            Returns:
                tuple: a tuple containing a candidate edge added to the graph and the updated candidate matrix
                after adding this edge to the graph.
            """
            # If there is no more candidate edge, then return an empty edge
            if numpy.count_nonzero(candidate) == 0:
                return tuple(), candidate

            # Initialize a zero matrix to record scores of candidate edges
            score_matrix = numpy.zeros((candidate.shape[0], candidate.shape[1]), dtype=int)
            # Calculate the scores of all candidate edges
            for i in range(candidate.shape[0]):
                for j in range(candidate.shape[1]):
                    if candidate[i][j] != 0:  # non-zero entry indicates a candidate edge
                        # Update the candidate matrix after adding the candidate edge (t_i, r_j)
                        candidate_copy = copy.deepcopy(candidate)
                        candidate_copy = self.__update_candidate_matrix(candidate_copy, i, j)
                        # Use the number of remaining candidate as the score of candidate edge (t_i, r_j)
                        # Plus one is used to distinguish candidate edges from non-candidate edges
                        score_matrix[i][j] += numpy.sum(candidate_copy) + 1

            # Identify the target edge as the candidate edge with the highest score
            if method == "deterministic":
                # Deterministically select the first one if multiple candidate edges share the highest score
                edge = numpy.unravel_index(numpy.argmax(score_matrix), score_matrix.shape)
                u, v = edge
            elif method == "random":
                # Randomly select one if multiple candidate edges share the highest score
                max_score_candidates = numpy.where(score_matrix == numpy.max(score_matrix))
                index = random.randint(0, len(max_score_candidates[0]) - 1)
                u = max_score_candidates[0][index]
                v = max_score_candidates[1][index]
                edge = (u, v)
            else:
                raise NotImplementedError

            # Update the candidate matrix after adding the target edge
            candidate = self.__update_candidate_matrix(candidate, u, v)

            return edge, candidate

        added_edges = []
        # Iteratively add candidate edges to the graph until the candidate matrix becomes a zero matrix
        flag = 1
        while flag > 0:
            new_edge, candidate_matrix = _add_one_edge_by_greedy(candidate_matrix, method=method)
            if len(new_edge) != 0:
                added_edge = (terminals[new_edge[0]], roots[new_edge[1]])
                added_edges.append(added_edge)
            flag = len(new_edge)

        return added_edges

    def reduce_by_greedy(self, method=None, shots=1, draw=False) -> None:
        r"""Compile a quantum circuit into an equivalent dynamic circuit with fewer qubits
        by the greedy heuristic algorithm.

        Args:
            method (optional, str): specific method to identify the candidate edge at each iteration
            shots: (optional, int): number of times to run random greedy algorithm
            draw: (optional, bool): whether to draw the modified graph with added edges

        Note:
            If there are multiple local optimal candidate edges during one iteration,
            there are two different methods to identify which one to choose:

            1. Deterministic selection: always select the first one encountered ("deterministic")
            2. Random selection: randomly select one from all local optima ("random")

            If no method is specified, then by default the random greedy algorithm will be executed once.

            If 'shots' is set to a value greater than one, the random greedy algorithm will be run multiple
            times. It will return the dynamic circuit with the minimal circuit width across all runs.

            Multiple runs of the deterministic greedy algorithm produce consistent result.
        """
        # Get the graph representation of the circuit
        graph, roots, terminals = self.to_dag()
        # Obtain the initial candidate matrix of the graph
        candidate_matrix = self._get_candidate_matrix_by_boolean_matrix()
        # Get a list of edges added to the graph with specified method
        if method == "random" or method is None:
            added_edges = []
            # Run random greedy algorithm multiple times and return the dynamic circuit with the minimal width
            for shot in range(shots):
                candidate_copy = copy.deepcopy(candidate_matrix)
                new_edges = self._greedy_heuristic(candidate_copy, roots, terminals, method="random")
                if len(new_edges) > len(added_edges):
                    added_edges = new_edges
        elif method == "deterministic":
            if shots > 1:
                print("\nThe deterministic greedy algorithm produces consistent results across multiple runs.")
            added_edges = self._greedy_heuristic(candidate_matrix, roots, terminals, method="deterministic")
        else:
            raise NotImplementedError

        # Compile the input static circuit into dynamic circuit with the modified graph and added edges
        new_graph = graph.copy()
        new_graph.add_edges_from(added_edges)
        if draw is True:  # draw the modified graph with added edges
            self._draw_dag(new_graph, roots, terminals, added_edges)
        self._reorder_by_dag(new_graph, added_edges)
        self.remap_indices(print_index=False)

    def reduce_by_hybrid(self, level: int, draw=False) -> None:
        r"""Compile a quantum circuit into an equivalent dynamic circuit with fewer qubits
        by the hybrid algorithm which combines the minimum remaining values heuristic algorithm
        with brute force search.

        Args:
            level (int): hierarchy level of the hybrid algorithm
            draw (optional, bool): whether to draw the modified graph with added edges
        """
        # Get the graph representation of the circuit
        graph, roots, terminals = self.to_dag()
        # Obtain the initial candidate matrix of the graph
        candidate_matrix = self._get_candidate_matrix_by_boolean_matrix()
        # Initialize the optimal solution
        optimal_edges = []

        # Calculate the number of candidate roots for each terminal
        candidate_num_t = numpy.sum(candidate_matrix, axis=1)
        # Identify terminals with non-zero candidate roots
        non_zero_indices = numpy.nonzero(candidate_num_t)[0]
        # Sort the terminals in ascending order of the number of candidate roots.
        sorted_indices = non_zero_indices[numpy.argsort(candidate_num_t[non_zero_indices])]
        enumerate_terminals = []  # indices of terminals to enumerate
        enumerate_roots = []  # indices of the candidate roots corresponding to terminals to be enumerated
        # Find all the terminals to enumerate and their corresponding candidate roots based on the hierarchy level
        for terminal_index in sorted_indices[:level]:
            enumerate_terminals.append(terminal_index)
            candidate_root_indices = list(numpy.nonzero(candidate_matrix[terminal_index, :])[0]) + [None]
            enumerate_roots.append(candidate_root_indices)
        search_space = []  # used to record all feasible solutions that need to be enumerated
        # Solutions without repeated candidate roots are feasible solutions
        for solution in itertools.product(*enumerate_roots):
            new_solution = list(filter(lambda x: x is not None, solution))
            if len(set(new_solution)) == len(new_solution):
                search_space.append(solution)

        # Brute force search overall feasible solution space
        for i, solution in enumerate(search_space):
            print_progress((i + 1) / len(search_space), "Enumeration Progress:")
            # Make a copy of the graph, roots and terminals list for updating
            new_graph = copy.deepcopy(graph)
            new_roots = copy.deepcopy(roots)
            new_terminals = copy.deepcopy(terminals)
            # Identify the index of roots and terminals need to be updated
            added_edges_indices = [(t, r) for t, r in zip(enumerate_terminals, solution)]
            # Identify all added edges in this enumeration
            added_edges = [(terminals[t], roots[r]) for t, r in added_edges_indices if r is not None]
            new_graph.add_edges_from(added_edges)

            # Check whether added edges will introduce cycle to the graph.
            # If not, all vertices corresponding to added edges are deleted from the copied roots and terminals lists
            if nx.is_directed_acyclic_graph(new_graph):
                for t, r in added_edges_indices:  # update roots and terminals after adding some edges
                    if r is not None:
                        new_roots.remove(roots[r])
                    new_terminals.remove(terminals[t])
                # Get the candidate matrix of the DAG with newly added edges by graph search
                new_candidate_matrix = self._get_candidate_matrix_by_graph_search(new_graph, new_roots, new_terminals)
                # Apply mrv heuristic algorithm to identify the remaining added edges
                new_edges_by_mrv = self._minimum_remaining_values_heuristic(
                    new_candidate_matrix, new_roots, new_terminals, method="terminal"
                )
                added_edges.extend(new_edges_by_mrv)
                # Compare the number of reduced qubits in this enumeration with the current optimal solution
                if len(added_edges) > len(optimal_edges):
                    optimal_edges = added_edges

        # Update the graph based on the optimal solution found
        optimal_graph = graph.copy()
        # Update the graph based on the optimal solution found
        optimal_graph.add_edges_from(optimal_edges)
        if draw is True:  # draw the modified graph with added edges
            self._draw_dag(optimal_graph, roots, terminals, added_edges=optimal_edges)
        # Compile the input static circuit into dynamic circuit with the modified graph and added edges
        self._reorder_by_dag(optimal_graph, optimal_edges)
        self.remap_indices(print_index=False)

    def reduce_by_brute_force_search(self, draw=False) -> None:
        r"""Compile a quantum circuit into an equivalent dynamic quantum circuit with fewer qubits
        by brute-force searching the optimal compilation.

        Note:
            For dynamic circuit compilation, we aim to add as many directed edges from terminal vertices
            to root vertices as possible to the directed acyclic graph representation of the input static circuit
            without introducing any directed cycle.

        Args:
            draw (bool, optional): whether to draw the optimal modified graph with added edges
        """

        def _find_by_brute_force(
            di_graph: DiGraph, roots_list: List[Any], terminals_list: List[Any]
        ) -> Tuple[DiGraph, list]:
            r"""Brute force search the optimal compilation.

            Note:
                For dynamic circuit compilation, we aim to add as many directed edges from terminal vertices
                to root vertices in the DAG representation of the input static circuit as possible, without
                introducing any directed cycles.

            Args:
                di_graph (networkx.Digraph): directed acyclic graph representation of a quantum circuit
                roots_list (List[Any]): a list of root vertices in the graph
                terminals_list (List[Any]): a list of terminal vertices in the graph

            Returns:
                tuple: a tuple containing the optimal modified graph and the list of edges added to the graph
            """
            enumerate_roots = []
            # Identify a list of candidate roots for each terminal
            for terminal in terminals_list:
                # None indicates this terminal is not connected to any root
                candidate_roots = [root for root in roots_list if not nx.has_path(di_graph, root, terminal)] + [None]
                enumerate_roots.append(candidate_roots)
            search_space = []  # search space is the Cartesian product of candidate roots of all terminals
            # Solutions without repeated candidate root are feasible solutions
            for solution in itertools.product(*enumerate_roots):
                feasible_solution = list(filter(lambda x: x is not None, solution))
                if len(feasible_solution) == len(set(feasible_solution)):
                    search_space.append({"solution": solution, "weight": len(feasible_solution)})
            # Sort the search space according to the number of edges in each solution in descending order
            sorted_search_space = sorted(search_space, key=lambda s: s["weight"], reverse=True)

            # Brute-Force search all solutions in search space
            for i, solution in enumerate(sorted_search_space):
                print_progress(i / len(sorted_search_space), "Brute-Force Progress:")
                new_graph = di_graph.copy()
                new_edges = [
                    (terminal, root) for terminal, root in zip(terminals_list, solution["solution"]) if root is not None
                ]
                new_graph.add_edges_from(new_edges)
                # The optimal solution corresponds to the one that maximizes the number of edges
                # without introducing any directed cycle.
                if nx.is_directed_acyclic_graph(new_graph):
                    return new_graph, new_edges

        # Get the graph representation of the circuit
        graph, roots, terminals = self.to_dag()
        # Find the optimal compilation by brute force searching
        modified_graph, added_edges = _find_by_brute_force(graph, roots, terminals)
        if draw is True:  # draw the modified graph with added edges
            self._draw_dag(modified_graph, roots, terminals, added_edges)
        # Compile the input static circuit into dynamic circuit with the modified graph and added edges
        self._reorder_by_dag(modified_graph, added_edges)
        self.remap_indices(print_index=False)

    def reduce(self, method: str, level=None, shots=1, draw=False) -> None:
        r"""Compile a quantum circuit into an equivalent dynamic circuit with fewer qubits by the specified method.

        Args:
            method (str): specific dynamic circuit compilation method
            level (optinal, int): the hierarchy level of the hybrid algorithm
            shots (optional, int): the number of times to run random greedy algorithm
            draw (optional, bool): whether to draw the modified graph with added edges

        Note:
            There are multiple algorithms supported for the dynamic circuit compilation, including:

            1. row-order-first algorithm ("row_order_first")
            2. minimum remaining values algorithm ("minimum_remaining_values")
            3. minimum remaining values algorithm with respect to root vertices ("minimum_remaining_values_root")
            4. minimum remaining values algorithm with respect
               to terminal vertices ("minimum_remaining_values_terminal")
            5. greedy algorithm ("greedy")
            6. deterministic greedy algorithm ("deterministic_greedy")
            7. random greedy algorithm ("random_greedy")
            8. hybrid algorithm ("hybrid")
            9. brute-force searching ("brute_force")

            Please see the reference [Dynamic quantum circuit compilation arXiv:2310.11021] for technical details.

            In this context, we presume that all quantum circuits start with reset operations applied
            to all qubits and ends with measurements performed on all qubits.

            Here is an example of dynamic circuit compilation:

            >>> from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit
            >>> cir = Circuit()
            >>> cir.h(0)
            >>> cir.h(1)
            >>> cir.h(2)
            >>> cir.h(3)
            >>> cir.rx(0, 0.1)
            >>> cir.ry(1, 0.2)
            >>> cir.rz(2, 0.3)
            >>> cir.u(3, 0.1, 0.2, 0.3)
            >>> cir.cnot([0, 1])
            >>> cir.cz([1, 2])
            >>> cir.cnot([2, 3])
            >>> cir.measure(0, mid='a')
            >>> cir.measure(1, mid='b')
            >>> cir.measure(2, mid='c')
            >>> cir.measure(3, mid='d')
            >>> cir.output_ids = ['a', 'b', 'c', 'd']
            >>> cir.print_circuit()
            >>> cir.reduce(method="minimum_remaining_values")
            >>> cir.print_circuit()

        Warning:
            For a dynamic circuit as input, it is recommended to start by calling the 'to_static' method to convert
            the dynamic circuit into a static circuit. Following this, the 'reduce' method can be used to explore
            whether the original dynamic circuit width can be further reduced.
        """
        if method == "row_order_first":
            self.reduce_by_row_order_first()
        elif method == "minimum_remaining_values":
            self.reduce_by_minimum_remaining_values(draw=draw)
        elif method == "minimum_remaining_values_root":
            self.reduce_by_minimum_remaining_values(method="root", draw=draw)
        elif method == "minimum_remaining_values_terminal":
            self.reduce_by_minimum_remaining_values(method="terminal", draw=draw)
        elif method == "greedy":
            self.reduce_by_greedy(method="random", shots=shots, draw=draw)
        elif method == "deterministic_greedy":
            self.reduce_by_greedy(method="deterministic", shots=shots, draw=draw)
        elif method == "random_greedy":
            self.reduce_by_greedy(method="random", shots=shots, draw=draw)
        elif method == "hybrid":
            if level is None:
                raise ArgumentTypeError("\nPlease specify the hierarchy level of the hybrid algorithm.")
            else:
                self.reduce_by_hybrid(level=level, draw=draw)
        elif method == "brute_force":
            self.reduce_by_brute_force_search(draw=draw)
        else:
            raise NotImplementedError

    def run(self, shots: int, backend: Optional[Enum] = None, token: Optional[str] = None) -> dict:
        r"""Run the quantum circuit with a given backend.

        Args:
            shots (int): the number of sampling
            backend (Enum, optional): backend to run the quantum circuit
            token (str, optional): your token for QCompute backend

        Returns:
            dict: circuit results, including the circuit's name, sampling shots and sampling results
        """
        if backend is not None and not hasattr(backend, "name"):
            raise ArgumentTypeError(
                f"{backend} has no attribute 'name'. " f"Please assign a specific backend for {backend}."
            )

        backend = Backend.QNET.StateVector if backend is None else backend

        if backend.name in Backend.QNET.__members__:
            results = self.run_circuit(shots=shots, backend=backend)
        elif backend.name in Backend.QCompute.__members__:
            results = qcompute.run_circuit(self, shots=shots, backend=backend, token=token)
        else:
            raise ArgumentTypeError(f"Cannot find the backend {backend}.")

        cir_results = {
            "backend": backend.name,
            "circuit_name": self.name,
            "shots": shots,
            "counts": self.sort_results(results),
        }
        return cir_results

    def run_circuit(self, shots: int, backend: Enum) -> dict:
        r"""Run a quantum circuit by QNET StateVector backend.

        Warnings:
            We should check if the circuit has sequential registers first.
            If not, we need to perform remapping before running the circuit.

        Args:
            shots (int): number of sampling
            backend (Enum): specific QNET backend

        Returns:
            dict: circuit sampling results
        """
        # MBQC backend
        if backend == Backend.QNET.MBQC:
            return mbqc.run_circuit(self, shots=shots)

        # Remap the indices to sequential integers starting from zero
        if self.width != max(self.occupied_indices) + 1:
            remap_circuit = self.copy()
            remap_circuit.remap_indices()
            self = remap_circuit

        samples = []

        for i in range(shots):
            # Initialize a quantum state
            if backend == Backend.QNET.StateVector:
                global_state = PureState(substates=[PureState.SubState(Zero.SV, [i]) for i in range(self.width)])
            elif backend == Backend.QNET.DensityMatrix:
                global_state = MixedState(substates=[MixedState.SubState(Zero.DM, [i]) for i in range(self.width)])

            # Evolve the quantum state by all gates in the circuit
            global_state.evolve_by_gates(self.gate_history)
            outcome = global_state.outcome

            # Suppose all qubits are eventually measured
            if self.output_ids is None:  # use the last measurement result
                s = ""
                for j in range(self.width):
                    outcome_by_row = {k: v for k, v in outcome.items() if k[0] == j or self.get_qubit_by_mid(k) == j}
                    max_tuple = max(outcome_by_row.keys(), key=lambda tup: tup[1])
                    s += str(outcome[max_tuple])
                samples.append(s)
            else:  # obtain the result from the specific output information, e.g. output of a pattern
                s = ""
                for mid in self.output_ids:
                    try:
                        s += str(outcome[mid])
                    except KeyError:  # 'mid' in 'output_ids' is not specified before
                        raise ArgumentTypeError(f"Invalid measurement ID in 'output_ids'.")
                samples.append(s)

        sample_dict = {}
        for string in list(set(samples)):
            sample_dict[string] = samples.count(string)

        counts = self.sort_results(sample_dict)

        return counts

    @staticmethod
    def reduce_results(results: dict, indices: List[int]) -> dict:
        r"""Reduce the circuit sampling results with specific indices.

        Args:
            results (dict): circuit sampling results to reduce
            indices (List[int]): global indices for reducing circuit results

        Returns:
            dict: reduced results
        """
        assert all(
            indices[i] < indices[i + 1] for i in range(len(indices) - 1)
        ), f"The indices should be in ascending order."

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
        r"""Print the events scheduled for the circuit."""
        from Extensions.QuantumNetwork.qcompute_qnet.core.des import Event

        df = Event.events_to_dataframe(self.agenda)
        print(f"\nAgenda of {self.name} (unsorted):\n{df.to_string()}")

    def print_list(self) -> None:
        r"""Print the quantum circuit list."""
        df = pd.DataFrame(columns=["name", "which_qubit", "signature", "params"])

        for i, gate in enumerate(self.gate_history):
            gate_params = {
                key: gate[key] for key in list(filter(lambda x: x not in ["name", "which_qubit", "signature"], gate))
            }
            circuit_info = pd.DataFrame(
                {
                    "name": gate["name"],
                    "which_qubit": str(gate["which_qubit"]),
                    "signature": gate["signature"].name if gate.get("signature") is not None else None,
                    "params": str(gate_params),
                },
                index=[f"Gate {i + 1}"],
            )
            df = pd.concat([df, circuit_info])

        print(f"\nCircuit details:\n{df.to_string()}")

    def print_circuit(self, color: Optional[bool] = False, colors: Optional[dict] = None) -> None:
        r"""Print the circuit on the terminal.

        Args:
            color (bool, optional): whether to print a colored circuit
            colors (dict, optional): specified colors for different nodes

        Note:
            Only the following colors are supported for the circuit print.

            1. red
            2. blue
            3. yellow
            4. green
            5. purple
            6. cyan
            7. white
            8. grey (default)

        Examples:

            >>> from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit
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
        # Gate strings and widths for terminal print
        gate_strings = {
            # Wires
            "qw": "---",
            "cw": "===",
            "qwx": "|",
            "cwx": "‖",
            # Others
            "cctrl": "○",
            "qctrl": "●",
            "space": "   ",  # 'barrier': '≡',
            # Single-qubit gates
            "id": "I",
            "h": "H",
            "x": "X",
            "y": "Y",
            "z": "Z",
            "s": "S",
            "t": "T",
            "rx": "Rx",
            "ry": "Ry",
            "rz": "Rz",
            "u": "U",
            "u3": "U3",
            "m": "MEAS",
            "r": "RESET",
            # Double-qubit gates
            "ch": ("●", "H"),
            "cx": ("●", "X"),
            "cy": ("●", "Y"),
            "cz": ("●", "Z"),
            "crx": ("●", "Rx"),
            "cry": ("●", "Ry"),
            "crz": ("●", "Rz"),
            "cu": ("●", "U"),
            "cu3": ("●", "U3"),
            # Triple-qubit gates
            "ccx": ("●", "●", "X"),
            # Noise
            "bit_flip": "BFLIP",
            "phase_flip": "PFLIP",
            "bit_phase_flip": "BPFLIP",
            "amplitude_damping": "ADAMP",
            "phase_damping": "PDAMP",
            "depolarizing": "DEPOL",
        }
        gate_widths = {
            # Single-qubit gates
            "id": 1,
            "h": 1,
            "x": 1,
            "y": 1,
            "z": 1,
            "s": 1,
            "t": 1,
            "rx": 2,
            "ry": 2,
            "rz": 2,
            "u": 1,
            "u3": 2,
            "m": 4,
            "r": 5,
            # Double-qubit gates
            "ch": 1,
            "cx": 1,
            "cy": 1,
            "cz": 1,
            "crx": 2,
            "cry": 2,
            "crz": 2,
            "cu": 1,
            "cu3": 2,
            # Triple-qubit gates
            "ccx": 1,
            # Noise
            "bit_flip": 5,
            "phase_flip": 5,
            "bit_phase_flip": 6,
            "amplitude_damping": 5,
            "phase_damping": 5,
            "depolarizing": 5,
        }

        def _check_classically_controlled_gate(gate: dict) -> bool:
            r"""Check whether the given gate is a classically conditioned gate.

            Args:
                gate (dict): gate to check

            Returns:
                bool: whether the gate is a classically conditioned gate
            """
            return True if gate.get("condition") is not None else False

        def _check_quantum_controlled_gate(gate: dict) -> bool:
            r"""Check whether the given gate is a quantum controlled gate.

            Args:
                gate (dict): gate to check

            Returns:
                bool: whether the gate is a quantum controlled gate
            """
            return True if gate["name"] in {"ch", "cx", "cy", "cz", "cu", "crx", "cry", "crz", "cu3", "ccx"} else False

        def _check_controlled_gate(gate: dict) -> bool:
            r"""Check whether the given gate is a controlled gate.

            Args:
                gate (dict): gate to check

            Returns:
                bool: whether the gate is a controlled gate
            """
            return _check_quantum_controlled_gate(gate) or _check_classically_controlled_gate(gate)

        def _gate_related_qubits(gate: dict) -> list:
            r"""Find qubits related to the given gate.

            Args:
                gate (dict): gate to find related qubits

            Returns:
                list: related qubit indices
            """
            related_qubits = (
                gate["which_qubit"]
                if not _check_classically_controlled_gate(gate)
                else [self.get_qubit_by_mid(gate["condition"]), gate["which_qubit"][0]]
            )
            return related_qubits

        def _check_gates_span_overlap(gate_1: dict, gate_2: dict) -> bool:
            r"""Check whether the spans of given gates are overlapped.

            Args:
                gate_1 (dict): one gate for check
                gate_2 (dict): another gate for check

            Returns:
                bool: whether the two gates have overlapped spans
            """
            # Record related indices for two gates, respectively
            gate_1_span = set(range(min(_gate_related_qubits(gate_1)), max(_gate_related_qubits(gate_1)) + 1))
            gate_2_span = set(range(min(_gate_related_qubits(gate_2)), max(_gate_related_qubits(gate_2)) + 1))
            return not gate_1_span.isdisjoint(gate_2_span)

        def __to_dag(cir: "Circuit") -> MultiDiGraph:
            r"""Convert the circuit into a directed graph with multi-edges.

            Args:
                cir (Circuit): circuit to convert

            Returns:
                networkx.MultiDiGraph: directed graph with multi-edges converted from the circuit
            """
            temp_gates = copy.deepcopy(cir.gate_history)
            num_gates = len(temp_gates)
            dag = MultiDiGraph()
            # Add gate nodes to the graph in ascending order
            dag.add_nodes_from(range(num_gates))
            # Construct gate edges (related to the time dependency of gates) from back to front
            while len(temp_gates) > 0:
                back_gate = temp_gates.pop()
                back_idx = len(temp_gates)
                back_qubits = set(_gate_related_qubits(back_gate))
                union_set = set()
                for front_idx in range(back_idx - 1, -1, -1):
                    front_gate = temp_gates[front_idx]
                    front_qubits = set(_gate_related_qubits(front_gate))
                    # Check the time dependency between the two gates
                    if len(front_qubits & back_qubits) > 0 and len(front_qubits & union_set) == 0:
                        # Add an edge between two dependent gates
                        dag.add_edge(front_idx, back_idx)
                        union_set = union_set.union(front_qubits)
            return dag

        def _to_layers(cir: "Circuit") -> list:
            r"""Convert the circuit to layers.

            Args:
                cir (Circuit): circuit to convert

            Returns:
                list: a list containing several layers where each layer consists of several gates
            """

            def __split_overlapped_layers(layers: list) -> list:
                r"""Check if certain gates in the same layer have overlapped spans. If so, split the layer.

                Args:
                    layers (list): layers to split

                Returns:
                    list: split layers with no gates overlapping
                """
                split_layers = []
                while len(layers) > 0:
                    layer = layers.pop(0)
                    split_layer = []
                    while len(layer) > 0:
                        current_gate = layer.pop(0)
                        split_layer.append(current_gate)
                        # Judge if the current gate overlaps with the remainder of the layer
                        if any(map(partial(_check_gates_span_overlap, current_gate), layer)):
                            layers.insert(0, layer)  # put back the remainder of the layer
                            break
                    split_layers.append(split_layer)  # save split part as a separate layer

                return split_layers

            def __align_measurement_gates(layers: list) -> list:
                r"""Align all the backmost measurement gates if they are also the backmost gates operated on the qubits.

                Args:
                    layers (list): layers to align measurement gates

                Returns:
                    list: layers after the alignment
                """
                alignment_manipulable = [True] * self.width  # whether the follow-up measurement can be aligned
                # Lists for saving aligned measurement gates
                aligned_positions, aligned_qubits, aligned_gates = [], [], []

                # 1. Collect the positions and related qubits of the measurements that can be aligned
                for layer_idx, layer in enumerate(layers[::-1]):
                    for gate_idx, gate in enumerate(layer[::-1]):
                        related_qubits = _gate_related_qubits(gate)
                        # If there is no gate after the measurement, record its position and related qubits
                        if gate["name"] == "m" and alignment_manipulable[related_qubits[0]]:
                            aligned_positions.append((len(layers) - layer_idx - 1, len(layer) - gate_idx - 1))
                            aligned_qubits.append(related_qubits[0])
                            aligned_gates.append(gate)
                        # Mark the manipulability status of the measurement alignment
                        for i in related_qubits:
                            alignment_manipulable[i] = False
                    if not any(alignment_manipulable):
                        break

                # 2. Create a unique layer for alignment if
                # the qubits of measurement gates to align overlap with those of the backmost layer
                if any(set(aligned_qubits) & set(_gate_related_qubits(gate)) for gate in layers[-1]):
                    layers.append([])

                # 3. Move the collected measurement gates from their original positions to the last layer for alignment
                for pos, meas_gate in zip(aligned_positions, aligned_gates):
                    del layers[pos[0]][pos[1]]  # remove the gates from their original positions
                    layers[-1].append(meas_gate)  # align in the last layer

                layers = [layer for layer in layers if layer != []]  # remove empty layers
                return layers

            dag = __to_dag(cir)
            layers = []
            while len(dag.nodes) > 0:
                front_idx = []
                for gate in dag.nodes:
                    # Find the foremost gate
                    if len(list(dag.predecessors(gate))) == 0:
                        front_idx.append(gate)
                layers.append([self.gate_history[idx] for idx in front_idx])
                dag.remove_nodes_from(front_idx)

            split_layers = __split_overlapped_layers(layers)
            layers = __align_measurement_gates(split_layers)
            return layers

        def _filled_str(gate_str: str, layer_width: int) -> str:
            r"""Concatenate the gate string with placeholders for filling the circuit line.

            Args:
                gate_str (str): gate string to concatenate
                layer_width (int): width of the layer where the gate is

            Returns:
                str: concatenated string
            """
            placeholder = " " if gate_str == gate_strings["qwx"] or gate_str == gate_strings["cwx"] else "-"
            appended_str = (layer_width - len(gate_str)) * placeholder
            return gate_str + appended_str

        # 1. Sanity check and initialization
        cir_width = self.width
        if cir_width == 0:  # empty circuit
            print(f"\nIn 'print_circuit': The given circuit '{self.name}' is empty!")
            return
        line_colors = ["end"] * cir_width  # no specific line colors for initialization

        if color is True:
            # Sanity check for signatures
            for gate in self.gate_history:
                if gate["signature"] is None:
                    print(
                        "\nIn 'print_circuit': Not all gates are assigned signatures. "
                        "A monochrome circuit will be printed."
                    )
                    color = False
                    break

        if color is True:
            # Sanity check for input colors
            if colors is not None:
                assert isinstance(colors, dict), (
                    "\nIn 'print_circuit': " "'colors' should be a dictionary mapping nodes to different colors."
                )
                colors = {node: color.lower() for node, color in colors.items()}  # save colors in lowercase
                for node_color in colors.values():
                    assert node_color in COLOR_TABLE, (
                        f"\nIn 'print_circuit': " f"The {node_color} color is not a supported color."
                    )
            # Assign colors for different signatures
            else:
                # Filter signatures with no duplicates
                signatures = []
                for gate in self.gate_history:
                    if gate["signature"] not in signatures:
                        signatures.append(gate["signature"])
                assigned_colors = list(COLOR_TABLE)[: len(signatures)]
                colors = {signature: assigned_color for signature, assigned_color in zip(signatures, assigned_colors)}

            # Set the initial line colors according to the foremost gates of each qubit
            for gate in self.gate_history:
                for i in gate["which_qubit"]:
                    line_colors[i] = colors[gate["signature"]] if line_colors[i] == "end" else line_colors[i]
                if all(line_colors[i] != "end" for i in range(cir_width)):
                    break

        # Horizontal strings for circuit lines
        str_circuit_lines = [
            (f"{i:2d}: " if cir_width > 10 else f"{i}: ")
            + COLOR_TABLE[line_colors[i]]
            + gate_strings["qw"]
            + COLOR_TABLE["end"]
            for i in range(cir_width)
        ]
        # Vertical strings for space lines (including control wires and blank spaces)
        str_space_lines = [" " * (4 if cir_width > 10 else 3) + gate_strings["space"]] * (cir_width - 1)

        # 2. Decompose the quantum circuit into layers for comfortable print
        layers = _to_layers(self)

        # 3. Append strings for circuit print layer by layer
        for layer_idx, layer in enumerate(layers):
            # Restrict layer width for circuit alignment
            layer_width = max([gate_widths[gate["name"]] for gate in layer])
            # Record whether the circuit lines and space lines have been filled: 0 for blank; 1 for filled
            circuit_lines_filled = numpy.zeros(cir_width)
            space_lines_filled = numpy.zeros(cir_width - 1)

            # 3.1 Append string for each gate in a certain layer
            for gate in layer:
                gate_str = gate_strings[gate["name"]]

                if _check_controlled_gate(gate):
                    SGL_CTRL = True

                    if _check_quantum_controlled_gate(gate):
                        if len(gate["which_qubit"]) == 2:
                            ctrl, targ = gate["which_qubit"]
                            ctrl_str, targ_str = gate_str
                        elif len(gate["which_qubit"]) == 3:
                            SGL_CTRL = False
                            ctrl1, ctrl2, targ = gate["which_qubit"]
                            ctrl1, ctrl2 = min(ctrl1, ctrl2), max(ctrl1, ctrl2)
                            ctrl1_str, ctrl2_str, targ_str = gate_str
                        ctrl_wire = gate_strings["qwx"]
                    elif _check_classically_controlled_gate(gate):
                        ctrl, targ = self.get_qubit_by_mid(gate["condition"]), gate["which_qubit"][0]
                        ctrl_str, targ_str = gate_strings["cctrl"], gate_str
                        ctrl_wire = gate_strings["cwx"]

                    if SGL_CTRL:
                        ctrl_color = line_colors[ctrl]
                    else:
                        ctrl1_color, ctrl2_color = line_colors[ctrl1], line_colors[ctrl2]
                    targ_color = line_colors[targ]

                    # Fill the circuit lines
                    if SGL_CTRL:
                        str_circuit_lines[ctrl] += (
                            COLOR_TABLE[ctrl_color] + _filled_str(ctrl_str, layer_width) + COLOR_TABLE["end"]
                        )  # string for control qubit(s)
                    else:
                        str_circuit_lines[ctrl1] += (
                            COLOR_TABLE[ctrl1_color] + _filled_str(ctrl1_str, layer_width) + COLOR_TABLE["end"]
                        )  # string for control qubit(s)
                        str_circuit_lines[ctrl2] += (
                            COLOR_TABLE[ctrl2_color] + _filled_str(ctrl2_str, layer_width) + COLOR_TABLE["end"]
                        )  # string for control qubit(s)
                    str_circuit_lines[targ] += (
                        COLOR_TABLE[targ_color] + _filled_str(targ_str, layer_width) + COLOR_TABLE["end"]
                    )  # string for target qubit
                    circuit_lines_filled[[ctrl, targ] if SGL_CTRL else [ctrl1, ctrl2, targ]] = 1  # record the filling

                    # Fill the space lines
                    filling_range = range(min(_gate_related_qubits(gate)), max(_gate_related_qubits(gate)))
                    if SGL_CTRL:
                        for i in filling_range:
                            str_space_lines[i] += (
                                COLOR_TABLE[ctrl_color] + _filled_str(ctrl_wire, layer_width) + COLOR_TABLE["end"]
                            )  # string for control wires
                    else:
                        lowest, medium, highest = sorted(gate["which_qubit"])
                        for i in range(lowest, medium):
                            str_space_lines[i] += (
                                COLOR_TABLE[ctrl1_color] + _filled_str(ctrl_wire, layer_width) + COLOR_TABLE["end"]
                            )  # string for control wires
                        for i in range(medium, highest):
                            str_space_lines[i] += (
                                COLOR_TABLE[ctrl2_color] + _filled_str(ctrl_wire, layer_width) + COLOR_TABLE["end"]
                            )  # string for control wires
                    space_lines_filled[filling_range] = 1  # record the filling event

                else:  # other non-controlled gates (including measurement and reset gates)
                    which_qubit = gate["which_qubit"][0] if len(gate["which_qubit"]) == 1 else gate["which_qubit"]
                    gate_color = line_colors[which_qubit]
                    str_circuit_lines[which_qubit] += (
                        COLOR_TABLE[gate_color] + _filled_str(gate_str, layer_width) + COLOR_TABLE["end"]
                    )
                    circuit_lines_filled[which_qubit] = 1  # record the filling event

            # 3.2 Update line colors for following print
            if color is True:
                remaining_gates = sum(layers[layer_idx + 1 :], [])
                for idx in range(cir_width):
                    # Search the foremost gate of remaining parts for each qubit
                    for gate in remaining_gates:
                        if idx in gate["which_qubit"]:
                            if line_colors[idx] != colors[gate["signature"]]:  # different color from the previous one
                                line_colors[idx] = colors[gate["signature"]]  # change the line color
                            break

            # 3.3 Append quantum wires and spaces for blank lines
            for i in numpy.where(circuit_lines_filled == 0)[0]:  # quantum wires
                str_circuit_lines[i] += COLOR_TABLE[line_colors[i]] + "-" * layer_width + COLOR_TABLE["end"]
            for i in numpy.where(space_lines_filled == 0)[0]:  # spaces
                str_space_lines[i] += " " * layer_width

            # 3.4 Placeholders for distinguishing layers
            str_circuit_lines = [
                str_circuit_lines[i] + COLOR_TABLE[line_colors[i]] + gate_strings["qw"] + COLOR_TABLE["end"]
                for i in range(cir_width)
            ]  # '---'
            str_space_lines = [str_space_lines[i] + gate_strings["space"] for i in range(cir_width - 1)]  # '   '

        # 4. Concatenation for terminal print
        str_print = [[str_circuit_lines[i], str_space_lines[i]] for i in range(cir_width - 1)]
        str_print.append([str_circuit_lines[-1]])
        str_print = sum(str_print, [])  # merge into one list for convenience
        str_print = "\n".join(str_print)
        print(f"\n{self.name}: \n\n" + str_print)
        if color is True:
            print("\nColors:", {node.name: color for node, color in colors.items()})
