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
Module for quantum registers.
"""

from typing import List, Union, Optional, Any
import numpy

from qcompute_qnet.core.des import Entity, EventHandler
from qcompute_qnet.quantum.circuit import Circuit
from qcompute_qnet.quantum.state import QuantumState

__all__ = [
    "QuantumRegister"
]


class QuantumRegister(Entity):
    r"""Quantum register is a logical device that stores and processes quantum information.

    Note:
        Since quantum states can be entangled :math:`\rho_{AB} \neq \rho_A \otimes \rho_B`,
        storing the local states :math:`\rho_A` and :math:`\rho_B` cannot faithfully represent all the information.
        So a quantum circuit is built up in the `Network` that reflects the evolution of the network.

    Attributes:
        size (int): size of the quantum register
        units (List[dict]): list for information about the qubits stored locally
        circuit_index (int): index of the current circuit related to the quantum register
    """

    def __init__(self, name: str, size: int, env: Optional["DESEnv"] = None):
        r"""Constructor for QuantumRegister class.

        Args:
            name (str): name of the quantum register
            size (int): size of the quantum register
            env (DESEnv, optional): discrete-event simulation environment
        """
        super().__init__(name, env)
        self.size = size

        self.units = []  # each unit stores a qubit, with its local address, quantum state and source
        for i in range(self.size):
            self.units.append({"address": i, "qubit": None, "identifier": self.owner, 'outcome': None})
        self.circuit_index = -1
        self.__op_counter = 0

    def init(self) -> None:
        r"""Initialization of the quantum register.
        """
        assert self.owner != self, f"{self.name} has no owner!"

    @property
    def __count(self) -> int:
        r"""Count the number of operations of the quantum register.

        Returns:
            int: index of the current operation in the quantum register
        """
        self.__op_counter += 1
        return self.__op_counter

    @property
    def circuit(self) -> "Circuit":
        r"""Get the quantum circuit of the network according to the circuit index.

        Important:
            If the index is specified before, this method will return the circuit with the index from the ``circuits``
            list of the network; otherwise it will return the default circuit of the network.

        Returns:
            Circuit: default circuit or circuit with the specified index
        """
        if self.circuit_index == -1:  # return the default circuit
            return self.env.network.default_circuit
        else:  # return the circuit with the specified index
            return self.env.network.circuits[self.circuit_index]

    def create_circuit(self, name: str) -> None:
        r"""Create a new circuit and add it to the ``circuits`` list of the network.

        Args:
            name (str): name of the new created circuit
        """
        self.env.network.circuits.append(Circuit(name=name))
        self.circuit_index += 1

    def init_unit(self, address: int) -> None:
        r"""Initialize the quantum register unit with the given address.

        Args:
            address (int): unit address of the quantum register
        """
        index = self.circuit.init_new_qreg_unit()
        self.units[address]['qubit'] = QuantumState(state=numpy.array([[index]]))

    def reset(self, address: int) -> None:
        r"""Reset the qubit with the given address.

        Args:
            address (int): local address of the qubit to reset
        """
        self.units[address]['qubit'] = None
        self.units[address]['identifier'] = self.owner

    def get_qubit(self, address: int) -> "QuantumState":
        r"""Get the qubit with the given address.

        Warnings:
            Due to the no-cloning theorem, a unit gets reset when we fetch its qubit.

        Args:
            address (int): address of the qubit

        Returns:
            QuantumState: quantum state of the qubit with the given address
        """
        qubit = self.units[address]['qubit']
        self.reset(address)

        return qubit

    def get_address(self, identifier: Any) -> int:
        r"""Get the qubit address with a given id.

        Args:
            identifier (Any): identity to fetch the qubit

        Returns:
            int: local address of the qubit that matches the id
        """
        for unit in self.units:
            if unit['identifier'] == identifier:
                return unit['address']

    def store_qubit(self, qubit: "QuantumState", identifier: Any) -> None:
        r"""Store a received qubit to the quantum register.

        Note:
            This method will store the qubit to a quantum register unit with the smallest available address.
            The identifier is used to uniquely identify the qubit. An identifier is required to fetch this qubit later.

        Args:
            qubit (QuantumState): received qubit
            identifier (Any): identity to store the qubit
        """
        assert any(unit['qubit'] is None for unit in self.units), f"No available units in {self.name}."

        for unit in self.units:
            if unit['qubit'] is None:
                unit['qubit'] = qubit
                unit['identifier'] = identifier
                return

    def __add_single_qubit_gate(self, operation: str, address: int, **params) -> None:
        r"""Add a single qubit gate to the circuit list.

        Args:
            operation (str): gate operation
            address (int): unit address of the quantum register
            **params (Any): gate parameters
        """
        if self.units[address]['qubit'] is None:
            self.init_unit(address)

        which_qubit = int(self.units[address]['qubit'].state[0][0])

        # Single-qubit gate with condition
        if "condition" in params.keys():
            # Parameterized single-qubit gate with condition
            if operation in ['rx', 'ry', 'rz']:
                handler = EventHandler(self.circuit, operation, [which_qubit, params['angle']],
                                       signature=self.owner, condition=params['condition'])
            elif operation == 'u3':
                handler = EventHandler(self.circuit, operation, [which_qubit, *params['angles']],
                                       signature=self.owner, condition=params['condition'])
            # Fixed single-qubit gate with condition
            else:
                handler = EventHandler(self.circuit, operation, [which_qubit], signature=self.owner, **params)
        # Single-qubit gate without condition
        else:
            # Parameterized single-qubit gate without condition
            if operation in ['rx', 'ry', 'rz']:
                handler = EventHandler(self.circuit, operation, [which_qubit, params['angle']], signature=self.owner)
            elif operation == 'u3':
                handler = EventHandler(self.circuit, operation, [which_qubit, *params['angles']], signature=self.owner)
            # Fixed single-qubit gate without condition (including measurement)
            else:
                handler = EventHandler(self.circuit, operation, [which_qubit], signature=self.owner)

        # Use the operation count as the priority so that the gate are executed in the correct order
        self.scheduler.schedule_now(handler, priority=self.__count)

    def __add_double_qubit_gate(self, operation: str, address: List[int], **params) -> None:
        r"""Add a double qubit gate to the circuit list.

        Args:
            operation (str): gate operation
            address (list): unit address of the quantum register in the order of [control, target]
            **params (Any): gate parameters
        """
        if self.units[address[0]]['qubit'] is None:
            self.init_unit(address[0])
        if self.units[address[1]]['qubit'] is None:
            self.init_unit(address[1])

        ctrl, targ = int(self.units[address[0]]['qubit'].state[0][0]), int(self.units[address[1]]['qubit'].state[0][0])

        # Parameterized double-qubit gate
        if operation in ['crx', 'cry', 'crz']:
            handler = EventHandler(self.circuit, operation, [[ctrl, targ], params['angle']], signature=self.owner)
        elif operation == 'cu3':
            handler = EventHandler(self.circuit, operation, [[ctrl, targ], *params['angles']], signature=self.owner)
        # Fixed double-qubit gate
        else:
            handler = EventHandler(self.circuit, operation, [[ctrl, targ]], signature=self.owner)

        self.scheduler.schedule_now(handler, priority=self.__count)

    def id(self, address: int) -> None:
        r"""Apply an identity gate on the local qubit with a given address.

        Args:
            address (int): local address of the qubit to act on in the quantum register
        """
        self.__add_single_qubit_gate('id', address)

    def h(self, address: int, **condition) -> None:
        r"""Apply a Hadamard gate on the local qubit with a given address.

        Args:
            address (int): local address of the qubit to act on in the quantum register
            **condition (int): control qubit address
        """
        self.__add_single_qubit_gate('h', address, **condition)

    def x(self, address: int, **condition) -> None:
        r"""Apply a X gate on the local qubit with a given address.

        Args:
            address (int): local address of the qubit to act on in the quantum register
            **condition (int): control qubit address
        """
        self.__add_single_qubit_gate('x', address, **condition)

    def y(self, address: int, **condition) -> None:
        r"""Apply a Y gate on the local qubit with a given address.

        Args:
            address (int): local address of the qubit to act on in the quantum register
            **condition (int): control qubit address
        """
        self.__add_single_qubit_gate('y', address, **condition)

    def z(self, address: int, **condition) -> None:
        r"""Apply a Z gate on the local qubit with a given address.

        Args:
            address (int): local address of the qubit to act on in the quantum register
            **condition (int): control qubit address
        """
        self.__add_single_qubit_gate('z', address, **condition)

    def s(self, address: int) -> None:
        r"""Apply a S gate on the local qubit with a given address.

        Args:
            address (int): local address of the qubit to act on in the quantum register
        """
        self.__add_single_qubit_gate('s', address)

    def t(self, address: int) -> None:
        r"""Apply a T gate on the local qubit with a given address.

        Args:
            address (int): local address of the qubit to act on in the quantum register
        """
        self.__add_single_qubit_gate('t', address)

    def rx(self, address: int, theta: Union[float, int], **condition) -> None:
        r"""Apply a rotation gate around x-axis on the local qubit with a given address.

        Args:
            address (int): local address of the qubit to act on in the quantum register
            theta (Union[float, int]): rotation angle
            **condition (int): control qubit address
        """
        self.__add_single_qubit_gate('rx', address, angle=theta, **condition)

    def ry(self, address: int, theta: Union[float, int], **condition) -> None:
        r"""Apply a rotation gate around y-axis on the local qubit a the given address.

        Args:
            address (int): local address of the qubit to act on in the quantum register
            theta (Union[float, int]): rotation angle
            **condition (int): control qubit address
        """
        self.__add_single_qubit_gate('ry', address, angle=theta, **condition)

    def rz(self, address: int, theta: Union[float, int], **condition) -> None:
        r"""Apply a rotation gate around z-axis on the local qubit with the given address.

        Args:
            address (int): local address of the qubit to act on in the quantum register
            theta (Union[float, int]): rotation angle
            **condition (int): control qubit address
        """
        self.__add_single_qubit_gate('rz', address, angle=theta, **condition)

    def u3(self, address: int, theta: Union[float, int], phi: Union[float, int], gamma: Union[float, int],
           **condition) -> None:
        r"""Apply a rotation gate on the local qubit with a given address.

        Args:
            address (int): local address of the qubit to act on in the quantum register
            theta (Union[float, int]): the rotation angle of the Rx gate
            phi (Union[float, int]): the rotation angle of the left Rz gate
            gamma (Union[float, int]): the rotation angle of the right Rz gate
            **condition (int): control qubit address
        """
        angles = [theta, phi, gamma]
        self.__add_single_qubit_gate('u3', address, angles=angles, **condition)

    def ch(self, address: List[int]) -> None:
        r"""Apply a Controlled-Hadamard gate on the local qubits with given addresses.

        Args:
            address (List[int]): local addresses of the qubits to act on in the quantum register
        """
        self.__add_double_qubit_gate('ch', address)

    def cx(self, address: List[int]) -> None:
        r"""Apply a Controlled-X gate on the local qubits with given addresses.

        Args:
            address (List[int]): local addresses of the qubits to act on in the quantum register
        """
        self.__add_double_qubit_gate('cx', address)

    def cnot(self, address: List[int]) -> None:
        r"""Apply a Controlled-NOT gate on the local qubits with given addresses.

        Args:
            address (List[int]): local addresses of the qubits to act on in the quantum register
        """
        self.__add_double_qubit_gate('cx', address)

    def cy(self, address: List[int]) -> None:
        r"""Apply a Controlled-Y gate on the local qubits with given addresses.

        Args:
            address (List[int]): local addresses of the qubits to act on in the quantum register
        """
        self.__add_double_qubit_gate('cy', address)

    def cz(self, address: List[int]) -> None:
        r"""Apply a Controlled-Z gate on the local qubits with given addresses.

        Args:
            address (List[int]): local addresses of the qubits to act on in the quantum register
        """
        self.__add_double_qubit_gate('cz', address)

    def crx(self, address: List[int], theta: Union[float, int]) -> None:
        r"""Apply a Controlled-rotation gate around x-axis on the local qubit with given addresses.

        Args:
            address (List[int]): local addresses of the qubits to act on in the quantum register
            theta (Union[float, int]): rotation angle
        """
        self.__add_double_qubit_gate('crx', address, angle=theta)

    def cry(self, address: List[int], theta: Union[float, int]) -> None:
        r"""Apply a Controlled-rotation gate around y-axis on the local qubit with given addresses.

        Args:
            address (List[int]): local addresses of the qubits to act on in the quantum register
            theta (Union[float, int]): rotation angle
        """
        self.__add_double_qubit_gate('cry', address, angle=theta)

    def crz(self, address: List[int], theta: Union[float, int]) -> None:
        r"""Apply a Controlled-rotation gate around z-axis on the local qubit with given addresses.

        Args:
            address (List[int]): local addresses of the qubits to act on in the quantum register
            theta (Union[float, int]): rotation angle
        """
        self.__add_double_qubit_gate('crz', address, angle=theta)

    def cu3(self, address: List[int],
            theta: Union[float, int], phi: Union[float, int], gamma: Union[float, int]) -> None:
        r"""Apply a Controlled-rotation gate on the local qubit with given addresses.

        Args:
            address (List[int]): local addresses of the qubits to act on in the quantum register
            theta (Union[float, int]): the rotation angle of the Rx gate
            phi (Union[float, int]): the rotation angle of the left Rz gate
            gamma (Union[float, int]): the rotation angle of the right Rz gate
        """
        angles = [theta, phi, gamma]
        self.__add_double_qubit_gate('cu3', address, angles=angles)

    def swap(self, address: List[int]) -> None:
        r"""Apply a SWAP gate on the local qubits with given addresses.

        Args:
            address (List[int]): local addresses of the qubits to act on in the quantum register
        """
        self.__add_double_qubit_gate('cx', address)
        self.__add_double_qubit_gate('cx', [address[1], address[0]])
        self.__add_double_qubit_gate('cx', address)

    def measure(self, address: int, basis: Optional[str] = "z") -> None:
        r"""Measure the qubit with a given address in the computational basis.

        Args:
            address (int): local address of the qubit to act on in the quantum register
            basis (str): on which basis should the measurement performs, either Z-basis or X-basis
        """
        if basis.casefold() == "z":
            self.__add_single_qubit_gate("measure", address)
        elif basis.casefold() == "x":
            self.h(address)
            self.__add_single_qubit_gate("measure", address)

        # Side effect for measurement, the trick here is to use the qubit register as its measurement outcome
        which_qubit = int(self.units[address]['qubit'].state[0][0])
        self.units[address]['outcome'] = which_qubit

        # Rest the unit once measured
        self.reset(address)

    def bsm(self, address: List[int]) -> None:
        r"""Perform Bell state measurement on the qubits with given addresses.

        Args:
            address (List[int]): local addresses of the qubits to perform Bell state measurement in the quantum register
        """
        self.cnot([address[0], address[1]])
        self.h(address[0])
        self.measure(address[0])
        self.measure(address[1])
