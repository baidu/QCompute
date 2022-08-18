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
Module for quantum nodes.
"""

from typing import Optional

from qcompute_qnet.topology.node import Node
from qcompute_qnet.devices.register import QuantumRegister
from qcompute_qnet.models.qpu.message import QuantumMsg
from qcompute_qnet.protocols.protocol import ProtocolStack

__all__ = [
    "QuantumNode"
]


class QuantumNode(Node):
    r"""Node with a quantum register built-in.

    Attributes:
        qreg (QuantumRegister): quantum register installed
        protocol_type (type): protocol to install in the protocol stack
    """

    def __init__(self, name: str, qreg_size: int, protocol: Optional[type] = None, env: Optional["DESEnv"] = None):
        r"""Constructor for QuantumNode class.

        Args:
            name (str): name of the quantum node
            qreg_size (int): size of the local quantum register
            protocol (type, optional): protocol to install in the local protocol stack
            env (DESEnv, optional): discrete-event simulation environment
        """
        super().__init__(name, env)
        self.protocol_type = protocol

        self.qreg = QuantumRegister(name=f"{name}_qreg", size=qreg_size)  # create a quantum register
        self.install(self.qreg)
        ps = ProtocolStack(name=f"ps_{self.name}",
                           protocol=protocol(f"{protocol.__name__}_{self.name}") if protocol is not None else None)
        self.load_protocol(ps)

    @property
    def protocol(self) -> "Protocol":
        r"""Get the protocol from the protocol stack with the pre-specified type.

        Returns:
            Protocol: protocol with the pre-specified type
        """
        assert self.protocol_type is not None, f"Should set a protocol first."

        return self.protocol_stack.get_protocol(self.protocol_type)

    def send_quantum_msg(self, dst: "Node", qreg_address: int, priority=None) -> None:
        r"""Get the quantum state from the quantum register with a given address
        and send it as a quantum message to a given destination.

        Args:
            dst (Node): destination of the quantum message
            qreg_address (int): unit address in the quantum register
            priority (int, optional): priority of the quantum message
        """
        quantum_msg = QuantumMsg(self.qreg.get_qubit(qreg_address), self.qreg.circuit_index)
        super().send_quantum_msg(dst=dst, msg=quantum_msg, priority=priority)

    def receive_classical_msg(self, src: "Node", msg: "ClassicalMessage") -> None:
        r"""Receive a classical message from the classical channel.

        The source node is inferred by the receiver and is not part of the transmitted message.

        Args:
            src (Node): source of the classical message
            msg (ClassicalMessage): classical message to receive
        """
        assert self.protocol_type is not None, f"Should set a specific protocol type for the node first."

        self.protocol_stack.get_protocol(self.protocol_type).receive_classical_msg(msg)

    def receive_quantum_msg(self, src: "Node", msg: "QuantumMsg") -> None:
        r"""Receive a quantum message from the quantum channel.

        The source node is inferred by the receiver and is not part of the transmitted message.

        Args:
            src (Node): source of the quantum message
            msg (QuantumMessage): quantum message to receive
        """
        assert self.protocol_type is not None, f"Should set a specific protocol type for the node first."

        self.protocol_stack.get_protocol(self.protocol_type).receive_quantum_msg(msg, src=src)
