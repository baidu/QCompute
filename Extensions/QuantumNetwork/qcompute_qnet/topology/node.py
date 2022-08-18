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
Module for nodes in a network.
"""

from abc import ABC
from typing import Any
from qcompute_qnet.core.des import Entity
from qcompute_qnet.functionalities.mobility import Mobility
from qcompute_qnet.protocols.protocol import ProtocolStack
from qcompute_qnet.protocols.routing import Routing
from qcompute_qnet.messages.message import ClassicalMessage, QuantumMessage

__all__ = [
    "Node",
    "Satellite"
]


class Node(Entity):
    r"""Class for creating a node.

    Attributes:
        links (Dict[Node, Link]): links connecting the node with other nodes
        protocol_stack (ProtocolStack): protocol stack installed
        functionalities (list): functionalities of the node
        is_mobile (bool): whether the node is a mobile node
    """

    def __init__(self, name: str, env=None):
        r"""Constructor for Node class.

        Args:
            name (str): name of the node
            env (DESEnv): related discrete-event simulation environment
        """
        super().__init__(name, env)
        self.links = {}
        self.protocol_stack = None
        self.functionalities = []
        self.is_mobile = False

    def init(self) -> None:
        r"""Node initialization.

        This method will do a sanity check to confirm that the node is installed to a ``Network``.
        """
        assert self.owner != self, f"The node {self.name} should be installed to a 'Network' first!"

    def load_protocol(self, protocol_stack: "ProtocolStack") -> None:
        r"""Load a protocol stack to the node.

        ``load_protocol`` method will set the node for all protocols in the protocol stack and synchronize
        their scheduler's simulation environment as the one of node.

        Args:
            protocol_stack (ProtocolStack): protocol stack to load
        """
        self.protocol_stack = protocol_stack
        protocol_stack.owner = self
        protocol_stack.sync_env(self)

    def assign_functionality(self, functionality: Any) -> None:
        r"""Assign a functionality to the node.

        Args:
            functionality (Any): functionality to assign
        """
        self.functionalities.append(functionality)

        from qcompute_qnet.functionalities.mobility import Mobility
        if isinstance(functionality, Mobility):
            self.is_mobile = True

    def cchannel(self, dst: "Node") -> "ClassicalChannel":
        r"""Find the classical channel with a given destination.

        Args:
            dst (Node): destination node of the classical channel

        Returns:
            ClassicalChannel: the classical channel with the given destination
        """
        return self.links[dst].cchannel(dst)

    def qchannel(self, dst: "Node") -> "QuantumChannel":
        r"""Find the quantum channel with a given destination.

        Args:
            dst (Node): destination node of the quantum channel

        Returns:
            QuantumChannel: the quantum channel with the given destination
       """
        return self.links[dst].qchannel(dst)

    def send_classical_msg(self, dst: "Node", msg: "ClassicalMessage", priority=None) -> None:
        r"""Send a classical message through the classical channel.

        Args:
            dst (Node): destination of the message
            msg (ClassicalMessage): classical message to send
            priority (int): priority of the transmission event
        """
        self.cchannel(dst).transmit(msg, priority)

    def send_quantum_msg(self, dst: "Node", msg: "QuantumMessage", priority=None) -> None:
        r"""Send a quantum message through the quantum channel.

        Args:
            dst (Node): destination of the message
            msg (QuantumMessage): quantum message to send
            priority (int): priority of the transmission event
        """
        self.qchannel(dst).transmit(msg, priority)

    def receive_classical_msg(self, src: "Node", msg: "ClassicalMessage") -> None:
        r"""Receive a classical message from the classical channel.

        The source node is inferred by the receiver and is not part of the transmitted message.

        Args:
            src (Node): source of the classical message
            msg (ClassicalMessage): classical message to receive
        """
        pass

    def receive_quantum_msg(self, src: "Node", msg: "QuantumMessage") -> None:
        r"""Receive a quantum message from the quantum channel.

        The source node is inferred by the receiver and is not part of the transmitted message.

        Args:
            src (Node): source of the quantum message
            msg (QuantumMessage): quantum message to receive
        """
        pass

    def start(self, **kwargs) -> None:
        r"""Start the protocol stack of the node.

        Args:
            **kwargs: keyword arguments to start the protocol stack
        """
        assert self.protocol_stack is not None, f"Should load a protocol stack first."
        self.protocol_stack.start(**kwargs)


class Satellite(Node, ABC):
    r"""Abstract class for satellite nodes.

    A satellite node is a mobile node that keeps moving on its orbit.
    """

    def __init__(self, name: str, env=None):
        r"""Constructor for Satellite class.

        Args:
            name (str): name of the satellite node
            env (DESEnv): related discrete-event simulation environment
        """
        super().__init__(name, env)
        self.mobility = Mobility()
        self.assign_functionality(self.mobility)
