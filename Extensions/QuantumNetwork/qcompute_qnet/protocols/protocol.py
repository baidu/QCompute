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
Module for protocols.
"""

from abc import ABC
from typing import List, Union
import networkx as nx
import matplotlib.pyplot as plt
from qcompute_qnet.core.des import Event, Scheduler, EventHandler

__all__ = [
    "Protocol",
    "SubProtocol",
    "ProtocolStack"
]


class Protocol(ABC):
    r"""Abstract protocol class for designing a protocol.

    Attributes:
        name (str): name of the protocol
        owner (ProtocolStack): owner of the protocol
        node (Node): node where the protocol stack is loaded
        scheduler (Scheduler): event scheduler of the protocol
        agenda (List): events scheduled for the protocol
        signed_events (List): events scheduled by the protocol
    """

    def __init__(self, name=None):
        r"""Constructor for Protocol class.

        Args:
            name (str): name of the protocol
        """

        self.name = 'Protocol' if name is None else name
        self.owner = self
        self.node = None
        self.scheduler = Scheduler(owner=self)
        self.agenda = []
        self.signed_events = []

    @property
    def is_top(self) -> bool:
        r"""Check if the protocol is at the top of the protocol stack.

        Returns:
            bool: whether the protocol is a top protocol
        """
        return len(self.upper_protocols) == 0

    @property
    def is_bottom(self) -> bool:
        r"""Check if the protocol is at the bottom of the protocol stack.

        Returns:
            bool: whether the protocol is a bottom protocol
        """
        return len(self.lower_protocols) == 0

    @property
    def upper_protocols(self) -> List["Protocol"]:
        r"""Return the upper protocols of the current protocol.

        Returns:
            List: a list of upper protocols
        """
        assert isinstance(self.owner, ProtocolStack), f"Not attached to any protocol stack yet!"
        return list(self.owner.stack.predecessors(self))

    @property
    def lower_protocols(self) -> List["Protocol"]:
        r"""Return the lower protocols of the current protocol.

        Returns:
            List: a list of lower protocols
        """
        assert isinstance(self.owner, ProtocolStack), f"Not attached to any protocol stack yet!"
        return list(self.owner.stack.successors(self))

    def start(self, **kwargs) -> None:
        r"""Start the protocol.

        This should be overridden in a specific protocol.

        Args:
            **kwargs: keyword arguments to start the protocol
        """
        pass

    def send_upper(self, upper_protocol: type, **kwargs) -> None:
        r"""Send a message to upper protocols of a specified type.

        Args:
            upper_protocol (type): type of the upper protocols to receive the message
            **kwargs: keyword arguments of the message
        """
        assert not self.is_top, f"'{self.name}' has no upper protocols."

        for proto in self.upper_protocols:
            if isinstance(proto, upper_protocol):
                proto.receive_lower(type(self), **kwargs)

    def send_lower(self, lower_protocol: type, **kwargs) -> None:
        r"""Send a message to lower protocols of a specified type.

        Args:
            lower_protocol (type): type of the lower protocols to receive the message
            **kwargs: keyword arguments of the message
        """
        assert not self.is_bottom, f"'{self.name}' has no lower protocols."

        for proto in self.lower_protocols:
            if isinstance(proto, lower_protocol):
                proto.receive_upper(type(self), **kwargs)

    def receive_upper(self, upper_protocol: type, **kwargs) -> None:
        r"""Receive a message from an upper protocol.

        Args:
            upper_protocol (type): type of the upper protocol
            **kwargs: keyword arguments of the message
        """
        pass

    def receive_lower(self, lower_protocol: type, **kwargs) -> None:
        r"""Receive a message from a lower protocol.

        Args:
            lower_protocol (type): type of the lower protocol
            **kwargs: keyword arguments of the message
        """
        pass

    def receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        r"""Receive a classical message from the node.

        Args:
            msg (ClassicalMessage): received classical message
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        pass

    def receive_quantum_msg(self, msg: "QuantumMessage", **kwargs) -> None:
        r"""Receive a quantum message from the node.

        Args:
            msg (QuantumMessage): received quantum message
            **kwargs (Any): keyword arguments for receiving the quantum message
        """
        pass

    def print_agenda(self) -> None:
        r"""Print the events scheduled for the protocol.
        """
        df = Event.events_to_dataframe(self.agenda)
        print(f"\nAgenda of {self.name} (unsorted):\n{df.to_string()}")

    def print_signed_events(self) -> None:
        r"""Print the events scheduled by the protocol.
        """
        df = Event.events_to_dataframe(self.signed_events)
        print(f"\nSigned events by {self.name} (unsorted):\n{df.to_string()}")


class SubProtocol:
    r"""Class for the sub-protocols built in a protocol.

    Note:
        A sub-protocol defines the behaviors of a specific role in a protocol.
        Taking teleportation protocol as an example, we can define sub-protocols for sender and receiver respectively.
        Then these sub-protocols together forms a full protocol for teleportation.

    Attributes:
        super_protocol (Protocol): protocol in which the sub-protocol is built
    """

    def __init__(self, super_protocol: Protocol):
        r"""Constructor for SubProtocol class.

        Args:
            super_protocol (Protocol): protocol in which the sub-protocol is built
        """
        self.super_protocol = super_protocol

    @property
    def node(self) -> "Node":
        r"""Get the node of its super protocol.

        Returns:
            Node: node of super protocol
        """
        return self.super_protocol.node


class ProtocolStack:
    r"""Class for building a protocol stack.

    Note that the stack relation is maintained as a directed graph.

    Attributes:
        name (str): name of the ProtocolStack
        owner (Node): owner of the ProtocolStack
        stack (networkx.DiGraph): stack relation of protocols
        config (Dict[str, Dict]): configurations of the protocols
    """

    def __init__(self, name=None, protocol=None):
        r"""Constructor for ProtocolStack class.

        Args:
            name (str): name of the ProtocolStack
        """

        self.name = 'Protocol Stack' if name is None else name
        self.owner = self
        self.stack = nx.DiGraph()
        self.config = {}
        if protocol is not None:
            self.build(protocol)

    @property
    def protocols(self) -> List["Protocol"]:
        r"""Return a list of all protocols in the protocol stack.

        Returns:
            List[Protocol]: protocols in the protocol stack
        """
        return list(self.stack.nodes)

    def get_protocol(self, proto_type: type) -> "Protocol":
        r"""Get a protocol from the protocol stack by its type.

        Args:
            proto_type (type): type of the protocol

        Returns:
            Protocol: protocol of the given type
        """
        assert any(isinstance(proto, proto_type) for proto in self.protocols), "Cannot find a protocol with the given type."
        for proto in self.protocols:
            if isinstance(proto, proto_type):
                return proto

    def set(self, protocol_type: str, **kwargs) -> None:
        r"""Set configuration of a specified protocol type by given parameters.

        Args:
            protocol_type (str): type of the protocol to set
            **kwargs: keyword arguments to set
        """
        assert isinstance(protocol_type, str), "'protocol' should be a str."

        self.config[protocol_type] = kwargs

    def build(self, relation: Union["Protocol", List["Protocol"]]) -> None:
        r"""Build a protocol stack from the protocol relation.

        Args:
            relation (Union[Protocol, List[Protocol]]): a single protocol or a list of protocol relations
        """
        if isinstance(relation, Protocol):
            self.stack.add_node(relation)
            relation.owner = self
        elif isinstance(relation, list):
            self.__update_from_relations(relation)
        else:
            raise TypeError("Should input a single protocol or a list of protocol relations!")
        from qcompute_qnet.topology.node import Node
        if isinstance(self.owner, Node):
            self.sync_env(self.owner)

    def __update_from_relations(self, relation: list) -> None:
        r"""Update the protocol stack from the protocol relation.

        Args:
            relation (list): a list of protocol relations
        """
        assert all(edge[0] != edge[1] for edge in relation), "Invalid relation input."

        # Build or update the relation graph from edges, nodes will be added automatically
        self.stack.add_edges_from(relation)
        # Update protocol owners
        for protocol in self.protocols:
            protocol.owner = self

    def sync_env(self, node: "Node") -> None:
        r"""Synchronize the simulation environment for each protocol.

        Args:
            node (Node): node that the protocol stack is loaded to
        """
        for protocol in self.protocols:
            protocol.node = node
            protocol.scheduler.env = node.env

    def update(self, protocol: "Protocol", upper_protocols=None, lower_protocols=None) -> None:
        r"""Add a protocol to the protocol stack.

        Args:
            protocol (Protocol): protocol to add
            upper_protocols (List[Protocol]): upper protocols of the protocol
            lower_protocols (List[Protocol]): lower protocols of the protocol
        """
        assert upper_protocols is not None or lower_protocols is not None,\
            f"Should assign at least an upper protocol or a lower protocol!"

        from qcompute_qnet.topology.node import Node
        if isinstance(self.owner, Node):
            protocol.node = self.owner
            protocol.scheduler.env = self.owner.env
        relation = []
        if upper_protocols is not None:
            for upper in upper_protocols:
                relation.append((upper, protocol))
        if lower_protocols is not None:
            for lower in lower_protocols:
                relation.append((protocol, lower))
        self.__update_from_relations(relation)

    def print_stack(self) -> None:
        r"""Print the stack relation as a graph.
        """
        labels = {proto: proto.name for proto in self.protocols}
        pos = nx.spring_layout(self.stack)
        nx.draw(self.stack, pos=pos, labels=labels, with_labels=True, edge_color='r', arrowsize=20)
        plt.show()

    def start(self, **kwargs) -> None:
        r"""Start the protocol stack.

        Args:
            **kwargs: keyword arguments to start the protocol stack
        """
        assert self.owner is not None, f"Should load the protocol stack to a node first!"

        for proto in self.protocols:
            if proto.is_top:
                handler = EventHandler(proto, "start", **kwargs)
                proto.scheduler.schedule_now(handler)
