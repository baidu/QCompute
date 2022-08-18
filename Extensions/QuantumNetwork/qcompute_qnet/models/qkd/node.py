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
Module for QKD node templates with different functionalities.
"""

from typing import List
from qcompute_qnet.topology.node import Node, Satellite
from qcompute_qnet.devices.source import PhotonSource
from qcompute_qnet.devices.detector import PolarizationDetector
from qcompute_qnet.protocols.protocol import ProtocolStack
from qcompute_qnet.models.qkd.application import QKDApp
from qcompute_qnet.models.qkd.routing import QKDRouting
from qcompute_qnet.models.qkd.key_generation import BB84, DecoyBB84, KeyGeneration

__all__ = [
    "QKDNode",
    "EndNode",
    "TrustedRepeaterNode",
    "BackboneNode",
    "QKDSatellite"
]


class QKDNode(Node):
    r"""Class for the simulation of a QKD node.

    Attributes:
        location (Tuple): geographical location of the QKD node
        photon_source (PhotonSource): photon source of the QKD node
        polar_detector (PolarizationDetector): polarization detector of the QKD node
        keys (Dict[Node, List]): generated keys with other QKD nodes
    """

    def __init__(self, name: str, env=None, location=None):
        r"""Constructor for QKDNode class.

        Args:
            name (str): name of the QKD node
            env (DESEnv): related discrete-event simulation environment
            location (Tuple): geographical location of the QKD node
        """
        super().__init__(name, env)
        self.location = location
        self.photon_source = PhotonSource("photon_source")
        self.polar_detector = PolarizationDetector("polar_detector")
        self.install([self.photon_source, self.polar_detector])
        ps = ProtocolStack(f"ps_{self.name}")
        self.load_protocol(ps)
        self.keys = {}

    def set_key_generation(self, peer=None, **kwargs) -> "KeyGeneration":
        r"""Set a key generation protocol with given parameters.

        Args:
            peer (Node): peer node of the key generation protocol
            **kwargs: keyword arguments to set

        Returns:
            KeyGeneration: key generation protocol instance created

        Note:
            This method will set the peer node and other related parameters of the key generation protocol of a
            ``QKDNode`` and return a protocol instance.

        Hint:
            If the parameters are not specified, default parameters are used in the key generation protocol.

        Examples:
            It is a universal approach to parse the parameters as a dictionary.

            >>> alice = QKDNode("Alice")
            >>> bob = QKDNode("Bob")
            >>> intensities = {"prob": [0.5, 0.25, 0.25], "mean_photon_num": [0.8, 0.1, 0]}
            >>> tx_options = {"protocol": "DecoyBB84", "tx_bases_ratio": [0.8, 0.2], "intensities": intensities}
            >>> rx_options = {"protocol": "DecoyBB84", "rx_bases_ratio": [0.6, 0.4]}
            >>> bb84_alice = alice.set_key_generation(peer=bob, **tx_options)
            >>> bb84_bob = bob.set_key_generation(peer=alice, **rx_options)
        """
        # If the parameters are specified, reset related configurations in the protocol stack
        if kwargs != {}:
            for attr in kwargs:
                if attr == "protocol":
                    assert isinstance(kwargs[attr], str), "'protocol' should be a str value."
                elif attr == "tx_bases_ratio":
                    assert isinstance(kwargs[attr], list), "'tx_bases_ratio' should be a list."
                elif attr == "rx_bases_ratio":
                    assert isinstance(kwargs[attr], list), "'rx_bases_ratio' should be a list."
                elif attr == "intensities":
                    assert isinstance(kwargs[attr], dict), "'intensities' should be a dict."
                    assert "prob" in kwargs[attr].keys() and "mean_photon_num" in kwargs[attr].keys(), \
                        "'intensities' should include 'prob' and 'mean_photon_num'."
                    assert isinstance(kwargs[attr]["prob"], list), "'prob' should be a list"
                    assert isinstance(kwargs[attr]["mean_photon_num"], list), "'mean_photon_num' should be a list"
                else:
                    raise TypeError(f"Setting {attr} is not allowed.")
            self.protocol_stack.set(protocol_type="key_generation", **kwargs)

        # Else, set with default parameters if not configure before
        elif kwargs == {} and "key_generation" not in self.protocol_stack.config.keys():
            print(f"Note: Default parameters are set for {self.name}'s key generation protocol.")
            self.protocol_stack.set(protocol_type="key_generation", protocol="BB84")

        if peer is not None:
            if self.protocol_stack.config['key_generation']['protocol'].casefold() == "BB84".casefold():
                key_gen_proto = BB84(f"BB84_{self.name}_{peer.name}", peer)
                if 'tx_bases_ratio' in self.protocol_stack.config['key_generation'].keys():
                    key_gen_proto.set(tx_bases_ratio=self.protocol_stack.config['key_generation']['tx_bases_ratio'])
                if 'rx_bases_ratio' in self.protocol_stack.config['key_generation'].keys():
                    key_gen_proto.set(rx_bases_ratio=self.protocol_stack.config['key_generation']['rx_bases_ratio'])

            elif self.protocol_stack.config['key_generation']['protocol'].casefold() == "DecoyBB84".casefold():
                key_gen_proto = DecoyBB84(f"DecoyBB84_{self.name}_{peer.name}", peer)
                if 'tx_bases_ratio' in self.protocol_stack.config['key_generation'].keys():
                    key_gen_proto.set(tx_bases_ratio=self.protocol_stack.config['key_generation']['tx_bases_ratio'])
                if 'rx_bases_ratio' in self.protocol_stack.config['key_generation'].keys():
                    key_gen_proto.set(rx_bases_ratio=self.protocol_stack.config['key_generation']['rx_bases_ratio'])
                if 'intensities' in self.protocol_stack.config['key_generation'].keys():
                    key_gen_proto.set_intensity(self.protocol_stack.config['key_generation']['intensities'])

            else:
                raise TypeError(f"Protocol {self.protocol_stack.config['key_generation']['protocol']}"
                                f" is not supported in this version.")

            return key_gen_proto

    def receive_classical_msg(self, src: "Node", msg: "ClassicalMessage") -> None:
        r"""Receive a classical message from the classical channel.

        The source node is inferred by the receiver and is not part of the transmitted message.

        Args:
            src (Node): source of the classical message
            msg (ClassicalMessage): classical message to receive
        """
        if msg.dst == self:
            for proto in self.protocol_stack.protocols:
                if isinstance(proto, msg.protocol):
                    proto.receive_classical_msg(msg)
        else:  # find the routing protocol to deal with the message
            for proto in self.protocol_stack.protocols:
                if isinstance(proto, QKDRouting):
                    proto.receive_classical_msg(msg)

    def receive_quantum_msg(self, src: "Node", msg: "QuantumMessage") -> None:
        r"""Receive a quantum message from the quantum channel.

        Args:
            src (Node): sender node of the quantum message
            msg (QuantumMessage): quantum message to receive
        """
        for proto in self.protocol_stack.protocols:
            if isinstance(proto, KeyGeneration):
                proto.receive_quantum_msg(msg)


class EndNode(QKDNode):
    r"""Class for the simulation of an end node.

    Note that ``EndNode`` usually represents a user node.

    Attributes:
        direct_repeater (TrustedRepeaterNode): directly connected repeater
    """

    def __init__(self, name: str, env=None, location=None):
        r"""Constructor for EndNode class.

        Args:
            name (str): name of the end node
            env (DESEnv): related discrete-event simulation environment
            location (Tuple): geographical location of the end node
        """
        super().__init__(name, env, location)
        self.direct_repeater = None
        routing = QKDRouting(f"QKDRouting_{self.name}")
        self.protocol_stack.build(routing)
        self.load_protocol(self.protocol_stack)

    def init(self) -> None:
        r"""Node initialization and assign the direct repeater if there is one.
        """
        super().init()
        for node in self.links.keys():
            if isinstance(node, TrustedRepeaterNode):
                self.direct_repeater = node

    @property
    def direct_nodes(self) -> List:
        r"""Return nodes that directly connect with the end node.

        Returns:
            List: a list of nodes that are directly connected with the node
        """
        return list(self.links.keys())

    def key_request(self, **kwargs) -> None:
        r"""Generate a QKD request.

        Args:
            **kwargs: keyword arguments for a key request
        """
        qkd_app = self.set_qkd_app()
        qkd_app.start(**kwargs)

    def set_qkd_app(self) -> "QKDApp":
        r"""Set QKDApp protocol and update the protocol stack.

        Returns:
            QKDApp: QKDApp protocol instance created
        """
        for proto in self.protocol_stack.protocols:
            # TODO: only consider QKDRouting as a lower protocol in this version
            if isinstance(proto, QKDRouting):
                qkd_routing = proto
        qkd_app = QKDApp(f"QKDApp_{self.name}")
        self.protocol_stack.update(qkd_app, lower_protocols=[qkd_routing])

        return qkd_app

    def set_key_generation(self, peer=None, **kwargs) -> "KeyGeneration":
        r"""Set a key generation protocol with given parameters.

        Args:
            peer (Node): peer node of the key generation protocol
            **kwargs: keyword arguments to set

        Returns:
            KeyGeneration: key generation protocol instance created
        """
        key_gen_proto = super().set_key_generation(peer, **kwargs)

        if key_gen_proto is not None:
            for proto in self.protocol_stack.protocols:
                # TODO: only consider QKDRouting as a lower protocol in this version
                if isinstance(proto, QKDRouting):
                    qkd_routing = proto
            self.protocol_stack.update(key_gen_proto, upper_protocols=[qkd_routing])

            return key_gen_proto


class TrustedRepeaterNode(QKDNode):
    r"""Class for the simulation of a trusted repeater node.

    Attributes:
        classical_routing_table (Dict[Node, Node]): classical routing table the trusted repeater holds
        quantum_routing_table (Dict[Node, Node]): quantum routing table the trusted repeater holds
    """

    def __init__(self, name: str, env=None, location=None):
        r"""Constructor for TrustedRepeaterNode class.

        Args:
            name (str): name of the trusted repeater
            env (DESEnv): related discrete-event simulation environment
            location (Tuple): geographical location of the trusted repeater
        """
        super().__init__(name, env, location)
        self.classical_routing_table = {}
        self.quantum_routing_table = {}
        routing = QKDRouting(f"QKDRouting_{self.name}")
        self.protocol_stack.build(routing)
        self.load_protocol(self.protocol_stack)

    def set_key_generation(self, peer=None, **kwargs) -> "KeyGeneration":
        r"""Set a key generation protocol with given parameters.

        Args:
            peer (Node): peer node of the key generation protocol
            **kwargs: keyword arguments to set

        Returns:
            KeyGeneration: key generation protocol instance created
        """
        key_gen_proto = super().set_key_generation(peer, **kwargs)

        if key_gen_proto is not None:
            for proto in self.protocol_stack.protocols:
                # TODO: only consider QKDRouting as an upper protocol in this version
                if isinstance(proto, QKDRouting):
                    qkd_routing = proto
            self.protocol_stack.update(key_gen_proto, upper_protocols=[qkd_routing])

            return key_gen_proto


class BackboneNode(TrustedRepeaterNode):
    r"""Class for the simulation of a backbone node.
    """

    def __init__(self, name: str, env=None, location=None):
        r"""Constructor for BackboneNode class.

        Args:
            name (str): name of the backbone node
            env (DESEnv): related discrete-event simulation environment
            location (Tuple): geographical location of the backbone node
        """
        super().__init__(name, env, location)


class QKDSatellite(QKDNode, Satellite):
    r"""Class for satellites capable of implementing quantum key distribution.
    """

    def __init__(self, name: str, env=None, location=None):
        r"""Constructor for QKDSatellite class.

        Args:
            name (str): name of the QKD satellite node
            env (DESEnv): related discrete-event simulation environment
            location (Tuple): geographical location of the QKD satellite node
        """
        super().__init__(name, env, location)
