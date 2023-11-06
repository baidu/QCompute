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
from Extensions.QuantumNetwork.qcompute_qnet.core.des import Entity, EventHandler
from Extensions.QuantumNetwork.qcompute_qnet.topology.node import Node, Satellite
from Extensions.QuantumNetwork.qcompute_qnet.protocols.protocol import ProtocolStack
from Extensions.QuantumNetwork.qcompute_qnet.devices.source import PhotonSource
from Extensions.QuantumNetwork.qcompute_qnet.devices.detector import PolarizationDetector
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.application import QKDApp, QKDAppL4
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.resource_management import QKDRMP
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.routing import QKDRouting, QKDRoutingL2
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.key_generation import BB84, DecoyBB84, KeyGeneration
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.message import QKDMessage

__all__ = ["QKDNode", "EndNode", "TrustedRepeaterNode", "BackboneNode", "QKDSatellite", "RMPEndNode", "RMPRepeaterNode"]


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
                    assert (
                        "prob" in kwargs[attr].keys() and "mean_photon_num" in kwargs[attr].keys()
                    ), "'intensities' should include 'prob' and 'mean_photon_num'."
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
            if self.protocol_stack.config["key_generation"]["protocol"].casefold() == "BB84".casefold():
                key_gen_proto = BB84(f"BB84_{self.name}_{peer.name}", peer)
                if "tx_bases_ratio" in self.protocol_stack.config["key_generation"].keys():
                    key_gen_proto.set(tx_bases_ratio=self.protocol_stack.config["key_generation"]["tx_bases_ratio"])
                if "rx_bases_ratio" in self.protocol_stack.config["key_generation"].keys():
                    key_gen_proto.set(rx_bases_ratio=self.protocol_stack.config["key_generation"]["rx_bases_ratio"])

            elif self.protocol_stack.config["key_generation"]["protocol"].casefold() == "DecoyBB84".casefold():
                key_gen_proto = DecoyBB84(f"DecoyBB84_{self.name}_{peer.name}", peer)
                if "tx_bases_ratio" in self.protocol_stack.config["key_generation"].keys():
                    key_gen_proto.set(tx_bases_ratio=self.protocol_stack.config["key_generation"]["tx_bases_ratio"])
                if "rx_bases_ratio" in self.protocol_stack.config["key_generation"].keys():
                    key_gen_proto.set(rx_bases_ratio=self.protocol_stack.config["key_generation"]["rx_bases_ratio"])
                if "intensities" in self.protocol_stack.config["key_generation"].keys():
                    key_gen_proto.set_intensity(self.protocol_stack.config["key_generation"]["intensities"])

            else:
                raise TypeError(
                    f"Protocol {self.protocol_stack.config['key_generation']['protocol']}"
                    f" is not supported in this version."
                )

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

    def __init__(self, name: str, env=None, location=None, rmp: bool = False):
        r"""Constructor for EndNode class.

        Args:
            name (str): name of the end node
            env (DESEnv): related discrete-event simulation environment
            location (Tuple): geographical location of the end node
            rmp (bool): whether to adopt the resource management protocol for quantum key distribution
        """
        super().__init__(name, env, location)
        self.direct_repeater = None
        routing = QKDRoutingL2(f"QKDRoutingL2_{self.name}") if rmp is True else QKDRouting(f"QKDRouting_{self.name}")
        self.protocol_stack.build(routing)
        self.load_protocol(self.protocol_stack)

    def init(self) -> None:
        r"""Node initialization and assign the direct repeater if there is one."""
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

    def __init__(self, name: str, env=None, location=None, rmp: bool = False):
        r"""Constructor for TrustedRepeaterNode class.

        Args:
            name (str): name of the trusted repeater
            env (DESEnv): related discrete-event simulation environment
            location (Tuple): geographical location of the trusted repeater
            rmp (bool): whether to adopt the resource management protocol for quantum key distribution
        """
        super().__init__(name, env, location)
        self.classical_routing_table = {}
        self.quantum_routing_table = {}
        routing = QKDRoutingL2(f"QKDRoutingL2_{self.name}") if rmp is True else QKDRouting(f"QKDRouting_{self.name}")
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
    r"""Class for the simulation of a backbone node."""

    def __init__(self, name: str, env=None, location=None):
        r"""Constructor for BackboneNode class.

        Args:
            name (str): name of the backbone node
            env (DESEnv): related discrete-event simulation environment
            location (Tuple): geographical location of the backbone node
        """
        super().__init__(name, env, location)


class QKDSatellite(QKDNode, Satellite):
    r"""Class for satellites capable of implementing quantum key distribution."""

    def __init__(self, name: str, env=None, location=None):
        r"""Constructor for QKDSatellite class.

        Args:
            name (str): name of the QKD satellite node
            env (DESEnv): related discrete-event simulation environment
            location (Tuple): geographical location of the QKD satellite node
        """
        super().__init__(name, env, location)


class RMPEndNode(EndNode):
    r"""Class for the simulation of an end node in resource management architecture.

    Note that ``RMPEndNode`` usually represents a user node.

    Attributes:
        idle (bool): whether the node is idle
        random_request (bool): whether to generate random requests automatically
        reqs_delivered (int): number of the delivered requests
        reqs_discarded (int): number of the discarded requests
        keys_delivered (int): number of the delivered keys
        keys_discarded (int): number of the discarded keys
    """

    def __init__(self, name: str, env=None, location=None):
        r"""Constructor for RMPEndNode class.

        Args:
            name (str): name of the RMPEndNode
            env (DESEnv): related discrete-event simulation environment
            location (Tuple): geographical location of the RMPEndNode
        """
        super().__init__(name, env, location, rmp=True)
        self.idle = True
        self.random_request = False
        self.reqs_delivered = 0
        self.reqs_discarded = 0
        self.keys_delivered = 0
        self.keys_discarded = 0

        rmp = QKDRMP(f"QKDRMP_{self.name}", self)
        for proto in self.protocol_stack.protocols:
            if isinstance(proto, QKDRoutingL2):
                qkd_routing = proto
        self.protocol_stack.update(rmp, lower_protocols=[qkd_routing])

    def init(self) -> None:
        r"""RMPEndNode initialization."""
        super().init()
        if self.random_request:
            # Generate a random request
            self.random_req()

    def random_req(self) -> None:
        r"""Random key request generation for the demo of Beijing quantum metropolitan area network."""
        import random
        import numpy as np

        if self.reqs_delivered == self.reqs_discarded == 0:
            # Start time for the first request is sampled from a Gaussian distribution
            start_time = int(abs(random.gauss(1e10, 5e9)))
        else:
            # Start time for other requests is sampled from a Poisson distribution
            start_time = int(abs(np.random.poisson(1e7, 1)) * 2e3)

        # Requested key number and key length
        key_num = 10
        key_length = 32

        # Search for other idle RMPEndNodes
        end_list = [
            node for node in self.owner.nodes if isinstance(node, RMPEndNode) and node != self and node.idle is True
        ]

        if len(end_list) != 0:
            # Shuffle the list of end nodes, and randomly choose one peer node for QKD request
            r = random.Random(start_time)
            r.shuffle(end_list)
            dst_node = end_list.pop()

            # Generate a QKD request
            handler = EventHandler(self, "key_request", dst=dst_node, key_num=key_num, key_length=key_length)
            self.scheduler.schedule_after(start_time, handler)
        else:
            return

    def key_request(self, **kwargs) -> None:
        r"""Generate a QKD request.

        Args:
            **kwargs (Any): keyword arguments for a key request
        """
        if self.idle and kwargs["dst"].idle:
            if kwargs["dst"].random_request is False and self.random_request is True:
                self.random_req()
                return
            # Set the status of the sender and receiver to be busy
            self.set_busy()
            kwargs["dst"].set_busy()
            qkd_app = self.set_qkd_app()
            qkd_app.start(**kwargs)
        else:
            # Set a random request for another time point
            self.random_req()

    def set_qkd_app(self) -> "QKDAppL4":
        r"""Set QKDAppL4 protocol and update the protocol stack.

        Returns:
            QKDAppL4: the QKDAppL4 protocol instance created
        """
        for proto in self.protocol_stack.protocols:
            if isinstance(proto, QKDRMP):
                qkd_rmp = proto
        qkd_app = QKDAppL4(f"QKDApp_{self.name}")
        self.protocol_stack.update(qkd_app, lower_protocols=[qkd_rmp])

        return qkd_app

    def set_busy(self) -> None:
        r"""Set the end node to be busy."""
        self.idle = False

    def set_idle(self) -> None:
        r"""Set the end node to be idle."""
        self.idle = True


class RMPRepeaterNode(TrustedRepeaterNode):
    r"""Class for the simulation of a repeater node with resource management.

    Note:
        For each connected link of a ``RMPRepeaterNode``, there is a pair of photon source and polarization detector
        for quantum key distribution.

    Attributes:
        key_pools (Dict): a list of the key pools owned by the node
        reqs_delivered (int): number of delivered requests
        keys_delivered (int): number of rejected requests
        reqs_discarded (int): number of discarded requests
        keys_discarded (int): number of discarded keys
        reqs_rejected (int): number of rejected requests
        keys_rejected (int): number of rejected keys
        reqs_invalid (int): number of invalid requests
        keys_invalid (int): number of invalid keys
        key_pool_volume (int): maximum volume of the key pools
        key_length (int): length of the keys stored in the key pools
    """

    def __init__(self, name: str, env=None, location=None, key_pool_volume: int = 20, key_length: int = 32):
        r"""Constructor of the RMPRepeaterNode class.

        Args:
            name (str): name of the RMPRepeaterNode
            env (DESEnv): related discrete-event simulation environment
            location (Tuple): geographical location of the RMPRepeaterNode
            key_pool_volume (int): maximum volume of the key pools
            key_length (int): length of the keys stored in the key pools
        """
        super().__init__(name, env, location, rmp=True)
        self.key_pools = {}

        self.reqs_delivered = 0
        self.reqs_discarded = 0
        self.reqs_invalid = 0
        self.reqs_rejected = 0

        self.keys_delivered = 0
        self.keys_discarded = 0
        self.keys_invalid = 0
        self.keys_rejected = 0

        self.key_pool_volume = key_pool_volume
        self.key_length = key_length

        rmp = QKDRMP(f"QKDRMP_{self.name}", self)
        self.protocol_stack.build(rmp)
        for proto in self.protocol_stack.protocols:
            if isinstance(proto, QKDRoutingL2):
                qkd_routing = proto
        self.protocol_stack.update(rmp, lower_protocols=[qkd_routing])

    def init(self) -> None:
        r"""RMPRepeaterNode initialization."""
        self.init_photon_sources()
        self.init_polar_detectors()
        self.init_key_pools(max_volume=self.key_pool_volume, key_length=self.key_length)

    def init_photon_sources(self) -> None:
        r"""Initialization of the photon sources. One photon source is set for each connected link."""
        photon_sources = {}
        for neighbor_node in self.links:
            photon_source = PhotonSource(f"photon_source_to_{neighbor_node.name}")
            self.install(photon_source)
            photon_sources[neighbor_node] = photon_source

        self.photon_source = photon_sources

    def init_polar_detectors(self) -> None:
        r"""Initialization of the polar detectors. One polar detector is set for each connected link."""
        polar_detectors = {}
        for neighbor_node in self.links:
            polar_detector = PolarizationDetector(f"polar_detector_from_{neighbor_node.name}")
            self.install(polar_detector)
            polar_detectors[neighbor_node] = polar_detector

        self.polar_detector = polar_detectors

    def init_key_pools(self, max_volume: int, key_length: int) -> None:
        r"""Initialization of the key pools with other RMPRepeaterNodes.

        Args:
            max_volume (int): maximum volume of the key pools
            key_length (int): length of keys of the key pools
        """
        for neighbor_node in self.links:
            if isinstance(neighbor_node, RMPRepeaterNode):
                # Set two key pools for an upstream role and a downstream role in key generation, respectively
                upstream_key_pool = self.KeyPool(
                    node=self, peer_node=neighbor_node, key_length=key_length, max_volume=max_volume
                )
                downstream_key_pool = self.KeyPool(
                    node=self, peer_node=neighbor_node, key_length=key_length, max_volume=max_volume
                )
                self.key_pools[neighbor_node] = [upstream_key_pool, downstream_key_pool]
                self.install([upstream_key_pool, downstream_key_pool])

                # Start key generation
                handler_up = EventHandler(upstream_key_pool, "key_gen", peer=neighbor_node, towards="up")
                handler_down = EventHandler(downstream_key_pool, "key_gen", peer=neighbor_node, towards="down")
                self.scheduler.schedule_now(handler_up)
                self.scheduler.schedule_now(handler_down)

    def set_key_generation(self, peer=None, **kwargs) -> "KeyGeneration":
        r"""Set a key generation protocol with given parameters.

        Args:
            peer (Node): peer node of the key generation protocol
            **kwargs (Any): keyword arguments to set

        Returns:
            KeyGeneration: key generation protocol instance created
        """
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.key_generation import PrepareAndMeasure

        towards = kwargs.pop("towards") if kwargs.get("towards") else None
        role = PrepareAndMeasure.Role.TRANSMITTER if towards == "up" else PrepareAndMeasure.Role.RECEIVER

        for proto in self.protocol_stack.protocols:
            if isinstance(proto, QKDRoutingL2):
                qkd_routing_l2 = proto

        # Reuse the existed key generation instance
        for proto in qkd_routing_l2.lower_protocols:
            if proto.peer == peer and proto.role == role:
                self.env.logger.info(
                    f"{self.name} found an existed {proto.name}.{proto.role},"
                    f"reused it as the key generation protocol"
                )
                proto.role = None
                return proto

        key_gen_proto = super().set_key_generation(peer, **kwargs)
        return key_gen_proto

    def receive_quantum_msg(self, src: "Node", msg: "QuantumMessage") -> None:
        r"""Receive a quantum message from the quantum channel.

        Args:
            src (Node): sender node of the quantum message
            msg (QuantumMessage): quantum message to receive
        """
        for proto in self.protocol_stack.protocols:
            if isinstance(proto, KeyGeneration):
                proto.receive_quantum_msg(msg, src=src)

    class KeyPool(Entity):
        r"""Class for the structure that stores the sifted keys between repeaters.

        Attributes:
            node (Node): owner of the key pool
            peer_node (Node): peer node for key generation
            key_length (int): length of the keys in the key pool
            max_volume (int): maximum number of keys the key pool can accommodate
            current_volume (int): number of available keys
            regenerate_threshold (int): minimum number of keys needed to be held for providing service
            keys (List[str]): list of the sifted keys
            idle (bool): whether the key pool is idle
            ready_to_generate (bool): whether the key pool is ready for key generation
            generating (bool): whether the key pool is generating keys
        """

        def __init__(
            self, node: "Node", peer_node: "Node", key_length: int, max_volume: int, name: str = None, env=None
        ):
            r"""Constructor of KeyPool class.

            Args:
                node (Node): owner of the key pool
                peer_node (Node): peer repeater node for key generation
                key_length (int): length of the keys in the key pool
                max_volume (int): maximum number of keys the key pool can accommodate
                name (str): name of the key pool
                env (DESEnv): simulation environment of the key pool
            """
            super().__init__(name, env)
            self.node = node
            self.peer_node = peer_node

            self.key_length = key_length
            self.max_volume = max_volume
            self.regenerate_threshold = max_volume // 4
            self.current_volume = 0

            self.keys = []
            self.idle = True
            self.ready_to_generate = True
            self.generating = False

        def init(self) -> None:
            r"""KeyPool initialization."""
            self.owner = self

        def key_gen(self, peer: "Node", towards: str, key_num: int = None) -> None:
            r"""Key generation function for the key pool.

            Args:
                peer (Node): the node sharing the keys
                towards (str): denote the direction of the key pool
                key_num (int): the number of sifted keys to be generated
            """
            self.node.env.logger.info(
                f"{self.node.name}'s {towards}stream key pool" f" started key generation with {peer.name}"
            )

            # Set the status of the key pool
            self.idle = False
            self.generating = True
            self.ready_to_generate = False

            key_num = key_num if key_num is not None else self.max_volume
            # Create a 'GENERATE' message to inform QKDRoutingL2 to generate keys
            generate_msg = QKDMessage(
                src=self.node,
                dst=peer,
                protocol=QKDRoutingL2,
                data={"type": QKDMessage.Type.GENERATE, "key_num": key_num, "key_length": self.key_length},
            )

            for proto in self.node.protocol_stack.protocols:
                if isinstance(proto, QKDRoutingL2):
                    qkd_routing_l2 = proto
                    # Schedule an event for QKDRoutingL2 to generate keys
                    scheduler = EventHandler(qkd_routing_l2, "receive_classical_msg", msg=generate_msg, towards=towards)
                    self.node.scheduler.schedule_now(scheduler)
                    break

        def fulfill(self, request: "QKDMessage") -> bool:
            r"""Judge if a request can be fulfilled with current volume of the key pool.

            Args:
                request (QKDMessage): request to execute

            Returns:
                bool: whether the request can be fulfilled
            """
            res = 0
            if request.data["type"] == QKDMessage.Type.PATH:
                res = self.max_volume - request.data["key_num"]
            elif request.data["type"] == QKDMessage.Type.RESERVE:
                res = self.current_volume - request.data["key_num"]

            return True if res > 0 else False

        def allocate(self, key_num: int, key_length: int, **kwargs) -> List:
            r"""Allocate keys to a request.

            Args:
                key_num (int): number of required keys
                key_length (int): length of required keys
                **kwargs: keyword arguments to check the volume and generate new keys

            Returns:
                List: a list of keys allocated for the request
            """
            # Calculate the number of keys to be allocated
            allocate_num = key_num * key_length // self.key_length
            assert self.current_volume > allocate_num, "No enough keys to allocate, should be already filtered."

            # Update current volume
            self.current_volume -= allocate_num
            # Allocate the keys
            allocate_keys = self.keys[0:allocate_num]
            self.keys = self.keys[allocate_num:]
            # Check if the key pools needs to start key generation
            towards = kwargs["towards"]
            self.check_volume(towards)

            return allocate_keys

        def check_volume(self, towards: str) -> None:
            r"""Check the volume of the key pool to decide whether to start key generation.

            Args:
                towards (str): denote the direction of the key pool
            """
            if self.current_volume <= self.regenerate_threshold:
                # Prepare to start key generation for the key pool
                key_num = self.max_volume - self.current_volume
                # Reverse the towards for later operations
                towards = "down" if towards == "up" else "up"
                dual_key_pool_index = 1 if towards == "up" else 0
                dual_key_pool = self.peer_node.key_pools[self.node][dual_key_pool_index]

                self.ready_to_generate = True

                if dual_key_pool.ready_to_generate is True:
                    # Set a delay for the key generation
                    delay = self.node.links[self.peer_node].components[-1].channel2_1.delay * 5
                    handler = EventHandler(self, "key_gen", [self.peer_node, towards, key_num])
                    self.node.scheduler.schedule_after(delay, handler, None)
                    # Reverse direction for the peer node's key generation
                    towards = "down" if towards == "up" else "up"
                    dual_handler = EventHandler(dual_key_pool, "key_gen", [self.node, towards, key_num])
                    self.peer_node.scheduler.schedule_after(delay, dual_handler, None)
