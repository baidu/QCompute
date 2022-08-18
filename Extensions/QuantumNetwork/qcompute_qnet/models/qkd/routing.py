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
Module for the QKD routing protocol.
"""

from typing import List
from qcompute_qnet.core.des import EventHandler
from qcompute_qnet.protocols.routing import Routing
from qcompute_qnet.models.qkd.key_generation import PrepareAndMeasure
from qcompute_qnet.models.qkd.message import QKDMessage

__all__ = [
    "QKDRouting"
]


class QKDRouting(Routing):
    r"""Class for the routing protocol in quantum key distribution.

    Attributes:
        path (List[Node]): path of key swapping
        upstream_node (Node): upstream node in the path
        downstream_node (Node): downstream node in the path
        keys (Dict[Node, List]): keys generated with upstream and downstream nodes
    """

    def __init__(self, name: str):
        r"""Constructor for QKDRouting class.

        Args:
            name (str): name of the QKDRouting protocol
        """
        super().__init__(name)
        self.path = None
        self.upstream_node = None
        self.downstream_node = None
        self.keys = {}

    def receive_upper(self, upper_protocol: type, **kwargs) -> None:
        r"""Receive a message from an upper protocol.

        Args:
            upper_protocol (type): upper protocol that sends the message
            **kwargs: received keyword arguments
        """
        msg = kwargs['msg']

        # Alice: Send a 'REQUEST' message
        if msg.data['type'] == QKDMessage.Type.REQUEST:
            # Check if the destination is directly connected to the node
            if msg.dst in self.node.direct_nodes:
                self.node.send_classical_msg(dst=msg.dst, msg=msg)
            else:
                self.node.send_classical_msg(dst=self.node.direct_repeater, msg=msg)

        # Bob: Send back an 'ACCEPT' message
        elif msg.data['type'] == QKDMessage.Type.ACCEPT:
            self.path = msg.data['path']  # save the path for key swapping
            self.node.env.logger.info(f"Path of the QKD request: {[node.name for node in self.path]}")

            self.upstream_node = self.path[-2]  # get the upstream node from the path
            key_num = msg.data['key_num']
            key_length = msg.data['key_length']

            self.node.send_classical_msg(dst=self.upstream_node, msg=msg)  # forward the 'ACCEPT' to the upstream node

            # Create an instance for key generation with the upstream node
            self.node.set_key_generation(self.upstream_node)
            self.send_lower(PrepareAndMeasure, peer=self.upstream_node, role=PrepareAndMeasure.Role.RECEIVER,
                            key_num=key_num, key_length=key_length)

        # Bob: Send back the 'ACKNOWLEDGE' message to the repeater node
        elif msg.data['type'] == QKDMessage.Type.ACKNOWLEDGE:
            if msg.dst in self.node.direct_nodes:
                self.node.send_classical_msg(dst=msg.dst, msg=msg)
            else:
                self.node.send_classical_msg(dst=self.node.direct_repeater, msg=msg)

        # Bob: Send back the 'DONE' message to Alice
        elif msg.data['type'] == QKDMessage.Type.DONE:
            if msg.dst in self.node.direct_nodes:
                self.node.send_classical_msg(dst=msg.dst, msg=msg)
            else:
                self.node.send_classical_msg(dst=self.node.direct_repeater, msg=msg)

            self.reset()

    def receive_lower(self, lower_protocol: type, **kwargs) -> None:
        r"""Receive a message from a lower protocol.

        Args:
            lower_protocol (type): lower protocol that sends the message
            **kwargs: received keyword arguments
        """
        from qcompute_qnet.models.qkd.node import TrustedRepeaterNode, EndNode
        from qcompute_qnet.models.qkd.application import QKDApp

        peer = kwargs['peer']
        self.keys[peer] = kwargs['sifted_keys']

        # EndNode: Deliver the sifted keys to the upper protocol
        if isinstance(self.node, EndNode):
            # If the two users are directly connected, then finish the QKD
            finish = True if len(self.path) == 2 else False
            ready_msg = QKDMessage(src=None,
                                   dst=None,
                                   protocol=QKDApp,
                                   data={'type': QKDMessage.Type.READY,
                                         'sifted_keys': self.keys[peer],
                                         'finish': finish})
            self.send_upper(QKDApp, msg=ready_msg)

        # Trusted repeater node: Do key swapping
        elif isinstance(self.node, TrustedRepeaterNode):
            # Check if already generated keys for both directions
            if self.upstream_node in self.keys.keys() and self.downstream_node in self.keys.keys():
                ciphertext_list = []
                # Do 'xor' operation to each pair of keys to generate a ciphertext list
                for key_down, key_up in zip(self.keys[self.downstream_node], self.keys[self.upstream_node]):
                    ciphertext_list.append(key_down ^ key_up)

                # Send the ciphertext list to Bob
                cipher_msg = QKDMessage(src=self.node,
                                        dst=self.path[-1],
                                        protocol=QKDRouting,
                                        data={'type': QKDMessage.Type.CIPHERTEXT,
                                              'ciphertext_list': ciphertext_list})
                next_hop = self.node.classical_routing_table[cipher_msg.dst]
                self.node.send_classical_msg(dst=next_hop, msg=cipher_msg)

    def receive_classical_msg(self, msg: "QKDMessage", **kwargs) -> None:
        r"""Receive a QKDMessage from the node.

        Args:
            msg (QKDMessage): received QKDMessage
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        from qcompute_qnet.models.qkd.node import EndNode, TrustedRepeaterNode
        from qcompute_qnet.models.qkd.application import QKDApp

        if msg.data['type'] == QKDMessage.Type.REQUEST:
            self.node.env.logger.info(f"{self.node.name} received 'REQUEST' of QKDApp "
                                      f"with {msg.dst.name} from {msg.src.name}")

            # Repeater: Receive the 'REQUEST' from Alice
            if isinstance(self.node, TrustedRepeaterNode):
                msg.data['path'].append(self.node)  # append 'self.node' to the path

                # Find the next hop node and forward the 'REQUEST' message
                next_hop = self.node.quantum_routing_table[msg.dst]
                self.node.send_classical_msg(dst=next_hop, msg=msg)

            # Bob: Receive the 'REQUEST' from Alice
            elif isinstance(self.node, EndNode):
                # Create a QKDApp instance and send the request
                self.node.set_qkd_app()
                self.send_upper(QKDApp, msg=msg)

        elif msg.data['type'] == QKDMessage.Type.ACCEPT:
            self.node.env.logger.info(f"{self.node.name} received 'ACCEPT' of QKDApp from {msg.src.name}")

            self.path = msg.data['path']  # save the path for key swapping
            key_num = msg.data['key_num']
            key_length = msg.data['key_length']

            # Repeater: Receive the 'ACCEPT' message
            if isinstance(self.node, TrustedRepeaterNode):
                # Get upstream node and downstream node according to the path
                index = msg.data['path'].index(self.node)
                self.upstream_node, self.downstream_node = msg.data['path'][index - 1], msg.data['path'][index + 1]

                self.node.send_classical_msg(self.upstream_node, msg=msg)  # forward 'ACCEPT' to the upstream node

                # Set lower 'PrepareAndMeasure' protocols for key generation with the upstream node and downstream node
                self.node.set_key_generation(self.upstream_node)
                self.send_lower(PrepareAndMeasure, peer=self.upstream_node, role=PrepareAndMeasure.Role.RECEIVER,
                                key_num=key_num, key_length=key_length)

                self.node.set_key_generation(self.downstream_node)
                self.send_lower(PrepareAndMeasure, peer=self.downstream_node, role=PrepareAndMeasure.Role.TRANSMITTER,
                                key_num=key_num, key_length=key_length)

            # Alice: Receive the 'ACCEPT' message from Bob
            elif isinstance(self.node, EndNode):
                self.downstream_node = self.path[1]  # get downstream node from the path

                # Begin key generation with downstream node
                self.node.set_key_generation(self.downstream_node)
                self.send_lower(PrepareAndMeasure, peer=self.downstream_node, role=PrepareAndMeasure.Role.TRANSMITTER,
                                key_num=key_num, key_length=key_length)

        elif msg.data['type'] == QKDMessage.Type.CIPHERTEXT:
            # Bob: Receive ciphertext list from a repeater node
            if msg.dst == self.node:
                self.node.env.logger.info(f"{self.node.name} received 'CIPHERTEXT' of QKDApp from {msg.src.name}")
                self.send_upper(QKDApp, msg=msg)

            # Repeater: Receive the 'CIPHERTEXT' from other repeaters, directly forward the message to the destination
            else:
                next_hop = self.node.classical_routing_table[msg.dst]
                self.node.send_classical_msg(dst=next_hop, msg=msg)

        # Repeater: Receive the 'ACKNOWLEDGE' from Bob
        elif msg.data['type'] == QKDMessage.Type.ACKNOWLEDGE:
            if msg.dst == self.node:
                self.node.env.logger.info(f"{self.node.name} received 'ACKNOWLEDGE' for ciphertext of QKDApp"
                                          f" from {msg.src.name}")
                self.reset()  # reset the QKD routing protocol

            else:
                next_hop = self.node.classical_routing_table[msg.dst]
                self.node.send_classical_msg(dst=next_hop, msg=msg)

        elif msg.data['type'] == QKDMessage.Type.DONE:
            # Alice: Receive the 'DONE' from Bob
            if msg.dst == self.node:
                self.node.env.logger.info(f"{self.node.name} received 'DONE' of QKDApp from {msg.src.name}")
                handler = EventHandler(self, "send_upper", [QKDApp], msg=msg)
                self.node.scheduler.schedule_now(handler)
                self.reset()  # reset the QKD routing protocol

            # Repeater: Directly forward the 'DONE' message
            else:
                next_hop = self.node.classical_routing_table[msg.dst]
                self.node.send_classical_msg(dst=next_hop, msg=msg)

    def reset(self) -> None:
        r"""Reset the QKD routing protocol.
        """
        for proto in self.lower_protocols:
            if isinstance(proto, PrepareAndMeasure) and \
                    (proto.peer == self.upstream_node or proto.peer == self.downstream_node):
                # Remove the PrepareAndMeasure protocol instances from the protocol stack
                self.owner.stack.remove_node(proto)
                proto.owner = proto
        self.path = None
        self.upstream_node = None
        self.downstream_node = None
        self.keys = {}
