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
Module for the QKD application protocol.
"""

from typing import Tuple

from qcompute_qnet.core.des import EventHandler
from qcompute_qnet.protocols.protocol import Protocol
from qcompute_qnet.models.qkd.routing import QKDRouting
from qcompute_qnet.models.qkd.message import QKDMessage

__all__ = [
    "QKDApp"
]


class QKDApp(Protocol):
    r"""Class for an application protocol in the quantum key distribution.

    Attributes:
        start_time (int): start time of QKDApp protocol
        request_from (Node): node that requests for key generation
        request_to (Node): node that receives a QKD request
        key_num (int): number of requested keys
        key_length (int): length of requested keys
        keys (list): generated keys
        ciphertext_lists (Dict[Node, list]): ciphertext lists received from trusted repeater nodes
    """

    def __init__(self, name: str):
        r"""Constructor for QKDApp class.

        Args:
            name (str): name of the QKDApp protocol
        """
        super().__init__(name)
        self.start_time = 0
        self.request_from = None
        self.request_to = None
        self.key_num = 0
        self.key_length = 0
        self.keys = None
        self.ciphertext_lists = {}

    def start(self, **kwargs) -> None:
        r"""Start QKDApp protocol.

        Args:
            **kwargs: keyword arguments for initialization
        """
        self.start_time = self.node.env.now
        self.request_to = kwargs['dst']
        self.key_num = kwargs['key_num']
        self.key_length = kwargs['key_length']

        handler = EventHandler(self, "send_request")
        self.scheduler.schedule_now(handler)

    def send_request(self) -> None:
        r"""Send a request to the destination.
        """
        self.node.env.logger.info(f"{self.node.name} started a QKD request with {self.request_to.name}")

        request_msg = QKDMessage(src=self.node,
                                 dst=self.request_to,
                                 protocol=QKDRouting,
                                 data={'type': QKDMessage.Type.REQUEST,
                                       'path': [self.node],
                                       'key_num': self.key_num,
                                       'key_length': self.key_length})
        self.send_lower(QKDRouting, msg=request_msg)

    def receive_lower(self, lower_protocol: type, **kwargs) -> None:
        r"""Receive a message from a lower protocol.

        Args:
            lower_protocol (type): lower protocol that sends the message
            **kwargs: received keyword arguments
        """
        msg = kwargs['msg']

        # Bob: Receive the 'REQUEST' message from Alice
        if msg.data['type'] == QKDMessage.Type.REQUEST:
            # Save sender of the QKD request
            self.request_from = msg.src
            # Append 'self.node' to the path
            path = msg.data['path']
            path.append(self.node)
            self.key_num = msg.data['key_num']
            self.key_length = msg.data['key_length']

            # If the request passed repeaters, set 'ciphertext_lists' and wait for key swapping
            if len(path) > 2:
                for node in path[1:-1]:
                    self.ciphertext_lists[node] = None

            # Bob: Send back an 'ACCEPT' message
            accept_msg = QKDMessage(src=self.node,
                                    dst=self.request_from,
                                    protocol=QKDRouting,
                                    data={'type': QKDMessage.Type.ACCEPT,
                                          'path': path,
                                          'key_num': self.key_num,
                                          'key_length': self.key_length})
            self.send_lower(QKDRouting, msg=accept_msg)

        # Receive the 'READY' message from the lower protocol
        elif msg.data['type'] == QKDMessage.Type.READY:
            # Save the keys in 'self.keys'
            self.keys = msg.data['sifted_keys']
            finish = msg.data['finish']

            if finish:
                self.finish()

        # Bob: Receive ciphertext list from a repeater node
        elif msg.data['type'] == QKDMessage.Type.CIPHERTEXT:
            self.ciphertext_lists[msg.src] = msg.data['ciphertext_list']  # save incoming ciphertext list

            # Return an 'ACKNOWLEDGE' message to the repeater node for acknowledgement
            ack_msg = QKDMessage(src=self.node,
                                 dst=msg.src,
                                 protocol=QKDRouting,
                                 data={'type': QKDMessage.Type.ACKNOWLEDGE})
            self.send_lower(QKDRouting, msg=ack_msg)

            # If receive all ciphertext lists, begin decryption for swapped keys
            if all(ciphertext_list is not None for ciphertext_list in self.ciphertext_lists.values()):
                swapped_keys = self.keys
                for ciphertext_list in self.ciphertext_lists.values():
                    # Do 'xor' operation for decryption
                    swapped_keys = [key ^ ciphertext for key, ciphertext in zip(swapped_keys, ciphertext_list)]
                self.keys = swapped_keys

                # Send back a 'DONE' message to Alice
                done_msg = QKDMessage(src=self.node,
                                      dst=self.request_from,
                                      protocol=QKDRouting,
                                      data={'type': QKDMessage.Type.DONE})
                self.send_lower(QKDRouting, msg=done_msg)
                self.finish()

        # Alice: Receive the 'DONE' message from Bob
        elif msg.data['type'] == QKDMessage.Type.DONE:
            self.finish()

    def finish(self) -> None:
        r"""Finish the QKDApp protocol.
        """
        # Alice: Finish her protocol
        if self.request_to is not None:
            # Deliver the keys to the node
            self.node.keys[self.request_to] = self.keys
            self.node.env.logger.info(f"{self.node.name} generated end-to-end keys with {self.request_to.name}")
            self.node.env.logger.info(f"{self.node.name}'s keys (with {self.request_to.name} in decimal): {self.keys}")
            print(f"{self.node.name}'s keys (with {self.request_to.name} in decimal): {self.keys}")
            self.key_rate_estimation()
            if self.node in self.request_to.keys.keys():
                self.error_rate_estimation()  # estimate error rate

        # Bob: Finish his protocol
        elif self.request_from is not None:
            self.node.keys[self.request_from] = self.keys
            self.node.env.logger.info(f"{self.node.name} generated end-to-end keys with {self.request_from.name}")
            self.node.env.logger.info(f"{self.node.name}'s keys "
                                      f"(with {self.request_from.name} in decimal): {self.keys}")
            print(f"{self.node.name}'s keys (with {self.request_from.name} in decimal): {self.keys}")

            if self.node in self.request_from.keys.keys():
                self.error_rate_estimation()  # estimate error rate

        # Remove the protocol itself from the protocol stack
        self.owner.stack.remove_node(self)
        self.owner = self

    def key_rate_estimation(self) -> float:
        r"""Estimate key rate of generated keys.

        Returns:
            float: estimated end-to-end key rate
        """
        key_rate = (self.key_num * self.key_length / 1000) / ((self.node.env.now - self.start_time) * 1e-12)
        self.node.env.logger.info(f"End-to-end key rate: {key_rate:.4f} kbit/s")
        print(f"End-to-end key rate: {key_rate:.4f} kbit/s")
        return key_rate

    def error_rate_estimation(self) -> float:
        r"""Estimate error rate of generated keys.

        Returns:
            float: estimated error rate of the end-to-end keys
        """
        # Calculate error rate of all generated keys
        num_errors = 0
        for i in range(self.key_num):
            if self.request_to is not None:
                key_xor = self.keys[i] ^ self.request_to.keys[self.node][i]  # in decimal
            elif self.request_from is not None:
                key_xor = self.keys[i] ^ self.request_from.keys[self.node][i]  # in decimal
            num_errors += bin(key_xor).count('1')  # count the number of errors
        error_rate = num_errors / (self.key_num * self.key_length)
        self.node.env.logger.info(f"End-to-end error rate: {error_rate:.4f}")
        print(f"End-to-end error rate: {error_rate:.4f}")
        return error_rate

    def statistics_estimation(self) -> Tuple[float, float]:
        r"""Estimate key rate and error rate of generated keys.

        Returns:
            Tuple[float, float]: estimated key rate and error rate of the end-to-end keys
        """
        key_rate = self.key_rate_estimation()
        error_rate = self.error_rate_estimation()
        return key_rate, error_rate
