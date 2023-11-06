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
Module for the QKD resource management protocol.
"""

from Extensions.QuantumNetwork.qcompute_qnet.protocols.resource import RMP, RequestQueue
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.message import QKDMessage

__all__ = ["QKDRMP"]


class QKDRMP(RMP):
    r"""Class for the resource management protocol for quantum key distribution.

    Attributes:
        request_queue (RequestQueue): the request queue of the QKDRMP protocol
        confirm_records (dict): records for the received CONFIRM message from repeater nodes
    """

    def __init__(self, name: str, node: "Node", max_volume: int = 10) -> None:
        r"""Constructor for QKDRMP class.

        Args:
            name (str): name of the QKDRMP protocol
            node (Node): the node that holds the request queue
            max_volume (int): maximum number of requests the request queue can accommodate
        """
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.node import RMPRepeaterNode, RMPEndNode

        super().__init__(name)
        self.node = node
        self.request_queue = RequestQueue(self.node, max_volume) if isinstance(self.node, RMPRepeaterNode) else None
        self.confirm_records = {} if isinstance(self.node, RMPEndNode) else None

    def receive_upper(self, upper_protocol: type, **kwargs) -> None:
        r"""Receive a message from an upper protocol.

        Args:
            upper_protocol (type): upper protocol that sends the message
            **kwargs (Any): received keyword arguments
        """
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.routing import QKDRoutingL2

        msg = kwargs["msg"]

        if msg.data["type"] == QKDMessage.Type.PATH:
            request_msg = QKDMessage(
                src=self.node,
                dst=msg.dst,
                protocol=QKDRoutingL2,
                data={
                    "type": QKDMessage.Type.REQUEST,
                    "req_id": msg.data["req_id"],
                    "path": msg.data["path"],
                    "key_num": msg.data["key_num"],
                    "key_length": msg.data["key_length"],
                },
            )
            self.send_lower(QKDRoutingL2, msg=request_msg)

        elif msg.data["type"] == QKDMessage.Type.RESERVE:
            for node in msg.data["path"][1:-1]:
                self.confirm_records[node] = None

            accept_msg = QKDMessage(
                src=msg.src,
                dst=msg.dst,
                protocol=QKDRoutingL2,
                data={
                    "type": QKDMessage.Type.ACCEPT,
                    "req_id": msg.data["req_id"],
                    "path": msg.data["path"],
                    "key_num": msg.data["key_num"],
                    "key_length": msg.data["key_length"],
                },
            )
            self.send_lower(QKDRoutingL2, msg=accept_msg, option="forward")

        elif msg.data["type"] == QKDMessage.Type.ACKNOWLEDGE:
            # Generate an 'ACKNOWLEDGE' message and send it to QKDRoutingL2
            ack_msg = QKDMessage(
                src=msg.src,
                dst=msg.dst,
                protocol=QKDRoutingL2,
                data={
                    "type": QKDMessage.Type.ACKNOWLEDGE,
                    "req_id": msg.data["req_id"],
                    "path": msg.data["path"],
                    "key_num": msg.data["key_num"],
                    "key_length": msg.data["key_length"],
                },
            )
            self.send_lower(QKDRoutingL2, msg=ack_msg)

        elif msg.data["type"] == QKDMessage.Type.DONE:
            # Send 'DONE' to QKDRoutingL2
            done_msg = QKDMessage(
                src=msg.src,
                dst=msg.dst,
                protocol=QKDRoutingL2,
                data={
                    "type": QKDMessage.Type.DONE,
                    "req_id": msg.data["req_id"],
                    "path": msg.data["path"],
                    "key_num": msg.data["key_num"],
                    "key_length": msg.data["key_length"],
                },
            )
            self.send_lower(QKDRoutingL2, msg=done_msg)

    def receive_lower(self, lower_protocol: "type", **kwargs) -> None:
        r"""Receive a message from a lower protocol.

        Args:
            lower_protocol (type): lower protocol that sends the message
            **kwargs (Any):  received keyword arguments
        """
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.application import QKDAppL4
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.routing import QKDRoutingL2
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.node import RMPEndNode, RMPRepeaterNode

        msg = kwargs["msg"]

        if msg.data["type"] == QKDMessage.Type.PATH:
            if isinstance(self.node, RMPRepeaterNode):
                # Forward the 'PATH' to the next hop
                next_hop = self.node.quantum_routing_table[msg.dst]
                if self.node not in msg.data["path"]:
                    msg.data["path"].append(self.node)

                if isinstance(next_hop, RMPRepeaterNode):
                    # If the request can be fulfilled, forward the 'REQUEST' message
                    if self.node.key_pools[next_hop][0].fulfill(msg):
                        request_msg = QKDMessage(
                            src=msg.src,
                            dst=msg.dst,
                            protocol=QKDRoutingL2,
                            data={
                                "type": QKDMessage.Type.REQUEST,
                                "req_id": msg.data["req_id"],
                                "path": msg.data["path"],
                                "key_num": msg.data["key_num"],
                                "key_length": msg.data["key_length"],
                            },
                        )
                        self.send_lower(QKDRoutingL2, msg=request_msg)
                    else:
                        self.node.env.logger.info(
                            f"{self.node.name} rejected the QKD request from {msg.src.name} to {msg.dst.name}"
                        )

                        reject_msg = QKDMessage(
                            src=self.node,
                            dst=msg.src,
                            protocol=QKDRoutingL2,
                            data={
                                "type": QKDMessage.Type.REJECT,
                                "req_id": msg.data["req_id"],
                                "path": msg.data["path"],
                                "err_type": 1,
                                "key_num": msg.data["key_num"],
                                "key_length": msg.data["key_length"],
                            },
                        )
                        self.send_lower(QKDRoutingL2, msg=reject_msg)

                elif isinstance(next_hop, RMPEndNode):
                    # Send the 'REQUEST' to QKDRoutingL2 to forward directly
                    request_msg = QKDMessage(
                        src=msg.src,
                        dst=msg.dst,
                        protocol=QKDRoutingL2,
                        data={
                            "type": QKDMessage.Type.REQUEST,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_lower(QKDRoutingL2, msg=request_msg)

            elif isinstance(self.node, RMPEndNode):
                # Receiver: Create a QKDAppL4 instance and send the key request
                if msg.dst == self.node:
                    self.node.set_qkd_app()
                    # Generate a 'RESERVE' message and send it to QKDAppL4
                    reserve_msg = QKDMessage(
                        src=self.node,
                        dst=msg.src,
                        protocol=QKDAppL4,
                        data={
                            "type": QKDMessage.Type.RESERVE,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_upper(QKDAppL4, msg=reserve_msg, towards="down")

                else:  # operations for missent request
                    self.node.env.logger.info(
                        f"{self.node.name} received a missent QKD request from {msg.src.name} to {msg.dst.name}"
                    )
                    msg.data["path"].append(self.node)
                    reject_msg = QKDMessage(
                        src=self.node,
                        dst=msg.src,
                        protocol=QKDRoutingL2,
                        data={
                            "type": QKDMessage.Type.REJECT,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "err_type": 1,
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_lower(QKDRoutingL2, msg=reject_msg)

        elif msg.data["type"] == QKDMessage.Type.RESERVE:
            if self.node == msg.dst:  # operations for request sender
                # Send the 'RESERVE' to QKDAppL4
                reserve_msg = QKDMessage(
                    src=msg.src,
                    dst=msg.dst,
                    protocol=QKDAppL4,
                    data={
                        "type": QKDMessage.Type.RESERVE,
                        "req_id": msg.data["req_id"],
                        "path": msg.data["path"],
                        "key_num": msg.data["key_num"],
                        "key_length": msg.data["key_length"],
                    },
                )
                self.send_upper(QKDAppL4, msg=reserve_msg, towards="up")

            else:  # operations of repeater nodes
                pattern = 0
                push_flag = None

                index = msg.data["path"].index(self.node)
                upstream_node, downstream_node = msg.data["path"][index - 1], msg.data["path"][index + 1]
                for neighbor_node in [downstream_node, upstream_node]:
                    if isinstance(neighbor_node, RMPEndNode):
                        pattern += 1

                # Pattern 0: RMPRepeaterNodes on both sides
                if pattern == 0:
                    # Check if the corresponding key pools on both sides are idle and check their volumes
                    if (
                        self.node.key_pools[upstream_node][1].idle is True
                        and self.node.key_pools[upstream_node][1].fulfill(msg) is True
                        and self.node.key_pools[downstream_node][0].idle is True
                        and self.node.key_pools[downstream_node][0].fulfill(msg) is True
                    ):
                        # Set the status of corresponding key pools to be busy
                        self.node.key_pools[upstream_node][1].idle = False
                        self.node.key_pools[downstream_node][0].idle = False
                        # Generate an 'ACCEPT' message
                        accept_msg = QKDMessage(
                            src=msg.src,
                            dst=msg.dst,
                            protocol=QKDRoutingL2,
                            data={
                                "type": QKDMessage.Type.ACCEPT,
                                "req_id": msg.data["req_id"],
                                "path": msg.data["path"],
                                "key_num": msg.data["key_num"],
                                "key_length": msg.data["key_length"],
                            },
                        )
                        # Call the QKDRoutingL2 to fetch the keys and calculate the ciphertext
                        self.send_lower(QKDRoutingL2, msg=accept_msg, option="operate", pattern=pattern)
                        # Reset the status of the corresponding key pools
                        self.node.key_pools[upstream_node][1].idle = True
                        self.node.key_pools[downstream_node][0].idle = True

                    else:
                        # Push the request to the request queue
                        msg.data["pattern"] = pattern
                        push_flag = self.request_queue.push(msg)

                # Pattern 1: RMPEndNode on one side and RMPRepeaterNode on the other side
                elif pattern == 1:
                    if isinstance(upstream_node, RMPEndNode):
                        end_peer, repeater_peer, towards, key_pool_index = upstream_node, downstream_node, "down", 0
                    else:
                        end_peer, repeater_peer, towards, key_pool_index = downstream_node, upstream_node, "up", 1

                    # Prepare for key generation with the end node
                    generate_msg = QKDMessage(
                        src=self.node,
                        dst=end_peer,
                        protocol=QKDRoutingL2,
                        data={
                            "type": QKDMessage.Type.GENERATE,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_lower(QKDRoutingL2, msg=generate_msg, towards=towards)

                    # Check if the corresponding key pool is idle and check the volume of corresponding key pools
                    if (
                        self.node.key_pools[repeater_peer][key_pool_index].idle is True
                        and self.node.key_pools[repeater_peer][key_pool_index].fulfill(msg) is True
                    ):
                        self.node.key_pools[repeater_peer][key_pool_index].idle = False

                        # Generate an 'ACCEPT' message
                        accept_msg = QKDMessage(
                            src=msg.src,
                            dst=msg.dst,
                            protocol=QKDRoutingL2,
                            data={
                                "type": QKDMessage.Type.ACCEPT,
                                "req_id": msg.data["req_id"],
                                "path": msg.data["path"],
                                "key_num": msg.data["key_num"],
                                "key_length": msg.data["key_length"],
                            },
                        )
                        # Call the QKDRoutingL2 to fetch the keys
                        self.send_lower(
                            QKDRoutingL2, msg=accept_msg, option="operate", towards=towards, pattern=pattern
                        )
                        # Reset the status of the corresponding key pool
                        self.node.key_pools[repeater_peer][key_pool_index].idle = True

                    else:
                        # Push the request to the request queue for waiting
                        msg.data["pattern"] = pattern
                        push_flag = self.request_queue.push(msg)

                # Pattern 2: RMPEndNode on both sides
                elif pattern == 2:
                    # Prepare for key generation with the upstream node
                    generate_msg = QKDMessage(
                        src=self.node,
                        dst=upstream_node,
                        protocol=QKDRoutingL2,
                        data={
                            "type": QKDMessage.Type.GENERATE,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_lower(QKDRoutingL2, msg=generate_msg, towards="down")

                    # Prepare for key generation with the downstream node
                    generate_msg = QKDMessage(
                        src=self.node,
                        dst=downstream_node,
                        protocol=QKDRoutingL2,
                        data={
                            "type": QKDMessage.Type.GENERATE,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_lower(QKDRoutingL2, msg=generate_msg, towards="up")

                if push_flag in [True, None]:
                    # Generate an 'ACCEPT' message and send it to QKDRoutingL2
                    accept_msg = QKDMessage(
                        src=msg.src,
                        dst=msg.dst,
                        protocol=QKDRoutingL2,
                        data={
                            "type": QKDMessage.Type.ACCEPT,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_lower(QKDRoutingL2, msg=accept_msg, option="forward")

                    # Generate an 'CONFIRM' message and send it to QKDRoutingL2
                    confirm_msg = QKDMessage(
                        src=self.node,
                        dst=msg.src,
                        protocol=QKDRoutingL2,
                        data={
                            "type": QKDMessage.Type.CONFIRM,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_lower(QKDRoutingL2, msg=confirm_msg)

                elif push_flag is False:
                    self.node.env.logger.info(
                        f"The key request from {msg.dst.name} to {msg.src.name} was rejected "
                        f"for the request queue is full"
                    )
                    # Generate a 'REJECT' message and send it to QKDRoutingL2
                    reject_msg = QKDMessage(
                        src=self.node,
                        dst=msg.dst,
                        protocol=QKDRoutingL2,
                        data={
                            "type": QKDMessage.Type.REJECT,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "err_type": 2,
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_lower(QKDRoutingL2, msg=reject_msg)

                    # Generate another 'REJECT' message and ask the downstream nodes to cancel the resource reservation
                    cancel_msg = QKDMessage(
                        src=self.node,
                        dst=msg.src,
                        protocol=QKDRoutingL2,
                        data={
                            "type": QKDMessage.Type.REJECT,
                            "req_id": msg.data["req_id"],
                            "path": list(reversed(msg.data["path"])),
                            "err_type": 3,
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_lower(QKDRoutingL2, msg=cancel_msg)

        elif msg.data["type"] == QKDMessage.Type.CONFIRM:
            # Save the 'CONFIRM' message
            self.confirm_records[msg.src] = msg
            self.node.env.logger.info(
                f"{self.node.name} received 'CONFIRM' of request {msg.data['req_id']}" f" from repeater {msg.src.name}"
            )

        elif msg.data["type"] == QKDMessage.Type.REJECT:
            err_type = msg.data["err_type"]

            if err_type in [1, 2]:
                towards = "up"

            else:
                towards = "down"
                if isinstance(self.node, RMPRepeaterNode):
                    for req in self.request_queue.requests:
                        # If the request is still in the request queue, remove it from the queue and record the event
                        if req.data["req_id"] == msg.data["req_id"]:
                            self.request_queue.requests.remove(req)

                            self.node.keys_rejected += msg.data["key_num"]
                            self.node.reqs_rejected += 1
                        # If the request is already processed, record the event
                        else:
                            self.node.keys_invalid += msg.data["key_num"]
                            self.node.reqs_invalid += 1
                            self.node.keys_delivered -= msg.data["key_num"]
                            self.node.reqs_delivered -= 1

            if self.node in [msg.src, msg.dst]:
                self.send_upper(QKDAppL4, msg=msg, towards=towards)  # inform QKDAppL4 protocol to discard the request

        elif msg.data["type"] == QKDMessage.Type.POP:
            pop_counter = 0  # pop counter of the request queue
            while pop_counter < self.request_queue.max_volume:
                if self.request_queue.current_volume == 0:  # no request in the queue
                    self.node.env.logger.info(f"{self.node.name}'s request queue was empty, no request popped")
                    break
                else:
                    pop_msg = self.request_queue.pop()
                    pattern = pop_msg.data["pattern"]
                    index = pop_msg.data["path"].index(self.node)
                    upstream_node, downstream_node = pop_msg.data["path"][index - 1], pop_msg.data["path"][index + 1]

                    if pattern == 0:
                        # Check if the corresponding key pools on both sides are idle and check their volumes
                        if (
                            self.node.key_pools[upstream_node][1].idle is True
                            and self.node.key_pools[upstream_node][1].fulfill(pop_msg) is True
                            and self.node.key_pools[downstream_node][0].idle is True
                            and self.node.key_pools[downstream_node][0].fulfill(pop_msg) is True
                        ):
                            # Set the status of corresponding key pools to be busy
                            self.node.key_pools[upstream_node][1].idle = False
                            self.node.key_pools[downstream_node][0].idle = False
                            # Generate an 'ACCEPT' message
                            accept_msg = QKDMessage(
                                src=pop_msg.src,
                                dst=pop_msg.dst,
                                protocol=QKDRoutingL2,
                                data={
                                    "type": QKDMessage.Type.ACCEPT,
                                    "req_id": pop_msg.data["req_id"],
                                    "path": pop_msg.data["path"],
                                    "key_num": pop_msg.data["key_num"],
                                    "key_length": pop_msg.data["key_length"],
                                },
                            )
                            # Call the QKDRoutingL2 to fetch the keys and calculate the ciphertext
                            self.send_lower(QKDRoutingL2, msg=accept_msg, option="operate", pattern=pattern)
                            # Reset the status of the corresponding key pools
                            self.node.key_pools[upstream_node][1].idle = True
                            self.node.key_pools[downstream_node][0].idle = True
                            break

                        else:
                            self.node.env.logger.info(
                                f"The key pools of {self.node.name} couldn't serve for the"
                                f" request now, pushed the request back, pattern = {pattern}"
                            )

                            self.request_queue.push(pop_msg)  # push the request back
                            pop_counter += 1

                    elif pattern == 1:
                        if isinstance(upstream_node, RMPRepeaterNode):
                            repeater_peer = upstream_node
                            key_pool_index = 1
                            towards = "up"
                        else:
                            repeater_peer = downstream_node
                            key_pool_index = 0
                            towards = "down"

                        # Check if the corresponding key pool is idle and check the volume of corresponding key pools
                        if (
                            self.node.key_pools[repeater_peer][key_pool_index].idle is True
                            and self.node.key_pools[repeater_peer][key_pool_index].fulfill(pop_msg) is True
                        ):
                            self.node.key_pools[repeater_peer][key_pool_index].idle = False

                            # Generate a 'ACCEPT' message
                            accept_msg = QKDMessage(
                                src=pop_msg.src,
                                dst=pop_msg.dst,
                                protocol=QKDRoutingL2,
                                data={
                                    "type": QKDMessage.Type.ACCEPT,
                                    "req_id": pop_msg.data["req_id"],
                                    "path": pop_msg.data["path"],
                                    "key_num": pop_msg.data["key_num"],
                                    "key_length": pop_msg.data["key_length"],
                                },
                            )
                            self.send_lower(
                                QKDRoutingL2, msg=accept_msg, option="operate", towards=towards, pattern=pattern
                            )

                            # Reset the status of the corresponding key pool
                            self.node.key_pools[repeater_peer][key_pool_index].idle = True

                        else:
                            self.node.env.logger.info(
                                f"The key pool of {self.node.name} couldn't serve for the"
                                f" request now, pushed the request back, pattern = {pattern}"
                            )

                            self.request_queue.push(pop_msg)  # push the request back
                            pop_counter += 1

        elif msg.data["type"] == QKDMessage.Type.CIPHERTEXT:
            # Send the 'CIPHERTEXT' to QKDAppL4
            self.send_upper(QKDAppL4, msg=msg, towards="down")

        elif msg.data["type"] == QKDMessage.Type.ACKNOWLEDGE:
            # Record the request
            self.node.reqs_delivered += 1
            self.node.keys_delivered += msg.data["key_num"]

        elif msg.data["type"] == QKDMessage.Type.DONE:
            # Send the 'DONE' to QKDAppL4
            self.send_upper(QKDAppL4, msg=msg, towards="up")

        elif msg.data["type"] == QKDMessage.Type.READY:
            # Send the 'READY' to QKDAppL4
            self.send_upper(QKDAppL4, **kwargs)
