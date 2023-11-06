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
from Extensions.QuantumNetwork.qcompute_qnet.core.des import EventHandler
from Extensions.QuantumNetwork.qcompute_qnet.protocols.routing import Routing
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.key_generation import PrepareAndMeasure
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.message import QKDMessage
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.resource_management import QKDRMP

__all__ = ["QKDRouting", "QKDRoutingL2"]


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
        msg = kwargs["msg"]

        # Alice: Send a 'REQUEST' message
        if msg.data["type"] == QKDMessage.Type.REQUEST:
            # Check if the destination is directly connected to the node
            if msg.dst in self.node.direct_nodes:
                self.node.send_classical_msg(dst=msg.dst, msg=msg)
            else:
                self.node.send_classical_msg(dst=self.node.direct_repeater, msg=msg)

        # Bob: Send back an 'ACCEPT' message
        elif msg.data["type"] == QKDMessage.Type.ACCEPT:
            self.path = msg.data["path"]  # save the path for key swapping
            self.node.env.logger.info(f"Path of the QKD request: {[node.name for node in self.path]}")

            self.upstream_node = self.path[-2]  # get the upstream node from the path
            key_num = msg.data["key_num"]
            key_length = msg.data["key_length"]

            self.node.send_classical_msg(dst=self.upstream_node, msg=msg)  # forward the 'ACCEPT' to the upstream node

            # Create an instance for key generation with the upstream node
            self.node.set_key_generation(self.upstream_node)
            self.send_lower(
                PrepareAndMeasure,
                peer=self.upstream_node,
                role=PrepareAndMeasure.Role.RECEIVER,
                key_num=key_num,
                key_length=key_length,
            )

        # Bob: Send back the 'ACKNOWLEDGE' message to the repeater node
        elif msg.data["type"] == QKDMessage.Type.ACKNOWLEDGE:
            if msg.dst in self.node.direct_nodes:
                self.node.send_classical_msg(dst=msg.dst, msg=msg)
            else:
                self.node.send_classical_msg(dst=self.node.direct_repeater, msg=msg)

        # Bob: Send back the 'DONE' message to Alice
        elif msg.data["type"] == QKDMessage.Type.DONE:
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
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.node import TrustedRepeaterNode, EndNode
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.application import QKDApp

        peer = kwargs["peer"]
        self.keys[peer] = kwargs["sifted_keys"]

        # EndNode: Deliver the sifted keys to the upper protocol
        if isinstance(self.node, EndNode):
            # If the two users are directly connected, then finish the QKD
            finish = True if len(self.path) == 2 else False
            ready_msg = QKDMessage(
                src=None,
                dst=None,
                protocol=QKDApp,
                data={"type": QKDMessage.Type.READY, "sifted_keys": self.keys[peer], "finish": finish},
            )
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
                cipher_msg = QKDMessage(
                    src=self.node,
                    dst=self.path[-1],
                    protocol=QKDRouting,
                    data={"type": QKDMessage.Type.CIPHERTEXT, "ciphertext_list": ciphertext_list},
                )
                next_hop = self.node.classical_routing_table[cipher_msg.dst]
                self.node.send_classical_msg(dst=next_hop, msg=cipher_msg)

    def receive_classical_msg(self, msg: "QKDMessage", **kwargs) -> None:
        r"""Receive a QKDMessage from the node.

        Args:
            msg (QKDMessage): received QKDMessage
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.node import EndNode, TrustedRepeaterNode
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.application import QKDApp

        if msg.data["type"] == QKDMessage.Type.REQUEST:
            self.node.env.logger.info(
                f"{self.node.name} received 'REQUEST' of QKDApp " f"with {msg.dst.name} from {msg.src.name}"
            )

            # Repeater: Receive the 'REQUEST' from Alice
            if isinstance(self.node, TrustedRepeaterNode):
                msg.data["path"].append(self.node)  # append 'self.node' to the path

                # Find the next hop node and forward the 'REQUEST' message
                next_hop = self.node.quantum_routing_table[msg.dst]
                self.node.send_classical_msg(dst=next_hop, msg=msg)

            # Bob: Receive the 'REQUEST' from Alice
            elif isinstance(self.node, EndNode):
                # Create a QKDApp instance and send the request
                self.node.set_qkd_app()
                self.send_upper(QKDApp, msg=msg)

        elif msg.data["type"] == QKDMessage.Type.ACCEPT:
            self.node.env.logger.info(f"{self.node.name} received 'ACCEPT' of QKDApp from {msg.src.name}")

            self.path = msg.data["path"]  # save the path for key swapping
            key_num = msg.data["key_num"]
            key_length = msg.data["key_length"]

            # Repeater: Receive the 'ACCEPT' message
            if isinstance(self.node, TrustedRepeaterNode):
                # Get upstream node and downstream node according to the path
                index = msg.data["path"].index(self.node)
                self.upstream_node, self.downstream_node = msg.data["path"][index - 1], msg.data["path"][index + 1]

                self.node.send_classical_msg(self.upstream_node, msg=msg)  # forward 'ACCEPT' to the upstream node

                # Set lower 'PrepareAndMeasure' protocols for key generation with the upstream node and downstream node
                self.node.set_key_generation(self.upstream_node)
                self.send_lower(
                    PrepareAndMeasure,
                    peer=self.upstream_node,
                    role=PrepareAndMeasure.Role.RECEIVER,
                    key_num=key_num,
                    key_length=key_length,
                )

                self.node.set_key_generation(self.downstream_node)
                self.send_lower(
                    PrepareAndMeasure,
                    peer=self.downstream_node,
                    role=PrepareAndMeasure.Role.TRANSMITTER,
                    key_num=key_num,
                    key_length=key_length,
                )

            # Alice: Receive the 'ACCEPT' message from Bob
            elif isinstance(self.node, EndNode):
                self.downstream_node = self.path[1]  # get downstream node from the path

                # Begin key generation with downstream node
                self.node.set_key_generation(self.downstream_node)
                self.send_lower(
                    PrepareAndMeasure,
                    peer=self.downstream_node,
                    role=PrepareAndMeasure.Role.TRANSMITTER,
                    key_num=key_num,
                    key_length=key_length,
                )

        elif msg.data["type"] == QKDMessage.Type.CIPHERTEXT:
            # Bob: Receive ciphertext list from a repeater node
            if msg.dst == self.node:
                self.node.env.logger.info(f"{self.node.name} received 'CIPHERTEXT' of QKDApp from {msg.src.name}")
                self.send_upper(QKDApp, msg=msg)

            # Repeater: Receive the 'CIPHERTEXT' from other repeaters, directly forward the message to the destination
            else:
                next_hop = self.node.classical_routing_table[msg.dst]
                self.node.send_classical_msg(dst=next_hop, msg=msg)

        # Repeater: Receive the 'ACKNOWLEDGE' from Bob
        elif msg.data["type"] == QKDMessage.Type.ACKNOWLEDGE:
            if msg.dst == self.node:
                self.node.env.logger.info(
                    f"{self.node.name} received 'ACKNOWLEDGE' for ciphertext of QKDApp" f" from {msg.src.name}"
                )
                self.reset()  # reset the QKD routing protocol

            else:
                next_hop = self.node.classical_routing_table[msg.dst]
                self.node.send_classical_msg(dst=next_hop, msg=msg)

        elif msg.data["type"] == QKDMessage.Type.DONE:
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
        r"""Reset the QKD routing protocol."""
        for proto in self.lower_protocols:
            if isinstance(proto, PrepareAndMeasure) and (
                proto.peer == self.upstream_node or proto.peer == self.downstream_node
            ):
                # Remove the PrepareAndMeasure protocol instances from the protocol stack
                self.owner.stack.remove_node(proto)
                proto.owner = proto
        self.path = None
        self.upstream_node = None
        self.downstream_node = None
        self.keys = {}


class QKDRoutingL2(QKDRouting):
    r"""Class for the routing protocol in resource management architecture for quantum key distribution.

    Attributes:
        keys_up (Dict): save keys with upstream nodes
        keys_down (Dict): save keys with downstream nodes
        serving_req_id (str): identifier of the serving request
    """

    def __init__(self, name: str):
        r"""Constructor for QKDRoutingL2 class.

        Args:
            name (str): name of the QKDRoutingL2 protocol
        """
        super().__init__(name)
        self.keys_up = {}
        self.keys_down = {}
        self.serving_req_id = None

    def receive_upper(self, upper_protocol: type, **kwargs) -> None:
        r"""Receive a message from an upper protocol.

        Args:
            upper_protocol (type): upper protocol that sends the message
            **kwargs: received keyword arguments
        """
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.node import RMPEndNode, RMPRepeaterNode

        msg = kwargs["msg"]

        if msg.data["type"] == QKDMessage.Type.REQUEST:
            if isinstance(self.node, RMPEndNode):
                # Check if the destination is directly connected to the node
                if msg.dst in self.node.direct_nodes:
                    self.node.send_classical_msg(dst=msg.dst, msg=msg)
                elif isinstance(self.node.direct_repeater, RMPRepeaterNode):
                    self.node.send_classical_msg(dst=self.node.direct_repeater, msg=msg)
                else:
                    self.node.env.logger.info(
                        f"Illegal key request: " f"no RMP repeater between {msg.src.name} and {msg.dst.name}"
                    )

                    # Send a 'REJECT' to QKDRMP protocol
                    reject_msg = QKDMessage(
                        src=msg.src,
                        dst=msg.dst,
                        protocol=QKDRMP,
                        data={
                            "type": QKDMessage.Type.REJECT,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "err_type": 1,
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_upper(upper_protocol=QKDRMP, msg=reject_msg)

            elif isinstance(self.node, RMPRepeaterNode):
                next_hop = self.node.quantum_routing_table[msg.dst]
                self.node.send_classical_msg(dst=next_hop, msg=msg)

        elif msg.data["type"] == QKDMessage.Type.ACCEPT:
            option = kwargs["option"]

            # Condition 1: Forward the message
            if option == "forward":
                # Forward the 'ACCEPT' to the upstream node
                self.path = msg.data["path"]
                index = msg.data["path"].index(self.node)
                upstream_node = msg.data["path"][index - 1]
                self.node.send_classical_msg(upstream_node, msg=msg)

                if self.node == msg.src:
                    key_num = msg.data["key_num"]
                    key_length = msg.data["key_length"]
                    # Set the lower protocol for key generation with its upstream node
                    self.node.set_key_generation(upstream_node)
                    self.send_lower(
                        PrepareAndMeasure,
                        peer=upstream_node,
                        role=PrepareAndMeasure.Role.RECEIVER,
                        key_num=key_num,
                        key_length=key_length,
                    )

            # Condition 2: Process the request
            elif option == "operate":
                pattern = kwargs["pattern"]

                # Pattern 0: RMPRepeaterNode on both sides
                if pattern == 0:
                    self.path = msg.data["path"]
                    index = self.path.index(self.node)
                    upstream_node, downstream_node = msg.data["path"][index - 1], msg.data["path"][index + 1]

                    key_num = msg.data["key_num"]
                    key_length = msg.data["key_length"]

                    # Fetch sifted keys from key pools at both side and generate the ciphertext
                    ciphertext_list = []
                    keys_up = self.node.key_pools[upstream_node][1].allocate(key_num, key_length, towards="up")
                    keys_down = self.node.key_pools[downstream_node][0].allocate(key_num, key_length, towards="down")

                    # Record the fetched keys
                    self.node.env.logger.info(
                        f"{self.node.name} fetched the keys with upstream node "
                        f"{upstream_node.name} from the key pool: {keys_up}"
                    )
                    self.node.env.logger.info(
                        f"{self.node.name} fetched the keys with downstream node "
                        f"{downstream_node.name} from the key pool: {keys_down}"
                    )

                    for key_up, key_down in zip(keys_up, keys_down):
                        ciphertext_list.append(key_up ^ key_down)

                    # Send the ciphertext list to the receiver
                    cipher_msg = QKDMessage(
                        src=self.node,
                        dst=self.path[-1],
                        protocol=QKDRoutingL2,
                        data={
                            "type": QKDMessage.Type.CIPHERTEXT,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "ciphertext_list": ciphertext_list,
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.node.send_classical_msg(dst=downstream_node, msg=cipher_msg)

                    # Record the key delivery
                    self.node.reqs_delivered += 1
                    self.node.keys_delivered += key_num

                    self.serving_req_id = None

                    # Call the QKDRMP to pop a request
                    pop_msg = QKDMessage(
                        src=self.node, dst=self.node, protocol=QKDRMP, data={"type": QKDMessage.Type.POP}
                    )
                    self.send_upper(QKDRMP, msg=pop_msg)

                # Pattern 1: RMPEndNode on one side and RMPRepeaterNode on the other side
                elif pattern == 1:
                    self.serving_req_id = msg.data["req_id"]
                    self.path = msg.data["path"]
                    index = self.path.index(self.node)
                    upstream_node, downstream_node = msg.data["path"][index - 1], msg.data["path"][index + 1]

                    key_num = msg.data["key_num"]
                    key_length = msg.data["key_length"]
                    towards = kwargs["towards"]

                    if towards == "up":
                        key_pool_index = 1
                        repeater_peer = upstream_node
                        end_peer = downstream_node
                        key_end_dict = self.keys_up

                    elif towards == "down":
                        key_pool_index = 0
                        repeater_peer = downstream_node
                        end_peer = upstream_node
                        key_end_dict = self.keys_down

                    # Fetch the keys from the corresponding key pool
                    ciphertext_list = []
                    keys_repeater = self.node.key_pools[repeater_peer][key_pool_index].allocate(
                        key_num, key_length, towards=towards
                    )
                    self.node.env.logger.info(
                        f"{self.node.name} fetched the keys with {repeater_peer.name}"
                        f" from the key pool: {keys_repeater}"
                    )

                    # Check if the key generation on the other side is finished
                    if key_end_dict[end_peer]["keys"] is None:
                        # Save the delivered keys and wait
                        if key_pool_index == 0:
                            self.keys_up[tuple([repeater_peer, end_peer])] = {"path": self.path, "keys": keys_repeater}
                        else:
                            self.keys_down[tuple([repeater_peer, end_peer])] = {
                                "path": self.path,
                                "keys": keys_repeater,
                            }

                    else:
                        # Calculate the ciphertext
                        for key_repeater, key_end in zip(keys_repeater, key_end_dict[end_peer]["keys"]):
                            ciphertext_list.append(key_repeater ^ key_end)
                        # Clear the key dict
                        key_end_dict.pop(end_peer)

                        # Send the ciphertext list to the receiver
                        cipher_msg = QKDMessage(
                            src=self.node,
                            dst=self.path[-1],
                            protocol=QKDRoutingL2,
                            data={
                                "type": QKDMessage.Type.CIPHERTEXT,
                                "req_id": msg.data["req_id"],
                                "path": msg.data["path"],
                                "ciphertext_list": ciphertext_list,
                                "key_num": msg.data["key_num"],
                                "key_length": msg.data["key_length"],
                            },
                        )
                        self.node.send_classical_msg(dst=downstream_node, msg=cipher_msg)
                        self.node.env.logger.info(
                            f"{self.node.name} generated the ciphertext with {repeater_peer.name}"
                            f" and {end_peer.name}, and sent it to {cipher_msg.dst.name}"
                        )

                        # Record the key delivery
                        self.node.reqs_delivered += 1
                        self.node.keys_delivered += key_num

                        self.serving_req_id = None

                        # Ask QKDRMP to pop a request
                        pop_msg = QKDMessage(
                            src=self.node, dst=self.node, protocol=QKDRMP, data={"type": QKDMessage.Type.POP}
                        )
                        self.send_upper(QKDRMP, msg=pop_msg)

        elif msg.data["type"] == QKDMessage.Type.REJECT:
            err_type = msg.data["err_type"]

            if err_type == 1:
                # Forward the 'REJECT' to the upstream node
                upstream_node = msg.data["path"][-2]
                self.node.send_classical_msg(upstream_node, msg=msg)

            elif err_type == 2:
                # Forward the 'REJECT' to the upstream node
                self.path = msg.data["path"]
                index = msg.data["path"].index(self.node)
                upstream_node = msg.data["path"][index - 1]

                self.node.send_classical_msg(upstream_node, msg=msg)

            elif err_type == 3:
                # Forward the 'REJECT' to the downstream nodes to cancel the resource reservation
                self.path = msg.data["path"]
                index = msg.data["path"].index(self.node)
                downstream_node = msg.data["path"][index + 1]

                self.node.send_classical_msg(downstream_node, msg=msg)

        elif msg.data["type"] == QKDMessage.Type.CONFIRM:
            # Forward the 'CONFIRM' to the downstream node
            self.path = msg.data["path"]
            index = msg.data["path"].index(self.node)
            downstream_node = msg.data["path"][index + 1]

            self.node.send_classical_msg(downstream_node, msg=msg)

        elif msg.data["type"] == QKDMessage.Type.ACKNOWLEDGE:
            # Send the message to the next hop
            upstream_node = msg.data["path"][-2]
            self.node.send_classical_msg(dst=upstream_node, msg=msg)

        elif msg.data["type"] == QKDMessage.Type.DONE:
            # Send 'DONE' to the upstream node
            upstream_node = msg.data["path"][-2]
            self.node.send_classical_msg(upstream_node, msg=msg)

        elif msg.data["type"] == QKDMessage.Type.GENERATE:
            towards = kwargs["towards"]
            if isinstance(msg.dst, RMPEndNode):
                if towards == "up":
                    self.keys_up[msg.dst] = {"path": msg.data["path"], "keys": None}
                elif towards == "down":
                    self.keys_down[msg.dst] = {"path": msg.data["path"], "keys": None}

            # Call the key generation protocol to start key generation
            self.node.set_key_generation(msg.dst, towards=towards)
            role = PrepareAndMeasure.Role.TRANSMITTER if towards == "up" else PrepareAndMeasure.Role.RECEIVER
            self.send_lower(
                PrepareAndMeasure,
                peer=msg.dst,
                role=role,
                key_num=msg.data["key_num"],
                key_length=msg.data["key_length"],
            )

    def receive_lower(self, lower_protocol: type, **kwargs) -> None:
        r"""Receive a message from a lower protocol.

        Args:
          lower_protocol (type): lower protocol that sends the message
          **kwargs: received keyword arguments
        """
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.node import RMPRepeaterNode, RMPEndNode

        peer = kwargs["peer"]
        self.keys[peer] = kwargs["sifted_keys"]
        towards = kwargs["towards"]

        if isinstance(self.node, RMPRepeaterNode):
            if isinstance(peer, RMPRepeaterNode):
                # Store the generated keys in the corresponding key pool
                index = 0 if towards == "up" else 1
                for key in self.keys[peer]:
                    self.node.key_pools[peer][index].keys.append(key)
                    self.node.key_pools[peer][index].current_volume += 1
                # Record the key delivery
                self.node.env.logger.info(
                    f"The sifted keys between {self.node.name} and {peer.name} at "
                    f"{self.node.name}'s key pools: {self.node.key_pools[peer][index].keys}"
                )
                # Reset the status of the corresponding key pool
                self.node.key_pools[peer][index].generating = False
                self.node.key_pools[peer][index].idle = True

                # Check if all the key pools of the node are ready for delivering keys
                ready_to_deliver = True
                for _, key_pools in self.node.key_pools.items():
                    for key_pool in key_pools:
                        if key_pool.generating is True:
                            ready_to_deliver = False
                            break

                if ready_to_deliver is True:
                    # Call QKDRMP to pop a request
                    self.node.env.logger.info(
                        f"Key pools at {self.node.name} were all ready to allocate keys,"
                        f" asked the request queue of RMP protocol to pop a request"
                    )

                    pop_msg = QKDMessage(
                        src=self.node, dst=self.node, protocol=QKDRMP, data={"type": QKDMessage.Type.POP}
                    )
                    self.send_upper(QKDRMP, msg=pop_msg)

            elif isinstance(peer, RMPEndNode):
                # Check if keys at another side is delivered
                if towards == "up":
                    key_end_dict = self.keys_up
                    key_repeater_dict = self.keys_down
                elif towards == "down":
                    key_end_dict = self.keys_down
                    key_repeater_dict = self.keys_up
                # Get the repeater_peer
                if peer == key_end_dict[peer]["path"][0]:
                    repeater_peer = key_end_dict[peer]["path"][2]
                elif peer == key_end_dict[peer]["path"][-1]:
                    repeater_peer = key_end_dict[peer]["path"][-3]

                if isinstance(repeater_peer, RMPRepeaterNode):
                    if key_repeater_dict.get(tuple([repeater_peer, peer])) is not None:
                        cipher_text_list = []
                        # Fetch keys from the corresponding key pool
                        for key_repeater, key_end in zip(
                            self.keys[peer], key_repeater_dict[tuple([repeater_peer, peer])]["keys"]
                        ):
                            cipher_text_list.append(key_repeater ^ key_end)

                        key_repeater_dict.pop(tuple([repeater_peer, peer]))
                        self.keys.pop(peer)
                        self.path = key_end_dict[peer]["path"]
                        key_end_dict.pop(peer)

                        # Generate the 'CIPHERTEXT' message
                        cipher_msg = QKDMessage(
                            src=self.node,
                            dst=self.path[-1],
                            protocol=QKDRoutingL2,
                            data={
                                "type": QKDMessage.Type.CIPHERTEXT,
                                "req_id": self.serving_req_id,
                                "path": self.path,
                                "ciphertext_list": cipher_text_list,
                                "key_num": len(cipher_text_list),
                                "key_length": len(str(cipher_text_list[0])),
                            },
                        )
                        # Record the delivery
                        self.node.reqs_delivered += 1
                        self.node.keys_delivered += cipher_msg.data["key_num"]
                        self.serving_req_id = None
                        # Record the delivery to log file
                        self.node.env.logger.info(
                            f"{self.node.name} fetched the allocated keys,"
                            f" generated the ciphertext with {peer.name} and"
                            f" {repeater_peer.name}, sent it to {cipher_msg.dst.name}"
                        )
                        # Send the ciphertext to the receiver
                        downstream_node_index = self.path.index(self.node) + 1
                        downstream_node = self.path[downstream_node_index]
                        self.node.send_classical_msg(dst=downstream_node, msg=cipher_msg)

                        # Call QKDRMP to pop a request
                        pop_msg = QKDMessage(
                            src=self.node, dst=self.node, protocol=QKDRMP, data={"type": QKDMessage.Type.POP}
                        )
                        self.send_upper(QKDRMP, msg=pop_msg)

                    else:
                        # Save the keys to the corresponding key pool
                        if towards == "up":
                            self.keys_up[peer]["keys"] = self.keys[peer]
                        else:
                            self.keys_down[peer]["keys"] = self.keys[peer]
                        self.node.env.logger.info(
                            f"Key generation at {self.node.name} with {peer.name} was ready,"
                            f" waiting the key pool at another side to get ready"
                        )

                elif isinstance(repeater_peer, RMPEndNode):
                    if key_repeater_dict[repeater_peer]["keys"] is not None:
                        cipher_text_list = []
                        # Fetch keys from the corresponding keypool
                        for key_repeater, key_end in zip(self.keys[peer], key_repeater_dict[repeater_peer]["keys"]):
                            cipher_text_list.append(key_repeater ^ key_end)

                        # Clear the two key dicts
                        key_repeater_dict.pop(repeater_peer)
                        self.keys.pop(peer)
                        self.path = key_end_dict[peer]["path"]
                        key_end_dict.pop(peer)

                        # Generate the 'CIPHERTEXT' message
                        cipher_msg = QKDMessage(
                            src=self.node,
                            dst=self.path[-1],
                            protocol=QKDRoutingL2,
                            data={
                                "type": QKDMessage.Type.CIPHERTEXT,
                                "req_id": self.serving_req_id,
                                "path": self.path,
                                "ciphertext_list": cipher_text_list,
                                "key_num": len(cipher_text_list),
                                "key_length": len(str(cipher_text_list[0])),
                            },
                        )
                        # Record the delivery
                        self.node.reqs_delivered += 1
                        self.node.keys_delivered += cipher_msg.data["key_num"]
                        self.serving_req_id = None
                        # Record the event to log file
                        self.node.env.logger.info(
                            f"{self.node.name} fetched the allocated keys,"
                            f" generated the ciphertext with {peer.name} and"
                            f" {repeater_peer.name}, sent it to {cipher_msg.dst.name}"
                        )
                        # Send the ciphertext to the receiver
                        downstream_node_index = self.path.index(self.node) + 1
                        downstream_node = self.path[downstream_node_index]
                        self.node.send_classical_msg(dst=downstream_node, msg=cipher_msg)

                        # Call QKDRMP to pop a request
                        pop_msg = QKDMessage(
                            src=self.node, dst=self.node, protocol=QKDRMP, data={"type": QKDMessage.Type.POP}
                        )
                        self.send_upper(QKDRMP, msg=pop_msg)

                    else:
                        # Save the keys to corresponding key pool
                        if towards == "up":
                            self.keys_up[peer]["keys"] = self.keys[peer]
                        else:
                            self.keys_down[peer]["keys"] = self.keys[peer]
                        # Record the waiting to log file
                        self.node.env.logger.info(
                            f"Key generation at {self.node.name} with {peer.name} was ready, waiting"
                            f" for the key pool at another side to get ready"
                        )

        elif isinstance(self.node, RMPEndNode):
            # If the two users are directly connected, end the QKD request
            finish = True if len(self.path) == 2 else False

            self.node.env.logger.info(f"{self.node.name} finished the key generation with {peer.name}")

            # Generate a 'READY' message and send it to the QKDRMP
            ready_msg = QKDMessage(
                src=None,
                dst=None,
                protocol=QKDRMP,
                data={"type": QKDMessage.Type.READY, "sifted_keys": self.keys[peer], "finish": finish},
            )
            self.send_upper(QKDRMP, msg=ready_msg, peer=peer, towards=towards)

    def receive_classical_msg(self, msg: "QKDMessage", **kwargs) -> None:
        r"""Receive QKDMessage from other nodes.

        Args:
            msg (QKDMessage): received QKDMessage
            **kwargs : keyword arguments for receiving the classical message
        """
        from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.node import RMPEndNode, RMPRepeaterNode

        if msg.data["type"] == QKDMessage.Type.REQUEST:
            self.node.env.logger.info(
                f"{self.node.name} received 'REQUEST' of QKD " f"with {msg.dst.name} from {msg.src.name}"
            )

            # Generate a 'PATH' message and send it to QKDRMP
            request_msg = QKDMessage(
                src=msg.src,
                dst=msg.dst,
                protocol=QKDRMP,
                data={
                    "type": QKDMessage.Type.PATH,
                    "req_id": msg.data["req_id"],
                    "path": msg.data["path"],
                    "key_num": msg.data["key_num"],
                    "key_length": msg.data["key_length"],
                },
            )
            self.send_upper(QKDRMP, msg=request_msg)

        elif msg.data["type"] == QKDMessage.Type.ACCEPT:
            self.node.env.logger.info(
                f"{self.node.name} received 'ACCEPT' of QKD " f"with {msg.dst.name} from {msg.src.name}"
            )

            if isinstance(self.node, RMPEndNode):
                # Sender: receive 'ACCEPT' message from the receiver
                self.path = msg.data["path"]
                downstream_node = self.path[1]
                key_num = msg.data["key_num"]
                key_length = msg.data["key_length"]

                # Create an instance for key generation with the upstream node
                self.node.set_key_generation(downstream_node)
                self.send_lower(
                    PrepareAndMeasure,
                    peer=downstream_node,
                    role=PrepareAndMeasure.Role.TRANSMITTER,
                    key_num=key_num,
                    key_length=key_length,
                )

            # Generate a 'RESERVE' message and send it to QKDRMP
            reserve_msg = QKDMessage(
                src=msg.src,
                dst=msg.dst,
                protocol=QKDRMP,
                data={
                    "type": QKDMessage.Type.RESERVE,
                    "req_id": msg.data["req_id"],
                    "path": msg.data["path"],
                    "key_num": msg.data["key_num"],
                    "key_length": msg.data["key_length"],
                },
            )
            self.send_upper(QKDRMP, msg=reserve_msg)

        elif msg.data["type"] == QKDMessage.Type.CONFIRM:
            if self.node == msg.dst:
                self.send_upper(QKDRMP, msg=msg)

            else:
                # Forward the 'CONFIRM' to the next hop
                self.path = msg.data["path"]
                index = self.path.index(self.node)
                self.downstream_node = msg.data["path"][index + 1]
                self.node.send_classical_msg(dst=self.downstream_node, msg=msg)

        elif msg.data["type"] == QKDMessage.Type.REJECT:
            self.node.env.logger.info(f"{self.node.name} received reject from {msg.src.name} to {msg.dst.name}")

            err_type = msg.data["err_type"]
            if err_type == 3:
                # Generate a 'REJECT' message and send to QKDRMP
                discard_msg = QKDMessage(
                    src=msg.src,
                    dst=msg.dst,
                    protocol=QKDRMP,
                    data={
                        "type": QKDMessage.Type.REJECT,
                        "req_id": msg.data["req_id"],
                        "path": msg.data["path"],
                        "err_type": msg.data["err_type"],
                        "key_num": msg.data["key_num"],
                        "key_length": msg.data["key_length"],
                    },
                )
                self.send_upper(QKDRMP, msg=discard_msg)

                # For repeater nodes, forward the message to the downstream node
                if self.node != msg.dst:
                    self.path = msg.data["path"]
                    index = self.path.index(self.node)
                    self.downstream_node = msg.data["path"][index + 1]
                    self.node.send_classical_msg(dst=self.downstream_node, msg=msg)

            else:
                if self.node == msg.dst:
                    # Generate a 'REJECT' message and send to QKDRMP
                    discard_msg = QKDMessage(
                        src=msg.src,
                        dst=msg.dst,
                        protocol=QKDRMP,
                        data={
                            "type": QKDMessage.Type.REJECT,
                            "req_id": msg.data["req_id"],
                            "path": msg.data["path"],
                            "err_type": msg.data["err_type"],
                            "key_num": msg.data["key_num"],
                            "key_length": msg.data["key_length"],
                        },
                    )
                    self.send_upper(QKDRMP, msg=discard_msg)

                else:
                    # Forward the message to downstream node
                    self.path = msg.data["path"]
                    index = self.path.index(self.node)
                    upstream_node = msg.data["path"][index - 1]
                    self.node.send_classical_msg(dst=upstream_node, msg=msg)

        elif msg.data["type"] == QKDMessage.Type.CIPHERTEXT:
            if msg.dst == self.node:
                self.node.env.logger.info(f"{self.node.name} received ciphertexts from {msg.src.name}")

                # Generate a 'CIPHERTEXT' message and send it to QKDRMP
                cipher_msg = QKDMessage(
                    src=msg.src,
                    dst=msg.dst,
                    protocol=QKDRMP,
                    data={
                        "type": QKDMessage.Type.CIPHERTEXT,
                        "req_id": msg.data["req_id"],
                        "path": msg.data["path"],
                        "ciphertext_list": msg.data["ciphertext_list"],
                        "key_num": msg.data["key_num"],
                        "key_length": msg.data["key_length"],
                    },
                )
                self.send_upper(QKDRMP, msg=cipher_msg)

            else:
                # Forward the message to the next hop
                if isinstance(self.node, RMPRepeaterNode):
                    self.node.env.logger.info(
                        f"{self.node.name} received ciphertext" f" from {msg.src.name} to {msg.dst.name}"
                    )

                    path = msg.data["path"]
                    index = path.index(self.node)
                    next_hop = path[index + 1]
                    self.node.send_classical_msg(dst=next_hop, msg=msg)

        elif msg.data["type"] == QKDMessage.Type.ACKNOWLEDGE:
            if msg.dst == self.node:
                # Send 'DONE' to QKDRMP
                ack_msg = QKDMessage(
                    src=msg.src,
                    dst=msg.dst,
                    protocol=QKDRMP,
                    data={
                        "type": QKDMessage.Type.ACKNOWLEDGE,
                        "req_id": msg.data["req_id"],
                        "path": msg.data["path"],
                        "key_num": msg.data["key_num"],
                        "key_length": msg.data["key_length"],
                    },
                )
                self.send_upper(QKDRMP, msg=ack_msg)

            else:
                # Forward the message to the upstream node
                index = msg.data["path"].index(self.node)
                next_hop = msg.data["path"][index - 1]
                self.node.send_classical_msg(next_hop, msg=msg)

        elif msg.data["type"] == QKDMessage.Type.GENERATE:
            towards = kwargs["towards"]
            role = PrepareAndMeasure.Role.TRANSMITTER if towards == "up" else PrepareAndMeasure.Role.RECEIVER

            # Get ready for key generation
            self.node.set_key_generation(msg.dst, towards=towards)
            self.send_lower(
                PrepareAndMeasure,
                peer=msg.dst,
                role=role,
                key_num=msg.data["key_num"],
                key_length=msg.data["key_length"],
            )

            if isinstance(msg.dst, RMPEndNode):
                if towards == "up":
                    self.keys_up[msg.dst] = {"path": msg.data["path"], "keys": None}
                elif towards == "down":
                    self.keys_down[msg.dst] = {"path": msg.data["path"], "keys": None}

        elif msg.data["type"] == QKDMessage.Type.DONE:
            if msg.dst == self.node:
                # Send a 'DONE' message to QKDRMP protocol
                done_msg = QKDMessage(
                    src=msg.src,
                    dst=msg.dst,
                    protocol=QKDRMP,
                    data={
                        "type": QKDMessage.Type.DONE,
                        "req_id": msg.data["req_id"],
                        "path": msg.data["path"],
                        "key_num": msg.data["key_num"],
                        "key_length": msg.data["key_length"],
                    },
                )
                self.send_upper(QKDRMP, msg=done_msg)

            else:
                # Forward the 'DONE' to the next hop
                if self.node in msg.data["path"]:
                    index = msg.data["path"].index(self.node)
                    next_hop = msg.data["path"][index - 1]
                    self.node.send_classical_msg(dst=next_hop, msg=msg)
                else:
                    self.node.env.logger.info(
                        f"wrong node {self.node.name} receive a DONE from {msg.src.name} to {msg.dst.name}"
                    )

    def send_lower(self, lower_protocol: type, **kwargs) -> None:
        r"""Send message to lower protocol.

        Args:
            lower_protocol (type): type of the lower protocol to receive the message
            **kwargs: keyword arguments of the message
        """
        assert not self.is_bottom, f"'{self.name}' has no lower protocols."

        for proto in self.lower_protocols:
            if isinstance(proto, lower_protocol) and proto.role is None:
                proto.receive_upper(type(self), **kwargs)
                break
