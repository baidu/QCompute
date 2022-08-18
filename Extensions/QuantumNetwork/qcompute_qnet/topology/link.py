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
Module for communication links in a network.
"""

from qcompute_qnet.core.des import Entity
from qcompute_qnet.devices.channel import ClassicalChannel, QuantumChannel, DuplexClassicalFiberChannel
from qcompute_qnet.devices.channel import DuplexQuantumFiberChannel, ClassicalFiberChannel, QuantumFiberChannel

__all__ = [
    "Link"
]


class Link(Entity):
    r"""Class for creating a link.

    Attributes:
        ends (Tuple[Node]): ends connected by the link

    Note:
        ``Link`` is an abstract concept indicating the connectivity between two nodes.
        The information transmission is realized by its channel components.
        So we should install classical and quantum channels to a link before running the simulation.

    Examples:
        We can create a link between two nodes (e.g. alice and bob) by two approaches.
        The first approach is to instantiate a ``Link`` with given ``name``.
        Then using the ``connect`` method to make a connection to alice and bob.

        >>> l_ab = Link(name="Link_Alice_Bob")
        >>> l_ab.connect(alice, bob)

        The second approach is to assign the two ends when instantiating.
        This will make your code more compact.

        >>> l_ab = Link(name="Link_Alice_Bob", ends=(alice, bob))

        Once we create a link, we can install some specific channels to it. Taking BB84 protocol as an example,
        we need a classical channel from alice to bob and a classical channel from bob to alice.
        We will also need a quantum channel from alice to bob. This can be done as follows:

        >>> l_ab = Link(name="Link_Alice_Bob", ends=(alice, bob))
        >>> c_a2b = ClassicalFiberChannel(name="c_a2b", distance=1e3, sender=alice, receiver=bob)
        >>> c_b2a = ClassicalFiberChannel(name="c_b2a", distance=1e3, sender=bob, receiver=alice)
        >>> q_a2b = QuantumFiberChannel(name="q_a2b", distance=1e3, sender=alice, receiver=bob)
        >>> l_ab.install([c_a2b, c_b2a, q_a2b])
    """

    def __init__(self, name: str, env=None, ends=None, distance=None):
        r"""Constructor for Link class.

        Args:
            name (str): name of the link
            env (DESEnv, optional): discrete-event simulation environment
            ends (Tuple[Node], optional): ends connected by the link
            distance (int, optional): length of the link in meters
        """

        super().__init__(name, env)
        self.ends = (None, None)
        if distance is not None:
            assert ends is not None, f"Should set the ends of the link once specify its distance."
            cchannel1_2 = ClassicalChannel(
                f"cchannel_{ends[0]}2{ends[1]}", distance=distance, sender=ends[0], receiver=ends[1]
            )
            cchannel2_1 = ClassicalChannel(
                f"cchannel_{ends[1]}2{ends[0]}", distance=distance, sender=ends[1], receiver=ends[0]
            )
            qchannel1_2 = QuantumChannel(
                f"qchannel_{ends[0]}2{ends[1]}", distance=distance, sender=ends[0], receiver=ends[1]
            )
            qchannel2_1 = QuantumChannel(
                f"qchannel_{ends[1]}2{ends[0]}", distance=distance, sender=ends[1], receiver=ends[0]
            )
            self.install([cchannel1_2, cchannel2_1, qchannel1_2, qchannel2_1])
        if ends is not None:
            self.connect(ends[0], ends[1])

    def init(self) -> None:
        r"""Link initialization.

        This method will do a sanity check to confirm that the link is installed to a ``Network``.
        """
        assert self.owner != self, f"The link {self.name} should be installed to a 'Network' first!"

    def cchannel(self, dst: "Node") -> "ClassicalChannel":
        r"""Find the classical channel with a given destination.

        Args:
            dst (Node): destination node of the classical channel

        Returns:
            ClassicalChannel: the classical channel with the given destination
        """
        for component in self.components:
            if isinstance(component, ClassicalChannel) and component.receiver == dst:
                return component
            elif isinstance(component, DuplexClassicalFiberChannel) and component.components[0].receiver == dst:
                return component.components[0]
            elif isinstance(component, DuplexClassicalFiberChannel) and component.components[1].receiver == dst:
                return component.components[1]
        raise Exception("No such a classical channel in the link!")

    def qchannel(self, dst: "Node") -> "QuantumChannel":
        r"""Find the quantum channel with a given destination.

        Args:
            dst (Node): destination node of the quantum channel

        Returns:
            QuantumChannel: the quantum channel with the given destination
        """
        for component in self.components:
            if isinstance(component, QuantumChannel) and component.receiver == dst:
                return component
            elif isinstance(component, DuplexQuantumFiberChannel) and component.components[0].receiver == dst:
                return component.components[0]
            elif isinstance(component, DuplexQuantumFiberChannel) and component.components[1].receiver == dst:
                return component.components[1]
        raise Exception("No such a quantum channel in the link!")

    def connect(self, node1: "Node", node2: "Node") -> None:
        r"""Connect the link to two given nodes.

        Args:
            node1 (Node): one end of the link
            node2 (Node): the other end of the link
        """
        assert self.ends[0] is None and self.ends[1] is None, \
            f"Link has been connected to '{self.ends[0].name}' and '{self.ends[1].name}'."

        self.ends = (node1, node2)
        node1.links[node2] = self
        node2.links[node1] = self
