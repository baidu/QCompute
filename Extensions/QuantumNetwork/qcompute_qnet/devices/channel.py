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
Module for communication channels.
"""

from abc import abstractmethod, ABC
from numpy import random
from qcompute_qnet.core.des import Entity, EventHandler

__all__ = [
    "Channel",
    "ClassicalChannel",
    "QuantumChannel",
    "ClassicalFiberChannel",
    "QuantumFiberChannel",
    "DuplexChannel",
    "DuplexClassicalFiberChannel",
    "DuplexQuantumFiberChannel",
    "ClassicalFreeSpaceChannel",
    "QuantumFreeSpaceChannel"
]


LIGHT_SPEED = 3e-4  # meters per picoseconds


class Channel(Entity, ABC):
    r"""Class for the simulation of communication channels.

    Attributes:
        sender (Node): sender of the channel
        receiver (Node): receiver of the channel
        delay (int): delay of message transmission in picoseconds
        loss (float): loss of the channel in decibels
        lossy_prob (float): lossy probability of the channel
    """

    def __init__(self, name: str, distance: float, loss=0, env=None, sender=None, receiver=None, delay=None):
        r"""Constructor for Channel class.

        Args:
            name (str): name of the channel
            distance (float): length of the channel in meters
            loss (float): loss of the channel in decibels
            env (DESEnv): discrete-event simulation environment
            sender (Node): sender of the channel
            receiver (Node): receiver of the channel
            delay (int): delay of message transmission in picoseconds
        """
        super().__init__(name, env)
        self.__distance = distance
        self.delay = round(distance / LIGHT_SPEED) if delay is None else delay
        self.loss = loss
        self.lossy_prob = 0

        self.sender, self.receiver = None, None
        if sender is not None and receiver is not None:
            self.connect(sender, receiver)

    def init(self) -> None:
        r"""Channel initialization.

        This method will do a sanity check to confirm that the channel is installed to a ``Link``.
        """
        assert self.owner != self, f"The channel {self.name} should be installed to a 'Link' first!"

    def set_distance(self, distance: float) -> None:
        r"""Set a distance for the channel.

        Args:
            distance (float): distance to set
        """
        self.__distance = distance

    def get_distance(self) -> float:
        r"""Get the distance of the channel.

        Returns:
            float: distance of the channel
        """
        return self.__distance

    def set_loss(self, loss: float) -> None:
        r"""Set a loss for the channel.

        Args:
            loss (float): loss to set
        """
        self.loss = loss

    def connect(self, sender: "Node", receiver: "Node") -> None:
        r"""Connect a sender node and a receiver node.

        Args:
            sender (Node): sender node
            receiver (Node): receiver node
        """
        assert self.sender is None and self.receiver is None, \
            f"Channel has been connected to '{self.sender.name}' and '{self.receiver.name}'."
        self.sender, self.receiver = sender, receiver

    @abstractmethod
    def transmit(self, msg: "Message", priority=None) -> None:
        r"""Abstract method for message transmission of the channel.

        This method should be implemented by a specific channel.

        Args:
            msg (Message): message to transmit
            priority (int): priority of the transmission
        """
        pass


class ClassicalChannel(Channel):
    r"""Class for the simulation of classical channels.

    Note:
        ``ClassicalChannel`` is a logical classical channel in which no transmission loss will take place.
         Thus, the lossy probability of a classical channel is set to be zero.

    Examples:
        We can create a classical channel between two nodes (e.g. alice and bob) by two approaches.
        The first approach is to instantiate a ``ClassicalChannel`` with given ``name`` and ``distance``.
        Then using the ``connect`` method to make a connection to alice and bob.

        >>> c_a2b = ClassicalChannel(name="c_a2b", distance=1e3)
        >>> c_a2b.connect(alice, bob)

        The second approach is to assign the sender and receiver when instantiating.
        This will make your code more compact.

        >>> c_a2b = ClassicalChannel(name="c_a2b", distance=1e3, sender=alice, receiver=bob)
    """

    def __init__(self, name: str, distance: float, loss=0, env=None, sender=None, receiver=None, delay=None):
        r"""Constructor for ClassicalChannel class.

        Args:
            name (str): name of the classical channel
            distance (float): length of the classical channel in meters
            loss (float): loss of the classical channel in decibels
            env (DESEnv): discrete-event simulation environment
            sender (Node): sender of the classical channel
            receiver (Node): receiver of the classical channel
            delay (int): delay of message transmission in picoseconds
        """
        super().__init__(name, distance, loss, env, sender, receiver, delay)

    def transmit(self, msg: "ClassicalMessage", priority=None) -> None:
        r"""Transmit classical message from the sender to the receiver.

        Args:
            msg (ClassicalMessage): classical message to transmit
            priority (int): priority of the transmission event
        """
        handler = EventHandler(self.receiver, "receive_classical_msg", [self.sender, msg])
        self.scheduler.schedule_after(self.delay, handler, priority)


class QuantumChannel(Channel):
    r"""Class for the simulation of quantum channels.

    Note:
        ``QuantumChannel`` is a logical quantum channel in which no transmission loss will take place.
         Thus, the lossy probability of a quantum channel is set to be zero.

    Examples:
        We can create a quantum channel between two nodes (e.g. alice and bob) by two approaches.
        The first approach is to instantiate a ``QuantumChannel`` with given ``name`` and ``distance``.
        Then using the ``connect`` method to make a connection to alice and bob.

        >>> q_a2b = QuantumChannel(name="q_a2b", distance=1e3)
        >>> q_a2b.connect(alice, bob)

        The second approach is to assign the sender and receiver when instantiating.
        This will make your code more compact.

        >>> q_a2b = QuantumChannel(name="q_a2b", distance=1e3, sender=alice, receiver=bob)
    """

    def __init__(self, name: str, distance: float, loss=0, env=None, sender=None, receiver=None, delay=None):
        r"""Constructor for QuantumChannel class.

        Args:
            name (str): name of the quantum channel
            distance (float): length of the quantum channel in meters
            loss (float): loss of the quantum channel in decibels
            env (DESEnv): discrete-event simulation environment
            sender (Node): sender of the quantum channel
            receiver (Node): receiver of the quantum channel
            delay (int): delay of message transmission in picoseconds
        """
        super().__init__(name, distance, loss, env, sender, receiver, delay)

    def transmit(self, msg: "QuantumMessage", priority=None) -> None:
        r"""Transmit quantum message from the sender to the receiver.

        Args:
            msg (QuantumMessage): quantum message to transmit
            priority (int): priority of the transmission event
        """
        handler = EventHandler(self.receiver, "receive_quantum_msg", [self.sender, msg])
        self.scheduler.schedule_after(self.delay, handler, priority)


class ClassicalFiberChannel(ClassicalChannel):
    r"""Class for the simulation of classical fiber channels.

    Note:
        The lossy probability of a classical fiber channel is set to be zero for simplicity.
    """

    def __init__(self, name: str, distance: float, loss=0, env=None, sender=None, receiver=None, delay=None):
        r"""Constructor for ClassicalFiberChannel class.

        Args:
            name (str): name of the classical fiber channel
            distance (float): length of the classical fiber channel in meters
            loss (float): loss of the classical fiber channel in decibels
            env (DESEnv): discrete-event simulation environment
            sender (Node): sender of the classical fiber channel
            receiver (Node): receiver of the classical fiber channel
            delay (int): delay of message transmission in picoseconds
        """
        super().__init__(name, distance, loss, env, sender, receiver, delay)


class QuantumFiberChannel(QuantumChannel):
    r"""Class for the simulation of quantum fiber channels.

    Note:
        We use the standard commercial single mode fiber (SMF) to model the channel loss.
        The lossy probability is given by :math:`1 - 10^{- loss / 10}`, where loss (dB) is calculated by
        :math:`\alpha * L`, :math:`\alpha` is the attenuation (dB/km, default 0.2) and
        :math:`L` is the channel length (km).
    """

    def __init__(self, name: str, distance: float, loss=None, env=None, sender=None, receiver=None, delay=None):
        r"""Constructor for QuantumFiberChannel class.

        Args:
            name (str): name of the quantum fiber channel
            distance (float): length of the quantum fiber channel in meters
            loss (float): loss of the quantum fiber channel in decibels
            env (DESEnv): discrete-event simulation environment
            sender (Node): sender of the quantum fiber channel
            receiver (Node): receiver of the quantum fiber channel
            delay (int): delay of message transmission in picoseconds
            loss (float): loss of the quantum fiber channel
        """
        super().__init__(name, distance, loss, env, sender, receiver, delay)
        self.loss = 0.2 * distance * 1e-3 if loss is None else loss  # set default attenuation as 0.2
        self.lossy_prob = 1 - 10 ** (- self.loss / 10)

    def transmit(self, msg: "QuantumMessage", priority=None) -> None:
        r"""Transmit quantum message from the sender to the receiver.

        Args:
            msg (QuantumMessage): quantum message to transmit
            priority (int): priority of the transmission event
        """
        # Schedule an event for the receiver if the message is not lost
        if random.random_sample() > self.lossy_prob:
            super().transmit(msg, priority)


class DuplexChannel(Entity, ABC):
    r"""Class for the simulation of duplex channels.

    A duplex channel contains two unidirectional channels. These two unidirectional channels are identical except for
    their directions. One must set the ends of two channels at the same time by 'connect' method.

    Attributes:
        node1 (Node): one end of the duplex channel
        node2 (Node): the other end of the duplex channel
        channel1_2 (Channel): the channel from node1 to node2
        channel2_1 (Channel): the channel from node2 to node1
        delay (int): delay of message transmission in picoseconds
        lossy_prob (float): lossy probability of the duplex channel
    """

    def __init__(self, name: str, distance: float, loss=0, env=None, node1=None, node2=None, delay=None):
        r"""Constructor for DuplexChannel class.

        Args:
            name (str): name of the duplex channel
            distance (float): length of the duplex channel in meters
            loss (float): loss of the duplex channel in decibels
            env (DESEnv): discrete-event simulation environment
            node1 (Node): one end of the duplex channel
            node2 (Node): the other end of the duplex channel
            delay (int): delay of message transmission in picoseconds
        """
        super().__init__(name, env)
        self.node1 = node1
        self.node2 = node2
        self.channel1_2 = None
        self.channel2_1 = None
        self.__distance = distance
        self.loss = loss
        self.delay = delay
        self.lossy_prob = 0

    def init(self) -> None:
        r"""Duplex channel initialization.

        This method will do a sanity check to confirm that the duplex channel is installed to a ``Link``.
        """
        assert self.owner != self, f"The duplex channel {self.name} should be installed to a 'Link' first!"

    def set_distance(self, distance: float) -> None:
        r"""Set a distance for the two unidirectional channels of the duplex channel.

        Args:
            distance (float): distance to set
        """
        self.channel1_2.set_distance(distance)
        self.channel2_1.set_distance(distance)

    def get_distance(self) -> float:
        r"""Get the distance of the duplex channel.

        Returns:
            float: distance of the channel
        """
        return self.__distance

    def set_loss(self, loss: float) -> None:
        r"""Set a loss for the two unidirectional channels of the duplex channel.

        Args:
            loss (float): loss to set
        """
        self.channel1_2.set_loss(loss)
        self.channel2_1.set_loss(loss)

    def connect(self, node1: "Node", node2: "Node") -> None:
        r"""Connect a sender node and a receiver node for both unidirectional channels of the duplex channel.

        Args:
            node1 (Node): one end of the duplex channel
            node2 (Node): the other end of the duplex channel
        """
        self.channel1_2.connect(node1, node2)
        self.channel1_2.name = node1.name + "->" + node2.name
        self.channel2_1.connect(node2, node1)
        self.channel2_1.name = node2.name + "->" + node1.name


class DuplexClassicalFiberChannel(DuplexChannel):
    r"""Class for the simulation of duplex classical fiber channels.

    A duplex classical fiber channel contains two unidirectional classical fiber channels. These two unidirectional
    channels are identical except for their directions. One must set the ends of two channels at the same time by
    'connect' method.
    """

    def __init__(self, name: str, distance: float, loss=0, env=None, node1=None, node2=None, delay=None):
        r"""Constructor for DuplexClassicalFiberChannel class.

        Args:
            name (str): name of the duplex classical fiber channel
            distance (float): length of the duplex classical fiber channel in meters
            loss (float): loss of the duplex classical fiber channel in decibels
            env (DESEnv): discrete-event simulation environment
            node1 (Node): one end of the duplex classical fiber channel
            node2 (Node): the other end of the duplex classical fiber channel
            delay (int): delay of message transmission in picoseconds
        """
        super().__init__(name, distance, loss, env, node1, node2, delay)
        self.channel1_2 = ClassicalFiberChannel("node1->node2", distance, loss, env=env,
                                                sender=node1, receiver=node2, delay=delay)
        self.channel2_1 = ClassicalFiberChannel("node2->node1", distance, loss, env=env,
                                                sender=node2, receiver=node1, delay=delay)
        self.install([self.channel1_2, self.channel2_1])


class DuplexQuantumFiberChannel(DuplexChannel):
    r"""Class for the simulation of duplex quantum fiber channels.

    A duplex quantum fiber channel contains two unidirectional quantum fiber channels. These two unidirectional
    channels are identical except for their directions. One must set the ends of two channels at the same time by
    'connect' method.
    """

    def __init__(self, name: str, distance: float, loss=None, env=None, node1=None, node2=None, delay=None):
        r"""Constructor for DuplexQuantumFiberChannel class.

        Args:
            name (str): name of the duplex quantum fiber channel
            distance (float): length of the duplex quantum fiber channel in meters
            loss (float): loss of the duplex quantum fiber channel in decibels
            env (DESEnv): discrete-event simulation environment
            node1 (Node): one end of the duplex quantum fiber channel
            node2 (Node): the other end of the duplex quantum fiber channel
            delay (int): delay of message transmission in picoseconds
        """
        super().__init__(name, distance, loss, env, node1, node2, delay)
        self.channel1_2 = QuantumFiberChannel("node1->node2", distance, loss, env=env,
                                              sender=node1, receiver=node2, delay=delay)
        self.channel2_1 = QuantumFiberChannel("node2->node1", distance, loss, env=env,
                                              sender=node2, receiver=node1, delay=delay)
        self.install([self.channel1_2, self.channel2_1])
        self.loss = 0.2 * distance * 1e-3 if loss is None else loss  # set default attenuation as 0.2
        self.lossy_prob = 1 - 10 ** (- self.loss / 10)


class ClassicalFreeSpaceChannel(ClassicalChannel):
    r"""Class for the simulation of a free space channel for transmitting classical message.

    Warning:
        1. If there is a mobile node, the value of ``distance`` is calculated according to the track of its mobile node.
        2. If both of the connected nodes are fixed, we should set a value for ``distance``.
    """

    def __init__(self, name: str, distance=0, loss=0, is_mobile=False,
                 env=None, sender=None, receiver=None, delay=None):
        r"""Constructor for ClassicalFreeSpaceChannel class.

        Args:
            name (str): name of the classical free space channel
            distance (float): length of the classical free space channel in meters
            loss (float): loss of the classical free space channel in decibels per kilometer
            is_mobile (bool): whether the classical free space channel is mobile
            env (DESEnv): discrete-event simulation environment
            sender (Node): sender of the classical free space channel
            receiver (Node): receiver of the classical free space channel
            delay (int): delay of message transmission in picoseconds
        """
        assert is_mobile or distance != 0, "Should set a distance if both connected nodes are fixed!"

        self.is_mobile = is_mobile
        self.mobile_node = None
        super().__init__(name, distance, loss, env, sender, receiver, delay)

    def connect(self, sender: "Node", receiver: "Node") -> None:
        r"""Connect a sender node and a receiver node.

        Args:
            sender (Node): sender node
            receiver (Node): receiver node
        """
        super().connect(sender, receiver)
        if self.is_mobile:
            assert not (sender.is_mobile and receiver.is_mobile), \
                f"We do not allow both nodes to be mobile in this version."
            if sender.is_mobile:
                self.mobile_node = sender
            elif receiver.is_mobile:
                self.mobile_node = receiver
            assert self.mobile_node is not None, "The channel is claimed to be mobile but both nodes are fixed."

            self.set_distance(self.mobile_node.mobility.track.time2distance(self.mobile_node.env.now))
            self.delay = round(self.get_distance() / LIGHT_SPEED)

    def update_params(self, distance: float) -> None:
        r"""Update the parameters of the classical free space channel.

        Args:
            distance (float): updated distance of the quantum free space channel
        """
        self.set_distance(distance)
        # Update transmission delay of the classical message
        self.delay = round(self.get_distance() / LIGHT_SPEED)

    def transmit(self, msg: "ClassicalMessage", priority=None) -> None:
        r"""Transmit classical message from the sender to the receiver.

        Args:
            msg (ClassicalMessage): classical message to transmit
            priority (int): priority of the transmission event
        """
        # Acquire the current distance between the two connected nodes
        distance = self.mobile_node.mobility.track.time2distance(self.env.now)
        self.update_params(distance)

        handler = EventHandler(self.receiver, "receive_classical_msg", [self.sender, msg])
        self.scheduler.schedule_after(self.delay, handler, priority)


class QuantumFreeSpaceChannel(QuantumChannel):
    r"""Class for the simulation of a free space channel for transmitting quantum message.

    Warning:
        1. If there is a mobile node, the value of ``distance`` is calculated according to the track of its mobile node.
        2. If both of the connected nodes are fixed, we should set a value for ``distance``.
    """

    def __init__(self, name: str, distance=0, loss=None, is_mobile=False,
                 env=None, sender=None, receiver=None, delay=None):
        r"""Constructor for QuantumFreeSpaceChannel class.

        Args:
            name (str): name of the quantum free space channel
            distance (float): length of the quantum free space channel in meters
            loss (float): loss of the quantum free space channel in decibels per kilometer
            is_mobile (bool): whether the quantum free space channel is mobile
            env (DESEnv): discrete-event simulation environment
            sender (Node): sender of the quantum free space channel
            receiver (Node): receiver of the quantum free space channel
            delay (int): delay of message transmission in picoseconds
            loss (float): loss of the quantum free space channel
        """
        assert is_mobile or distance != 0, "Should set a distance if both connected nodes are fixed!"
        assert is_mobile or loss is not None, "Should set a loss if both connected nodes are fixed!"

        self.is_mobile = is_mobile
        self.mobile_node = None
        super().__init__(name, distance, loss, env, sender, receiver, delay)
        self.lossy_prob = 1 - 10 ** (- self.loss / 10)

    def connect(self, sender: "Node", receiver: "Node") -> None:
        r"""Connect a sender node and a receiver node.

        Args:
            sender (Node): sender node
            receiver (Node): receiver node
        """
        super().connect(sender, receiver)
        if self.is_mobile:
            assert not (sender.is_mobile and receiver.is_mobile), \
                f"We do not allow both nodes to be mobile in this version."
            if sender.is_mobile:
                self.mobile_node = sender
            elif receiver.is_mobile:
                self.mobile_node = receiver
            assert self.mobile_node is not None, "The channel is claimed to be mobile but both nodes are fixed."

            self.set_distance(self.mobile_node.mobility.track.time2distance(self.mobile_node.env.now))
            self.delay = round(self.get_distance() / LIGHT_SPEED)
            self.set_loss(self.mobile_node.mobility.track.distance2loss(self.get_distance()))

    def update_params(self, distance: float, loss: float) -> None:
        r"""Update the parameters of the quantum free space channel.

        Args:
            distance (float): updated distance of the quantum free space channel
            loss (float): updated loss of the quantum free space channel
        """
        self.set_distance(distance)
        # Update transmission delay of the quantum message
        self.delay = round(self.get_distance() / LIGHT_SPEED)
        # Update loss and lossy probability of the quantum message
        self.set_loss(loss)
        self.lossy_prob = 1 - 10 ** (- self.loss / 10)

    def transmit(self, msg: "QuantumMessage", priority=None) -> None:
        r"""Transmit quantum message from the sender to the receiver.

        Args:
            msg (QuantumMessage): quantum message to transmit
            priority (int): priority of the transmission event
        """
        # Acquire the current distance and loss of the channel
        distance = self.mobile_node.mobility.track.time2distance(self.env.now)
        loss = self.mobile_node.mobility.track.distance2loss(distance)
        self.update_params(distance, loss)

        # Schedule an event for the receiver if the message is not lost
        if random.random_sample() > self.lossy_prob:
            handler = EventHandler(self.receiver, "receive_quantum_msg", [self.sender, msg])
            self.scheduler.schedule_after(self.delay, handler, priority)
