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
Module for key generation protocols.
"""

from abc import ABC, abstractmethod
from enum import Enum, unique
from typing import List, Dict, Tuple
import numpy as np
from qcompute_qnet.core.des import EventHandler
from qcompute_qnet.protocols.protocol import Protocol
from qcompute_qnet.messages.message import ClassicalMessage
from qcompute_qnet.quantum.basis import Basis

__all__ = [
    "KeyGeneration",
    "PrepareAndMeasure",
    "BB84",
    "DecoyBB84"
]


class KeyGeneration(Protocol, ABC):
    r"""Class for key generation protocols.

    Attributes:
        peer (QKDNode): peer node to implement the same task
        start_time (int): start time of the key generation protocol
        key_num (int): number of the sifted keys
        key_length (int): length of the sifted keys
        status (Status): the status of the key generation protocol
        sifted_keys (list): list of the sifted keys
        key_rate (float): rate of key generation
        error_rate (float): error rate of key generation
    """

    def __init__(self, name: str, peer=None):
        r"""Constructor for KeyGeneration class.

        Args:
            name (str): name of the key generation protocol
            peer (QKDNode): peer node to implement the same task
        """
        super().__init__(name)
        self.peer = peer
        self.start_time = 0
        self.key_num = float("inf")
        self.key_length = 256
        self.status = KeyGeneration.Status.READY
        self.sifted_keys = []
        self.key_rate = 0
        self.error_rate = 0

    @unique
    class Status(Enum):
        r"""Class for the status of the key generation protocol.
        """

        READY = "Ready"
        WORKING = "Working"
        DONE = "Done"

    def receive_upper(self, upper_protocol: type, **kwargs) -> None:
        r"""Receive a message from an upper protocol.

        Args:
            upper_protocol (type): upper protocol that sends the message
            **kwargs: received keyword arguments
        """
        if kwargs['peer'] == self.peer:
            self.start(**kwargs)

    def receive_quantum_msg(self, msg: "QuantumMessage", **kwargs) -> None:
        r"""Receive a quantum message from the node.

        Args:
            msg (Message): received quantum message
            **kwargs (Any): keyword arguments for receiving the quantum message
        """
        self.node.polar_detector.receive(msg)

    def finish(self) -> None:
        r"""Finish the key generation protocol.
        """
        from qcompute_qnet.models.qkd.routing import QKDRouting
        self.node.env.logger.info(f"{self.node.name} finished key generation with {self.peer.name}")
        self.node.env.logger.info(f"{self.node.name}'s sifted keys (with {self.peer.name} in decimal): "
                                  f"{self.sifted_keys}")

        self.status = KeyGeneration.Status.DONE

        if self.is_top:
            self.node.keys[self.peer] = self.sifted_keys
        else:
            # TODO: only consider QKDRouting as an upper protocol in this version
            handler = EventHandler(self, "send_upper", [QKDRouting], peer=self.peer, sifted_keys=self.sifted_keys)
            self.node.scheduler.schedule_now(handler)


class PrepareAndMeasure(KeyGeneration, ABC):
    r"""Class for the prepare-and-measure type of quantum key distribution protocols.

    Attributes:
        role (Role): role of the prepare-and-measure protocol
        round (int): round of photon emission
        pulse_num (int): number of photon pulses
        light_duration (int): duration of photon emissions in picoseconds
        emission_rest (int): gap between photon emissions
        tx_bases_ratio (list): ratio of the preparation bases
        tx_bases_list (list): preparation bases for the transmitter
        tx_bits_list (list): chosen bits for the transmitter
        tx_key_bits (list): key bits for the transmitter
        rx_bases_ratio (list): ratio of the measurement bases
        rx_bases_list (list): measurement bases for the receiver
        rx_bits_list (list): chosen bits for the receiver
        rx_key_bits (list): key bits for the receiver
    """

    def __init__(self, name: str, peer=None):
        r"""Constructor for PrepareAndMeasure class.

        Args:
            name (str): name of the prepare-and-measure protocol
            peer (QKDNode): peer node to implement the same task
        """
        super().__init__(name, peer)
        self.role = None
        self.round = 0
        self.pulse_num = 0
        self.light_duration = 0
        self.emission_rest = 0

        self.tx_bases_ratio = None
        self.tx_bases_list = None
        self.tx_bits_list = None
        self.tx_key_bits = None

        self.rx_bases_ratio = None
        self.rx_bases_list = None
        self.rx_bits_list = None
        self.rx_key_bits = None

    @unique
    class Role(Enum):
        r"""Class for the role of the prepare-and-measure protocol.
        """

        TRANSMITTER = "Transmitter"
        RECEIVER = "Receiver"

    def set(self, **kwargs) -> None:
        r"""Set given parameters.

        Args:
            **kwargs: keyword arguments to set
        """
        for attr in kwargs:
            if attr == "tx_bases_ratio":
                assert isinstance(kwargs[attr], list), "'tx_bases_ratio' should be a list."
                self.__setattr__(attr, kwargs[attr])
            elif attr == "rx_bases_ratio":
                assert isinstance(kwargs[attr], list), "'rx_bases_ratio' should be a list."
                self.__setattr__(attr, kwargs[attr])
            else:
                raise TypeError(f"Setting {attr} is not allowed in {self.name}")

    def receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        r"""Node receives a classical message.

        Args:
            msg (ClassicalMessage): classical message used for the prepare-and-measure protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        # Check if the message is for self and the protocol is working
        if msg.src == self.peer and self.status == KeyGeneration.Status.WORKING:
            # Different roles lead to different operations
            if self.role == PrepareAndMeasure.Role.TRANSMITTER:
                self.transmitter_receive_classical_msg(msg)
            elif self.role == PrepareAndMeasure.Role.RECEIVER:
                self.receiver_receive_classical_msg(msg)

    @abstractmethod
    def transmitter_receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        r"""Transmitter receives a classical message from the receiver.

        Args:
            msg (ClassicalMessage): classical message for the prepare-and-measure protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        pass

    @abstractmethod
    def receiver_receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        r"""Receiver receives a classical message from the transmitter.

        Args:
            msg (ClassicalMessage): classical message for the prepare-and-measure protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        pass

    def receiver_get_bits(self) -> List:
        r"""Get bits from the detectors' time records.

        Warnings:
            We regard the detection events as invalid if both detectors click.
            This may not consist with some literatures where a random choice is made if both detectors click.
            However, this makes no much difference to the key rate's estimation.
            Users can override this method if necessary.

        Returns:
            list: receiver's bits list
        """
        bits = [None] * self.pulse_num

        records_zero = self.node.polar_detector.detectors[0].pop_records()
        records_one = self.node.polar_detector.detectors[1].pop_records()

        for record in records_zero:
            index = record[0]
            bits[index] = 0

        for record in records_one:
            index = record[0]
            bits[index] = None if bits[index] == 0 else 1

        return bits

    def key_rate_estimation(self) -> float:
        r"""Estimate sifted key rate.

        Returns:
            float: estimated sifted key rate
        """
        self.key_rate = ((len(self.sifted_keys) * self.key_length + len(self.tx_key_bits)) / 1000) / \
                        ((self.node.env.now - self.start_time) * 1e-12)
        self.node.env.logger.info(f"Sifted key rate: {self.key_rate:.4f} kbit/s")
        return self.key_rate

    def error_rate_estimation(self) -> float:
        r"""Estimate error rate of the sifted keys.

        Returns:
            float: estimated error rate of the sifted keys
        """
        # Find the peer protocol
        for proto in self.peer.protocol_stack.protocols:
            if isinstance(proto, KeyGeneration) and proto.peer == self.node:
                peer_proto = proto

        # Calculate error rate of all sifted keys
        num_errors = 0
        for i in range(self.key_num):
            key_xor = self.sifted_keys[i] ^ peer_proto.sifted_keys[i]  # in decimal
            num_errors += bin(key_xor).count('1')  # count the number of errors
        self.error_rate = num_errors / (self.key_num * self.key_length)
        self.node.env.logger.info(f"Key error rate: {self.error_rate:.4f}")
        return self.error_rate

    def statistics_estimation(self) -> Tuple[float, float]:
        r"""Estimate key rate and error rate of the sifted keys.

        Returns:
            Tuple[float, float]: estimated key rate and error rate
        """
        key_rate = self.key_rate_estimation()
        error_rate = self.error_rate_estimation()
        return key_rate, error_rate


class BB84(PrepareAndMeasure):
    r"""Class for the BB84 key generation protocol.

    Attributes:
        basis_encoding (dict): map an int to a measurement basis
    """

    def __init__(self, name: str, peer=None):
        r"""Constructor for BB84 class.

        Args:
            name (str): name of the BB84 protocol
            peer (QKDNode): peer node to implement the same task
        """
        super().__init__(name, peer)
        self.basis_encoding = {0: Basis.Z(), 1: Basis.X()}

    class Message(ClassicalMessage):
        r"""Class for the classical control messages in BB84 protocol.
        """

        def __init__(self, src: "Node", dst: "Node", protocol: type, data: Dict):
            r"""Constructor for Message class.

            Args:
                src (Node): source of Message
                dst (Node): destination of Message
                protocol (type): protocol of Message
                data (Dict): message content
            """
            super().__init__(src, dst, protocol, data)

        @unique
        class Type(Enum):
            r"""Class for Message types.
            """

            START = "Start"
            BASES_REQUEST = "Bases request"
            BASES = "Bases"
            MATCHING = "Matching"

    def start(self, **kwargs) -> None:
        r"""Start the BB84 protocol.

        Args:
            **kwargs: keyword arguments to start the protocol
        """
        assert self.node is not None and self.peer is not None, f"Should load the protocol stack to a node first!"

        # Initial settings
        self.role = kwargs['role']
        if "key_num" in kwargs.keys():
            self.key_num = kwargs['key_num']
        if "key_length" in kwargs.keys():
            self.key_length = kwargs['key_length']
        self.status = KeyGeneration.Status.WORKING
        self.sifted_keys = []
        self.round = 0

        if self.role == PrepareAndMeasure.Role.TRANSMITTER:
            self.start_time = self.node.env.now  # record the start time of BB84 protocol
            if self.tx_bases_ratio is None:
                self.tx_bases_ratio = [0.5, 0.5]
            self.tx_bases_list = []
            self.tx_bits_list = []
            self.tx_key_bits = []
            self.transmitter_start_protocol()

        elif self.role == PrepareAndMeasure.Role.RECEIVER:
            if self.rx_bases_ratio is None:
                self.rx_bases_ratio = [0.5, 0.5]
            self.rx_bases_list = []
            self.rx_bits_list = []
            self.rx_key_bits = []

    def transmitter_start_protocol(self) -> None:
        r"""Transmitter starts the BB84 protocol.

        Transmitter sends a classical message to the receiver for device synchronization
        and prepares to emit photon pulses.
        """
        self.node.env.logger.info(f"{self.node.name} started BB84 protocol with {self.peer.name}")

        self.pulse_num = round(self.key_length / self.node.photon_source.mean_photon_num)
        self.light_duration = round(self.pulse_num / self.node.photon_source.frequency * 1e12)
        self.emission_rest = round(1e12 / self.node.photon_source.frequency)

        transmitter_emit_time = self.node.env.now + self.node.cchannel(self.peer).delay
        clock = transmitter_emit_time + self.node.qchannel(self.peer).delay  # clock time for photon reception

        start_msg = BB84.Message(src=self.node,
                                 dst=self.peer,
                                 protocol=BB84,
                                 data={'type': BB84.Message.Type.START,
                                       'clock': clock,
                                       'source_frequency': self.node.photon_source.frequency,
                                       'light_duration': self.light_duration,
                                       'emission_rest': self.emission_rest})
        self.node.send_classical_msg(dst=self.peer, msg=start_msg)

        # Transmitter begins photon emission as soon as the receiver receives the message
        handler = EventHandler(self, "transmitter_start_emission")
        self.scheduler.schedule_after(self.node.cchannel(self.peer).delay, handler)

    def transmitter_start_emission(self) -> None:
        r"""Transmitter starts to emit photon pulses.

        Transmitter randomly generates bits and encoding bases for photon emission.
        """
        if self.status == KeyGeneration.Status.WORKING:
            self.round += 1
            self.node.env.logger.debug(f"{self.node.name} began photon emission for BB84 with {self.peer.name}"
                                       f" (Round {self.round})")

            # Generate transmitter's random bases and bits
            bases = list(np.random.choice([0, 1], self.pulse_num, p=self.tx_bases_ratio))
            bits = list(np.random.choice([0, 1], self.pulse_num))  # uniform distribution
            self.tx_bases_list.append(bases)
            self.tx_bits_list.append(bits)

            bases = [self.basis_encoding[basis] for basis in bases]
            states = []
            for i, basis in enumerate(bases):
                # Prepare states |0>, |1>, |+>, |-> (in the form of density matrices) for photon emission
                states.append(basis[bits[i]] @ np.conjugate(basis[bits[i]]).T)

            handler = EventHandler(self.node.photon_source, "emit", [self.peer, states])
            self.scheduler.schedule_now(handler)

            # Schedule next round of photon emission
            handler = EventHandler(self, "transmitter_start_emission")
            # Rest for a while to avoid scheduling the event during cool-down period of the photon source
            self.scheduler.schedule_after(self.light_duration + self.emission_rest, handler)

    def receiver_stop_reception(self) -> None:
        r"""Receiver stops photon reception.

        Receiver pops the measurement outcomes and resets the measurement bases to
        prepare for receiving photon pulses in the next round.
        """
        if self.status == KeyGeneration.Status.WORKING:
            self.round += 1
            self.node.env.logger.debug(f"{self.node.name} finished receiving photons from {self.peer.name}"
                                       f" (Round {self.round})")

            self.rx_bits_list.append(self.receiver_get_bits())
            # Reset measurement bases for the next round of photon reception
            new_bases = list(np.random.choice([0, 1], self.pulse_num, p=self.rx_bases_ratio))
            self.rx_bases_list.append(new_bases)
            clock = self.node.env.now + self.emission_rest  # set the next clock time

            new_bases = [self.basis_encoding[basis] for basis in new_bases]
            # Reset parameters for polarization detector
            self.node.polar_detector.set_beamsplitter(clock=clock, bases=new_bases)

            # Schedule next round of stopping photon reception
            handler = EventHandler(self, "receiver_stop_reception")
            self.scheduler.schedule_at(clock + self.light_duration, handler)

            # Ask the transmitter to announce the bases for key sifting
            request_bases_msg = BB84.Message(src=self.node,
                                             dst=self.peer,
                                             protocol=BB84,
                                             data={'type': BB84.Message.Type.BASES_REQUEST,
                                                   'round': self.round})
            self.node.send_classical_msg(dst=self.peer, msg=request_bases_msg)

        elif self.status == KeyGeneration.Status.DONE:
            self.node.polar_detector.detectors[0].turn_off()
            self.node.polar_detector.detectors[1].turn_off()

    def transmitter_receive_classical_msg(self, msg: "BB84.Message", **kwargs) -> None:
        r"""Transmitter receives a BB84 message from the receiver.

        Args:
            msg (BB84.Message): message for BB84 protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        photon_round = msg.data['round']

        if msg.data['type'] == BB84.Message.Type.BASES_REQUEST:
            self.node.env.logger.debug(f"{self.node.name} received bases request from"
                                       f" {self.peer.name} (Round {photon_round})")

            # Pop the earliest bases and send to the receiver
            bases = self.tx_bases_list.pop(0)
            bases_msg = BB84.Message(src=self.node,
                                     dst=self.peer,
                                     protocol=BB84,
                                     data={'type': BB84.Message.Type.BASES,
                                           'peer_bases': bases,
                                           'round': photon_round})
            self.node.send_classical_msg(dst=self.peer, msg=bases_msg)

        elif msg.data['type'] == BB84.Message.Type.MATCHING:
            self.node.env.logger.debug(f"{self.node.name} received matching indices from {self.peer.name}"
                                       f" (Round {photon_round})")

            matching_list = msg.data['matching_list']
            bits = self.tx_bits_list.pop(0)  # pop the earliest bits for index comparison
            for i in matching_list:
                self.tx_key_bits.append(bits[i])

            # Transmitter generates a valid key successfully
            if len(self.tx_key_bits) >= self.key_length:
                self.node.env.logger.info(f"{self.node.name} generated a valid key with {self.peer.name}"
                                          f" (Round {photon_round})")
                self.node.env.logger.info(f"{self.node.name}'s sifted key: {self.tx_key_bits[0: self.key_length]}")

                # Store the generated key as a decimal integer
                transmitter_key = int("".join(str(self.tx_key_bits.pop(0)) for _ in range(self.key_length)), base=2)
                self.node.env.logger.info(f"{self.node.name}'s sifted key (Decimal): {transmitter_key}")
                self.sifted_keys.append(transmitter_key)

                # Check if it reaches the terminal condition
                if len(self.sifted_keys) >= self.key_num:
                    self.finish()  # transmitter finishes BB84 protocol
                    self.statistics_estimation()  # estimate sifted key rate and error rate

    def receiver_receive_classical_msg(self, msg: "BB84.Message", **kwargs) -> None:
        r"""Receiver receives a BB84 message from the transmitter.

        Args:
            msg (BB84.Message): message for BB84 protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        if msg.data['type'] == BB84.Message.Type.START:
            self.node.env.logger.debug(f"{self.node.name} received message 'START' from {self.peer.name}")

            clock = msg.data['clock']  # time for photon reception
            frequency = msg.data['source_frequency']  # frequency for beam splitter
            self.light_duration = msg.data['light_duration']
            self.emission_rest = msg.data['emission_rest']

            # Generate receiver's random measurement bases
            self.pulse_num = round(msg.data['light_duration'] * frequency * 1e-12)
            bases = list(np.random.choice([0, 1], self.pulse_num, p=self.rx_bases_ratio))
            self.rx_bases_list.append(bases)

            bases = [self.basis_encoding[basis] for basis in bases]  # map the bits to bases for measurement
            # Set parameters for polarization detector
            self.node.polar_detector.set_beamsplitter(clock=clock, frequency=frequency, bases=bases)
            self.node.polar_detector.detectors[0].turn_on()
            self.node.polar_detector.detectors[1].turn_on()

            # Stop receiving photons after a light duration
            handler = EventHandler(self, "receiver_stop_reception")
            self.scheduler.schedule_at(clock + self.light_duration, handler)

        elif msg.data['type'] == BB84.Message.Type.BASES:
            photon_round = msg.data['round']
            self.node.env.logger.debug(f"{self.node.name} received bases from {self.peer.name} "
                                       f"(Round: {photon_round})")

            peer_bases = msg.data['peer_bases']
            bases = self.rx_bases_list.pop(0)  # pop the earliest bases for basis comparison
            bits = self.rx_bits_list.pop(0)
            matching_list = []
            for i, basis in enumerate(bases):
                # Record valid bits of matching indices
                if basis == peer_bases[i] and bits[i] is not None:
                    matching_list.append(i)
                    self.rx_key_bits.append(bits[i])

            # Tell the transmitter the matching indices
            matching_msg = BB84.Message(src=self.node,
                                        dst=self.peer,
                                        protocol=BB84,
                                        data={'type': BB84.Message.Type.MATCHING,
                                              'matching_list': matching_list,
                                              'round': photon_round})
            self.node.send_classical_msg(dst=self.peer, msg=matching_msg)

            # Check if the length of key bits is enough for a new key
            if len(self.rx_key_bits) >= self.key_length:
                self.node.env.logger.info(f"{self.node.name} generated a valid key with {self.peer.name}"
                                          f" (Round {photon_round})")
                self.node.env.logger.info(f"{self.node.name}'s sifted key: {self.rx_key_bits[0: self.key_length]}")

                # Store the generated key as a decimal integer
                receiver_key = int("".join(str(self.rx_key_bits.pop(0)) for _ in range(self.key_length)), base=2)
                self.node.env.logger.info(f"{self.node.name}'s sifted key (Decimal): {receiver_key}")
                self.sifted_keys.append(receiver_key)

                # Check if enough keys are generated
                if len(self.sifted_keys) >= self.key_num:
                    self.finish()  # receiver finishes BB84 protocol


class DecoyBB84(PrepareAndMeasure):
    r"""Class for the decoy-state BB84 key generation protocol.

    Attributes:
        basis_encoding (dict): map an int to a measurement basis
        intensities (dict): intensities of the photon source (signal, decoy, vacuum)

    Examples:
        The ``intensities`` attribute of DecoyBB84 protocol is a dictionary which stores the probabilities and
        mean photon numbers of three different state types (signal, decoy, vacuum).
        Here is an example of setting ``intensities`` attribute of the DecoyBB84 protocol.

        >>> decoy_bb84 = DecoyBB84("Decoy BB84")
        >>> decoy_bb84.set_intensity({"prob": [0.5, 0.25, 0.25], "mean_photon_num": [0.8, 0.1, 0]})

    Warning:
        This is an example of decoy-state protocol.
        We assume the classical and quantum channels have the same distance so that the propogation delays of
        classical and quantum messages are the same.
    """

    def __init__(self, name: str, peer=None):
        r"""Constructor for DecoyBB84 class.

        Args:
            name (str): name of the decoy-state BB84 protocol
            peer (QKDNode): peer node to implement the same task
        """
        super().__init__(name, peer)
        self.basis_encoding = {0: Basis.Z(), 1: Basis.X()}
        self.intensities = {}
        self.tx_state_pos_list = []

    @unique
    class StateType(Enum):
        r"""Class for the state types of emitted pulses.
        """

        SIGNAL = 0
        DECOY = 1
        VACUUM = 2

    class Message(ClassicalMessage):
        r"""Class for the classical control messages in DecoyBB84 protocol.
        """

        def __init__(self, src: "Node", dst: "Node", protocol: type, data: Dict):
            r"""Constructor for Message class.

            Args:
                src (Node): source of Message
                dst (Node): destination of Message
                protocol (type): protocol of Message
                data (Dict): message content
            """
            super().__init__(src, dst, protocol, data)

        @unique
        class Type(Enum):
            r"""Class for Message types.
            """

            START = "Start"
            FINISH_EMISSION = "Finish emission"
            BASES = "Bases"
            MATCHING = "Matching"

    def set_intensity(self, intensities: dict) -> None:
        r"""Set the intensities of signal, decoy and vacuum pulses.

        Args:
            intensities (dict): intensity of the photon source (signal, decoy, vacuum)
        """
        self.intensities = intensities

    def start(self, **kwargs) -> None:
        r"""Start the decoy-state BB84 protocol.

        Args:
            **kwargs: keyword arguments to start the protocol
        """
        assert self.node is not None and self.peer is not None, f"Should load the protocol stack to a node first!"

        # Initial settings
        self.role = kwargs['role']
        if "key_num" in kwargs.keys():
            self.key_num = kwargs['key_num']
        if "key_length" in kwargs.keys():
            self.key_length = kwargs['key_length']

        self.status = KeyGeneration.Status.WORKING
        self.sifted_keys = []
        self.round = 0

        if self.role == PrepareAndMeasure.Role.TRANSMITTER:
            self.start_time = self.node.env.now  # record the start time of DecoyBB84 protocol
            if self.tx_bases_ratio is None:
                self.tx_bases_ratio = [0.5, 0.5]
            self.tx_bases_list = []
            self.tx_bits_list = []
            self.tx_key_bits = []
            self.transmitter_start_protocol()

        elif self.role == PrepareAndMeasure.Role.RECEIVER:
            if self.rx_bases_ratio is None:
                self.rx_bases_ratio = [0.5, 0.5]
            self.rx_bases_list = []
            self.rx_bits_list = []
            self.rx_key_bits = []

    def transmitter_start_protocol(self) -> None:
        r"""Transmitter starts the decoy-state BB84 protocol.

        Transmitter sends a classical message to the receiver for device synchronization
        and prepares to emit photon pulses.
        """
        self.node.env.logger.info(f"{self.node.name} started decoy-state BB84 protocol with {self.peer.name}")

        self.pulse_num = round(self.key_length / self.node.photon_source.mean_photon_num)
        self.light_duration = round(self.pulse_num / self.node.photon_source.frequency * 1e12)
        self.emission_rest = round(1e12 / self.node.photon_source.frequency)

        transmitter_emit_time = self.node.env.now + self.node.cchannel(self.peer).delay
        clock = transmitter_emit_time + self.node.qchannel(self.peer).delay  # clock time for photon reception

        start_msg = DecoyBB84.Message(src=self.node,
                                      dst=self.peer,
                                      protocol=DecoyBB84,
                                      data={'type': DecoyBB84.Message.Type.START,
                                            'clock': clock,
                                            'source_frequency': self.node.photon_source.frequency,
                                            'light_duration': self.light_duration,
                                            'emission_rest': self.emission_rest})
        self.node.send_classical_msg(dst=self.peer, msg=start_msg)

        # Transmitter begins photon emission as soon as the receiver receives the message
        handler = EventHandler(self, "transmitter_start_emission")
        self.scheduler.schedule_after(self.node.cchannel(self.peer).delay, handler)

    def transmitter_start_emission(self) -> None:
        r"""Transmitter starts to emit photon pulses.

        Transmitter randomly generates bits and encoding bases for photon emission.
        """
        if self.status == KeyGeneration.Status.WORKING:
            self.round += 1
            self.node.env.logger.debug(f"{self.node.name} began photon emission for decoy-state BB84 with"
                                       f" {self.peer.name} (Round {self.round})")

            # Generate transmitter's random bases and bits
            bases = list(np.random.choice([0, 1], self.pulse_num, p=self.tx_bases_ratio))
            bits = list(np.random.choice([0, 1], self.pulse_num))  # uniform distribution
            self.tx_bases_list.append(bases)
            self.tx_bits_list.append(bits)

            bases = [self.basis_encoding[basis] for basis in bases]
            states = []
            for i, basis in enumerate(bases):
                # Prepare states |0>, |1>, |+>, |-> (in the form of density matrices) for photon emission
                states.append(basis[bits[i]] @ np.conjugate(basis[bits[i]]).T)

            # Randomly choose state types (signal, decoy or vacuum) according to the probabilities of intensities
            state_types = list(np.random.choice([self.StateType.SIGNAL, self.StateType.DECOY, self.StateType.VACUUM],
                                                self.pulse_num, p=self.intensities["prob"]))
            # Set related mean photon numbers
            mean_photon_num_list = [self.intensities["mean_photon_num"][state_type.value] for state_type in state_types]

            # Record the positions of different state types (signal, decoy and vacuum)
            state_pos_dict = {"signal": [], "decoy": [], "vacuum": []}
            for i, state_type in enumerate(state_types):
                if state_type == self.StateType.SIGNAL:
                    state_pos_dict["signal"].append(i)
                elif state_type == self.StateType.DECOY:
                    state_pos_dict["decoy"].append(i)
                elif state_type == self.StateType.VACUUM:
                    state_pos_dict["vacuum"].append(i)
            self.tx_state_pos_list.append(state_pos_dict)

            handler = EventHandler(self.node.photon_source, "emit", [self.peer, states, mean_photon_num_list])
            self.scheduler.schedule_now(handler)
            finish_emission_msg = DecoyBB84.Message(src=self.node,
                                                    dst=self.peer,
                                                    protocol=DecoyBB84,
                                                    data={'type': DecoyBB84.Message.Type.FINISH_EMISSION,
                                                          'round': self.round})
            handler = EventHandler(self.node, "send_classical_msg", [self.peer, finish_emission_msg])
            self.scheduler.schedule_after(self.light_duration, handler)

            # Schedule next round of photon emission
            handler = EventHandler(self, "transmitter_start_emission")
            # Rest for a while to avoid scheduling the event during cool-down period of the photon source
            self.scheduler.schedule_after(self.light_duration + self.emission_rest, handler)

    def receiver_stop_reception(self) -> None:
        r"""Receiver stops photon reception.

        Receiver pops the measurement outcomes and resets the measurement bases to
        prepare for receiving photon pulses in the next round.
        """
        if self.status == KeyGeneration.Status.WORKING:
            self.round += 1
            self.node.env.logger.debug(f"{self.node.name} finished receiving photons from {self.peer.name}"
                                       f" (Round {self.round})")

            bases = self.rx_bases_list.pop(0)
            bits = self.receiver_get_bits()
            self.rx_bits_list.append(bits)
            for i, bit in enumerate(bits):
                if bit is None:  # if the bit is invalid, set the related basis as invalid
                    bases[i] = None

            # Send the measurement bases to the transmitter
            bases_msg = DecoyBB84.Message(src=self.node,
                                          dst=self.peer,
                                          protocol=DecoyBB84,
                                          data={'type': DecoyBB84.Message.Type.BASES,
                                                'peer_bases': bases,
                                                'round': self.round})
            self.node.send_classical_msg(dst=self.peer, msg=bases_msg)

            # Reset measurement bases for the next round of photon reception
            new_bases = list(np.random.choice([0, 1], self.pulse_num, p=self.rx_bases_ratio))
            self.rx_bases_list.append(new_bases)
            clock = self.node.env.now + self.emission_rest  # set the next clock time

            new_bases = [self.basis_encoding[basis] for basis in new_bases]
            # Reset parameters for polarization detector
            self.node.polar_detector.set_beamsplitter(clock=clock, bases=new_bases)

        elif self.status == KeyGeneration.Status.DONE:
            self.node.polar_detector.detectors[0].turn_off()
            self.node.polar_detector.detectors[1].turn_off()

    def transmitter_receive_classical_msg(self, msg: "DecoyBB84.Message", **kwargs) -> None:
        r"""Transmitter receives a DecoyBB84 message from the receiver.

        Args:
            msg (DecoyBB84.Message): message for DecoyBB84 protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        photon_round = msg.data['round']

        # if self.node.env.now < self.end_time:
        if msg.data['type'] == DecoyBB84.Message.Type.BASES:
            self.node.env.logger.debug(f"{self.node.name} received bases from {self.peer.name}"
                                       f" (Round {photon_round})")

            peer_bases = msg.data['peer_bases']
            # Pop the earliest bases and bits for index comparison
            bases = self.tx_bases_list.pop(0)
            bits = self.tx_bits_list.pop(0)
            state_pos = self.tx_state_pos_list.pop(0)

            matching_list = []  # store indices of matching bases
            # Sift out mismatching bases
            for ix, basis in enumerate(bases):
                if peer_bases[ix] is not None and basis == peer_bases[ix]:  # store the bases indices if matching
                    matching_list.append(ix)
                else:  # remove mismatching elements
                    if ix in state_pos["signal"]:
                        state_pos["signal"].remove(ix)  # remove mismatching signal states
                    elif ix in state_pos["decoy"]:
                        state_pos["decoy"].remove(ix)  # remove mismatching decoy states
                    elif ix in state_pos["vacuum"]:
                        state_pos["vacuum"].remove(ix)  # remove mismatching vacuum states

            # Randomly choose 10 percent of signal states for testing
            testing_signal_pos = list(np.random.choice(state_pos["signal"], round(0.1 * len(state_pos["signal"])),
                                                       replace=False))
            decoy_bits = []
            vacuum_bits = []
            testing_signal_bits = []
            # Sift key bits and bits for testing (decoy bits, vacuum bits, 10% signal bits)
            for ix, bit in enumerate(bits):
                if ix in state_pos["signal"]:
                    if ix in testing_signal_pos:
                        testing_signal_bits.append(bit)
                    else:
                        self.tx_key_bits.append(bit)
                elif ix in state_pos["decoy"]:
                    decoy_bits.append(bit)
                elif ix in state_pos["vacuum"]:
                    vacuum_bits.append(bit)

            # Send matching bases and testing bits with their positions to the receiver
            matching_msg = DecoyBB84.Message(src=self.node,
                                             dst=self.peer,
                                             protocol=DecoyBB84,
                                             data={'type': DecoyBB84.Message.Type.MATCHING,
                                                   'matching_list': matching_list,
                                                   'decoy_pos': state_pos["decoy"],
                                                   'vacuum_pos': state_pos["vacuum"],
                                                   "testing_signal_pos": testing_signal_pos,
                                                   "decoy_bits": decoy_bits,
                                                   "vacuum_bits": vacuum_bits,
                                                   "testing_signal_bits": testing_signal_bits,
                                                   'round': photon_round})
            self.node.send_classical_msg(dst=self.peer, msg=matching_msg)

            # Check if the length of key bits is enough for a new key
            if len(self.tx_key_bits) >= self.key_length:
                self.node.env.logger.info(f"{self.node.name} generated a valid key with {self.peer.name}"
                                          f" (Round {photon_round})")
                self.node.env.logger.info(f"{self.node.name}'s sifted key: {self.tx_key_bits[0: self.key_length]}")

                # Store the generated key as a decimal integer
                transmitter_key = int("".join(str(self.tx_key_bits.pop(0)) for _ in range(self.key_length)), base=2)
                self.node.env.logger.info(f"{self.node.name}'s sifted key (Decimal): {transmitter_key}")
                self.sifted_keys.append(transmitter_key)

                # Check if it reaches the terminal condition
                if len(self.sifted_keys) >= self.key_num:
                    self.finish()  # transmitter finishes DecoyBB84 protocol
                    # Calculate the overall sifted key rate in kilobit per second
                    self.key_rate_estimation()

    def receiver_receive_classical_msg(self, msg: "DecoyBB84.Message", **kwargs) -> None:
        r"""Receiver receives a DecoyBB84 message from the transmitter.

        Args:
            msg (DecoyBB84.Message): message for DecoyBB84 protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        if msg.data['type'] == DecoyBB84.Message.Type.START:
            self.node.env.logger.debug(f"{self.node.name} received message 'START' from {self.peer.name}")

            clock = msg.data['clock']  # time for photon reception
            frequency = msg.data['source_frequency']  # frequency for beam splitter
            self.light_duration = msg.data['light_duration']
            self.emission_rest = msg.data['emission_rest']

            # Generate receiver's random measurement bases
            self.pulse_num = round(msg.data['light_duration'] * frequency * 1e-12)
            bases = list(np.random.choice([0, 1], self.pulse_num, p=self.rx_bases_ratio))
            self.rx_bases_list.append(bases)

            bases = [self.basis_encoding[basis] for basis in bases]  # map the bits to bases for measurement
            # Set parameters for polarization detector
            self.node.polar_detector.set_beamsplitter(clock=clock, frequency=frequency, bases=bases)
            self.node.polar_detector.detectors[0].turn_on()
            self.node.polar_detector.detectors[1].turn_on()

        elif msg.data['type'] == DecoyBB84.Message.Type.FINISH_EMISSION:
            # Stop receiving photons after a light duration
            handler = EventHandler(self, "receiver_stop_reception")
            self.scheduler.schedule_now(handler)

        elif msg.data['type'] == DecoyBB84.Message.Type.MATCHING:
            photon_round = msg.data['round']
            self.node.env.logger.debug(f"{self.node.name} received matching indices from {self.peer.name} "
                                       f"(Round: {photon_round})")

            matching_list = msg.data['matching_list']
            decoy_pos = msg.data['decoy_pos']
            vacuum_pos = msg.data['vacuum_pos']
            testing_signal_pos = msg.data['testing_signal_pos']
            transmitter_decoy_bits = msg.data['decoy_bits']
            transmitter_vacuum_bits = msg.data['vacuum_bits']
            transmitter_testing_signal_bits = msg.data['testing_signal_bits']

            bits = self.rx_bits_list.pop(0)
            decoy_bits = []
            vacuum_bits = []
            testing_signal_bits = []
            # Extract bits of different state types for testing
            for i in matching_list:
                if i in decoy_pos:
                    decoy_bits.append(i)
                elif i in vacuum_pos:
                    vacuum_bits.append(i)
                elif i in testing_signal_pos:
                    testing_signal_bits.append(i)

            # Parameter estimation is done by simply comparing the decoy, vacuum and testing signal bits here.
            num_testing_errors = 0
            testing_error_rate = 0
            # Parameter estimation: calculate errors for eavesdropping detection
            for decoy_bit, transmitter_decoy_bit in zip(decoy_bits, transmitter_decoy_bits):
                if decoy_bit ^ transmitter_decoy_bit == 1:
                    num_testing_errors += 1
            for vacuum_bit, transmitter_vacuum_bit in zip(vacuum_bits, transmitter_vacuum_bits):
                if vacuum_bit ^ transmitter_vacuum_bit == 1:
                    num_testing_errors += 1
            for testing_signal_bit, transmitter_testing_signal_bit in \
                    zip(testing_signal_bits, transmitter_testing_signal_bits):
                if testing_signal_bit ^ transmitter_testing_signal_bit == 1:
                    num_testing_errors += 1
            # Avoid ZeroDivisionError
            if len(decoy_bits) + len(vacuum_bits) + len(testing_signal_bits) > 0:
                testing_error_rate = num_testing_errors / \
                                     (len(decoy_bits) + len(vacuum_bits) + len(testing_signal_bits))
            self.node.env.logger.debug(f"testing error rate: {testing_error_rate} (Round {photon_round})")

            # Set the remained signal bits as key bits
            key_bits_pos = list(set(matching_list) - set(decoy_pos) - set(vacuum_pos) - set(testing_signal_pos))
            key_bits_pos.sort()
            self.rx_key_bits.extend([bits[i] for i in key_bits_pos])

            # Check if the length of key bits is enough for a new key
            if len(self.rx_key_bits) >= self.key_length:
                self.node.env.logger.info(f"{self.node.name} generated a valid key with {self.peer.name}"
                                          f" (Round {photon_round})")
                self.node.env.logger.info(f"{self.node.name}'s sifted key: {self.rx_key_bits[0: self.key_length]}")

                # Store the generated key as a decimal integer
                receiver_key = int("".join(str(self.rx_key_bits.pop(0)) for _ in range(self.key_length)), base=2)
                self.node.env.logger.info(f"{self.node.name}'s sifted key (Decimal): {receiver_key}")
                self.sifted_keys.append(receiver_key)

                # Check if enough keys are generated
                if len(self.sifted_keys) >= self.key_num:
                    self.finish()  # receiver finishes BB84 protocol
                    self.error_rate_estimation()  # estimate the error of sifted keys


class B92(PrepareAndMeasure):
    r"""Class for the B92 key generation protocol.
    """

    def __init__(self, name: str, peer=None):
        r"""Constructor for B92 class.

        Args:
            name (str): name of the B92 protocol
            peer (QKDNode): peer node to implement the same task
        """
        super().__init__(name, peer)

    def transmitter_receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        pass

    def receiver_receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        pass


class SixState(PrepareAndMeasure):
    r"""Class for the six-state key generation protocol.
    """

    def __init__(self, name: str, peer=None):
        r"""Constructor for SixState class.

        Args:
            name (str): name of the six-state protocol
            peer (QKDNode): peer node to implement the same task
        """
        super().__init__(name, peer)

    def transmitter_receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        pass

    def receiver_receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        pass
