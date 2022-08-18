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
Module for basic quantum network protocols.
"""

from enum import Enum, unique
from typing import List, Dict

import numpy
import numpy.random

from qcompute_qnet.protocols.protocol import Protocol, SubProtocol
from qcompute_qnet.messages.message import ClassicalMessage
from qcompute_qnet.models.qpu.message import QuantumMsg

__all__ = [
    "Teleportation",
    "EntanglementSwapping",
    "BellTest",
    "CHSHGame",
    "MagicSquareGame"
]


class Teleportation(Protocol):
    r"""Class for the teleportation protocol.

    Attributes:
        role (SubProtocol): sub-protocol of a specific role
    """

    def __init__(self, name=None):
        r"""Constructor for Teleportation class.

        Args:
            name (str): name of the teleportation protocol
        """
        super().__init__(name)
        self.role = None

    class Message(ClassicalMessage):
        r"""Class for the classical control messages in teleportation protocol.
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

            ENT_REQUEST = "Entanglement request"
            OUTCOME_FROM_SENDER = "Measurement outcome from the sender"

    def start(self, **kwargs) -> None:
        r"""Start the teleportation protocol.

        Args:
            **kwargs: keyword arguments to start the protocol
        """
        role = kwargs['role']
        # Instantiate a sub-protocol by its role, e.g., if role == "Sender", instantiate a "Sender" sub-protocol
        self.role = getattr(Teleportation, role)(self)
        self.role.start(**kwargs)

    def receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        r"""Receive a classical message from the node.

        Args:
            msg (ClassicalMessage): classical message used for the teleportation protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        self.role.receive_classical_msg(msg)

    def receive_quantum_msg(self, msg: "QuantumMsg", **kwargs) -> None:
        r"""Receive a quantum message from the node.

        Note:
            It stores the received qubit first
            and some additional receiving actions are performed based on the role of this node.

        Args:
            msg (QuantumMsg): quantum message used for the teleportation protocol
            **kwargs (Any): keyword arguments for receiving the quantum message
        """
        self.node.env.logger.debug(f"{self.node.name} received an entangled qubit from {kwargs['src'].name}")
        self.node.qreg.store_qubit(msg.data, kwargs['src'])

        self.role.receive_quantum_msg()

    class Source(SubProtocol):
        r"""Class for the role of entanglement source in teleportation protocol.
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Source class.

            Args:
                super_protocol (Protocol): super protocol of the Source protocol
            """
            super().__init__(super_protocol)

        def start(self, **kwargs) -> None:
            pass

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the teleportation protocol
            """
            if msg.data['type'] == Teleportation.Message.Type.ENT_REQUEST:
                self.node.env.logger.debug(f"{self.node.name} received an entanglement request from {msg.src.name}"
                                           f" for teleportation")
                # Entanglement generation
                self.node.qreg.h(0)
                self.node.qreg.cnot([0, 1])

                # Entanglement distribution
                self.node.send_quantum_msg(dst=msg.src, qreg_address=0)
                self.node.send_quantum_msg(dst=msg.data['peer'], qreg_address=1)

    class Sender(SubProtocol):
        r"""Class for the role of sender in teleportation protocol.

        Attributes:
            peer (QuantumNode): peer node to implement the same task
            ent_source (QuantumNode): entanglement source for entanglement distribution
            address_to_teleport (int): address of the qubit to teleport in local quantum register
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Sender class.

            Args:
                super_protocol (Protocol): super protocol of the Sender protocol
            """
            super().__init__(super_protocol)
            self.peer = None
            self.ent_source = None
            self.address_to_teleport = None

        def start(self, **kwargs) -> None:
            r"""Start the teleportation protocol for the sender.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.peer = kwargs['peer']
            self.ent_source = kwargs['ent_source']
            self.address_to_teleport = kwargs['address_to_teleport']

            self.node.env.logger.info(f"{self.node.name} started the teleportation protocol with {self.peer.name}")
            self.request_entanglement()

        def request_entanglement(self) -> None:
            r"""Send an entanglement distribution request to the entanglement source.
            """
            ent_request_msg = Teleportation.Message(
                src=self.node, dst=self.ent_source, protocol=Teleportation,
                data={'type': Teleportation.Message.Type.ENT_REQUEST, 'peer': self.peer}
            )
            self.node.send_classical_msg(dst=self.ent_source, msg=ent_request_msg)

        def receive_quantum_msg(self) -> None:
            r"""Receive a quantum message from the node.
            """
            # Perform Bell state measurement
            address_reception = self.node.qreg.get_address(self.ent_source)  # get the address of received qubit
            self.node.qreg.bsm([self.address_to_teleport, address_reception])
            self.node.env.logger.debug(f"{self.node.name} finished local operations for teleportation")

            outcome_msg = Teleportation.Message(
                src=self.node, dst=self.peer, protocol=Teleportation,
                data={'type': Teleportation.Message.Type.OUTCOME_FROM_SENDER,
                      'outcome_from_sender': [self.node.qreg.units[self.address_to_teleport]['outcome'],
                                              self.node.qreg.units[address_reception]['outcome']]}
            )
            self.node.send_classical_msg(dst=self.peer, msg=outcome_msg)

    class Receiver(SubProtocol):
        r"""Class for the role of receiver in teleportation protocol.

        Attributes:
            peer (QuantumNode): peer node to implement the same task
            ent_source (QuantumNode): entanglement source for entanglement distribution
            outcome_from_sender (List[int]): measurement outcome received from the sender
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Receiver class.

            Args:
                super_protocol (Protocol): super protocol of the Receiver protocol
            """
            super().__init__(super_protocol)
            self.peer = None
            self.ent_source = None
            self.outcome_from_sender = None

        def start(self, **kwargs) -> None:
            r"""Start the teleportation protocol for the receiver.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.peer = kwargs['peer']
            self.ent_source = kwargs['ent_source']

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the teleportation protocol
            """
            if msg.data['type'] == Teleportation.Message.Type.OUTCOME_FROM_SENDER:
                self.node.env.logger.debug(f"{self.node.name} received the measurement outcome from {msg.src.name} "
                                           f"for teleportation")
                self.outcome_from_sender = msg.data['outcome_from_sender']

                address_reception = self.node.qreg.get_address(self.ent_source)
                if self.node.qreg.units[address_reception]['qubit'] is not None:
                    self.correct_state()

        def receive_quantum_msg(self) -> None:
            r"""Receive a quantum message from the node.
            """
            if self.outcome_from_sender is not None:
                self.correct_state()

        def correct_state(self) -> None:
            r"""Correct the quantum state with Pauli operators.
            """
            address_reception = self.node.qreg.get_address(self.ent_source)
            self.node.qreg.z(address_reception, condition=self.outcome_from_sender[0])
            self.node.qreg.x(address_reception, condition=self.outcome_from_sender[1])

            # After the state correction we measure the qubit state to check the teleported state
            self.node.qreg.measure(address_reception)

            self.node.env.logger.info(f"{self.node.name} finished state correction and recovered the quantum state")


class EntanglementSwapping(Protocol):
    r"""Class for the entanglement swapping protocol.

    Attributes:
        role (SubProtocol): sub-protocol of a specific role

    Note:
        Once the entanglement swapping is done, both nodes measure their qubits for verification of the protocol.
    """

    def __init__(self, name=None):
        r"""Constructor for EntanglementSwapping class.

        Args:
            name (str): name of the entanglement swapping protocol
        """
        super().__init__(name)
        self.role = None

    class Message(ClassicalMessage):
        r"""Class for the classical control messages in entanglement swapping protocol.
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

            SWAP_REQUEST = "Swapping request"
            ENT_REQUEST = "Entanglement request"
            OUTCOME_FROM_REPEATER = "Measurement outcome from the repeater"

    def start(self, **kwargs) -> None:
        r"""Start the entanglement swapping protocol.

        Args:
            **kwargs: keyword arguments to start the protocol
        """
        role = kwargs['role']
        self.role = getattr(EntanglementSwapping, role)(self)  # instantiate a sub-protocol by its role
        self.role.start(**kwargs)

    def receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        r"""Receive a classical message from the node.

        Args:
            msg (ClassicalMessage): classical message used for the entanglement swapping protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        self.role.receive_classical_msg(msg)

    def receive_quantum_msg(self, msg: "QuantumMsg", **kwargs) -> None:
        r"""Receive a quantum message from the node.

        Note:
            It stores the received qubit first
            and some additional receiving actions are performed based on the role of this node.

        Args:
            msg (QuantumMsg): quantum message used for the entanglement swapping protocol
            **kwargs (Any): keyword arguments for receiving the quantum message
        """
        self.node.env.logger.debug(f"{self.node.name} received an entangled qubit from {kwargs['src'].name}")
        self.node.qreg.store_qubit(msg.data, kwargs['src'])

        self.role.receive_quantum_msg()

    class Source(SubProtocol):
        r"""Class for the role of entanglement source in entanglement swapping protocol.
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Source class.

            Args:
                super_protocol (Protocol): super protocol of the Source protocol
            """
            super().__init__(super_protocol)

        def start(self, **kwargs) -> None:
            pass

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the entanglement swapping protocol
            """
            if msg.data['type'] == EntanglementSwapping.Message.Type.ENT_REQUEST:
                self.node.env.logger.debug(f"{self.node.name} received an entanglement request from {msg.src.name}")
                # Entanglement generation
                self.node.qreg.h(0)
                self.node.qreg.cnot([0, 1])

                # Entanglement distribution
                self.node.send_quantum_msg(dst=msg.src, qreg_address=0)
                self.node.send_quantum_msg(dst=msg.data['peer'], qreg_address=1)

    class UpstreamNode(SubProtocol):
        r"""Class for the role of upstream node in entanglement swapping protocol.

        Attributes:
            repeater (QuantumNode): quantum repeater for entanglement swapping
            peer (QuantumNode): peer node to share entanglement
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for UpstreamNode class.

            Args:
                super_protocol (Protocol): super protocol of the UpstreamNode protocol
            """
            super().__init__(super_protocol)
            self.repeater = None
            self.peer = None

        def start(self, **kwargs) -> None:
            r"""Start the entanglement swapping protocol for the upstream node.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.peer = kwargs['peer']
            self.repeater = kwargs['repeater']
            self.node.env.logger.info(f"{self.node.name} started entanglement swapping protocol with {self.peer.name}")
            self.request_swapping()

        def request_swapping(self) -> None:
            r"""Send an entanglement swapping request to the repeater.
            """
            swap_request_msg = EntanglementSwapping.Message(
                src=self.node, dst=self.repeater, protocol=EntanglementSwapping,
                data={'type': EntanglementSwapping.Message.Type.SWAP_REQUEST, 'peer': self.peer}
            )
            self.node.send_classical_msg(dst=self.repeater, msg=swap_request_msg)

        def receive_quantum_msg(self) -> None:
            r"""Receive a quantum message from the node.
            """
            # Measure the received qubit for verification of the protocol
            self.node.qreg.measure(0)
            self.node.env.logger.debug(f"{self.node.name} measured the received qubit for verification")

    class DownstreamNode(SubProtocol):
        r"""Class for the role of downstream node in entanglement swapping protocol.

        Attributes:
            peer (QuantumNode): peer node to share entanglement
            outcome_from_repeater (List[int]): measurement outcome received from the repeater
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for DownstreamNode class.

            Args:
                super_protocol (Protocol): super protocol of the DownstreamNode protocol
            """
            super().__init__(super_protocol)
            self.peer = None
            self.outcome_from_repeater = None

        def start(self, **kwargs) -> None:
            r"""Start the entanglement swapping protocol for the downstream node.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.peer = kwargs['peer']

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the entanglement swapping protocol
            """
            if msg.data['type'] == EntanglementSwapping.Message.Type.OUTCOME_FROM_REPEATER:
                self.node.env.logger.debug(f"{self.node.name} received the measurement outcome from {msg.src.name}")
                self.outcome_from_repeater = msg.data['outcome_from_repeater']

                if self.node.qreg.units[0]['qubit'] is not None:
                    self.correct_state()

        def receive_quantum_msg(self) -> None:
            r"""Receive a quantum message from the node.
            """
            if self.outcome_from_repeater is not None:
                self.correct_state()

        def correct_state(self) -> None:
            r"""Correct the quantum state with Pauli operators.
            """
            self.node.qreg.z(0, condition=self.outcome_from_repeater[0])
            self.node.qreg.x(0, condition=self.outcome_from_repeater[1])
            # Measure the received qubit for verification of the protocol
            self.node.qreg.measure(0)
            self.node.env.logger.debug(f"{self.node.name} measured the received qubit for verification")

    class Repeater(SubProtocol):
        r"""Class for the role of repeater in entanglement swapping protocol.

        Attributes:
            ent_sources (List[QuantumNode]): entanglement sources for distributing entanglement
            upstream_node (QuantumNode): upstream node of the entanglement swapping
            downstream_node (QuantumNode): downstream node of the entanglement swapping
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Repeater class.

            Args:
                super_protocol (Protocol): super protocol of the Repeater protocol
            """
            super().__init__(super_protocol)
            self.ent_sources = None
            self.upstream_node = None
            self.downstream_node = None

        def start(self, **kwargs) -> None:
            r"""Start the entanglement swapping protocol for the repeater.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.ent_sources = kwargs['ent_sources']

        def request_entanglement(self, source: "Node", peer: "Node") -> None:
            r"""Send an entanglement distribution request to the entanglement source.
            """
            ent_request_msg = EntanglementSwapping.Message(
                src=self.node, dst=source, protocol=EntanglementSwapping,
                data={'type': EntanglementSwapping.Message.Type.ENT_REQUEST, 'peer': peer}
            )
            self.node.send_classical_msg(dst=source, msg=ent_request_msg)

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the entanglement swapping protocol
            """
            if msg.data['type'] == EntanglementSwapping.Message.Type.SWAP_REQUEST:
                self.upstream_node, self.downstream_node = msg.src, msg.data['peer']

                self.request_entanglement(self.ent_sources[0], self.upstream_node)
                self.request_entanglement(self.ent_sources[1], self.downstream_node)

        def receive_quantum_msg(self) -> None:
            r"""Receive a quantum message from the node.
            """
            if self.node.qreg.units[0]['qubit'] is not None and self.node.qreg.units[1]['qubit'] is not None:
                address_0 = self.node.qreg.get_address(self.ent_sources[0])
                address_1 = self.node.qreg.get_address(self.ent_sources[1])
                self.node.qreg.bsm([address_0, address_1])

                outcome_msg = EntanglementSwapping.Message(
                    src=self.node, dst=self.downstream_node, protocol=EntanglementSwapping,
                    data={'type': EntanglementSwapping.Message.Type.OUTCOME_FROM_REPEATER,
                          'outcome_from_repeater': [self.node.qreg.units[address_0]['outcome'],
                                                    self.node.qreg.units[address_1]['outcome']]
                          }
                )
                self.node.send_classical_msg(dst=self.downstream_node, msg=outcome_msg)


class BellTest(Protocol):
    r"""Class for the BellTest protocol.

    Attributes:
        role (SubProtocol): sub-protocol of a specific role
    """

    def __init__(self, name=None):
        r"""Constructor for BellTest class.

        Args:
            name (str): name of the BellTest protocol
        """
        super().__init__(name)
        self.role = None

    def start(self, **kwargs) -> None:
        r"""Start the BellTest protocol.

        Args:
            **kwargs: keyword arguments to start the protocol
        """
        role = kwargs['role']
        self.role = getattr(BellTest, role)(self)  # instantiate a sub-protocol by its role
        self.role.start(**kwargs)

    def receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        r"""Receive a classical message from the node.

        Args:
            msg (ClassicalMessage): classical message used for the BellTest protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        self.role.receive_classical_msg(msg)

    def receive_quantum_msg(self, msg: "QuantumMsg", **kwargs) -> None:
        r"""Receive a quantum message from the node.

        Note:
            It stores the received qubit first
            and some additional receiving actions are performed based on the role of this node.

        Args:
            msg (QuantumMsg): quantum message used for the BellTest protocol
            **kwargs (Any): keyword arguments for receiving the quantum message
        """
        self.node.env.logger.debug(f"{self.node.name} received an entangled qubit from {kwargs['src'].name}")
        self.node.qreg.store_qubit(msg.data, kwargs['src'])
        self.node.qreg.circuit_index = msg.index  # synchronize the quantum circuit

        self.role.receive_quantum_msg()

    @staticmethod
    def estimate_statistics(results: List[Dict]) -> None:
        r"""Calculate the sum of the expectations.

        Args:
            results (List[Dict]): sample results of the circuits
        """
        print("\n" + "-" * 40)
        for i, result in enumerate(results):
            cir_name = result['circuit_name']
            shots = result['shots']
            counts = result['counts']

            if "QS" in cir_name or "SQ" in cir_name:
                avg_qs = (counts['00'] - counts['01'] - counts['10'] + counts['11']) / shots
                print(f"Average value of QS: {avg_qs:.4f}")
            elif "QT" in cir_name or "TQ" in cir_name:
                avg_qt = (counts['00'] - counts['01'] - counts['10'] + counts['11']) / shots
                print(f"Average value of QT: {avg_qt:.4f}")
            elif "RS" in cir_name or "SR" in cir_name:
                avg_rs = (counts['00'] - counts['01'] - counts['10'] + counts['11']) / shots
                print(f"Average value of RS: {avg_rs:.4f}")
            elif "RT" in cir_name or "TR" in cir_name:
                avg_rt = (counts['00'] - counts['01'] - counts['10'] + counts['11']) / shots
                print(f"Average value of RT: {avg_rt:.4f}")

        avg_sum = - avg_qs - avg_qt - avg_rs + avg_rt
        print(f"\n- <QS> - <QT> - <RS> + <RT> = {avg_sum:.4f}\n" + '-' * 40)

    class Sender(SubProtocol):
        r"""Class for the role of sender in BellTest protocol.

        Attributes:
            receivers (List[QuantumNode]): receivers of the qubits
            rounds (int): rounds for testing the Bell inequality
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Sender class.

            Args:
                super_protocol (Protocol): super protocol of the Sender protocol
            """
            super().__init__(super_protocol)
            self.receivers = None
            self.rounds = None

        def start(self, **kwargs) -> None:
            r"""Start the BellTest protocol for the sender.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.receivers = kwargs['receivers']
            self.rounds = kwargs['rounds']

            self.node.env.logger.info(
                f"{self.node.name} started BellTest protocol with {self.receivers[0].name} and {self.receivers[1].name}"
            )
            self.distribute_entanglement()

        def distribute_entanglement(self) -> None:
            r"""Prepare and distribute pairs of entangled qubits.
            """
            for i in range(self.rounds):
                self.node.qreg.create_circuit('CHSH_')

                self.node.qreg.h(0)
                self.node.qreg.cnot([0, 1])
                self.node.qreg.z(0)
                self.node.qreg.x(0)

                self.node.send_quantum_msg(dst=self.receivers[0], qreg_address=0)
                self.node.send_quantum_msg(dst=self.receivers[1], qreg_address=1)

    class ReceiverA(SubProtocol):
        r"""Class for the role of receiver A in BellTest protocol.
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for ReceiverA class.

            Args:
                super_protocol (Protocol): super protocol of the ReceiverA protocol
            """
            super().__init__(super_protocol)

        def start(self, **kwargs) -> None:
            pass

        def receive_quantum_msg(self) -> None:
            r"""Receive a quantum message from the node.
            """
            measurement_choice = numpy.random.choice([0, 1])

            if measurement_choice == 0:
                self.node.qreg.measure(0, basis="z")
                self.node.qreg.circuit.name += "Q"
            else:
                self.node.qreg.measure(0, basis="x")
                self.node.qreg.circuit.name += "R"

    class ReceiverB(SubProtocol):
        r"""Class for the role of receiver B in BellTest protocol.
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for ReceiverB class.

            Args:
                super_protocol (Protocol): super protocol of the ReceiverB protocol
            """
            super().__init__(super_protocol)

        def start(self, **kwargs) -> None:
            pass

        def receive_quantum_msg(self) -> None:
            r"""Receive a quantum message from the node.
            """
            measurement_choice = numpy.random.choice([0, 1])

            if measurement_choice == 0:
                self.node.qreg.ry(0, - numpy.pi / 4)  # (Z + X) / \sqrt{2}
                self.node.qreg.measure(0, basis="z")
                self.node.qreg.circuit.name += "S"
            else:
                self.node.qreg.ry(0, numpy.pi / 4)  # (Z - X) / \sqrt{2}
                self.node.qreg.measure(0, basis="z")
                self.node.qreg.circuit.name += "T"


class CHSHGame(Protocol):
    r"""Class for the CHSH game protocol.

    Attributes:
        role (SubProtocol): sub-protocol of a specific role
    """

    def __init__(self, name=None):
        r"""Constructor for CHSHGame class.

        Args:
            name (str): name of the CHSHGame protocol
        """
        super().__init__(name)
        self.role = None

    class Message(ClassicalMessage):
        r"""Class for the classical control messages in CHSHGame protocol.
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

            ENT_REQUEST = "Entanglement request"
            READY = "Ready"
            QUESTION = "Question from the referee"
            ANSWER = "Answer from the player"

    def start(self, **kwargs) -> None:
        r"""Start the CHSHGame protocol.

        Args:
            **kwargs: keyword arguments to start the protocol
        """
        role = kwargs['role']
        self.role = getattr(CHSHGame, role)(self)  # instantiate a sub-protocol by its role
        self.role.start(**kwargs)

    def receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        r"""Receive a classical message from the node.

        Args:
            msg (ClassicalMessage): classical message used for the CHSHGame protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        self.role.receive_classical_msg(msg)

    def receive_quantum_msg(self, msg: "QuantumMsg", **kwargs) -> None:
        r"""Receive a quantum message from the node.

        Note:
            It stores the received qubit first
            and some additional receiving actions are performed based on the role of this node.

        Args:
            msg (QuantumMsg): quantum message used for the CHSHGame protocol
            **kwargs (Any): keyword arguments for receiving the quantum message
        """
        self.node.env.logger.debug(f"{self.node.name} received an entangled qubit from {kwargs['src'].name}")
        self.node.qreg.store_qubit(msg.data, kwargs['src'])
        self.node.qreg.circuit_index = msg.index  # synchronize the quantum circuit

        self.role.receive_quantum_msg()

    def estimate_statistics(self, results: List[Dict]) -> None:
        r"""Calculate the winning probability of the CHSH game.

        Args:
            results (List[Dict]): sample results of the circuits
        """
        assert type(self.role).__name__ == "Referee", \
            f"The role of {type(self.role).__name__} has no right to calculate the winning probability of the game!"

        self.role.estimate_statistics(results)

    class Source(SubProtocol):
        r"""Class for the role of entanglement source in the CHSHGame protocol.
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Source class.

            Args:
                super_protocol (Protocol): super protocol of the Source protocol
            """
            super().__init__(super_protocol)

        def start(self, **kwargs) -> None:
            pass

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the CHSHGame protocol
            """
            if msg.data['type'] == CHSHGame.Message.Type.ENT_REQUEST:
                self.node.env.logger.debug(f"{self.node.name} received an entanglement request from {msg.src.name}")

                self.node.qreg.create_circuit(f"CHSHGame_")
                # Entanglement generation
                self.node.qreg.h(0)
                self.node.qreg.cnot([0, 1])

                # Entanglement distribution
                self.node.send_quantum_msg(dst=msg.src, qreg_address=0)
                self.node.send_quantum_msg(dst=msg.data['peer'], qreg_address=1)

    class Player1(SubProtocol):
        r"""Class for the role of player 1 in the CHSHGame protocol.

        Attributes:
            peer (Node): another player of the CHSH game
            ent_source (Node): entanglement source for distributing entanglement
            referee (Node): referee of the CHSH game
            rounds (int): number of the game rounds
            current_round (int): current round of the game
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Player1 class.

            Args:
                super_protocol (Protocol): super protocol of the Player1 protocol
            """
            super().__init__(super_protocol)
            self.peer = None
            self.ent_source = None
            self.referee = None
            self.rounds = None
            self.current_round = 0

        def start(self, **kwargs) -> None:
            r"""Start the CHSHGame protocol for the player 1.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.peer = kwargs['peer']
            self.ent_source = kwargs['ent_source']
            self.referee = kwargs['referee']
            self.rounds = kwargs['rounds']

            self.node.env.logger.info("Loading CHSH game...")
            self.node.env.logger.info(
                f"Referee: {self.referee.name}, Player1: {self.node.name}, Player2: {self.peer.name}"
            )
            self.prepare_for_game()

        def prepare_for_game(self) -> None:
            r"""Player 1 prepares for the CHSH game.
            """
            self.current_round += 1
            self.request_entanglement()

        def request_entanglement(self) -> None:
            r"""Send an entanglement distribution request to the entanglement source.
            """
            ent_request_msg = CHSHGame.Message(
                src=self.node, dst=self.ent_source, protocol=CHSHGame,
                data={'type': CHSHGame.Message.Type.ENT_REQUEST, 'peer': self.peer}
            )
            self.node.send_classical_msg(dst=self.ent_source, msg=ent_request_msg)

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the CHSHGame protocol
            """
            if msg.data['type'] == CHSHGame.Message.Type.QUESTION:
                self.node.env.logger.debug(f"{self.node.name} received the question from {self.referee.name}")

                x = msg.data['question']

                if x == 0:
                    self.node.qreg.measure(0, basis="z")
                    self.node.qreg.circuit.name += "x0"

                elif x == 1:
                    self.node.qreg.measure(0, basis="x")
                    self.node.qreg.circuit.name += "x1"

                answer_msg = CHSHGame.Message(
                    src=self.node, dst=self.referee, protocol=CHSHGame,
                    data={'type': CHSHGame.Message.Type.ANSWER,
                          'answer': self.node.qreg.units[0]['outcome']}
                )
                self.node.send_classical_msg(dst=self.referee, msg=answer_msg)

                if self.current_round < self.rounds:
                    self.prepare_for_game()

        def receive_quantum_msg(self) -> None:
            r"""Receive a quantum message from the node.
            """
            self.play_game()

        def play_game(self) -> None:
            r"""Send a READY message to the referee.
            """
            ready_msg = CHSHGame.Message(
                src=self.node, dst=self.referee, protocol=CHSHGame,
                data={'type': CHSHGame.Message.Type.READY}
            )
            self.node.send_classical_msg(dst=self.referee, msg=ready_msg)

    class Player2(SubProtocol):
        r"""Class for the role of player 2 in the CHSHGame protocol.

        Attributes:
            peer (Node): another player of the CHSH game
            ent_source (Node): entanglement source for distributing entanglement
            referee (Node): referee of the CHSH game
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Player2 class.

            Args:
                super_protocol (Protocol): super protocol of the Player2 protocol
            """
            super().__init__(super_protocol)
            self.peer = None
            self.ent_source = None
            self.referee = None

        def start(self, **kwargs) -> None:
            r"""Start the CHSHGame protocol for the player 2.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.peer = kwargs['peer']
            self.ent_source = kwargs['ent_source']
            self.referee = kwargs['referee']

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the CHSHGame protocol
            """
            if msg.data['type'] == CHSHGame.Message.Type.QUESTION:
                self.node.env.logger.debug(f"{self.node.name} received the question from {self.referee.name}")

                y = msg.data['question']

                if y == 0:
                    self.node.qreg.ry(0, - numpy.pi / 4)  # (Z + X) / \sqrt{2}
                    self.node.qreg.measure(0, basis="z")
                    self.node.qreg.circuit.name += "y0"

                elif y == 1:
                    self.node.qreg.ry(0, numpy.pi / 4)  # (Z - X) / \sqrt{2}
                    self.node.qreg.measure(0, basis="z")
                    self.node.qreg.circuit.name += "y1"

                answer_msg = CHSHGame.Message(
                    src=self.node, dst=self.referee, protocol=CHSHGame,
                    data={'type': CHSHGame.Message.Type.ANSWER,
                          'answer': self.node.qreg.units[0]['outcome']}
                )
                self.node.send_classical_msg(dst=self.referee, msg=answer_msg)

        def receive_quantum_msg(self) -> None:
            r"""Receive a quantum message from the node.
            """
            self.play_game()

        def play_game(self) -> None:
            r"""Send a READY message to the referee.
            """
            ready_msg = CHSHGame.Message(
                src=self.node, dst=self.referee, protocol=CHSHGame,
                data={'type': CHSHGame.Message.Type.READY}
            )
            self.node.send_classical_msg(dst=self.referee, msg=ready_msg)

    class Referee(SubProtocol):
        r"""Class for the role of referee in the CHSHGame protocol.

        Attributes:
            players (List[QuantumNode]): players of the CHSH game
            players_ready (List[bool]): list for checking if both players are ready for the game
            questions (List[list]): record questions for the two players generated in each game round
            answers_p1 (list): record the answer from player 1 in each game round
            answers_p2 (list): record the answer from player 2 in each game round
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Referee class.

            Args:
                super_protocol (Protocol): super protocol of the Referee protocol
            """
            super().__init__(super_protocol)
            self.players = None
            self.players_ready = [False, False]
            self.questions = []
            self.answers_p1 = []
            self.answers_p2 = []

        def start(self, **kwargs) -> None:
            r"""Start the CHSHGame protocol for the referee.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.players = kwargs['players']

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the CHSHGame protocol
            """
            if msg.data['type'] == CHSHGame.Message.Type.READY:
                self.node.env.logger.debug(f"{msg.src.name} was ready for the CHSH game")
                self.players_ready[self.players.index(msg.src)] = True

                if all(self.players_ready):
                    self.node.env.logger.debug(f"All players were ready. CHSH game started")
                    self.send_questions()

                    self.players_ready = [False, False]  # reset the player's status for the next round

            elif msg.data['type'] == CHSHGame.Message.Type.ANSWER:
                self.node.env.logger.debug(f"{self.node.name} received the answer from {msg.src.name}")

                if msg.src == self.players[0]:
                    self.answers_p1.append(msg.data['answer'])
                elif msg.src == self.players[1]:
                    self.answers_p2.append(msg.data['answer'])

        def send_questions(self) -> None:
            r"""Randomly choose two questions and send them to the two players.
            """
            questions = numpy.random.choice([0, 1], size=2)  # randomly generate two bits as questions
            self.questions.append(questions)

            for i, player in enumerate(self.players):
                question_msg = CHSHGame.Message(
                    src=self.node, dst=player, protocol=CHSHGame,
                    data={'type': CHSHGame.Message.Type.QUESTION,
                          'question': questions[i]}
                )
                self.node.send_classical_msg(dst=player, msg=question_msg)

        def estimate_statistics(self, results: List[Dict]) -> None:
            r"""Calculate the winning probability of the CHSH game.

            Args:
                results (List[Dict]): sample results of the circuits
            """
            num_wins = 0

            for result in results:
                cir_name = result['circuit_name']
                counts = result['counts']

                if "x1" in cir_name and "y1" in cir_name:  # both questions are 1
                    for count in counts:
                        answer_p1, answer_p2 = self.answers_p1[0], self.answers_p2[0]
                        if int(count[answer_p1]) ^ int(count[answer_p2]) == 1:
                            num_wins += counts[count]
                else:
                    for count in counts:
                        answer_p1, answer_p2 = self.answers_p1[0], self.answers_p2[0]
                        if int(count[answer_p1]) ^ int(count[answer_p2]) == 0:
                            num_wins += counts[count]

            winning_prob = num_wins / sum(result['shots'] for result in results)
            print(f"\n{'-' * 55}\nThe winning probability of the CHSH game is {winning_prob:.4f}.\n{'-' * 55}")


class MagicSquareGame(Protocol):
    r"""Class for the MagicSquareGame protocol.

    Attributes:
        role (SubProtocol): sub-protocol of a specific role
    """

    def __init__(self, name=None):
        r"""Constructor for MagicSquareGame class.

        Args:
            name (str): name of the MagicSquareGame protocol
        """
        super().__init__(name)
        self.role = None

    class Message(ClassicalMessage):
        r"""Class for the classical control messages in MagicSquareGame protocol.
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

            ENT_REQUEST = "Entanglement request"
            READY = "Ready"
            QUESTION = "Question from the referee"
            ANSWER = "Answer from the player"

    def start(self, **kwargs) -> None:
        r"""Start the MagicSquareGame protocol.

        Args:
            **kwargs: keyword arguments to start the protocol
        """
        role = kwargs['role']
        self.role = getattr(MagicSquareGame, role)(self)  # instantiate a sub-protocol by its role
        self.role.start(**kwargs)

    def receive_classical_msg(self, msg: "ClassicalMessage", **kwargs) -> None:
        r"""Receive a classical message from the node.

        Args:
            msg (ClassicalMessage): classical message used for the MagicSquareGame protocol
            **kwargs (Any): keyword arguments for receiving the classical message
        """
        self.role.receive_classical_msg(msg)

    def receive_quantum_msg(self, msg: "QuantumMsg", **kwargs) -> None:
        r"""Receive a quantum message from the node.

        Note:
            It stores the received qubit first
            and some additional receiving actions are performed based on the role of this node.

        Args:
            msg (QuantumMsg): quantum message used for the MagicSquareGame protocol
            **kwargs (Any): keyword arguments for receiving the quantum message
        """
        self.node.env.logger.debug(f"{self.node.name} received an entangled qubit from {kwargs['src'].name}")
        self.node.qreg.store_qubit(msg.data, kwargs['src'])
        self.node.qreg.circuit_index = msg.index  # synchronize the quantum circuit

        self.role.receive_quantum_msg()

    def estimate_statistics(self, results: List[Dict]) -> None:
        r"""Calculate the winning probability of the magic square game.

        Args:
            results (List[Dict]): sample results of the circuits
        """
        assert type(self.role).__name__ == "Referee",\
            f"The role of {type(self.role).__name__} has no right to calculate the winning probability of the game!"

        self.role.estimate_statistics(results)

    class Source(SubProtocol):
        r"""Class for the role of entanglement source in the MagicSquareGame protocol.
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Source class.

            Args:
                super_protocol (Protocol): super protocol of the Source protocol
            """
            super().__init__(super_protocol)

        def start(self, **kwargs) -> None:
            pass

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the MagicSquareGame protocol
            """
            if msg.data['type'] == MagicSquareGame.Message.Type.ENT_REQUEST:
                self.node.env.logger.debug(f"{self.node.name} received an entanglement request from {msg.src.name}")

                self.node.qreg.create_circuit(f"MagicSquareGame_")
                # Entanglement generation
                self.node.qreg.h(0)
                self.node.qreg.cnot([0, 1])
                self.node.qreg.h(2)
                self.node.qreg.cnot([2, 3])

                # Entanglement distribution
                self.node.send_quantum_msg(dst=msg.src, qreg_address=0, priority=0)
                self.node.send_quantum_msg(dst=msg.src, qreg_address=2, priority=1)
                self.node.send_quantum_msg(dst=msg.data['peer'], qreg_address=1, priority=0)
                self.node.send_quantum_msg(dst=msg.data['peer'], qreg_address=3, priority=1)

    class Player1(SubProtocol):
        r"""Class for the role of player 1 in the MagicSquareGame protocol.

        Attributes:
            peer (Node): another player of the magic square game
            ent_source (Node): entanglement source for distributing entanglement
            referee (Node): referee of the magic square game
            rounds (int): number of the game rounds
            current_round (int): current round of the game
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Player1 class.

            Args:
                super_protocol (Protocol): super protocol of the Player1 protocol
            """
            super().__init__(super_protocol)
            self.peer = None
            self.ent_source = None
            self.referee = None
            self.rounds = None
            self.current_round = 0

        def start(self, **kwargs) -> None:
            r"""Start the MagicSquareGame protocol for the player 1.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.peer = kwargs['peer']
            self.ent_source = kwargs['ent_source']
            self.referee = kwargs['referee']
            self.rounds = kwargs['rounds']

            self.node.env.logger.info("Loading magic square game...")
            self.node.env.logger.info(
                f"Referee: {self.referee.name}, Player1: {self.node.name}, Player2: {self.peer.name}"
            )
            self.prepare_for_game()

        def prepare_for_game(self) -> None:
            r"""Player 1 prepares for the magic square game.
            """
            self.current_round += 1
            self.request_entanglement()

        def request_entanglement(self) -> None:
            r"""Send an entanglement distribution request to the entanglement source.
            """
            ent_request_msg = MagicSquareGame.Message(
                src=self.node, dst=self.ent_source, protocol=MagicSquareGame,
                data={'type': MagicSquareGame.Message.Type.ENT_REQUEST, 'peer': self.peer}
            )
            self.node.send_classical_msg(dst=self.ent_source, msg=ent_request_msg)

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the MagicSquareGame protocol
            """
            if msg.data['type'] == MagicSquareGame.Message.Type.QUESTION:
                self.node.env.logger.debug(f"{self.node.name} received the question from {self.referee.name}")

                row = msg.data['question']

                if row == 0:
                    self.node.qreg.measure(0, basis="z")
                    self.node.qreg.measure(1, basis="z")
                    self.node.qreg.circuit.name += "r0"

                elif row == 1:
                    self.node.qreg.measure(0, basis="x")
                    self.node.qreg.measure(1, basis="x")
                    self.node.qreg.circuit.name += "r1"

                elif row == 2:
                    self.node.qreg.z(0)
                    self.node.qreg.z(1)
                    self.node.qreg.cz([0, 1])
                    self.node.qreg.h(0)
                    self.node.qreg.h(1)
                    self.node.qreg.measure(0, basis="z")
                    self.node.qreg.measure(1, basis="z")
                    self.node.qreg.circuit.name += "r2"

                answer_msg = MagicSquareGame.Message(
                    src=self.node, dst=self.referee, protocol=MagicSquareGame,
                    data={'type': MagicSquareGame.Message.Type.ANSWER,
                          'answer': [self.node.qreg.units[0]['outcome'],
                                     self.node.qreg.units[1]['outcome']]
                          }
                )
                self.node.send_classical_msg(dst=self.referee, msg=answer_msg)

                if self.current_round < self.rounds:
                    self.prepare_for_game()

        def receive_quantum_msg(self) -> None:
            r"""Receive a quantum message from the node.
            """
            if all(unit['qubit'] is not None for unit in self.node.qreg.units):
                self.play_game()

        def play_game(self) -> None:
            r"""Send a READY message to the referee.
            """
            ready_msg = MagicSquareGame.Message(
                src=self.node, dst=self.referee, protocol=MagicSquareGame,
                data={'type': MagicSquareGame.Message.Type.READY}
            )
            self.node.send_classical_msg(dst=self.referee, msg=ready_msg)

    class Player2(SubProtocol):
        r"""Class for the role of player 2 in the MagicSquareGame protocol.

        Attributes:
            peer (Node): another player of the magic square game
            ent_source (Node): entanglement source for distributing entanglement
            referee (Node): referee of the magic square game
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Player2 class.

            Args:
                super_protocol (Protocol): super protocol of the Player2 protocol
            """
            super().__init__(super_protocol)
            self.peer = None
            self.ent_source = None
            self.referee = None

        def start(self, **kwargs) -> None:
            r"""Start the MagicSquareGame protocol for the player 2.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.peer = kwargs['peer']
            self.ent_source = kwargs['ent_source']
            self.referee = kwargs['referee']

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the MagicSquareGame protocol
            """
            if msg.data['type'] == MagicSquareGame.Message.Type.QUESTION:
                self.node.env.logger.debug(f"{self.node.name} received the question from {self.referee.name}")

                column = msg.data['question']

                if column == 0:
                    self.node.qreg.measure(0, basis="x")
                    self.node.qreg.measure(1, basis="z")
                    self.node.qreg.circuit.name += "c0"

                elif column == 1:
                    self.node.qreg.measure(0, basis="z")
                    self.node.qreg.measure(1, basis="x")
                    self.node.qreg.circuit.name += "c1"

                elif column == 2:
                    self.node.qreg.bsm([0, 1])
                    self.node.qreg.circuit.name += "c2"

                answer_msg = MagicSquareGame.Message(
                    src=self.node, dst=self.referee, protocol=MagicSquareGame,
                    data={'type': MagicSquareGame.Message.Type.ANSWER,
                          'answer': [self.node.qreg.units[0]['outcome'],
                                     self.node.qreg.units[1]['outcome']]
                          }
                )
                self.node.send_classical_msg(dst=self.referee, msg=answer_msg)

        def receive_quantum_msg(self) -> None:
            r"""Receive a quantum message from the node.
            """
            if all(unit['qubit'] is not None for unit in self.node.qreg.units):
                self.play_game()

        def play_game(self) -> None:
            r"""Send a READY message to the referee.
            """
            ready_msg = MagicSquareGame.Message(
                src=self.node, dst=self.referee, protocol=MagicSquareGame,
                data={'type': MagicSquareGame.Message.Type.READY}
            )
            self.node.send_classical_msg(dst=self.referee, msg=ready_msg)

    class Referee(SubProtocol):
        r"""Class for the role of referee in the MagicSquareGame protocol.

        Attributes:
            players (List[QuantumNode]): players of the magic square game
            players_ready (List[bool]): list for checking if both players are ready for the game
            questions (List[list]): record questions for the two players generated in each game round
            answers_p1 (list): record the answer from player 1 in each game round
            answers_p2 (list): record the answer from player 2 in each game round
        """

        def __init__(self, super_protocol: Protocol):
            r"""Constructor for Referee class.

            Args:
                super_protocol (Protocol): super protocol of the Referee protocol
            """
            super().__init__(super_protocol)
            self.players = None
            self.players_ready = [False, False]
            self.questions = []
            self.answers_p1 = []
            self.answers_p2 = []

        def start(self, **kwargs) -> None:
            r"""Start the MagicSquareGame protocol for the referee.

            Args:
                **kwargs: keyword arguments to start the protocol
            """
            self.players = kwargs['players']

        def receive_classical_msg(self, msg: "ClassicalMessage") -> None:
            r"""Receive a classical message from the node.

            Args:
                msg (ClassicalMessage): classical message used for the MagicSquareGame protocol
            """
            if msg.data['type'] == MagicSquareGame.Message.Type.READY:
                self.node.env.logger.debug(f"{msg.src.name} was ready for the magic square game")
                self.players_ready[self.players.index(msg.src)] = True

                if all(self.players_ready):
                    self.node.env.logger.debug(f"All players were ready. Magic square game started")
                    self.send_questions()

                    self.players_ready = [False, False]  # reset the players' status for the next round

            elif msg.data['type'] == MagicSquareGame.Message.Type.ANSWER:
                self.node.env.logger.debug(f"{self.node.name} received the answer from {msg.src.name}")

                if msg.src == self.players[0]:
                    self.answers_p1.append(msg.data['answer'])
                elif msg.src == self.players[1]:
                    self.answers_p2.append(msg.data['answer'])

        def send_questions(self) -> None:
            r"""Randomly choose the row and the column to fill and send the indices to the two players respectively.
            """
            questions = numpy.random.choice([0, 1, 2], size=2)
            self.questions.append(questions)

            for i, player in enumerate(self.players):
                question_msg = MagicSquareGame.Message(
                    src=self.node, dst=self.players[i], protocol=MagicSquareGame,
                    data={'type': MagicSquareGame.Message.Type.QUESTION,
                          'question': questions[i]}
                )
                self.node.send_classical_msg(dst=self.players[i], msg=question_msg)

        def estimate_statistics(self, results: List[Dict]) -> None:
            r"""Calculate the winning probability of the magic square game.

            Args:
                results (List[Dict]): sample results of the circuits
            """
            def p1_answer(row: int, outcome: List[int]) -> list:
                r"""Answer of the player 1.

                Args:
                    row (int): row index
                    outcome (List[int]): measurement outcome of the player 1

                Returns:
                    list: numbers to fill the row
                """
                if row == 0:
                    if outcome == [0, 0]:
                        return [1, 1, 1]
                    elif outcome == [0, 1]:
                        return [-1, 1, -1]
                    elif outcome == [1, 0]:
                        return [1, -1, -1]
                    elif outcome == [1, 1]:
                        return [-1, -1, 1]

                elif row == 1 or row == 2:
                    if outcome == [0, 0]:
                        return [1, 1, 1]
                    elif outcome == [0, 1]:
                        return [1, -1, -1]
                    elif outcome == [1, 0]:
                        return [-1, 1, -1]
                    elif outcome == [1, 1]:
                        return [-1, -1, 1]

            def p2_answer(column: int, outcome: List[int]) -> list:
                r"""Answer of the player 2.

                Args:
                    column (int): column index
                    outcome (List[int]): measurement outcome of the player 2

                Returns:
                    list: numbers to fill the column
                """
                if column == 0 or column == 2:
                    if outcome == [0, 0]:
                        return [1, 1, -1]
                    elif outcome == [0, 1]:
                        return [-1, 1, 1]
                    elif outcome == [1, 0]:
                        return [1, -1, 1]
                    elif outcome == [1, 1]:
                        return [-1, -1, -1]

                if column == 1:
                    if outcome == [0, 0]:
                        return [1, 1, -1]
                    elif outcome == [0, 1]:
                        return [1, -1, 1]
                    elif outcome == [1, 0]:
                        return [-1, 1, 1]
                    elif outcome == [1, 1]:
                        return [-1, -1, -1]

            def is_winning(row: int, column: int, answer_p1: List[int], answer_p2: List[int]) -> bool:
                r"""Determine if the players win the game.

                args:
                    row (int): row index
                    column (int): column index
                    answer_p1 (list[int]): answer of the player 1
                    answer_p2 (List[int]): answer of the player 2

                Returns:
                    bool: whether the players win
                """
                if numpy.prod(answer_p1) == 1 and numpy.prod(answer_p2) == -1 and answer_p1[column] == answer_p2[row]:
                    return True
                else:
                    return False

            num_wins = 0

            for i, result in enumerate(results):
                cir_name = result['circuit_name']
                counts = result['counts']

                if "r0" in cir_name:
                    row = 0
                elif "r1" in cir_name:
                    row = 1
                elif "r2" in cir_name:
                    row = 2

                if "c0" in cir_name:
                    column = 0
                elif "c1" in cir_name:
                    column = 1
                elif "c2" in cir_name:
                    column = 2

                for count in counts:
                    answer_p1, answer_p2 = self.answers_p1[0], self.answers_p2[0]
                    outcome_p1 = [int(count[answer_p1[0]]), int(count[answer_p1[1]])]
                    outcome_p2 = [int(count[answer_p2[0]]), int(count[answer_p2[1]])]

                    if is_winning(row, column, p1_answer(row, outcome_p1), p2_answer(column, outcome_p2)):
                        num_wins += counts[count]

            winning_prob = num_wins / sum(result['shots'] for result in results)
            print(f"\n{'-' * 60}\nThe winning probability of the magic square game is {winning_prob:.4f}.\n{'-' * 60}")
