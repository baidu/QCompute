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
An example of the workflow for quantum network simulation.
"""

import sys
sys.path.append('..')

from qcompute_qnet.core.des import DESEnv
from qcompute_qnet.topology.network import Network
from qcompute_qnet.models.qkd.node import QKDNode
from qcompute_qnet.topology.link import Link
from qcompute_qnet.devices.channel import ClassicalFiberChannel, QuantumFiberChannel
from qcompute_qnet.models.qkd.key_generation import PrepareAndMeasure


# 1. Create the simulation environment and set as default
env = DESEnv("BB84", default=True)

# 2. Create a quantum network
network = Network("BB84 Network")

# 3. Create the QKD nodes and load protocols
alice = QKDNode("Alice")
bob = QKDNode("Bob")

# Set the protocols and build the protocol stacks
bb84_alice = alice.set_key_generation(bob)
alice.protocol_stack.build(bb84_alice)

bb84_bob = bob.set_key_generation(alice)
bob.protocol_stack.build(bb84_bob)

# 4. Create and connect the link
link_ab = Link("A_B", ends=(alice, bob))

# 4.1 Create the channels and connect the nodes
cchannel1 = ClassicalFiberChannel("c_A2B", sender=alice, receiver=bob, distance=1e3)
cchannel2 = ClassicalFiberChannel("c_B2A", sender=bob, receiver=alice, distance=1e3)
qchannel = QuantumFiberChannel("q_A2B", sender=alice, receiver=bob, distance=1e3)

# 4.2 Install the channels to the link
link_ab.install([cchannel1, cchannel2, qchannel])

# 5. Install the nodes and the link to the network
network.install([alice, bob, link_ab])

# 6. Set parameters (role, key number, key length) for the start of protocol stacks
alice.protocol_stack.start(role=PrepareAndMeasure.Role.TRANSMITTER, key_num=2, key_length=256)
bob.protocol_stack.start(role=PrepareAndMeasure.Role.RECEIVER, key_num=2, key_length=256)

# 7. Initialize the environment and run simulation, save the log records
env.init()
env.run(logging=True)
