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
An example of Bell test experiment.
"""

import sys
sys.path.append('..')

from qcompute_qnet.models.qpu.env import QuantumEnv
from qcompute_qnet.models.qpu.node import QuantumNode
from qcompute_qnet.topology.network import Network
from qcompute_qnet.topology.link import Link
from qcompute_qnet.models.qpu.protocol import BellTest
from qcompute_qnet.quantum.backends import Backend


# Total rounds of the Bell test experiment
rounds = 1024

# Create an environment for simulation
env = QuantumEnv("Bell test", default=True)

# Create quantum nodes with quantum registers and specify their pre-installed protocols
alice = QuantumNode("Alice", qreg_size=2, protocol=BellTest)
bob = QuantumNode("Bob", qreg_size=1, protocol=BellTest)
charlie = QuantumNode("Charlie", qreg_size=2, protocol=BellTest)

# Create the communication links
link_ab = Link("link_ab", ends=(alice, bob), distance=1e3)
link_ac = Link("link_ac", ends=(alice, charlie), distance=1e3)
link_bc = Link("link_bc", ends=(bob, charlie), distance=1e3)

# Create a network, install the nodes and links
network = Network("Bell test network")
network.install([alice, bob, charlie, link_ab, link_ac, link_bc])

# Start the Bell test protocol
alice.start(role="ReceiverA")
bob.start(role="ReceiverB")
charlie.start(role="Sender", receivers=[alice, bob], rounds=rounds)

# Initialize the environment and run simulation
env.init()
results = env.run(backend=Backend.QCompute.LocalBaiduSim2, summary=False)

# Statistics estimation for the Bell test experiment
BellTest.estimate_statistics(results)
