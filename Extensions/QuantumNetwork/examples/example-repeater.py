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
An example of entanglement swapping protocol.
"""

import sys
sys.path.append('..')

from qcompute_qnet.models.qpu.env import QuantumEnv
from qcompute_qnet.topology.network import Network
from qcompute_qnet.topology.link import Link
from qcompute_qnet.models.qpu.node import QuantumNode
from qcompute_qnet.models.qpu.protocol import EntanglementSwapping
from qcompute_qnet.quantum.backends import Backend


# Create an environment for simulation
env = QuantumEnv("Repeater", default=True)

# Create nodes with quantum registers and specify their pre-installed protocols
alice = QuantumNode("Alice", qreg_size=1, protocol=EntanglementSwapping)
repeater = QuantumNode("Repeater", qreg_size=2, protocol=EntanglementSwapping)
bob = QuantumNode("Bob", qreg_size=1, protocol=EntanglementSwapping)
source_ar = QuantumNode("Source_AR", qreg_size=2, protocol=EntanglementSwapping)
source_br = QuantumNode("Source_BR", qreg_size=2, protocol=EntanglementSwapping)

# Create the communication links
link_ar = Link("Link_ar", ends=(alice, repeater), distance=1e3)
link_br = Link("Link_br", ends=(bob, repeater), distance=1e3)
link_alice_source_ar = Link("Link_alice_source_ar", ends=(alice, source_ar), distance=1e3)
link_repeater_source_ar = Link("Link_repeater_source_ar", ends=(repeater, source_ar), distance=1e3)
link_bob_source_br = Link("Link_bob_source_ar", ends=(bob, source_br), distance=1e3)
link_repeater_source_br = Link("Link_repeater_source_br", ends=(repeater, source_br), distance=1e3)

# Create a network, install the nodes and links
network = Network("Repeater network")
network.install([alice, bob, repeater, source_ar, source_br,
                 link_ar, link_br, link_alice_source_ar, link_repeater_source_ar,
                 link_bob_source_br, link_repeater_source_br])

# Start the entanglement swapping protocol
alice.start(role="UpstreamNode", peer=bob, repeater=repeater)
bob.start(role="DownstreamNode", peer=alice)
source_ar.start(role="Source")
source_br.start(role="Source")
repeater.start(role="Repeater", ent_sources=[source_ar, source_br])

# Initialize the environment and run simulation
env.init()
results = env.run(shots=1024, backend=Backend.QCompute.LocalBaiduSim2, summary=False)
# Print the running results
print(f"\nCircuit results:\n", results)

# Check the measurement results of Alice and Bob
reduced_indices = [1, 3]
reduced_results = network.default_circuit.reduce_results(results['counts'], indices=reduced_indices)
print(f"\nMeasurement results of Alice, Bob:\n", reduced_results)
