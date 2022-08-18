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
An example of quantum teleportation protocol.
"""

import sys
sys.path.append('..')

import numpy

from qcompute_qnet.models.qpu.env import QuantumEnv
from qcompute_qnet.topology.network import Network
from qcompute_qnet.models.qpu.node import QuantumNode
from qcompute_qnet.models.qpu.protocol import Teleportation
from qcompute_qnet.topology.link import Link
from qcompute_qnet.quantum.circuit import Circuit
from qcompute_qnet.quantum.backends import Backend


# Create an environment for simulation
env = QuantumEnv("Teleportation", default=True)

# Create nodes with quantum registers and specify their pre-installed protocols
alice = QuantumNode("Alice", qreg_size=2, protocol=Teleportation)
bob = QuantumNode("Bob", qreg_size=1, protocol=Teleportation)
charlie = QuantumNode("Charlie", qreg_size=2, protocol=Teleportation)

# Create the communication links
link_ab = Link("Link_ab", ends=(alice, bob), distance=1e3)
link_ac = Link("Link_ac", ends=(alice, charlie), distance=1e3)
link_bc = Link("Link_bc", ends=(bob, charlie), distance=1e3)

# Create a network, install the nodes and links
network = Network("Teleportation network")
network.install([alice, bob, charlie, link_ab, link_ac, link_bc])

# Randomly prepare a quantum state to teleport
theta, phi, gamma = numpy.random.randn(3)
print(f"Rotation angles (rad) of U3: theta: {theta:.4f}, phi: {phi:.4f}, gamma: {gamma:.4f}")
alice.qreg.u3(0, theta, phi, gamma)

# Start the teleportation protocol
alice.start(role="Sender", peer=bob, ent_source=charlie, address_to_teleport=0)
bob.start(role="Receiver", peer=alice, ent_source=charlie)
charlie.start(role="Source")

# Initialize the environment and run simulation
env.init()
results = env.run(shots=1024, backend=Backend.QCompute.LocalBaiduSim2, summary=False)
# Print the running results
print(f"\nCircuit results:\n", results)

# Check the measurement results of the receiver
reduced_indices = [2]
reduced_results = network.default_circuit.reduce_results(results['counts'], indices=reduced_indices)
print(f"\nMeasurement results of the receiver:\n", reduced_results)

# Create a circuit and apply a U3 gate with the same parameters
comp_cir = Circuit("Circuit for verification")
comp_cir.u3(0, theta, phi, gamma)
comp_cir.measure(0)

# Check the results of the circuit
results = comp_cir.run(shots=1024, backend=Backend.QCompute.LocalBaiduSim2)
print(f"\nMeasurement results of the origin state for verification:\n", results['counts'])
