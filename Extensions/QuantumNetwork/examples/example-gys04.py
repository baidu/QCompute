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
An example of QKD experiment simulation.
We study the tradeoff between channel distance and the sifted key rate.

See reference:
Gobby, C., ZL Yuan, and A. J. Shields.
"Quantum key distribution over 122 km of standard telecom fiber."
Applied Physics Letters 84.19 (2004): 3762-3764.
"""

import sys
sys.path.append('..')

import pandas as pd
from qcompute_qnet.core.des import DESEnv
from qcompute_qnet.topology.network import Network
from qcompute_qnet.models.qkd.node import QKDNode
from qcompute_qnet.topology.link import Link
from qcompute_qnet.devices.channel import ClassicalFiberChannel, QuantumFiberChannel
from qcompute_qnet.models.qkd.key_generation import PrepareAndMeasure


# Experimental devices configuration
source_options = {"frequency": 2e6, "mean_photon_num": 0.1}
detector_options = {"efficiency": 0.045, "dark_count": 0.0, "count_rate": 10e6, "resolution": 10}

key_rate = []


def gys04(distance: float) -> float:
    r"""Compute the sifted key rate for given distance.

    Args:
        distance (float): distance of the experiment

    Returns:
        float: sifted key rate
    """
    # Create a simulation environment
    env = DESEnv("GYS04", default=True)

    # Create the QKD nodes
    alice = QKDNode("Alice")
    bob = QKDNode("Bob")

    # Set parameters of the physical devices
    alice.photon_source.set(**source_options)
    bob.polar_detector.set_detectors(**detector_options)

    # Set key generation protocols and load to the protocol stacks
    bb84_alice = alice.set_key_generation(bob)
    alice.protocol_stack.build(bb84_alice)
    bb84_bob = bob.set_key_generation(alice)
    bob.protocol_stack.build(bb84_bob)

    # Create the link and communication channels and connect to the nodes
    l_ab = Link("A_B", ends=(alice, bob))
    c1 = ClassicalFiberChannel("c_A2B", sender=alice, receiver=bob, distance=distance)
    c2 = ClassicalFiberChannel("c_B2A", sender=bob, receiver=alice, distance=distance)
    q = QuantumFiberChannel("q_A2B", sender=alice, receiver=bob, distance=distance)

    # Install the channels to the link
    l_ab.install([c1, c2, q])

    # Create a network and install the nodes and links
    network = Network("End to End BB84")
    network.install([alice, bob, l_ab])

    # Start the protocol stacks
    alice.protocol_stack.start(peer=bob, role=PrepareAndMeasure.Role.TRANSMITTER, key_num=1, key_length=256)
    bob.protocol_stack.start(peer=alice, role=PrepareAndMeasure.Role.RECEIVER, key_num=1, key_length=256)

    # Initialize the environment and run
    env.init()
    env.run(summary=False)

    # Calculate the sifted key rate
    return bb84_alice.key_rate_estimation() * 1000


distances = [d * 10e3 for d in range(4)]

for distance in distances:
    rate = gys04(distance)
    key_rate.append(rate)

data = pd.DataFrame({"distance (m)": distances, "sifted key rate (bits/s)": key_rate})
print("-" * 50)
print("Tradeoff between distance and sifted key rate")
print("-" * 50)
print(data.to_string())
