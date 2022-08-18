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
An example of simple network simulation.
"""

import sys
sys.path.append('..')

from qcompute_qnet.core.des import DESEnv
from qcompute_qnet.topology import Network, Node, Link


# Create a simulation environment
env = DESEnv("Simulation Environment", default=True)
# Create a network
network = Network("First Network")
# Create a node named Alice
alice = Node("Alice")
# Create another node named Bob
bob = Node("Bob")
# Create a link between Alice and Bob
link = Link("Alice_Bob", ends=(alice, bob))
# Build up the network from nodes and links
network.install([alice, bob, link])
# Initialize the simulation environment
env.init()
# Run the network simulation
env.run()
