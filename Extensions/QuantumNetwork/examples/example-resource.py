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
An example of resource management architecture in QKD network simulation.
"""

import os
from Extensions.QuantumNetwork.qcompute_qnet.core.des import DESEnv
from Extensions.QuantumNetwork.qcompute_qnet.topology.network import Network
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.node import RMPEndNode
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.utils import summary, print_traffic

# Create the simulation environment and the network
env = DESEnv("QKD Resource Management", default=True)
network = Network("Beijing QMAN")

# Load network topology from the configuration JSON file
filename = "data/beijing_qman_topology_rmp.json"
filename = os.path.abspath(filename)
network.load_topology_from(filename)

# Activate random request generation for end nodes
for node in network.nodes:
    if isinstance(node, RMPEndNode):
        node.random_request = True

# Initialize the simulation environment and run the simulation
env.init()
env.run(end_time=5e11, logging=True)

# Summarize the requests and keys delivered in the whole simulation
summary(network)
# Display the network traffic
print_traffic(network, num_color=10)
