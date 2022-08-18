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
An example of quantum network architecture simulation.
"""

import sys
sys.path.append('..')

import os
from qcompute_qnet.core.des import DESEnv
from qcompute_qnet.topology.network import Network


# Create an environment for simulation
env = DESEnv("QKD Network Architecture", default=True)
# Create the network for Beijing quantum metropolitan area network
network = Network("Beijing QMAN")

# Set path of the JSON file for network topology configuration
filename = "data/beijing_qman_topology.json"
filename = os.path.abspath(filename)
# Load the network topology from the file
network.load_topology_from(filename)
# Print the quantum network topology by the geographical locations of the nodes
network.print_quantum_topology(geo=True)

# Get end nodes by their names
en13 = env.get_node("EN13")
en15 = env.get_node("EN15")
# EN13 sends a QKD request to EN15
en13.key_request(dst=en15, key_num=10, key_length=256)

# Initialize and run the simulation
env.init()
env.run(logging=True)
