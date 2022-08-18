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
An example of magic square game.
"""

import sys
sys.path.append('..')

from qcompute_qnet.models.qpu.env import QuantumEnv
from qcompute_qnet.models.qpu.node import QuantumNode
from qcompute_qnet.topology.network import Network
from qcompute_qnet.topology.link import Link
from qcompute_qnet.models.qpu.protocol import MagicSquareGame
from qcompute_qnet.quantum.backends import Backend


# Total rounds of the game
game_rounds = 1024

# Create an environment for simulation
env = QuantumEnv("Magic square game", default=True)

# Create quantum nodes with quantum registers and specify their pre-installed protocols
alice = QuantumNode("Alice", qreg_size=2, protocol=MagicSquareGame)
bob = QuantumNode("Bob", qreg_size=2, protocol=MagicSquareGame)
source = QuantumNode("Source", qreg_size=4, protocol=MagicSquareGame)
referee = QuantumNode("Referee", qreg_size=0, protocol=MagicSquareGame)

# Create the communication links
link_as = Link("link_as", ends=(alice, source), distance=1e3)
link_bs = Link("link_bs", ends=(bob, source), distance=1e3)
link_ar = Link("link_ar", ends=(alice, referee), distance=1e3)
link_br = Link("link_br", ends=(bob, referee), distance=1e3)

# Create a network, install the nodes and links
network = Network("Magic square game network")
network.install([alice, bob, referee, source, link_as, link_bs, link_ar, link_br])

# Start the magic square game protocol
alice.start(role="Player1", peer=bob, ent_source=source, referee=referee, rounds=game_rounds)
bob.start(role="Player2", peer=alice, ent_source=source, referee=referee)
source.start(role="Source")
referee.start(role="Referee", players=[alice, bob])

# Initialize the environment and run simulation
env.init()
results = env.run(backend=Backend.QCompute.LocalBaiduSim2, summary=False)

# Calculate the winning probability of the magic square game
referee.protocol.estimate_statistics(results)
