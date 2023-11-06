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
Numerical experiment of the Max-Cut QAOA circuit on random unweighted 3-regular graph.
"""

from math import pi
import networkx as nx
from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

# Set the number of random graph for each qubit number
seed_num = 5
# Set the random shots of the random greedy algorithm
random_shots = 5
# Set the range for the number of qubits
min_qubit_num = 10
max_qubit_num = 20
# Set the number of QAOA unitary layers
layer_num = 1

for q in range(min_qubit_num, max_qubit_num + 1, 2):
    for s in range(seed_num):
        # Generate the random U3R graph
        u3r_graph = nx.random_regular_graph(3, q, s)

        # Create a quantum circuit
        cir = Circuit()

        # Construct the Max-Cut QAOA circuit on the random graph
        # Prepare uniform superposition over all qubits
        for i in range(q):
            cir.h(i)
        # Apply the problem unitary and the mixing unitary alternatively multiple times
        # Here all rotation angles are set to pi for simplicity
        for p in range(layer_num):
            # The problem unitary for Max-Cut
            for edge in u3r_graph.edges():
                cir.cx([edge[0], edge[1]])
                cir.rz(edge[1], pi)
                cir.cx([edge[0], edge[1]])
            # The mixing unitary
            for node in u3r_graph.nodes:
                cir.rx(node, pi)

        # Measure all qubits
        cir.measure()
        original_width = cir.width
        # Apply random greedy heuristic algorithm to compile the QAOA circuit
        cir.reduce(method="random_greedy", shots=random_shots)

        # Print the result
        print("Original circuit width:", original_width, "\n", "Compiled circuit width:", cir.width, "\n")
