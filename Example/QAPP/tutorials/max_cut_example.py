#!/usr/bin/python3
# -*- coding: utf8 -*-

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

"""
MAX-CUT
"""

import sys

sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

sys.path.append('..')
import networkx as nx
import numpy as np

from QCompute.QPlatform import BackendName
from qapp.algorithm import QAOA
from qapp.application.optimization import MaxCut
from qapp.circuit import QAOAAnsatz
from qapp.optimizer import SPSA


# hyper-parameter setting
shots = 2048
num_vertices = 4  # num_vertices is the number of vertices in the graph G, which is also the number of qubits
layer = 2  # layer is the number of layers
iteration_num = 100

# Readers should get their tokens from quantum-hub.baidu.com to be connected to real quantum devices and cloud backend.
# from QCompute import Define
# Define.hubToken = 'your token'
# backend = BackendName.CloudIoPCAS
# backend = BackendName.CloudBaiduSim2Water
backend = BackendName.LocalBaiduSim2
measure = 'SimMeasure'  # Define pauli measurement method

# Initialize the target graph
G = nx.Graph()
V = range(num_vertices)
G.add_nodes_from(V)
E = [(0, 1), (1, 2), (2, 3), (3, 0)]
E.sort()
G.add_edges_from(E)

max_cut = MaxCut(num_qubits=num_vertices)
max_cut.graph_to_hamiltonian(G)
parameters = 2 * np.pi * np.random.rand(layer * 2)
ansatz = QAOAAnsatz(num_vertices, parameters, max_cut._hamiltonian, layer)
opt = SPSA(iteration_num, ansatz, a=0.5, c=0.15)  # Define optimizer

qaoa = QAOA(num_vertices, max_cut._hamiltonian, ansatz, opt, backend, measure)
qaoa.run(shots=shots)
print("The maximum eigenvalue: ", qaoa.maximum_eigenvalue)
# Repeat the measurement of the circuit output state 2048 times
counts = qaoa.get_measure(shots=2048)
# Find the most frequent bit string in the measurement results
cut_bitstring = max(counts, key=counts.get)
solution = max_cut.decode_bitstring(cut_bitstring)
print("The Max-Cut solution of this graph found is: ", solution)
