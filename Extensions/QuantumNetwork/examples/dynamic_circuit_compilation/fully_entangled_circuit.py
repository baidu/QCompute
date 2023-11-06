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
An example of fully entangled circuit.
"""

import numpy
from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

import numpy
from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

# As dynamic circuit compilation is independent of the type of double-qubit gates,
# all double-qubit gates employed here are controlled-NOT gates.

# Set the number of qubits and entanglement layers
qubit_num = 6
layer_num = 1

# Create a quantum circuit
cir = Circuit("Fully entangled circuit")

# Construct linearly entangled circuit
for i in range(layer_num):
    # Single qubit rotation layer, rotation gates can be arbitrarily selected
    for j in range(qubit_num):
        cir.h(j)
    # Entanglement layer
    for j in range(qubit_num - 1):
        for k in range(j + 1, qubit_num):
            cir.cx([j, k])
# Measure all qubits
cir.measure()
# Print quantum circuit
cir.print_circuit()


# Get the biadjacency matrix of the simplified graph of the quantum circuit through boolean matrix multiplication
b_circuit, _ = cir.get_biadjacency_and_candidate_matrices()

# Verify if the biadjacency matrix of the fully entangled circuit is an all-one matrix
print("\nThe biadjacency matrix of the fully entangled circuit is an all-one matrix:", numpy.all(b_circuit == 1))
