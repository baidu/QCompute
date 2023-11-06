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
An example of the quantum circuit implementation of the quantum Fourier transform.
"""

# As dynamic circuit compilation is independent of the type of double-qubit gates,
# all double-qubit gates employer here are controlled-NOT gates.

import numpy
from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

# Set the number of qubits
qubit_num = 4

# Create a quantum circuit
cir = Circuit("Quantum Fourier Transform")

# Implement the quantum Fourier transform
for i in range(qubit_num):
    cir.h(i)
    for j in range(2, qubit_num - i + 1):
        cir.cx([i, i + j - 1])

for i in range(qubit_num // 2):
    cir.cx([i, qubit_num - i - 1])

# Measure all qubits
cir.measure()
# Print the circuit
cir.print_circuit()


# Get the biadjacency matrix of the simplified graph of the quantum circuit through boolean matrix multiplication
b_circuit, _ = cir.get_biadjacency_and_candidate_matrices()

# Verify if the biadjacency matrix of the quantum Fourier transform circuit is an all-one matrix
print(
    "\nThe biadjacency matrix of the quantum Fourier transform circuit is an all-one matrix:", numpy.all(b_circuit == 1)
)
