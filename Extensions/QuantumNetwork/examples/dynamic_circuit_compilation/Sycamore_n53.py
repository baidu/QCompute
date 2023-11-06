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
Quantum supremacy circuit on Sycamore processor with 53 qubits.
"""

# As dynamic circuit compilation is independent of the type of double-qubit gates,
# all double-qubit gates employed here are controlled-NOT gates.

from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

# Define the double-qubit patterns
pattern_a = [
    [5, 11],
    [6, 12],
    [7, 13],
    [8, 14],
    [9, 15],
    [10, 16],
    [17, 23],
    [18, 24],
    [19, 25],
    [20, 26],
    [21, 27],
    [22, 28],
    [29, 35],
    [30, 36],
    [31, 37],
    [32, 38],
    [33, 39],
    [34, 40],
    [41, 47],
    [42, 48],
    [43, 49],
    [44, 50],
    [45, 51],
    [46, 52],
]

pattern_b = [
    [1, 5],
    [2, 6],
    [3, 8],
    [4, 9],
    [12, 17],
    [13, 18],
    [14, 19],
    [15, 20],
    [16, 21],
    [24, 29],
    [25, 30],
    [26, 31],
    [27, 32],
    [28, 33],
    [36, 41],
    [37, 42],
    [38, 43],
    [39, 44],
    [40, 45],
]

pattern_c = [
    [0, 5],
    [1, 6],
    [2, 7],
    [3, 9],
    [4, 10],
    [11, 17],
    [12, 18],
    [13, 19],
    [14, 20],
    [15, 21],
    [16, 22],
    [23, 29],
    [24, 30],
    [25, 31],
    [26, 32],
    [27, 33],
    [28, 34],
    [35, 41],
    [36, 42],
    [37, 43],
    [38, 44],
    [39, 45],
    [40, 46],
]

pattern_d = [
    [5, 12],
    [6, 13],
    [7, 14],
    [8, 15],
    [9, 16],
    [17, 24],
    [18, 25],
    [19, 26],
    [20, 27],
    [21, 28],
    [29, 36],
    [30, 37],
    [31, 38],
    [32, 39],
    [33, 40],
    [41, 48],
    [42, 49],
    [43, 50],
    [44, 51],
    [45, 52],
]

# Define the sequence of different patterns
patterns = [pattern_a, pattern_b, pattern_c, pattern_d, pattern_c, pattern_d, pattern_a, pattern_b]

# Set the range of cycle numbers
min_cycle = 2
max_cycle = 24

for cycle in range(min_cycle, max_cycle + 1, 2):
    # Create a quantum circuit
    cir = Circuit("Supremacy circuit on Sycamore with 53 qubits")

    # Alternatively apply a layer of single qubit gates and double-qubit gates
    for i in range(cycle):
        # A layer of single qubit gates, Hadamard gates are used for simplicity
        for j in range(53):
            cir.h(j)
        # A layer of double-qubit gates between qubit pairs within the given pattern
        for pair in patterns[i % 8]:
            cir.cx(pair)

    # Measure all qubits
    cir.measure()
    original_width = cir.width

    # Apply the deterministic greedy algorithm to compile the circuit
    cir.reduce(method="deterministic_greedy")
    compiled_width = cir.width

    # Calculate the reducibility factor
    reducibility_factor = 1 - compiled_width / original_width

    # Print the result
    print(
        "number of cycles:",
        cycle,
        "\n" "original circuit width:",
        original_width,
        "\n" "compiled circuit width",
        compiled_width,
        "\n" "reducibility factor:",
        "%.2f" % reducibility_factor,
        "\n",
    )
