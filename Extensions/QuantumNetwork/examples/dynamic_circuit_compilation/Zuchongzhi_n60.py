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
Quantum supremacy circuit on Zuchongzhi processor with 60 qubits.
"""

# As dynamic circuit compilation is independent of the type of double-qubit gates,
# all double-qubit gates employed here are controlled-NOT gates.

from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

# Define the double-qubit patterns
pattern_a = [
    [0, 4],
    [1, 5],
    [2, 6],
    [7, 11],
    [9, 14],
    [10, 15],
    [12, 17],
    [16, 20],
    [19, 25],
    [21, 27],
    [22, 28],
    [23, 29],
    [26, 31],
    [30, 35],
    [32, 38],
    [33, 39],
    [34, 40],
    [36, 42],
    [41, 46],
    [43, 49],
    [44, 50],
    [45, 51],
    [47, 53],
    [48, 54],
    [52, 57],
]

pattern_b = [
    [3, 7],
    [5, 9],
    [6, 10],
    [8, 12],
    [11, 16],
    [15, 19],
    [17, 21],
    [18, 23],
    [20, 26],
    [24, 30],
    [27, 32],
    [28, 33],
    [29, 34],
    [31, 37],
    [35, 41],
    [38, 43],
    [39, 44],
    [40, 45],
    [42, 47],
    [46, 52],
    [50, 55],
    [51, 56],
    [53, 58],
    [54, 59],
]

pattern_c = [
    [1, 6],
    [4, 9],
    [5, 10],
    [7, 12],
    [8, 13],
    [11, 17],
    [14, 19],
    [15, 20],
    [16, 21],
    [18, 24],
    [22, 29],
    [25, 31],
    [26, 32],
    [27, 33],
    [28, 34],
    [30, 36],
    [35, 42],
    [37, 43],
    [38, 44],
    [39, 45],
    [40, 46],
    [41, 47],
    [49, 55],
    [50, 56],
    [51, 57],
    [52, 58],
    [53, 59],
]

pattern_d = [
    [0, 5],
    [2, 7],
    [3, 8],
    [6, 11],
    [9, 15],
    [10, 16],
    [13, 18],
    [17, 22],
    [19, 26],
    [20, 27],
    [21, 28],
    [23, 30],
    [29, 35],
    [31, 38],
    [32, 39],
    [33, 40],
    [34, 41],
    [42, 48],
    [43, 50],
    [44, 51],
    [45, 52],
    [46, 53],
    [47, 54],
]

# Define the sequence of different patterns
patterns = [pattern_a, pattern_b, pattern_c, pattern_d, pattern_c, pattern_d, pattern_a, pattern_b]

# Set the range of cycle numbers
min_cycle = 2
max_cycle = 24

for cycle in range(min_cycle, max_cycle + 1, 2):
    # Create a quantum circuit
    cir = Circuit("Supremacy circuit on Zuchongzhi with 60 qubits")

    # Alternatively apply a layer of single qubit gates and double-qubit gates
    for i in range(cycle):
        # A layer of single qubit gates, Hadamard gates are used for simplicity
        for j in range(60):
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
