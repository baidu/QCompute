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
Quantum supremacy circuit on Zuchongzhi processor with 56 qubits.
"""

# As dynamic circuit compilation is independent of the type of double-qubit gates,
# all double-qubit gates employed here are controlled-NOT gates.

from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

# Define the double-qubit patterns
pattern_a = [
    [2, 7],
    [3, 8],
    [5, 11],
    [6, 12],
    [9, 15],
    [10, 16],
    [13, 19],
    [17, 22],
    [18, 23],
    [20, 26],
    [21, 27],
    [28, 34],
    [29, 35],
    [30, 36],
    [31, 37],
    [32, 38],
    [33, 39],
    [40, 46],
    [41, 47],
    [42, 48],
    [43, 49],
    [44, 50],
    [45, 51],
]

pattern_b = [
    [0, 5],
    [1, 6],
    [4, 10],
    [7, 13],
    [8, 14],
    [11, 17],
    [12, 18],
    [15, 20],
    [16, 21],
    [19, 24],
    [22, 28],
    [23, 29],
    [25, 31],
    [26, 32],
    [27, 33],
    [34, 40],
    [36, 42],
    [37, 43],
    [38, 44],
    [39, 45],
    [47, 52],
    [48, 53],
    [49, 54],
    [50, 55],
]

pattern_c = [
    [2, 6],
    [3, 7],
    [4, 9],
    [5, 10],
    [11, 16],
    [12, 17],
    [13, 18],
    [19, 23],
    [20, 25],
    [21, 26],
    [22, 27],
    [28, 33],
    [29, 34],
    [31, 36],
    [32, 37],
    [38, 43],
    [39, 44],
    [40, 45],
    [42, 47],
    [48, 52],
    [49, 53],
    [50, 54],
    [51, 55],
]

pattern_d = [
    [0, 4],
    [1, 5],
    [6, 11],
    [7, 12],
    [8, 13],
    [10, 15],
    [14, 19],
    [16, 20],
    [17, 21],
    [18, 22],
    [23, 28],
    [24, 29],
    [25, 30],
    [26, 31],
    [27, 32],
    [33, 38],
    [34, 39],
    [35, 40],
    [36, 41],
    [37, 42],
    [43, 48],
    [44, 49],
    [45, 50],
    [46, 51],
]

# Define the sequence of different patterns
patterns = [pattern_a, pattern_b, pattern_c, pattern_d, pattern_c, pattern_d, pattern_a, pattern_b]

# Set the range of cycle numbers
min_cycle = 2
max_cycle = 24

for cycle in range(min_cycle, max_cycle + 1, 2):
    # Create a quantum circuit
    cir = Circuit("Supremacy circuit on Zuchongzhi with 56 qubits")

    # Alternatively apply a layer of single qubit gates and double-qubit gates
    for i in range(cycle):
        # A layer of single qubit gates, Hadamard gates are used for simplicity
        for j in range(56):
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
