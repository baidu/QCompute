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
Quantum supremacy circuit on Sycamore processor with 70 qubits.
"""

# As dynamic circuit compilation is independent of the type of double-qubit gates,
# all double-qubit gates employed here are controlled-NOT gates.

from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

# Define the double-qubit patterns
pattern_a = [
    [0, 6],
    [1, 7],
    [2, 8],
    [3, 9],
    [10, 17],
    [11, 18],
    [12, 19],
    [13, 20],
    [14, 21],
    [22, 29],
    [23, 30],
    [24, 31],
    [25, 32],
    [26, 33],
    [34, 41],
    [35, 42],
    [36, 43],
    [37, 44],
    [38, 45],
    [46, 53],
    [47, 54],
    [48, 55],
    [49, 56],
    [50, 57],
    [58, 65],
    [59, 66],
    [60, 67],
    [61, 68],
    [62, 69],
]

pattern_b = [
    [5, 10],
    [6, 11],
    [7, 12],
    [8, 13],
    [9, 14],
    [16, 22],
    [17, 23],
    [18, 24],
    [19, 25],
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
    [52, 58],
    [53, 59],
    [54, 60],
    [55, 61],
    [56, 62],
    [57, 63],
]

pattern_c = [
    [0, 5],
    [1, 6],
    [2, 7],
    [3, 8],
    [4, 9],
    [10, 16],
    [11, 17],
    [12, 18],
    [13, 19],
    [14, 20],
    [15, 21],
    [22, 28],
    [23, 29],
    [24, 30],
    [25, 31],
    [26, 32],
    [27, 33],
    [34, 40],
    [35, 41],
    [36, 42],
    [37, 43],
    [38, 44],
    [39, 45],
    [46, 52],
    [47, 53],
    [48, 54],
    [49, 55],
    [50, 56],
    [51, 57],
    [58, 64],
    [59, 65],
    [60, 66],
    [61, 67],
    [62, 68],
    [63, 69],
]

pattern_d = [
    [6, 10],
    [7, 11],
    [8, 12],
    [9, 13],
    [17, 22],
    [18, 23],
    [19, 24],
    [20, 25],
    [21, 26],
    [29, 34],
    [30, 35],
    [31, 36],
    [32, 37],
    [33, 38],
    [41, 46],
    [42, 47],
    [43, 48],
    [44, 49],
    [45, 50],
    [53, 58],
    [54, 59],
    [55, 60],
    [56, 61],
    [57, 62],
]

# Define the sequence of different patterns
patterns = [pattern_a, pattern_b, pattern_c, pattern_d, pattern_c, pattern_d, pattern_a, pattern_b]

# Set the range of cycle numbers
min_cycle = 2
max_cycle = 24

for cycle in range(min_cycle, max_cycle + 1, 2):
    # Create a quantum circuit
    cir = Circuit("Supremacy circuit on Sycamore with 70 qubits")

    # Alternatively apply a layer of single qubit gates and double-qubit gates
    for i in range(cycle):
        # A layer of single qubit gates, Hadamard gates are used for simplicity
        for j in range(70):
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
