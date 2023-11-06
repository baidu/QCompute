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
Numerical experiment of the random quantum circuit.
"""

# As dynamic circuit compilation is independent of the type of double-qubit gates,
# all double-qubit gates employer here are controlled-NOT gates.

import math
import random
from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit


# Generate random quantum circuit
def random_circuits(num_qubit, num_gate) -> "Circuit":
    r"""Generate a random quantum circuit with the specified number of qubits
    and double-qubit gates.

    Args:
        num_qubit(int): the number of qubits in the random circuit
        num_gate(int): the number of double-qubit gates in the random circuit

    Returns:
        Circuit: a random circuit
    """
    # Create a quantum circuit
    circuit = Circuit()
    # Occupy all qubits with single qubit gate
    for i in range(num_qubit):
        circuit.h(i)
    # Randomly generate the specified number of double qubit gate
    for j in range(num_gate):
        qubit = random.sample(range(num_qubit), 2)
        circuit.cx(qubit)
    # Measure all qubits
    circuit.measure()

    return circuit


# Set the number of random circuits
run_times = 10
# Set the random shots for the random greedy algorithm
random_shots = 5
# Set the ratio between the number of double-qubit gates and qubits
gate_qubit_ratio = 0.5


for t in range(run_times):
    # Randomly select a qubit number
    qubit_num = random.randint(10, 80)

    # Calculate the number of double qubit gates
    gate_num = math.floor(qubit_num * gate_qubit_ratio)

    # Generate the random circuit
    cir = random_circuits(qubit_num, gate_num)
    original_width = cir.width

    # Apply random greedy heuristic algorithm to compile the circuit
    cir.reduce(method="random_greedy", shots=random_shots)
    compiled_width = cir.width

    # Calculate the reducibility factor
    reducibility_factor = 1 - compiled_width / original_width

    # Print the result
    print(
        "original circuit width:",
        original_width,
        "\n" "number of double qubit gates:",
        gate_num,
        "\n" "reducibility factor:",
        reducibility_factor,
        "\n",
    )
