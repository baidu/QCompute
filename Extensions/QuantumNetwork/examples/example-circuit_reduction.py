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
An example of reducing circuit width of a static quantum circuit through compiling it into an
equivalent dynamic quantum circuit.
"""

from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

# Create a quantum circuit
cir = Circuit()

# Quantum state evolution
cir.h(0)
cir.h(1)
cir.h(2)
cir.h(3)
cir.h(4)
cir.h(5)
cir.cnot([0, 1])
cir.cnot([1, 2])
cir.cnot([2, 3])
cir.cnot([3, 4])
cir.cnot([4, 5])
cir.cnot([5, 0])

# Set 'mid' for the measurement to obtain corresponding outcomes
cir.measure(0, mid="a")
cir.measure(1, mid="b")
cir.measure(2, mid="c")
cir.measure(3, mid="d")
cir.measure(4, mid="e")
cir.measure(5, mid="f")
# Set 'output_ids' for the final sampling results
cir.output_ids = ["a", "b", "c", "d", "e", "f"]

# Print the static quantum circuit
cir.print_circuit()

# Check whether the current circuit width can be reduced
if cir.is_reducible():
    # Reduce the circuit by 'minimum_remaining_values' algorithm
    cir.reduce("minimum_remaining_values")

# Print the dynamic quantum circuit
cir.print_circuit()
