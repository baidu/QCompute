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
An example of transpilation from a static quantum circuit into its equivalent measurement pattern and brickwork pattern,
and from these two patterns into a dynamic quantum circuit.
"""

from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

# Create a quantum circuit
cir = Circuit()

# Quantum gates for the 2-qubit Grover algorithm
cir.h(0)
cir.h(1)
cir.cz([0, 1])
cir.h(0)
cir.h(1)
cir.z(0)
cir.z(1)
cir.cz([0, 1])
cir.h(0)
cir.h(1)

# Set 'output_ids' for the final sampling results
cir.measure(0, mid="a")
cir.measure(1, mid="b")
cir.output_ids = ["a", "b"]

# Print the static quantum circuit
cir.print_circuit()

# Transpile the static quantum circuit into its corresponding measurement pattern
pattern = cir.to_pattern()

# Print the command list of the measurement pattern
pattern.print()

# Transpile the static quantum circuit into its corresponding brickwork pattern
brickwork_pattern = cir.to_brickwork_pattern()

# Print the command list of the brickwork pattern
brickwork_pattern.print()

# Transpile the measurement pattern into an equivalent dynamic quantum circuit
pat_dyn_cir = pattern.to_dynamic_circuit()

# Print the transpiled dynamic quantum circuit
pat_dyn_cir.print_circuit()

# Transpile the brickwork pattern into an equivalent dynamic quantum circuit
bw_pat_dyn_cir = brickwork_pattern.to_dynamic_circuit()

# Print the dynamic quantum circuit
bw_pat_dyn_cir.print_circuit()
