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
An example of dynamic circuit simulation.
"""

from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit
from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends import Backend

# Create a quantum circuit
cir = Circuit("Dynamic circuit")

# Quantum state evolution
cir.h(0)
cir.h(1)
cir.h(2)
cir.cnot([0, 1])
cir.cnot([0, 2])

# Mid-circuit measurement on qubit 0
cir.measure(0, mid="a")

# Reset the measured qubit
cir.reset(0)

# Quantum state evolution
cir.h(0)
cir.h(2)
cir.cnot([1, 2])
cir.cnot([2, 0])

# Measure all qubits at the end of the circuit
cir.measure(0, mid="b")
cir.measure(1, mid="c")
cir.measure(2, mid="d")

# Set 'output_ids' for the final sampling results
cir.output_ids = ["a", "b", "c", "d"]

# Print dynamic quantum circuit
cir.print_circuit()

# Run simulation and print results
results = cir.run(shots=1024, backend=Backend.QNET.StateVector)
print(f"\nCircuit results:\n", results)
