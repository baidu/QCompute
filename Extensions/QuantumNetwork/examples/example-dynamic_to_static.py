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
An example of compilation from a dynamic quantum circuit into a static one.
"""

from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit
from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends import Backend
from Extensions.QuantumNetwork.qcompute_qnet.quantum.utils import plot_results

# Create a quantum circuit
cir = Circuit()

# Quantum state evolution
cir.h(0)
cir.h(1)
cir.h(2)
cir.cnot([0, 1])

# Mid-circuit measurement on qubit 1
cir.measure(1, mid="a")

# Classically controlled operation on qubit 2 based on the measurement outcome of 'a'
cir.x(2, condition="a")

# Reset qubit 1
cir.reset(1)

# Quantum state evolution and measurements
cir.h(1)
cir.measure(2, mid="b")
cir.z(1, condition="b")
cir.cz([1, 0])
cir.measure(0, mid="c")
cir.measure(1, mid="d")

# Set 'output_ids' for the final sampling results
cir.output_ids = ["a", "b", "c", "d"]

# Print the dynamic quantum circuit
cir.print_circuit()
# Run simulation with QNET StateVector backend
counts1 = cir.run(shots=8192, backend=Backend.QNET.StateVector)["counts"]

# Compile the dynamic circuit into a static quantum circuit
cir.to_static()
cir.print_circuit()

# Run simulation for the compiled static quantum circuit
counts2 = cir.run(shots=8192, backend=Backend.QNET.StateVector)["counts"]

# Visualization of simulation results
plot_results([counts1, counts2], ["dynamic circuit", "static circuit"])
