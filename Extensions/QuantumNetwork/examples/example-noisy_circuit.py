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
An example of noisy circuit simulation.
"""

from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit
from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends import Backend
from Extensions.QuantumNetwork.qcompute_qnet.quantum.utils import plot_results

# Create two circuits for noiseless and noisy circuit simulation, respectively
noiseless_cir, noisy_cir = Circuit("Noiseless circuit"), Circuit("Noisy circuit")

# Entanglement preparation
for cir in [noiseless_cir, noisy_cir]:
    cir.h(1)
    cir.cnot([1, 2])

# Add a bit flip noise to the noisy circuit
noisy_cir.bit_flip(0, prob=0.2)

# Teleportation
for cir in [noiseless_cir, noisy_cir]:
    cir.cnot([0, 1])
    cir.h(0)
    cir.measure(0, mid="m0")
    cir.measure(1, mid="m1")
    cir.x(2, condition="m1")
    cir.z(2, condition="m0")
    cir.measure(2, mid="m2")
    cir.output_ids = ["m2"]

# Print circuits for comparison
noiseless_cir.print_circuit()
noisy_cir.print_circuit()

# Run simulation and plot results for comparison
counts1 = noiseless_cir.run(shots=2048, backend=Backend.QNET.DensityMatrix)["counts"]
counts2 = noisy_cir.run(shots=2048, backend=Backend.QNET.DensityMatrix)["counts"]
plot_results([counts1, counts2], ["Noiseless", "Noisy"])
