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

"""
An example to demonstrate the Interleaved Randomized Benchmarking protocol.
"""
import qiskit
from qiskit.providers.fake_provider import FakeSantiago, FakeParis
from qiskit.providers.aer import AerSimulator
from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error

import sys
sys.path.append('../..')

from QCompute import *
import qcompute_qep.benchmarking as rb
import qcompute_qep.utils.types as types
import qcompute_qep.quantum.clifford as clifford

# Set the token. You must set your VIP token in order to access the hardware.
Define.hubToken = "Token"

##########################################################################################
# Step 1. Set the quantum computer (instance of QComputer).
#         The QuantumComputer can be a simulator or a hardware interface.
##########################################################################################
# For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
qc = BackendName.LocalBaiduSim2

# For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian
# qc = QCompute.BackendName.CloudBaiduQPUQian

# For numeric test on the noisy simulator, change qc to Qiskit's FakeParis
qc = AerSimulator.from_backend(FakeParis())

# You can also use Qiskit's AerSimulator to customize noise
# noise_model = NoiseModel.from_backend(FakeSantiago())
# p1Q = 0.002
# p2Q = 0.02
# noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), ['s', 'h', 'sdg', ])
# noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), ['cz'])
# qc = AerSimulator(noise_model=noise_model)

##########################################################################################
# Step 2. Perform the randomized benchmarking protocol on the fourth-qubit
##########################################################################################

# Initialize a RandomizedBenchmarking instance
irb = rb.InterleavedRB()
qubits = [4]
n = len(qubits)
target_gate = clifford.Clifford(n)
irb.benchmark(target_gate=target_gate,
              qc=qc,
              qubits=qubits,
              seq_lengths=[1, 3, 5, 7, 10, 15, 20],
              repeats=10)

# Plot the randomized benchmarking results
fname = "InterleavedRB-{}.png".format(types.get_qc_name(qc))
irb.plot_results(show=True, fname=fname)
