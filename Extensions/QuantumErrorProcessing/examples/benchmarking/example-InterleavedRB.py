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
from QCompute import *
import qcompute_qep.benchmarking as rb
import qcompute_qep.utils.types as types
import qcompute_qep.quantum.clifford as clifford

##########################################################################################
# Step 1. Set the quantum computer (instance of QComputer).
#         The QuantumComputer can be a simulator or a hardware interface.
##########################################################################################
# For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
qc = BackendName.LocalBaiduSim2

# For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian.
# You must set your VIP token first in order to access the Baidu hardware.
# Define.hubToken = "Token"
# qc = BackendName.CloudBaiduQPUQian

# For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago simulator
# from qiskit.providers.fake_provider import FakeSantiago
# qc = FakeSantiago()

##########################################################################################
# Step 2. Perform the randomized benchmarking protocol on the fourth-qubit
##########################################################################################
# Initialize a RandomizedBenchmarking instance
irb = rb.InterleavedRB()
# Obtain a random Clifford gate as target
target_gate = clifford.Clifford(1)

# Call the randomized benchmarking procedure and obtain estimated error rate
qubits = [0]
irb.benchmark(target_gate=target_gate,
              qc=qc,
              qubits=qubits,
              seq_lengths=[1, 2, 3, 4, 5],
              repeats=5)
print("*******************************************************************************")
print("* Interleaved RB on qubits: {}".format(qubits))
print("* Estimated interleaved gate fidelity parameter: {}".format(irb.results['InterleavedRB']['f']))
print("* Standard deviation error of interleaved estimation: {}".format(irb.results['InterleavedRB']['f_err']))
print("* Estimated average gate fidelity parameter: {}".format(irb.results['StandardRB']['f']))
print("* Standard deviation error of average estimation: {}".format(irb.results['StandardRB']['f_err']))
fname = "InterleavedRB-{}-qubits{}.png".format(types.get_qc_name(qc), qubits)
print("* InterleavedRB data is visualized in figure '{}'.".format(fname))
# Plot the randomized benchmarking results
irb.plot_results(show=True, fname=fname)
print("*******************************************************************************")

# You can also perform the randomized benchmarking procedure on other qubits, for example [q2]
qubits = [2]
irb.benchmark(target_gate=target_gate,
              qc=qc,
              qubits=qubits,
              seq_lengths=[1, 2, 3, 4, 5],
              repeats=5)
print("*******************************************************************************")
print("* Interleaved RB on qubits: {}".format(qubits))
print("* Estimated interleaved gate fidelity parameter: {}".format(irb.results['InterleavedRB']['f']))
print("* Standard deviation error of interleaved estimation: {}".format(irb.results['InterleavedRB']['f_err']))
print("* Estimated average gate fidelity parameter: {}".format(irb.results['StandardRB']['f']))
print("* Standard deviation error of average estimation: {}".format(irb.results['StandardRB']['f_err']))
fname = "InterleavedRB-{}-qubits{}.png".format(types.get_qc_name(qc), qubits)
print("* InterleavedRB data is visualized in figure '{}'.".format(fname))
# Plot the randomized benchmarking results
irb.plot_results(show=True, fname=fname)
print("*******************************************************************************")
