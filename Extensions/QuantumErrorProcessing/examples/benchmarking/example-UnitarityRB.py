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
An example to demonstrate the Unitarity Randomized Benchmarking protocol.
"""

import qiskit
from qiskit.providers.fake_provider import FakeSantiago, FakeParis

import sys
sys.path.append('../..')

import QCompute
import qcompute_qep.benchmarking.unitarityrb as unitarityrb
import qcompute_qep.utils.types as types

# Set the token. You must set your VIP token in order to access the hardware.
QCompute.Define.hubToken = "Token"

##########################################################################################
# Step 1. Set the quantum computer (instance of QComputer).
#         The QuantumComputer can be a simulator or a hardware interface.
##########################################################################################
# For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
# qc = QCompute.BackendName.LocalBaiduSim2
# QCompute.Define.Settings.drawCircuitControl = []
# QCompute.Define.Settings.outputInfo = False
# For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian
# qc = QCompute.BackendName.CloudBaiduQPUQian

# For numeric test on the noisy simulator, change qc to Qiskit's FakeParis
qc = qiskit.providers.aer.AerSimulator.from_backend(FakeParis())

# You can also use Qiskit's AerSimulator to customize noise
# noise_model = NoiseModel()
# p1Q = 0.02
# p2Q = 0.05
# noise_model.add_all_qubit_quantum_error(depolarizing_error(p1Q, 1), ['s', 'h', 'id'])
# noise_model.add_all_qubit_quantum_error(depolarizing_error(p2Q, 2), ['cz'])
# qc = AerSimulator(noise_model=noise_model)

##########################################################################################
# Step 2. Perform the randomized benchmarking protocol.
##########################################################################################

# Initialize a RandomizedBenchmarking instance
urb = unitarityrb.UnitarityRB()
urb.benchmark(qc=qc,
              qubits=[0],
              seq_lengths=[1, 10, 25, 50, 75, 100, 150],
              repeats=10, )
# Plot the randomized benchmarking results
fname = "UnitarityRB-{}.png".format(types.get_qc_name(qc))
urb.plot_results(show=True, fname=fname)
