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
An example to demonstrate the Cross Entropy Randomized Benchmarking protocol.
"""
import qiskit
from qiskit.providers.fake_provider import FakeSantiago

import sys
sys.path.append('../..')

from QCompute import *
import qcompute_qep.benchmarking as rb
import qcompute_qep.utils.types as types


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

# For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago
# qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())

##########################################################################################
# Step 2. Perform the randomized benchmarking protocol.
##########################################################################################

# Initialize a XEB instance
xeb = rb.XEB()
xeb.benchmark(qc=qc,
              qubits=[0],
              shots=8192,
              seq_lengths=[1, 5, 10, 15, 25, 40, 75, 100],
              repeats=10)

# Plot the cross-entropy benchmarking results
fname = "XEB-{}.png".format(types.get_qc_name(qc))
xeb.plot_results(show=True, fname=fname)
