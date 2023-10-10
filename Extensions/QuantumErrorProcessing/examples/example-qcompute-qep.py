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
This is a simple example to test if you have successfully installed the QEP module.
"""

from QCompute import *
import Extensions.QuantumErrorProcessing.qcompute_qep.tomography as tomography


# Step 1. Initialize the quantum program for preparing the Bell state
qp = QEnv()  # qp is short for "quantum program", instance of QProgram
qp.Q.createList(2)
H(qp.Q[0])
CX(qp.Q[0], qp.Q[1])

# Step 2. Set the quantum computer (instance of QComputer).
# For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
qc = BackendName.LocalBaiduSim2

# For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian.
# You must set your VIP token first in order to access the Baidu hardware.
# Define.hubToken = "Token"
# qc = BackendName.CloudBaiduQPUQian

# For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago simulator
# from qiskit.providers.fake_provider import FakeSantiago
# qc = FakeSantiago()

# Step 3. Perform State Tomography, check how well the Bell state
# Initialize a StateTomography instance
st = tomography.StateTomography()

# Alternatively, you may initialize the StateTomography instance as follows:
# st = StateTomography(qp, qc, method='inverse', shots=4096)

# Call the tomography procedure and obtain the noisy quantum state
st.fit(qp, qc, method='inverse', shots=4096)

print("***********************************************************************")
print("Testing whether 'qcompute-qep' is successfully installed or not now ...\n")

print('Fidelity of the Bell state is: F = {:.5f}'.format(st.fidelity))
print("Please change 'qc' to other quantum computers for more tests.\n")

print("Package 'qcompute-qep' is successfully installed! Please enjoy!")
print("***********************************************************************")
