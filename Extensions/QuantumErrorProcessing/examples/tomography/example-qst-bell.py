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
Example to demonstrate quantum state tomography on the Bell state.
"""
from QCompute import *
import Extensions.QuantumErrorProcessing.qcompute_qep.tomography as tomography


# Step 1. Set up the quantum program for preparing the Bell state
# We aim to investigate how well the Bell state is prepared.
qp = QEnv()
qp.Q.createList(2)
H(qp.Q[0])
CX(qp.Q[0], qp.Q[1])

# Step 2. Set the quantum computer (instance of QComputer).
# The QuantumComputer can be a simulator or a hardware interface.

# For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
qc = BackendName.LocalBaiduSim2

# For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian.
# You must set your VIP token first in order to access the Baidu hardware.
# Define.hubToken = "Token"
# qc = BackendName.CloudBaiduQPUQian

# For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago simulator
# from qiskit.providers.fake_provider import FakeSantiago
# qc = FakeSantiago()

# Step 3. Execute quantum state tomography,
# check how well the Bell state is implemented in the QuantumComputer.
st = tomography.StateTomography()

# Call the tomography procedure and obtain the noisy quantum state
qubits = [0, 1]
st.fit(qp, qc, qubits=qubits, method="lstsq", shots=4096, ptm=False)
print("Fidelity of the Bell state on qubits {} is: F = {:.5f}".format(qubits, st.fidelity))

# You can also perform quantum state tomography on other qubits, for example [q1, q2]
qubits = [1, 2]
st.fit(qp, qc, qubits=qubits, method="lstsq", shots=4096, ptm=False)
print("Fidelity of the Bell state on qubits {} is: F = {:.5f}".format(qubits, st.fidelity))

# You can also perform quantum state tomography on other qubits, for example [q0, q2]
qubits = [0, 2]
st.fit(qp, qc, qubits=qubits, method="lstsq", shots=4096, ptm=False)
print("Fidelity of the Bell state on qubits {} is: F = {:.5f}".format(qubits, st.fidelity))
