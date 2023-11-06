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
An example to demonstrate the Direct Fidelity Estimation protocol on the Bell state.
"""
import time
import QCompute
import Extensions.QuantumErrorProcessing.qcompute_qep.estimation as estimation


# Step 1. Set up the quantum program for preparing the Bell state
# We aim to investigate how well the Bell state is prepared.
qp = QCompute.QEnv()
qp.Q.createList(2)
QCompute.H(qp.Q[0])
QCompute.CX(qp.Q[0], qp.Q[1])

# Step 2. Set the quantum computer (instance of QComputer).
# The QuantumComputer can be a simulator or a hardware interface.

# For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
qc = QCompute.BackendName.LocalBaiduSim2

# For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian
# You must set your VIP token first in order to access the Baidu hardware.
# QCompute.Define.hubToken = "Token"
# qc = QCompute.BackendName.CloudBaiduQPUQian

# For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago simulator
# from qiskit.providers.fake_provider import FakeSantiago
# qc = FakeSantiago()

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))

# Step 3. Execute direct fidelity estimation,
# check how well the Bell state is implemented in the QuantumComputer.
dfe = estimation.DFEState()

# Call the fidelity estimation procedure and obtain the noisy quantum state
qubits = [0, 1]
dfe.estimate(qp, qc, qubits=qubits)
print("Fidelity of the Bell state on qubits {} is: F = {:.5f}".format(qubits, dfe.fidelity))

# You can also perform the fidelity estimation procedure on other qubits, for example [q1, q2]
qubits = [1, 2]
dfe.estimate(qp, qc, qubits=qubits)
print("Fidelity of the Bell state on qubits {} is: F = {:.5f}".format(qubits, dfe.fidelity))

# You can also set the estimation error and failure probability
qubits = [0, 2]
dfe.estimate(qp, qc, qubits=qubits, eps=0.025, delta=0.025)
print("Fidelity of the Bell state on qubits {} is: F = {:.5f}".format(qubits, dfe.fidelity))

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time())))
