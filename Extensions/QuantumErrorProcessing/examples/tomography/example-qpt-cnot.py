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
Example to demonstrate quantum process tomography on the CNOT gate.
"""
from QCompute import *
import qcompute_qep.tomography as tomography
import qcompute_qep.utils.types as types


# Step 1. Set the quantum program for the CNOT gate.
qp = QEnv()  # qp is short for "quantum program", instance of QProgram
qp.Q.createList(2)
# We manually decompose the CNOT gate using the CZ gate
H(qp.Q[0])
CZ(qp.Q[1], qp.Q[0])
H(qp.Q[0])

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

qc_name = types.get_qc_name(qc)

# Step 3. Perform Quantum Process Tomography, check how well the CNOT gate is implemented in the QuantumComputer.
qpt = tomography.ProcessTomography()

# Perform quantum process tomography on qubits [q0, q1]
qubits = [0, 1]
noisy_ptm = qpt.fit(qp, qc, qubits=qubits, prep_basis='Pauli', meas_basis='Pauli',
                    method='inverse', shots=4096, ptm=True).data
print('Fidelity of the CNOT gate on qubits {} is: F = {:.5f}'.format(qubits, qpt.fidelity))

# Visualize the Pauli transfer matrices of the theoretical and noisy quantum gates
tomography.compare_process_ptm(ptms=[qpt.ideal_ptm, noisy_ptm.data, qpt.ideal_ptm - noisy_ptm],
                               titles=['Theoretical', qc_name, 'Difference'],
                               show_labels=True,
                               fig_name="QPT-CNOT-qubits{}-{}.png".format(qubits, qc_name))

# You can also perform quantum process tomography on other qubits, for example [q1, q2]
qubits = [1, 2]
noisy_ptm = qpt.fit(qp, qc, qubits=qubits, prep_basis='Pauli', meas_basis='Pauli',
                    method='inverse', shots=4096, ptm=True).data
print('Fidelity of the CNOT gate on qubits {} is: F = {:.5f}'.format(qubits, qpt.fidelity))

# Visualize the Pauli transfer matrices of the theoretical and noisy quantum gates
tomography.compare_process_ptm(ptms=[qpt.ideal_ptm, noisy_ptm.data, qpt.ideal_ptm - noisy_ptm],
                               titles=['Theoretical', qc_name, 'Difference'],
                               show_labels=True,
                               fig_name="QPT-CNOT-qubits{}-{}.png".format(qubits, qc_name))
