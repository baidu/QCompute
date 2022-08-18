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
import qiskit
from qiskit.providers.fake_provider import FakeSantiago
import numpy as np

import sys
sys.path.append('../..')

import QCompute
import qcompute_qep.tomography as tomography
import qcompute_qep.quantum.pauli as pauli
import qcompute_qep.utils.circuit
import qcompute_qep.utils.types as types
import qcompute_qep.quantum.metrics as metrics

# Set the token. You must set your VIP token in order to access the hardware.
QCompute.Define.hubToken = "Token"

##########################################################################################
# Step 1. Setup the quantum program for the CNOT gate.
##########################################################################################

qp = QCompute.QEnv()  # qp is short for "quantum program", instance of QProgram
qp.Q.createList(3)

# Manually decompose the CNOT gate using the CZ gate, where CNOT: q2 -> q1
QCompute.H(qp.Q[1])
QCompute.CZ(qp.Q[2], qp.Q[1])
QCompute.H(qp.Q[1])

##########################################################################################
# Step 2. Set the quantum computer (instance of QComputer).
#         The QuantumComputer can be a simulator or a hardware interface.
##########################################################################################
# For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
qc = QCompute.BackendName.LocalBaiduSim2

# For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian
# qc = QCompute.BackendName.CloudBaiduQPUQian

# For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago
# qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())

qc_name = types.get_qc_name(qc)
##########################################################################################
# Step 3. Perform Quantum Process Tomography, check how well the CNOT gate
#         is implemented in the QuantumComputer.
##########################################################################################

# Initialize a ProcessTomography instance
st = tomography.ProcessTomography()
# Call the tomography procedure and obtain the noisy CZ gate
noisy_ptm = st.fit(qp, qc, qubits=[1, 2], prep_basis='Pauli', meas_basis='Pauli',
                   method='inverse', shots=8192, ptm=True).data

# Compute numerically the ideal CNOT for reference
ideal_ptm = st.ideal_ptm

##########################################################################################
# Step 4. Analyze the experimental data.
##########################################################################################
# To check the correctness of the tomography results, we compute the distance between these two PTMs
# Step 3. Analyze the data: compute the average gate fidelity of two quantum maps
print("****** The average gate fidelity between these two PTMs is: {}".format(st.fidelity))
# Visualize these PTMs
diff_ptm = ideal_ptm - noisy_ptm
tomography.compare_process_ptm(ptms=[ideal_ptm, noisy_ptm.data, diff_ptm],
                               titles=['Simulator', qc_name, 'Difference'],
                               show_labels=True,
                               fig_name="QPT-CNOT-{}.png".format(qc_name))