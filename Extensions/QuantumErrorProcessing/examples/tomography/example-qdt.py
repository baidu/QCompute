# !/usr/bin/python3
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
Example to demonstrate quantum detector tomography.
"""

from QCompute import *
import qcompute_qep.tomography.detector_tomography as detector_tomography


##########################################################################################
# Step 1. Setup the quantum program for measurement
##########################################################################################

qp = QEnv()  # qp is short for "quantum program", instance of QProgram
qp.Q.createList(3)

##########################################################################################
# Step 2. Set the quantum computer (instance of QComputer).
#         The QuantumComputer can be a simulator or a hardware interface.
##########################################################################################
# For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
qc = BackendName.LocalBaiduSim2

# For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian.
# You must set your VIP token first in order to access the Baidu hardware.
# Define.hubToken = "Token"
# qc = BackendName.CloudBaiduQPUQian

# For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago
# from qiskit.providers.fake_provider import FakeSantiago
# qc = FakeSantiago()

##########################################################################################
# Step 3. Perform Detector Tomography, check how well the measurement
#         is implemented in the QuantumComputer.
##########################################################################################
# Initialize a DetectorTomography instance
qdt = detector_tomography.DetectorTomography()
# Alternatively, you may initialize the StateTomography instance as follows:
# qdt = DetectorTomography(qp, qc, method='inverse', shots=4096)

# Call the tomography procedure and obtain the noisy quantum state
meas = qdt.fit(qp, qc, method='mle', shots=4096, ptm=False, tol=1e-5, qubits=[1, 2])

# Compute the fidelity
print('Measurement fidelity: F = {:.5f}'.format(qdt.fidelity))

# meas1 = qdt.fit(qp, qc, method='mle', shots=4096, ptm=False, tol=1e-5)
# print('Fidelity between the ideal and noisy measurement is: F = {:.5f}'.format(qdt.fidelity))
# The main difference lies in that the former specifies the list of target qubits while the latter
# will tomography the full list of qubits (which are three qubits in total).
# You can tell the difference comparing `meas` and `meas1`.

# Visualize the POVM elements
detector_tomography.visualization(meas[0])
