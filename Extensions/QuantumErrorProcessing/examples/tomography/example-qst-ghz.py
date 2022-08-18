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
Example to demonstrate quantum state tomography on the four-qubit GHZ state.
"""
import qiskit
from qiskit.providers.fake_provider import FakeSantiago

import sys
sys.path.append('../..')

from QCompute import *
import qcompute_qep.tomography as tomography

# Set the token. You must set your VIP token in order to access the hardware.
Define.hubToken = "Token"

##########################################################################################
# Step 1. Setup the quantum program for preparing the four-qubit GHZ state
#         We aim to investigate how well the GHZ state is prepared.
##########################################################################################

qp = QEnv()  # qp is short for "quantum program", instance of QProgram
qp.Q.createList(4)
H(qp.Q[0])
CX(qp.Q[0], qp.Q[1])
CX(qp.Q[0], qp.Q[2])
CX(qp.Q[0], qp.Q[3])

##########################################################################################
# Step 2. Set the quantum computer (instance of QComputer).
#         The QuantumComputer can be a simulator or a hardware interface.
##########################################################################################
# For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
# qc = BackendName.LocalBaiduSim2

# For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian
# qc = BackendName.CloudBaiduQPUQian

# For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago
qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())

##########################################################################################
# Step 3. Perform State Tomography, check how well the Bell state
#         is implemented in the QuantumComputer.
##########################################################################################
# Initialize a StateTomography instance
st = tomography.StateTomography()
# Alternatively, you may initialize the StateTomography instance as follows:
# st = StateTomography(qp, qc, method='inverse', shots=8192)

# Call the tomography procedure and obtain the noisy quantum state
noisy_state = st.fit(qp, qc, method='lstsq', shots=8192, ptm=False)

# Compute the fidelity
print('Fidelity between the ideal and noisy states is: F = {:.5f}'.format(st.fidelity))
