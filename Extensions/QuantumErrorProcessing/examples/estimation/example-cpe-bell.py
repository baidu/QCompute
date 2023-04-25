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
An example to demonstrate the Cross-Platform Estimation of Quantum States protocol.
"""

import QCompute
import qcompute_qep.estimation.cpe_state as cp

# Step 1:Read the data of sampled unitaries and measurement results collected from
# different quantum devices
# ideal_baidu1 = QuantumSnapshot(qc_name='Baidu ideal1',
#                                qc=QCompute.BackendName.LocalBaiduSim2,
#                                qubits=[0])

ideal_baidu1 = cp.read_quantum_snapshot('Baidu ideal1_08_15_15', 'baidu ideal1')

# ideal_baidu2 = QuantumSnapshot(qc_name='Baidu ideal2',
#                                qc=QCompute.BackendName.LocalBaiduSim2,
#                                qubits=[0])

ideal_baidu2 = cp.read_quantum_snapshot('Baidu ideal2_08_15_16', 'baidu ideal2')

ideal_baidu3 = cp.read_quantum_snapshot('Baidu ideal3_08_16_09', 'baidu ideal3')

dev_list = [ideal_baidu1,
            ideal_baidu2,
            ideal_baidu3,
            # read_quantum_snapshot('Baidu ideal3_08_16_09', 'baidu ideal4'),
            # read_quantum_snapshot('Baidu ideal1_08_15_15', 'baidu ideal5'),
            # read_quantum_snapshot('Baidu ideal2_08_15_16', 'baidu ideal6'),
            ]
# Step 2: Set up the quantum program
qp = QCompute.QEnv()
qp.Q.createList(1)

# Step 3: Perform cross-platform estimation of different quantum devices.
est = cp.CPEState()
result = est.estimate(dev_list, qp, samples=100, shots=50, show=True, filename='test.png')

print(result)

