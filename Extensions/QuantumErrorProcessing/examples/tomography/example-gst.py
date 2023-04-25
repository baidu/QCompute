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
from typing import List

import QCompute
import qcompute_qep.tomography as tomography
import qcompute_qep.utils.types as types


def gateset_tomography(qc: types.QComputer, gate_set: tomography.GateSet, qubits: List[int]):
    qc_name = types.get_qc_name(qc)
    tomo = tomography.GateSetTomography()
    gate_set = tomo.fit(qc, gate_set=gate_set, qubits=qubits)
    print("*******************************************************************************")
    print("GateSet Tomography Done. Information:")
    print("+ GateSet name: {}".format(gate_set.name))
    print("+ Quantum computer name: {}".format(types.get_qc_name(qc)))
    print("+ Working qubits: {}".format(qubits))
    # Show the information of the tomographic quantum gates
    for name in gate_set.gate_names:
        # Get the ideal gate (in PTM representation)
        ideal = gate_set.gateset_ptm[name]
        # Get the noisy gate (in PTM representation)
        noisy = gate_set.gateset_opt[name]
        tomography.compare_process_ptm(ptms=[ideal, noisy, ideal - noisy],
                                       titles=['Ideal', qc_name, 'Difference'],
                                       show_labels=True,
                                       fig_name="GST-{}-{}.png".format(qc_name, name))
        print("+ Average gate fidelity of gate {}: {:.3f}".format(name, gate_set.fidelity(name)))

    print("+ Preparation state fidelity: {:.3f}".format(gate_set.fidelity('rho')))
    print("+ Computational basis measurement fidelity: {:.3f}".format(gate_set.fidelity('meas')))
    print("*******************************************************************************")


#######################################################################################################################
# Set the quantum hardware in process tomography
#######################################################################################################################
# For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
qc = QCompute.BackendName.LocalBaiduSim2

# For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian
# QCompute.Define.hubToken = ""
# qc = QCompute.BackendName.CloudBaiduQPUQian

# For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago
# import qiskit
# from qiskit.providers.fake_provider import FakeSantiago
# qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())

#######################################################################################################################
# Example 1: Single-qubit gate set tomography
#######################################################################################################################
gate_set = tomography.STD1Q_GATESET_RXRY
# Other available single-qubit gate sets
# gate_set = tomography.STD1Q_GATESET_RXRYID
# gate_set = tomography.STD1Q_GATESET_RXRYRX
gateset_tomography(qc=qc, gate_set=gate_set, qubits=[1])

#######################################################################################################################
# Example 1: Two-qubit gate set tomography
#######################################################################################################################
gate_set = tomography.STD2Q_GATESET_RXRYCZ
# Other available two-qubit gate sets
# gate_set = tomography.STD2Q_GATESET_RXRYCX
# gate_set = tomography.STD2Q_GATESET_RXRYSWAP
gateset_tomography(qc=qc, gate_set=gate_set, qubits=[2, 4])
