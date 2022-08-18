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
import numpy as np

import sys
sys.path.append('../..')

import QCompute
import qcompute_qep.tomography as tomography
import QCompute.QPlatform.QOperation as QOperation
import qcompute_qep.utils.types as types


def gateset_tomography(qc: types.QComputer, gate_set: tomography.GateSet):
    qc_name = types.get_qc_name(qc)

    tomo = tomography.GateSetTomography()

    gate_set = tomo.fit(qc, gate_set=gate_set)

    for i, key in enumerate(gate_set.noisy_gate.keys()):
        print(key)
        # print(gate_set.noisy_gate[key])
        ideal = tomo.result['ideal gateset'][i]
        noisy = gate_set.noisy_gate[key]
        tomography.compare_process_ptm(ptms=[ideal, noisy, ideal - noisy],
                                       titles=['Simulator', qc_name, 'Difference'],
                                       show_labels=True,
                                       fig_name="QPT-{}-{}.png".format(qc_name, key))
        print("The average gate fidelity between these two PTMs is: {}".format(gate_set.fidelity[key]))
        print("**************************")

    print("The state fidelity is: {}".format(gate_set.fidelity['rho']))
    print("***************")
    print("The measurement fidelity is: {}".format(gate_set.fidelity['meas']))


#######################################################################################################################
# Set the quantum hardware in process tomography
#######################################################################################################################
# For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
qc = QCompute.BackendName.LocalBaiduSim2

# For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian
# qc = QCompute.BackendName.CloudBaiduQPUQian

# For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago
# qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())


#######################################################################################################################
# Example 1.
#######################################################################################################################
prep_circuit = []
meas_circuit = []

rx_90 = QOperation.CircuitLine(QCompute.RX(np.pi / 2), [0])
ry_90 = QOperation.CircuitLine(QCompute.RY(np.pi / 2), [0])

gate_set = tomography.GateSet(prep_circuit, meas_circuit,
                              gates={'rx_90_ex0': rx_90, 'ry_90_ex0': ry_90},
                              prep_gates=[['rx_90_ex0'], ['ry_90_ex0'], ['rx_90_ex0', 'rx_90_ex0']],
                              meas_gates=[['rx_90_ex0'], ['ry_90_ex0'], ['rx_90_ex0', 'rx_90_ex0']])

gateset_tomography(qc, gate_set)

#######################################################################################################################
# Example 2.
#######################################################################################################################
prep_circuit = []
meas_circuit = [QOperation.CircuitLine(QCompute.X, [0])]


rx_90 = QOperation.CircuitLine(QCompute.RX(np.pi / 2), [0])
ry_90 = QOperation.CircuitLine(QCompute.RY(np.pi / 2), [0])
rx_180 = QOperation.CircuitLine(QCompute.RX(np.pi), [0])

gate_set = tomography.GateSet(prep_circuit, meas_circuit,
                              gates={'rx_90_ex1': rx_90, 'ry_90_ex1': ry_90, 'rx_180_ex1': rx_180},
                              prep_gates=[['rx_90_ex1'], ['ry_90_ex1'], ['rx_180_ex1']],
                              meas_gates=[['rx_90_ex1'], ['ry_90_ex1'], ['rx_180_ex1']])

gateset_tomography(qc, gate_set)
#######################################################################################################################
# Example 3.
#######################################################################################################################
prep_circuit = []
meas_circuit = []

rx_90 = QOperation.CircuitLine(QCompute.RX(np.pi / 2), [0])
ry_90 = QOperation.CircuitLine(QCompute.RY(np.pi / 2), [0])
i = QOperation.CircuitLine(QCompute.ID, [0])

gate_set = tomography.GateSet(prep_circuit, meas_circuit,
                              gates={'rx_90_ex2': rx_90, 'ry_90_ex2': ry_90, 'i_ex2': i},
                              prep_gates=[['rx_90_ex2'], ['ry_90_ex2'],
                                          ['rx_90_ex2', 'rx_90_ex2'],
                                          ['rx_90_ex2', 'rx_90_ex2', 'rx_90_ex2'],
                                          ['ry_90_ex2', 'ry_90_ex2', 'ry_90_ex2']],
                              meas_gates=[['rx_90_ex2'], ['ry_90_ex2'],
                                          ['rx_90_ex2', 'rx_90_ex2'],
                                          ['rx_90_ex2', 'rx_90_ex2', 'rx_90_ex2'],
                                          ['ry_90_ex2', 'ry_90_ex2', 'ry_90_ex2']])

gateset_tomography(qc, gate_set)


#######################################################################################################################
# Example 4.
#######################################################################################################################
prep_circuit = []
meas_circuit = []

Gix = QOperation.CircuitLine(data=QCompute.RX(np.pi / 2), qRegList=[0])
Gxi = QOperation.CircuitLine(data=QCompute.RX(np.pi / 2), qRegList=[1])
Giy = QOperation.CircuitLine(data=QCompute.RY(np.pi / 2), qRegList=[0])
Gyi = QOperation.CircuitLine(data=QCompute.RY(np.pi / 2), qRegList=[1])
Gcnot = QOperation.CircuitLine(data=QCompute.CX, qRegList=[0, 1])

prep_gates = [['Gix'], ['Giy'], ['Gix', 'Gix'], ['Gxi'], ['Gxi', 'Gix'], ['Gxi', 'Giy'], ['Gxi', 'Gix', 'Gix'],
              ['Gyi'], ['Gyi', 'Gix'], ['Gyi', 'Giy'], ['Gyi', 'Gix', 'Gix'],
              ['Gxi', 'Gxi'], ['Gxi', 'Gxi', 'Gix'], ['Gxi', 'Gxi', 'Giy'], ['Gxi', 'Gxi', 'Gix', 'Gix']]
meas_gates = [['Gix'], ['Giy'], ['Gix', 'Gix'], ['Gxi'], ['Gxi', 'Gix'], ['Gxi', 'Giy'], ['Gxi', 'Gix', 'Gix'],
              ['Gyi'], ['Gyi', 'Gix'], ['Gyi', 'Giy'], ['Gyi', 'Gix', 'Gix'],
              ['Gxi', 'Gxi'], ['Gxi', 'Gxi', 'Gix'], ['Gxi', 'Gxi', 'Giy'], ['Gxi', 'Gxi', 'Gix', 'Gix']]

gate_set = tomography.GateSet(prep_circuit, meas_circuit,
                              gates={'Gix': Gix, 'Gxi': Gxi, 'Giy': Giy, 'Gyi': Gyi, 'Gcnot': Gcnot},
                              prep_gates=prep_gates,
                              meas_gates=meas_gates)

gateset_tomography(qc, gate_set)
