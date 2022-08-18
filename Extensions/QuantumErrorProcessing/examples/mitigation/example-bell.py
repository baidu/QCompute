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
Using the Bell State as an example to illustrate the folding technique
implemented in the Zero-Noise Extrapolation method.
"""
import sys
sys.path.append('../..')

from QCompute import *
import qcompute_qep.mitigation as mitigation
import qcompute_qep.utils.circuit as circuit

##########################################################################################
# Step 1. Setup the quantum program for preparing the Bell state
#         We aim to investigate how well the Bell state is prepared.
##########################################################################################
# qp is short for "quantum program", instance of QProgram
qp = QEnv()
qp.Q.createList(2)
H(qp.Q[0])
CX(qp.Q[0], qp.Q[1])
MeasureZ(*qp.Q.toListPair())

# Print the quantum circuit
circuit.print_circuit(qp.circuit)

##########################################################################################
# Step 2. Perform the folding technique on various levels.
##########################################################################################
print("*****************************************************************************")
print("Illustrating the circuit-level folder in the Zero-Noise Extrapolation method.")
# Initialize a ZNEMitigator instance with circuit-level folder
zne = mitigation.ZNEMitigator(folder='circuit', extrapolator='linear')

folded_qp = zne.folder.fold(qp=qp, scale_factor=3, method='left')

circuit.print_circuit(folded_qp.circuit)

folded_qp = zne.folder.fold(qp=qp, scale_factor=5, method='left')

circuit.print_circuit(folded_qp.circuit)

print("*****************************************************************************")
print("Illustrating the layer-level folder in the Zero-Noise Extrapolation method.")
# Initialize a ZNEMitigator instance with layer-level folder
zne = mitigation.ZNEMitigator(folder='layer', extrapolator='linear')

folded_qp = zne.folder.fold(qp=qp, scale_factor=3, method='left')

circuit.print_circuit(folded_qp.circuit)

folded_qp = zne.folder.fold(qp=qp, scale_factor=5, method='left')

circuit.print_circuit(folded_qp.circuit)

print("*****************************************************************************")
print("Illustrating the gate-level folder in the Zero-Noise Extrapolation method.")
# Initialize a ZNEMitigator instance with gate-level folder
zne = mitigation.ZNEMitigator(folder='gate', extrapolator='linear')

folded_qp = zne.folder.fold(qp=qp, scale_factor=3, method='left')

circuit.print_circuit(folded_qp.circuit)

folded_qp = zne.folder.fold(qp=qp, scale_factor=5, method='left')

circuit.print_circuit(folded_qp.circuit)
