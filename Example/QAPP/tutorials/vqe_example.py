# -*- coding: UTF-8 -*-
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
Solve LiH ground state energy with VQE
"""

import sys

sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

sys.path.append('..')
import numpy as np

from QCompute.QPlatform import BackendName
from qapp.application.chemistry import LiH_HAMILTONIAN, MolecularGroundStateEnergy
from qapp.algorithm import VQE
from qapp.circuit import RealEntangledCircuit
from qapp.optimizer import SMO


lih = MolecularGroundStateEnergy(num_qubits=LiH_HAMILTONIAN[0], hamiltonian=LiH_HAMILTONIAN[1])
DEPTH = 1
iteration = 3
# Initialize ansatz parameters
parameters = 2 * np.pi * np.random.rand(lih.num_qubits * DEPTH) - np.pi
ansatz = RealEntangledCircuit(lih.num_qubits, DEPTH, parameters)
# Initialize an optimizer
opt = SMO(iteration, ansatz)
# Choose a Pauli measurement method
measurement = 'SimMeasure'

# Fill in your Quantum-hub token if using cloud resources
# Define.hubToken = "your token"
backend = BackendName.LocalBaiduSim2
# Uncomment the line below to use a cloud simulator
# backend = BackendName.CloudBaiduSim2Water
# Uncomment the line below to use a real quantum device
# backend = BackendName.CloudIoPCAS
vqe = VQE(lih.num_qubits, lih.hamiltonian, ansatz, opt, backend, measurement=measurement)
vqe.run(shots=4096)
print("estimated ground state energy: {} Hartree".format(vqe.minimum_eigenvalue))
print("theoretical ground state energy: {} Hartree".format(lih.compute_ground_state_energy()))
