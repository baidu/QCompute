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
In this script, we recover the identity equivalent two-qubit Clifford sequences
example applied on a Bell State given in Figure 2.c of [KTC+19]_.
We use the Zero-Noise Extrapolation error mitigation technique to improve the
accuracy of the estimated expectation value.

References:

.. [KTC+19] Kandala, A., et al.
            "Error mitigation extends the computational reach of a noisy quantum processor."
            Nature 567.7749 (2019): 491-495.
"""
import copy
import qiskit
from qiskit.providers.fake_provider import FakeSantiago
import numpy as np

import sys
sys.path.append('../..')

import QCompute
from QCompute import *
from QCompute.QPlatform.QOperation import FixedGate
from qcompute_qep.utils import expval_from_counts
from qcompute_qep.quantum import clifford
from qcompute_qep.mitigation.utils import plot_zne_sequences
from qcompute_qep.mitigation import ZNEMitigator
from qcompute_qep.utils.types import QProgram, QComputer
from qcompute_qep.utils.circuit import execute

# Set the token. You must set your VIP token in order to access the hardware.
Define.hubToken = "Token"


def calculator(qp: QProgram = None, qc: QComputer = None) -> float:
    """
    The expectation value calculator for the two-qubit Clifford sequence example.

    :param qp: QProgram, describes the quantum program
    :param qc: QComputer, specifies the quantum computer
    :return: float, the estimated expectation value
    """
    counts = execute(qp, qc)
    # Set the quantum observable
    A = np.diag([1, -1, -1, 1]).astype(complex)
    return expval_from_counts(A, counts)


# Maximal length of the sequence
num_seq = 10
# The default folding coefficients
scale_factors = [1, 2]

# Initialize the ZNE mitigator
zne = ZNEMitigator(folder='circuit', extrapolator='richardson')

# Set the ideal quantum computer to `LocalBaiduSim2` and the noisy quantum computer to `CloudBaiduQPUQian`
qc_ideal = QCompute.BackendName.LocalBaiduSim2
# qc_noisy = QCompute.BackendName.CloudBaiduQPUQian
# For numeric test on the noisy simulator, change qc_noisy to Qiskit's FakeSantiago
qc_noisy = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())

expvals_ideal = []  # (num_seq, )
expvals_miti = []  # (num_seq, )
expvals_noisy = []  # (num_seq, num_scales)

# Construct the quantum programs
for i in range(1, num_seq + 1):

    qp = QEnv()
    qp.Q.createList(2)
    qubits = [0, 1]
    # Initial operations: Hadamard + CNOT
    FixedGate.H(qp.Q[0])
    FixedGate.CX(qp.Q[0], qp.Q[1])

    # Randomly generate and operate a list of Clifford gates
    cliff_seq = clifford.random_clifford(n=len(qubits), m=i)
    inv_cir = []
    for c in cliff_seq:
        c(qp.Q, qubits=qubits)
        inv_cir.append(c.get_inverse_circuit(qubits=qubits))

    # Operate the inverse gate, making the circuit ideally an identity-equivalent circuit
    qp.circuit += sum(inv_cir[::-1], [])
    MeasureZ(*qp.Q.toListPair())

    # Compute the ideal expectation value using the ideal quantum computer (the simulator)
    val = calculator(copy.deepcopy(qp), qc_ideal)
    expvals_ideal.append(val)

    # Compute the noisy and mitigated expectation values using the noisy quantum computer
    val = zne.mitigate(qp, qc_noisy, calculator, scale_factors=scale_factors)
    expvals_miti.append(val)
    expvals_noisy.append(zne.history['expectations'])

# Visualize the data. Convert the data to the required format for visualization
expvals_ideal = np.array(expvals_ideal).transpose()
expvals_miti = np.array(expvals_miti).transpose()
expvals_noisy = np.array(expvals_noisy).transpose()

print('===' * 10, 'Quantum Results:', '===' * 10)
print('Sequence size:\t', list(range(1, num_seq + 1)))
print('Ideal expectation values:\t', expvals_ideal)
print('Noisy expectation values:\t', expvals_noisy)
print('Mitigated expectation values:\t,', expvals_miti)

# plot the results
fig = plot_zne_sequences(expvals_ideal,
                         expvals_miti,
                         expvals_noisy,
                         scale_factors=scale_factors,
                         title='clifford-two')
