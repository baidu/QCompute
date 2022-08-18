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
In this script, we recover the identity equivalent single-qubit Clifford sequences
example given in Figure 2.a of [KTC+19]_. We use the Zero-Noise Extrapolation
error mitigation technique to improve the accuracy of the estimated expectation value.

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
from qcompute_qep.utils import expval_from_counts, decompose_yzy
from qcompute_qep.quantum import clifford
from qcompute_qep.mitigation import ZNEMitigator
from qcompute_qep.mitigation.utils import plot_zne_sequences
from qcompute_qep.utils.circuit import execute

# Set the token. You must set your VIP token in order to access the hardware.
Define.hubToken = "Token"


def calculator(qp: QEnv = None, qc: BackendName = None) -> float:
    """
    The expectation value calculator for the single Clifford circuit example.

    :param qp: instance of `QEnv`, describes the quantum program
    :param qc: instance of `backend`, specifies the quantum computer
    :return: the estimated expectation value
    """
    counts = execute(qp, qc)
    # Set the quantum observable
    A = np.diag([1, 0]).astype(complex)
    return expval_from_counts(A, counts)


# Maximal length of the sequence
num_seq = 10
# The default folding coefficients
scale_factors = [1, 2]

# Initialize different ZNE mitigators
zne_linear = ZNEMitigator(folder="circuit", extrapolator='linear')
zne_richard = ZNEMitigator(folder="circuit", extrapolator='richardson')

# Set the ideal quantum computer to `LocalBaiduSim2` and the noisy quantum computer to `CloudBaiduQPUQian`
qc_ideal = QCompute.BackendName.LocalBaiduSim2
# qc_noisy = QCompute.BackendName.CloudBaiduQPUQian
# For numeric test on the noisy simulator, change qc_noisy to Qiskit's FakeSantiago
qc_noisy = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())

expvals_ideal = []  # (num_seq, )
expvals_noisy_linear = []  # (num_seq, num_scales)
expvals_noisy_richard = []  # (num_seq, num_scales)
expvals_miti_linear = []  # (num_seq, )
expvals_miti_richard = []  # (num_seq, )

# Construct the quantum programs for each length
for i in range(1, num_seq + 1):
    qp = QEnv()
    qp.Q.createList(1)
    qubits = [0]

    # Randomly generate and operate a list of Clifford gates
    cliff_seq = clifford.random_clifford(n=len(qubits), m=i)
    inv_cir = []

    for c in cliff_seq:
        c(qp.Q, qubits=qubits)
        inv_cir.append(c.get_inverse_circuit(qubits=qubits))

    # Operate the inverse gates, making the overall circuit ideally an identity-equivalent circuit
    qp.circuit += sum(inv_cir[::-1], [])
    MeasureZ(*qp.Q.toListPair())

    # Compute the ideal expectation value using the ideal quantum computer
    val = calculator(copy.deepcopy(qp), qc_ideal)
    expvals_ideal.append(val)

    # Compute the noisy and mitigated expectation values using the noisy quantum computer
    val = zne_linear.mitigate(qp, qc_noisy, calculator, scale_factors=scale_factors)
    expvals_miti_linear.append(val)
    expvals_noisy_linear.append(zne_linear.history['expectations'])

    val = zne_richard.mitigate(qp, qc_noisy, calculator, scale_factors=scale_factors)
    expvals_miti_richard.append(val)
    expvals_noisy_richard.append(zne_richard.history['expectations'])

# Visualize the data. Convert the data to the required format for visualization
expvals_ideal = np.array(expvals_ideal).transpose()
expvals_miti_linear = np.array(expvals_miti_linear).transpose()
expvals_noisy_linear = np.array(expvals_noisy_linear).transpose()
expvals_miti_richard = np.array(expvals_miti_richard).transpose()
expvals_noisy_richard = np.array(expvals_noisy_richard).transpose()

# 1. Linear Extrapolation result
plot_zne_sequences(expvals_ideal,
                   expvals_miti_linear,
                   expvals_noisy_linear,
                   scale_factors,
                   title='clifford-single-linear')

# 2. Richardson Extrapolation result
plot_zne_sequences(expvals_ideal,
                   expvals_miti_richard,
                   expvals_noisy_richard,
                   scale_factors,
                   title='clifford-single-richardson')
