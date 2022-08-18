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
In this example, we recover the quantum trajectories in Bloch sphere example given in Figure 2.b of [KTC+19]_.
We use the ``qutip`` package to plot the Bloch sphere [QuTiP]_.

References:

.. [KTC+19] Kandala, A., et al.
            "Error mitigation extends the computational reach of a noisy quantum processor."
            Nature 567.7749 (2019): 491-495.

.. [QuTiP] Plotting on the Bloch Sphere.
           https://qutip.org/docs/latest/guide/guide-bloch.html#animating-with-the-bloch-sphere
"""
import math
from copy import deepcopy
import numpy as np
import qutip
import matplotlib.pyplot as plt
from matplotlib import pylab

import qiskit
from qiskit.providers.fake_provider import FakeSantiago

import sys
sys.path.append('../..')

from QCompute import *
from QCompute.QPlatform.QOperation.RotationGate import RX, RZ
from qcompute_qep.exceptions import ArgumentError
from qcompute_qep.mitigation import ZNEMitigator
from qcompute_qep.utils import execute, expval_z_from_counts
from qcompute_qep.utils.circuit import remove_measurement
from qcompute_qep.utils.types import QProgram, QComputer

# Set the token. You must set your VIP token in order to access the hardware.
Define.hubToken = "Token"

# Define the Pauli operators
__PAULI_OPERATORS__ = ['X', 'Y', 'Z']


def pauli_meas(qp: QProgram, pauli: str) -> None:
    r"""Add Pauli measurement to the quantum circuit.

    Use `---MEAS---` to denote a measurement in the computational basis. We need to append a `Pauli Basis`
    transformation circuit to the original quantum circuit in order to convert the Pauli basis measurement
    to the computational basis measurement. This procedure is illustrated as follows:

    ::

           ---------------------   -----------------
        ---|                   |---|               |---MEAS---
        ---|  Quantum Circuit  |---|  Pauli Basis  |---MEAS---
        ---|                   |---|               |---MEAS---
           ---------------------   -----------------

    :param qp: QProgram, the quantum circuit that generates the quantum state
    :param pauli: str, the single-qubit Pauli name. Valid names are {'X', 'Y', 'Z'}
    :return: None
    """
    # Remove the old measurement operations if exists
    remove_measurement(qp)
    # Add the Pauli basis transformation circuit
    if pauli == 'I' or pauli == 'Z':
        pass
    elif pauli == 'X':
        H(qp.Q[0])
    elif pauli == 'Y':
        SDG(qp.Q[0])
        H(qp.Q[0])
    else:
        raise ArgumentError("in pauli_meas(): valid pauli names are {}.".format(__PAULI_OPERATORS__))
    # Add the computational basis measurement
    MeasureZ(*qp.Q.toListPair())
    pass


def xz_rotation(qp: QProgram, k: int, n: int) -> None:
    r"""Apply a XZ-rotation gate to the quantum state.

    The single-qubit XZ-rotation gate determined by index :math:`k` and period :math:`n` is defined as:

        .. math:: U_k = R_z\left(\frac{4k\pi}{n}\right)
                        R_x\left(\frac{\pi}{n})
                        R_x\left(\frac{-(k-1)\pi}{n})
                        R_z\left(\frac{-4(k-1)\pi}{n}\right).

    :param qp: QProgram, the quantum circuit that generates the quantum state
    :param k: int, the index of the XZ rotation gate. See the above equation for details
    :param n: int, the period of the XZ rotation gate. See the above equation for details
    :return: None
    """
    if k > n:
        raise ArgumentError("The index {} must be less than the period {}!".format(k, n))

    # Do not apply any gate if k = 0
    if k == 0:
        return

    # Apply the single-qubit XZ-rotation gate
    RZ(-4 * (k-1) * math.pi / n)(qp.Q[0])
    RX(- (k-1) * math.pi / n)(qp.Q[0])
    RX(k * math.pi / n)(qp.Q[0])
    RZ(4 * k * math.pi / n)(qp.Q[0])

    pass


def calculator(qp: QProgram, qc: QComputer) -> float:
    """Calculate the expectation value of the quantum program.

    More precisely, the expectation value is estimated in the following steps:

    1. Run the quantum program on the quantum computer and collect the counts.

    2. Estimate the expectation value from the counts.

    :param qp: QProgram, the quantum circuit describing a quantum task
    :param qc: QComputer, the quantum computer
    :return: float, the estimated expectation value
    """
    counts = execute(qp, qc, shots=8192)
    return expval_z_from_counts(counts)


if __name__ == '__main__':

    # Set the period
    period = 100

    # Setup the single-qubit quantum circuit
    qp = QEnv()
    qp.Q.createList(1)

    # Setup noisy and ideal quantum computers
    qc_noisy = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())
    qc_ideal = BackendName.LocalBaiduSim2

    # Initialize the ZNE mitigator
    zne = ZNEMitigator(folder="circuit", extrapolator='richardson')

    # Set the scaling factors
    scale_factors = [1, 3, 5, 7]

    # Store the ideal, noisy, and mitigated expectation values
    pauli_ideal = np.zeros((period, len(__PAULI_OPERATORS__)), dtype=float)
    pauli_noisy = np.zeros((period, len(__PAULI_OPERATORS__), len(scale_factors)), dtype=float)
    pauli_mitig = np.zeros((period, len(__PAULI_OPERATORS__)), dtype=float)

    for k in range(period):

        # Apply the single-qubit XZ-rotation gate of index k to the original quantum circuit
        xz_rotation(qp, k=k, n=period)

        for i, pauli in enumerate(__PAULI_OPERATORS__):

            # Use temporary quantum circuit to estimate expectation value of Pauli operator.
            # !!!WARNING!!! Do NOT forget to deepcopy the original quantum circuit otherwise it is only a reference.
            # The temporary quantum circuit will be destroyed after the Pauli measurement.
            qp_pauli = deepcopy(qp)

            # Add the Pauli measurement
            pauli_meas(qp_pauli, pauli)

            # Estimate the ideal expectation value of the Pauli operator
            pauli_ideal[k][i] = calculator(deepcopy(qp_pauli), qc_ideal)

            # Estimate the noise-rescaled expectation value of the Pauli operator for each rescale factor
            for j, factor in enumerate(scale_factors):

                # Rescale the quantum circuit and estimate the corresponding expectation value
                qp_rescaled = zne.folder.fold(qp=qp_pauli, scale_factor=factor)
                pauli_noisy[k][i][j] = calculator(qp_rescaled, qc_noisy)

            # Extrapolate to obtain the mitigated expectation value of the Pauli operator
            pauli_mitig[k][i] = zne.extrapolator.extrapolate(xdata=scale_factors, ydata=pauli_noisy[k][i][:].tolist())

    # Visualize the ideal, noisy, and mitigated expectation values in the Bloch sphere
    fig = plt.figure(figsize=(8, 8))
    b = qutip.Bloch(fig=fig)
    b.view = [-40, 30]
    # Set the colors for different points
    cm = pylab.get_cmap('Set1')  # Dark2, Accent, Paired, set1
    b.point_color = [cm(i) for i in range(10)]

    # Add the ideal expectation values to the Bloch sphere
    b.add_points([pauli_ideal[:, 0], pauli_ideal[:, 1], pauli_ideal[:, 2]])
    # Add the noisy expectation values to the Bloch sphere
    for j, factor in enumerate(scale_factors):
        b.add_points([pauli_noisy[:, 0, j], pauli_noisy[:, 1, j], pauli_noisy[:, 2, j]])
    # Add the noisy expectation values to the Bloch sphere
    b.add_points([pauli_mitig[:, 0], pauli_mitig[:, 1], pauli_mitig[:, 2]])

    b.render()  # render to the correct subplot

    plt.savefig("Block-Sphere.png",
                format='png',
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.1)

    plt.show()
