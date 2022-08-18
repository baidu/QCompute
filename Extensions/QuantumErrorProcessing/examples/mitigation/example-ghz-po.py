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
Study the parity oscillations of the GHZ state on the noisy Superconducting quantum computer.
We use the gate error mitigation technique to improve the performance. Reference:

.. [MSB+11] Monz, Thomas, et al.
        "14-qubit entanglement: Creation and coherence."
        Physical Review Letters 106.13 (2011): 130506.
"""

import qiskit
from qiskit.providers.fake_provider import FakeSantiago
from qiskit.providers.aer.noise.errors.standard_errors import depolarizing_error
from qiskit.providers.aer.noise import NoiseModel
import copy
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib import rc
import pylab

import sys
sys.path.append('../..')

from QCompute import *
from QCompute.QPlatform.QOperation import RotationGate
from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.mitigation import Mitigator
from qcompute_qep.mitigation import ZNEMitigator
from qcompute_qep.utils import expval_z_from_counts
from qcompute_qep.utils.types import QComputer, QProgram, get_qc_name
from qcompute_qep.utils.circuit import execute

# Set the token. You must set your VIP token in order to access the hardware.
Define.hubToken = "Token"
# Set the default number of shots
NUMBER_OF_SHOTS = 8192


def calculator(qp: QEnv = None, qc: BackendName = None) -> float:
    """
    Run the quantum program on the quantum computer and estimate the expectation value.
    This function must be specified by the user.

    :param qp: QProgram, describes the quantum program
    :param qc: QComputer, specifies the quantum computer on which the quantum program runs
    :return: float, the evaluated expectation value
    """
    # Setup the noise model
    noise_model = NoiseModel()
    noise_model.add_all_qubit_quantum_error(depolarizing_error(0.1, 2), ['cx'])
    # Obtain the output raw counts
    counts = execute(qp, qc, noise_model=noise_model, shots=NUMBER_OF_SHOTS)
    # Compute the expectation value from counts w.r.t. to the observable :math:`Z\otimes Z`
    return expval_z_from_counts(counts)


def rotation_gate(phi: float) -> RotationGate.RotationGateOP:
    """
    Define the single-qubit rotation gate used in the "parity oscillations of GHZ state" protocol.
    Mathematically, the one-parameter rotation gate is defined as follows:

    .. math:: U(\\phi) := e^{i\\frac{\\pi}{4}\\left(\\cos\\phi\\sigma_x + \\sin\\phi\\sigma_y\\right)}

    To implement it in Quantum Leaf, we represent it using the native :math:`U_3` gate as:

    .. math:: U(\phi) := U_3(pi/2, \\phi + \\pi/2, - \\phi - \\pi/2)

    :param phi: float, the angle of the single-qubit rotation gate
    :return: RotationGateOP, the :math:`U_3` representation of the rotation gate
    """
    return RotationGate.U(math.pi / 2, phi + math.pi / 2, - phi - math.pi / 2)


def setup_po_circuit(n: int, phi: float) -> QProgram:
    """
    Given the number of qubits of the GHZ state and the rotation angle,
    setup a quantum program that creates the target GHZ state, with rotations added to the end.

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the parity oscillation angle
    :return: QProgram, the quantum program that creates the GHZ state
    """
    qp = QEnv()
    qp.Q.createList(n)
    # Setup the GHZ state generating quantum circuit
    H(qp.Q[0])
    for i in range(0, n - 1):
        CX(qp.Q[i], qp.Q[i + 1])

    # Add the rotation gates layer
    U = rotation_gate(phi)
    for i in range(0, n):
        U(qp.Q[i])

    # Measure the qubits
    MeasureZ(*qp.Q.toListPair())

    return qp


def theo_parity_oscillation(n: int, phi: float) -> float:
    """
    Given the number of qubits of the GHZ state and the rotation angle, compute theoretically the parity oscillation.

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the parity oscillation angle
    :return: float, the theoretical parity oscillation value
    """
    r = n % 4
    if r == 0:
        po = math.cos(n * phi)
    elif r == 1:
        po = math.sin(n * phi)
    elif r == 2:
        po = - math.cos(n * phi)
    elif r == 3:
        po = - math.sin(n * phi)
    else:
        raise ArgumentError("in theo_parity_oscillation(): the number of qubits {} is invalid!".format(n))
    return po


def parity_oscillation(n: int, phi: float, qc: QComputer = BackendName.LocalBaiduSim2) -> float:
    """
    Given the number of qubits of the GHZ state and the rotation angle, estimate its parity oscillation
    on the given quantum computer by collecting the outcomes and estimating the expectation value.

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the parity oscillation angle
    :param qc: QComputer, defaults to BackendName.LocalBaiduSim2,
        the quantum computer on which the GHZ state's parity oscillation is evaluated
    :return: float, the estimated parity oscillation value
    """
    qp = setup_po_circuit(n, phi)
    return calculator(copy.deepcopy(qp), qc)


def parity_oscillation_zne(n: int,
                           phi: float,
                           qc: QComputer = BackendName.CloudBaiduQPUQian,
                           mitigator: Mitigator = None) -> float:
    """
    Given the number of qubits of the GHZ state and the rotation angle, estimate its parity oscillation
    on the given quantum computer by collecting the outcomes and estimating the expectation value.
    To improve the estimation accuracy, we use the ZNE method to mitigate the quantum gate noise.

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the parity oscillation angle
    :param qc: QComputer, defaults to BackendName.LocalBaiduSim2,
        the quantum computer on which the GHZ state's parity oscillation is evaluated
    :param mitigator: Mitigator, the quantum gate error mitigator
    :return: float, the estimated parity oscillation value
    """
    qp = setup_po_circuit(n, phi)
    # Use the quantum gate error mitigator to mitigate the quantum gate noise
    po = mitigator.mitigate(copy.deepcopy(qp), qc, calculator, scale_factors=[1, 3, 5, 7, 9])
    return po


if __name__ == '__main__':

    n = 4
    START = 0
    STOP = 2 * math.pi / n
    phi_list = np.linspace(start=START, stop=STOP, num=100, endpoint=True, dtype=float)

    #######################################################################################################################
    # Set the quantum hardware for estimating the parity oscillation.
    #######################################################################################################################
    # For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
    qc = BackendName.LocalBaiduSim2

    # For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian
    # qc = BackendName.CloudBaiduQPUQian

    # For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago
    qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())
    qc_name = get_qc_name(qc)

    # Compute the theoretical parity oscillation values
    theo_po_list = [theo_parity_oscillation(n, phi) for phi in phi_list]

    # Compute the noisy parity oscillation values
    noisy_po_list = [parity_oscillation(n, phi, qc) for phi in phi_list]

    # Use different quantum gate error mitigation techniques to improve the estimation accuracy
    zne_mitigators = [ZNEMitigator(folder='circuit', extrapolator='linear'),
                      ZNEMitigator(folder='layer', extrapolator='linear'),
                      ZNEMitigator(folder='gate', extrapolator='linear'),
                      ZNEMitigator(folder='circuit', extrapolator='richardson'),
                      ZNEMitigator(folder='layer', extrapolator='richardson'),
                      ZNEMitigator(folder='gate', extrapolator='richardson')]

    mitigated_values = []
    for mitigator in zne_mitigators:
        print("====== Current Mitigator is {} ======".format(mitigator.__str__()))
        mitigated_values.append([parity_oscillation_zne(n, phi, qc=qc, mitigator=mitigator) for phi in phi_list])

    zne_mitigator_names = ['Circuit+Linear', 'Layer+Linear', 'Gate+Linear',
                           'Circuit+Richard', 'Layer+Richard', 'Gate+Richard',
                           'Circuit+Exp', 'Layer+Exp', 'Gate+Exp']
    ###################################################################################################################
    # The following plot visualizes the ideal, noisy, and corrected expectation values
    ###################################################################################################################
    plt.figure()
    ax = plt.gca()

    # Set the colormap: https://matplotlib.org/stable/tutorials/colors/colormaps.html
    cm = pylab.get_cmap('tab20')  # Dark2, Accent, Paired
    # Get the complete list of available markers and delete the 'pixel' marker
    markers = Line2D.markers
    markers.pop(',')
    markers = iter(Line2D.markers.keys())

    # Plot the theoretical reference line
    plt.plot(phi_list, theo_po_list, '-', color='red', alpha=0.8, linewidth=1, label='Theoretical', zorder=1)

    # Plot the noisy parity oscillation values
    plt.scatter(phi_list, noisy_po_list,
                marker=next(markers),
                color=cm(0),
                edgecolors='none',
                alpha=1.0,
                label=qc_name,
                s=32,
                zorder=2)

    # Plot the error mitigated parity oscillation values
    for i in range(len(zne_mitigators)):
        plt.scatter(phi_list, mitigated_values[i],
                    marker=next(markers),
                    color=cm(i + 1),
                    edgecolors='none',
                    alpha=0.9,
                    label=zne_mitigator_names[i],
                    s=16,
                    zorder=2)

    # Define the xticklables
    # https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
    ax = plt.gca()
    ax.set_xticks(np.arange(START, STOP + 0.01, math.pi / 16))
    labels = ['$0$', r'$\pi/16$', r'$\pi/8$', r'$3\pi/16$', r'$\pi/4$',
              r'$5\pi/16$', r'$3\pi/8$', r'$7\pi/16$', r'$\pi/2$']
    ax.set_xticklabels(labels)

    # Add the theoretical reference line
    plt.axhline(y=0, color='black', linestyle='-.', linewidth=1, zorder=1)

    # Give x and y axis labels
    plt.xlabel(r'Rotation Angle $\phi$', fontsize=14)
    plt.ylabel(r'Parity Oscillation Value', fontsize=14)
    # Legend
    plt.legend(loc='best', fontsize="small")

    plt.savefig("GHZ_PO_{}_GEM_N{}.png".format(get_qc_name(qc), n),
                format='png',
                dpi=600,
                bbox_inches='tight',
                pad_inches=0.1)

    plt.show()
