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
We use the measurement error mitigation technique to improve the performance. Reference:

.. [MSB+11] Monz, Thomas, et al.
        "14-qubit entanglement: Creation and coherence."
        Physical Review Letters 106.13 (2011): 130506.
"""
import copy
from typing import Tuple
import qiskit
from qiskit.providers.fake_provider import FakeSantiago
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import rc, pylab

import sys
sys.path.append('../..')

import QCompute
from QCompute.QPlatform.QOperation import RotationGate
from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.measurement import InverseCorrector, LeastSquareCorrector, IBUCorrector, NeumannCorrector
from qcompute_qep.measurement.correction import vector2dict, dict2vector
from qcompute_qep.measurement.utils import plot_histograms
from qcompute_qep.utils import expval_z_from_counts
from qcompute_qep.utils.types import QComputer, QProgram, number_of_qubits
from qcompute_qep.utils.circuit import execute, circuit_to_state

# Set the token. You must set your VIP token in order to access the hardware.
QCompute.Define.hubToken = "Token"
# Set the default number of shots
NUMBER_OF_SHOTS = 8192


def calculator(qp: QCompute.QEnv = None, qc: QCompute.BackendName = None) -> Tuple[float, dict]:
    """
    Run the quantum program on the quantum computer and estimate the expectation value.
    This function must be specified by the user.

    :param qp: QProgram, describes the quantum program
    :param qc: QComputer, specifies the quantum computer on which the quantum program runs
    :return: Tuple[float, dict], the evaluated expectation value and the counts dict
    """
    # Obtain the output raw counts
    counts = execute(qp, qc, shots=NUMBER_OF_SHOTS)
    # Compute the expectation value from counts w.r.t. to the observable :math:`Z\otimes Z`
    return expval_z_from_counts(counts), counts


def corrected_calculator(qp: QCompute.QEnv = None,
                         qc: QCompute.BackendName = None,
                         method: str = 'least square') -> Tuple[float, dict]:
    """
    Run the quantum program on the quantum computer and estimate the expectation value.
    We use the MEM (measurement error mitigation) method to mitigate the quantum measurement noise
    and improve the estimation precision.
    This function must be specified by the user.

    :param qp: QProgram, describes the quantum program
    :param qc: QComputer, specifies the quantum computer on which the quantum program runs
    :param method: str, the measurement error correction method
    :return: Tuple[float, dict], the evaluated expectation value and the counts dict
    """
    # Obtain the output raw counts
    counts = execute(qp, qc, shots=NUMBER_OF_SHOTS)
    # Correct the counts
    n = number_of_qubits(qp)
    # Correct the counts
    if method.lower() == 'inverse':  # Case-insensitive
        corr = InverseCorrector(qc, calibrator='complete', qubits=range(n))
    elif method.lower() == 'least square':  # Case-insensitive
        corr = LeastSquareCorrector(qc, calibrator='complete', qubits=range(n))
    elif method.lower() == 'ibu':  # Case-insensitive
        corr = IBUCorrector(qc, calibrator='complete', qubits=range(n))
    elif method.lower() == 'neu':  # Case-insensitive
        corr = NeumannCorrector(qc, calibrator='complete', qubits=range(n))
    else:
        raise ArgumentError("Corrector with name {} is not defined!".format(method))

    corr_counts = corr.correct(counts)
    # Compute the expectation value from counts w.r.t. to the observable :math:`Z\otimes Z`
    return expval_z_from_counts(corr_counts), corr_counts


def rotation_gate(phi: float) -> RotationGate.RotationGateOP:
    r"""
    Define the single-qubit rotation gate used in the "parity oscillations of GHZ state" protocol.
    Mathematically, the one-parameter rotation gate is defined as follows:

    :math:`U(\phi) := e^{i\frac{\pi}{4}\left(\cos\phi\sigma_x + \sin\phi\sigma_y\right)}`

    To implement it in Quantum Leaf, we represent it using the native :math:`U_3` gate as:

    :math:`U(\phi) := U_3(pi/2, \phi + \pi/2, - \phi - \pi/2)`

    :param phi: float, the angle of the single-qubit rotation gate
    :return: RotationGateOP, the :math:`U_3` representation of the rotation gate
    """
    return RotationGate.U(math.pi / 2, phi + math.pi / 2, - phi - math.pi / 2)


def setup_po_circuit(n: int, phi: float) -> QProgram:
    """
    Given the number of qubits of the GHZ state and the rotation angle,
    set up a quantum program that describes the GHZ circuit (with rotations added to the end).

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the parity oscillation angle
    :return: QProgram, the quantum program that creates the GHZ state
    """
    qp = QCompute.QEnv()
    qp.Q.createList(n)
    # Set up the GHZ state generating quantum circuit
    QCompute.H(qp.Q[0])
    for i in range(0, n - 1):
        QCompute.CX(qp.Q[i], qp.Q[i + 1])

    # Add the rotation gates layer
    QCompute.U = rotation_gate(phi)
    for i in range(0, n):
        QCompute.U(qp.Q[i])

    QCompute.MeasureZ(*qp.Q.toListPair())

    return qp


def theo_parity_oscillation(n: int, phi: float) -> Tuple[float, dict]:
    """
    Given the number of qubits of the GHZ state and the rotation angle, compute theoretically the parity oscillation.

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the parity oscillation angle
    :return: Tuple[float, dict], the theoretical parity oscillation value and the counts dict
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

    # Compute the ideal counts
    qp = setup_po_circuit(n, phi)
    state = circuit_to_state(qp)
    counts_v = np.rint(np.diag(np.real(state)) * NUMBER_OF_SHOTS)
    counts = vector2dict(counts_v)

    if not math.isclose(po, expval_z_from_counts(counts), rel_tol=0.05):
        print("po = {}, expval_z_from_counts(counts) = {}".format(po, expval_z_from_counts(counts)))
        raise ArgumentError("in theo_parity_oscillation(): the theoretical counts are incorrect!")
    return po, counts


def parity_oscillation(n: int, phi: float, qc: QComputer = QCompute.BackendName.LocalBaiduSim2) -> Tuple[float, dict]:
    """
    Given the number of qubits of the GHZ state and the rotation angle,
    estimate its parity oscillation on the given quantum computer.

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the parity oscillation angle
    :param qc: QComputer, defaults to BackendName.LocalBaiduSim2,
        the quantum computer on which the GHZ state's parity oscillation is evaluated
    :return: Tuple[float, dict], the evaluated expectation value and the counts dict
    """
    qp = setup_po_circuit(n, phi)
    return calculator(copy.deepcopy(qp), qc)


def corrected_parity_oscillation(n: int,
                                 phi: float,
                                 qc: QComputer = QCompute.BackendName.CloudBaiduQPUQian,
                                 method: str = 'least square') -> Tuple[float, dict]:
    """
    Given the number of qubits of the GHZ state and the rotation angle,
    estimate its parity oscillation on the given quantum computer.
    We use the MEM (measurement error mitigation) method to mitigate the quantum measurement noise
    and improve the estimation precision.

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the parity oscillation angle
    :param qc: QComputer, defaults to BackendName.LocalBaiduSim2,
        the quantum computer on which the GHZ state's parity oscillation is evaluated
    :param method: str, the measurement error correction method
    :return: Tuple[float, dict], the evaluated expectation value and the counts dict
    """
    qp = setup_po_circuit(n, phi)
    return corrected_calculator(copy.deepcopy(qp), qc, method=method)


if __name__ == '__main__':

    n = 4
    START = 0
    STOP = 2 * math.pi / n
    phi_list = np.linspace(start=START, stop=STOP, num=100, endpoint=True, dtype=float)

    theo_po_list = []
    theo_counts_list = []
    noisy_po_list = []
    noisy_counts_list = []
    noisy_val_diff_list = []
    noisy_euc_list = []
    inv_po_list = []
    inv_counts_list = []
    inv_val_diff_list = []
    inv_euc_list = []
    ls_po_list = []
    ls_counts_list = []
    ls_val_diff_list = []
    ls_euc_list = []
    ibu_po_list = []
    ibu_counts_list = []
    ibu_val_diff_list = []
    ibu_euc_list = []
    neu_po_list = []
    neu_counts_list = []
    neu_val_diff_list = []
    neu_euc_list = []

    #######################################################################################################################
    # Set the quantum hardware for estimating the parity oscillation.
    #######################################################################################################################
    # For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
    qc = QCompute.BackendName.LocalBaiduSim2

    # For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian
    # qc = QCompute.BackendName.CloudBaiduQPUQian

    # For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago
    # qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())

    for phi in phi_list:
        theo_val, theo_counts = theo_parity_oscillation(n, phi)
        theo_po_list.append(theo_val)
        theo_counts_list.append(theo_counts)

        noisy_val, noisy_counts = parity_oscillation(n, phi=phi, qc=qc)
        noisy_po_list.append(noisy_val)
        noisy_counts_list.append(noisy_counts)
        noisy_euc_list.append(np.linalg.norm((dict2vector(theo_counts) - dict2vector(noisy_counts))/NUMBER_OF_SHOTS))
        noisy_val_diff_list.append(abs(theo_val - noisy_val))

        inv_val, inv_counts = corrected_parity_oscillation(n, phi=phi, qc=qc, method='inverse')
        inv_po_list.append(inv_val)
        inv_counts_list.append(inv_counts)
        inv_euc_list.append(np.linalg.norm((dict2vector(theo_counts) - dict2vector(inv_counts))/NUMBER_OF_SHOTS))
        inv_val_diff_list.append(abs(theo_val - inv_val))

        ls_val, ls_counts = corrected_parity_oscillation(n, phi=phi, qc=qc, method='least square')
        ls_po_list.append(ls_val)
        ls_counts_list.append(ls_counts)
        ls_euc_list.append(np.linalg.norm((dict2vector(theo_counts) - dict2vector(ls_counts))/NUMBER_OF_SHOTS))
        ls_val_diff_list.append(abs(theo_val - ls_val))

        ibu_val, ibu_counts = corrected_parity_oscillation(n, phi=phi, qc=qc, method='ibu')
        ibu_po_list.append(ibu_val)
        ibu_counts_list.append(ibu_counts)
        ibu_euc_list.append(np.linalg.norm((dict2vector(theo_counts) - dict2vector(ibu_counts))/NUMBER_OF_SHOTS))
        ibu_val_diff_list.append(abs(theo_val - ibu_val))

        neu_val, neu_counts = corrected_parity_oscillation(n, phi=phi, qc=qc, method='neu')
        neu_po_list.append(neu_val)
        neu_counts_list.append(neu_counts)
        neu_euc_list.append(np.linalg.norm((dict2vector(theo_counts) - dict2vector(neu_counts))/NUMBER_OF_SHOTS))
        neu_val_diff_list.append(abs(theo_val - neu_val))

    ###################################################################################################################
    # The following plot visualizes the ideal, noisy, and corrected expectation values
    ###################################################################################################################
    VISUALIZE_EXPECTATION_VALUES = True
    if VISUALIZE_EXPECTATION_VALUES:
        plt.figure()
        ax = plt.gca()

        NUM_COLORS = 8
        cm = pylab.get_cmap('Paired')  # Dark2, Accent, Paired
        colors = {
            'simulator': cm(1. * 0 / NUM_COLORS),
            'hardware': cm(1. * 1 / NUM_COLORS),
            'inverse': cm(1. * 2 / NUM_COLORS),
            'least square': cm(1. * 3 / NUM_COLORS),
            'ibu': cm(1. * 4 / NUM_COLORS),
            'neu': cm(1. * 5 / NUM_COLORS)
        }
        markers = {
            'simulator': 'o',
            'hardware': 'o',
            'inverse': '<',
            'least square': '>',
            'ibu': '^',
            'neu': '*'
        }

        # Plot the theoretical reference line
        plt.plot(phi_list, theo_po_list, '-', color='red', alpha=0.8, linewidth=1, label='Theoretical', zorder=1)

        # Plot the noisy result
        plt.scatter(phi_list, noisy_po_list,
                    marker=markers['hardware'],
                    color=colors['hardware'],
                    edgecolors='none',
                    alpha=0.75,
                    label="Santiago",
                    s=16,
                    zorder=2)

        # Plot the mitigated result using the inverse approach
        plt.scatter(phi_list, inv_po_list,
                    marker=markers['inverse'],
                    color=colors['inverse'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago Inverse",
                    s=18,
                    zorder=2)

        # Plot the mitigated result using the least square approach
        plt.scatter(phi_list, ls_po_list,
                    marker=markers['least square'],
                    color=colors['least square'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago LS",
                    s=18,
                    zorder=2)

        # Plot the mitigated result using the ibu approach
        plt.scatter(phi_list, ibu_po_list,
                    marker=markers['ibu'],
                    color=colors['ibu'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago IBU",
                    s=18,
                    zorder=2)

        # Plot the mitigated result using the Neumann approach
        plt.scatter(phi_list, ibu_po_list,
                    marker=markers['neu'],
                    color=colors['neu'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago Neumann",
                    s=18,
                    zorder=2)

        # Define the xticklables
        # https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
        ax = plt.gca()
        ax.set_xticks(np.arange(START, STOP + 0.01, math.pi / 16))
        labels = ['$0$', r'$\pi/16$', r'$\pi/8$', r'$3\pi/16$', r'$\pi/4$',
                  r'$5\pi/16$', r'$3\pi/8$', r'$7\pi/16$', r'$\pi/2$']
        ax.set_xticklabels(labels)

        # Add the theoretical reference line
        plt.axhline(y=0, color='black', linestyle='-.', linewidth=1, label="Theoretical", zorder=1)

        # Give x and y axis labels
        plt.xlabel(r'Rotation Angle $\phi$', fontsize=14)
        plt.ylabel(r'Parity Oscillation Value', fontsize=14)
        # Legend
        plt.legend(loc='best')

        plt.savefig("GHZ_PO_N={}.png".format(n),
                    format='png',
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0.1)

        plt.show()
    pass
    ###################################################################################################################
    # The following plot visualizes the Euclidean distances between the ideal and noisy/corrected counts.
    ###################################################################################################################
    VISUALIZE_EUCLIDEAN_DISTANCES = True
    if VISUALIZE_EUCLIDEAN_DISTANCES:
        plt.figure()
        ax = plt.gca()

        NUM_COLORS = 8
        cm = pylab.get_cmap('Paired')  # Dark2, Accent, Paired
        colors = {
            'simulator': cm(1. * 0 / NUM_COLORS),
            'hardware': cm(1. * 1 / NUM_COLORS),
            'inverse': cm(1. * 2 / NUM_COLORS),
            'least square': cm(1. * 3 / NUM_COLORS),
            'ibu': cm(1. * 4 / NUM_COLORS),
            'neu': cm(1. * 5 / NUM_COLORS)
        }
        markers = {
            'simulator': 'o',
            'hardware': 'o',
            'inverse': '<',
            'least square': '>',
            'ibu': '^',
            'neu': '*'
        }

        x_range = range(1, len(phi_list)+1)
        # Plot the IBM Santiago quantum computer result

        indices = np.argsort(noisy_val_diff_list)
        plt.scatter(x_range, [noisy_euc_list[i] for i in indices],
                    marker=markers['hardware'],
                    color=colors['hardware'],
                    edgecolors='none',
                    alpha=0.75,
                    label="Santiago Euc",
                    s=16,
                    zorder=2)

        plt.scatter(x_range, [noisy_val_diff_list[i] for i in indices],
                    marker=markers['hardware'],
                    color=cm(1. * 5 / NUM_COLORS),
                    edgecolors='none',
                    alpha=0.75,
                    label="Santiago Diff",
                    s=16,
                    zorder=2)

        # Plot the Inverse mitigated quantum computer result
        indices = np.argsort(inv_val_diff_list)
        plt.scatter(x_range, [inv_euc_list[i] for i in indices],
                    marker=markers['inverse'],
                    color=colors['inverse'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago Inverse Euc",
                    s=18,
                    zorder=2)
        plt.scatter(x_range, [inv_val_diff_list[i] for i in indices],
                    marker=markers['inverse'],
                    color=cm(1. * 6 / NUM_COLORS),
                    edgecolors='none',
                    alpha=1,
                    label="Santiago Inverse Diff",
                    s=18,
                    zorder=2)

        # Plot the LS mitigated quantum computer result
        indices = np.argsort(ls_val_diff_list)
        plt.scatter(x_range, [ls_euc_list[i] for i in indices],
                    marker=markers['least square'],
                    color=colors['least square'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago LS Euc",
                    s=18,
                    zorder=2)
        plt.scatter(x_range, [ls_val_diff_list[i] for i in indices],
                    marker=markers['least square'],
                    color=cm(1. * 7 / NUM_COLORS),
                    edgecolors='none',
                    alpha=1,
                    label="Santiago LS Diff",
                    s=18,
                    zorder=2)

        # Plot the IBU mitigated quantum computer result
        indices = np.argsort(ibu_val_diff_list)
        plt.scatter(x_range, [ibu_euc_list[i] for i in indices],
                    marker=markers['ibu'],
                    color=colors['ibu'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago IBU Euc",
                    s=18,
                    zorder=2)
        plt.scatter(x_range, [ibu_val_diff_list[i] for i in indices],
                    marker=markers['ibu'],
                    color=cm(1. * 8 / NUM_COLORS),
                    edgecolors='none',
                    alpha=1,
                    label="Santiago IBU Diff",
                    s=18,
                    zorder=2)

        # Plot the Neumann mitigated quantum computer result
        indices = np.argsort(neu_val_diff_list)
        plt.scatter(x_range, [neu_euc_list[i] for i in indices],
                    marker=markers['neu'],
                    color=colors['neu'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago Neumann Euc",
                    s=18,
                    zorder=2)
        plt.scatter(x_range, [neu_val_diff_list[i] for i in indices],
                    marker=markers['neu'],
                    color=cm(1. * 9 / NUM_COLORS),
                    edgecolors='none',
                    alpha=1,
                    label="Santiago Neumann Diff",
                    s=18,
                    zorder=2)

        # Give x and y axis labels
        plt.xlabel(r'Rotation Angle $\phi$', fontsize=14)
        plt.ylabel(r'Euclidean Distance', fontsize=14)
        # Legend
        plt.legend(loc='best')

        plt.savefig("GHZ_PO2_N={}.png".format(n),
                    format='png',
                    dpi=600,
                    bbox_inches='tight',
                    pad_inches=0.1)

        plt.show()
    pass
    ###################################################################################################################
    # The following plot visualizes the Euclidean distances between the ideal and noisy/corrected expectation values.
    ###################################################################################################################
    VISUALIZE_EXP_EUCLIDEAN_DISTANCES = True
    if VISUALIZE_EXP_EUCLIDEAN_DISTANCES:
        vals_diff = np.zeros((5, len(phi_list)), dtype=float)
        vals_diff[0, :] = np.asarray(theo_po_list) - np.asarray(noisy_po_list)
        vals_diff[1, :] = np.asarray(theo_po_list) - np.asarray(inv_po_list)
        vals_diff[2, :] = np.asarray(theo_po_list) - np.asarray(ls_po_list)
        vals_diff[3, :] = np.asarray(theo_po_list) - np.asarray(ibu_po_list)
        vals_diff[4, :] = np.asarray(theo_po_list) - np.asarray(neu_po_list)

        legends = ['Theo - Noisy', 'Theo - INV', 'Theo - LSC', 'Theo - IBU', 'Theo - Neumann']
        plot_histograms(counts=vals_diff,
                        legends=['Theo - Noisy', 'Theo - INV', 'Theo - LSC', 'Theo - IBU', 'Theo - Neumann'],
                        binary_labels=False,
                        fig_name="GHZ_PO_FakeSantiago_MEM2_N={}.png".format(n))

        print("Euclidean distance between theoretical and noisy values: {}".format(np.linalg.norm(vals_diff[0, :])))
        print("Euclidean distance between theoretical and inverse values: {}".format(np.linalg.norm(vals_diff[1, :])))
        print("Euclidean distance between theoretical and least square values: {}".format(np.linalg.norm(
            vals_diff[2, :])))
        print("Euclidean distance between theoretical and ibu values: {}".format(np.linalg.norm(vals_diff[3, :])))
        print("Euclidean distance between theoretical and neu values: {}".format(np.linalg.norm(vals_diff[4, :])))
    pass
