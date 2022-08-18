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

.. [WLS+20] Wei, Ken X., et al.
    "Verifying multipartite entangled Greenberger-Horne-Zeilinger states via multiple quantum coherences."
    Physical Review A 101.3 (2020): 032343.
"""
import copy
import qiskit
from typing import Tuple
from qiskit.providers.fake_provider import FakeSantiago
import numpy as np
import functools
import math
import matplotlib.pyplot as plt
from matplotlib import rc, pylab

import sys
sys.path.append('../..')

import QCompute
from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.measurement import InverseCorrector, LeastSquareCorrector, IBUCorrector, NeumannCorrector
from qcompute_qep.measurement.correction import vector2dict, dict2vector
from qcompute_qep.measurement.utils import plot_histograms
from qcompute_qep.utils import expval_from_counts
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
    n = number_of_qubits(qp)

    # Compute the expectation value from counts w.r.t. to the observable :math:`|0\cdots 0><0\cdots 0|`
    proj0 = np.array([[1, 0], [0, 0]]).astype(complex)
    o = functools.reduce(np.kron, [proj0] * n)

    return expval_from_counts(o, counts), counts


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

    # Compute the expectation value from counts w.r.t. to the observable :math:`|0\cdots 0><0\cdots 0|`
    proj0 = np.array([[1, 0], [0, 0]]).astype(complex)
    o = functools.reduce(np.kron, [proj0] * n)

    return expval_from_counts(o, corr_counts), corr_counts


def setup_mqc_circuit(n: int, phi: float) -> QProgram:
    """
    Set up the quantum program that estimates the MQC of a GHZ state.

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the rotation angle
    :return: QProgram, the quantum program that estimates the MQC of a GHZ state
    """
    qp = QCompute.QEnv()
    qp.Q.createList(n)

    # Prepare the GHZ state
    QCompute.H(qp.Q[0])
    for i in range(0, n - 1):
        QCompute.CX(qp.Q[i], qp.Q[i + 1])

    # Add the rotation gates layer
    for i in range(0, n):
        QCompute.RZ(phi)(qp.Q[i])

    # Reverse the preparation procedure
    for i in reversed(range(0, n - 1)):
        QCompute.CX(qp.Q[i], qp.Q[i + 1])
    QCompute.H(qp.Q[0])

    # Measure
    QCompute.MeasureZ(*qp.Q.toListPair())

    return qp


def theo_mqc(n: int, phi: float) -> Tuple[float, dict]:
    """
    Given the number of qubits of the GHZ state and the rotation angle, compute theoretically the MQC.

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the rotation angle
    :return: float, the theoretically MQC value
    """
    val = (1 + math.cos(n*phi))/2

    # Compute the ideal counts
    qp = setup_mqc_circuit(n, phi)
    state = circuit_to_state(qp)
    counts_v = np.rint(np.diag(np.real(state)) * NUMBER_OF_SHOTS)
    counts = vector2dict(counts_v)

    # Compute the expectation value from counts w.r.t. to the observable :math:`|0\cdots 0><0\cdots 0|`
    proj0 = np.array([[1, 0], [0, 0]]).astype(complex)
    o = functools.reduce(np.kron, [proj0] * n)

    if not math.isclose(val, expval_from_counts(o, counts), rel_tol=0.05):
        print("val = {}, expval_from_counts(counts) = {}".format(val, expval_from_counts(o, counts)))
        raise ArgumentError("in theo_mqc(): the theoretical counts are incorrect!")
    return val, counts


def mqc(n: int, phi: float, qc: QComputer = QCompute.BackendName.LocalBaiduSim2) -> Tuple[float, dict]:
    """
    Given the number of qubits of the GHZ state and the rotation angle,
    estimate its MQC on the given quantum computer.

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the rotation angle
    :param qc: QComputer, defaults to BackendName.LocalBaiduSim2,
        the quantum computer on which the GHZ state's MQC is evaluated
    :return: float, the estimated MQC value
    """
    qp = setup_mqc_circuit(n, phi)
    return calculator(copy.deepcopy(qp), qc)


def corrected_mqc(n: int, phi: float, qc: QComputer = QCompute.BackendName.CloudBaiduQPUQian, method: str = 'least square')\
        -> Tuple[float, dict]:
    """
    Given the number of qubits of the GHZ state and the rotation angle,
    estimate its MQC on the given quantum computer.
    We use the MEM (measurement error mitigation) method to mitigate the quantum measurement noise
    and improve the estimation precision.

    :param n: int, the number of qubits of the GHZ state
    :param phi: float, the rotation angle
    :param qc: QComputer, defaults to BackendName.LocalBaiduSim2,
        the quantum computer on which the GHZ state's MQC is evaluated
    :param method: str, the measurement error correction method
    :return: float, the estimated MQC value
    """
    qp = setup_mqc_circuit(n, phi)
    return corrected_calculator(copy.deepcopy(qp), qc, method=method)


if __name__ == '__main__':

    n = 4
    START = 0
    STOP = 2 * math.pi / n
    phi_list = np.linspace(start=START, stop=STOP, num=100, endpoint=True, dtype=float)

    theo_mqc_list = []
    theo_counts_list = []
    noisy_mqc_list = []
    noisy_counts_list = []
    noisy_euc_list = []
    inv_mqc_list = []
    inv_counts_list = []
    inv_euc_list = []
    ls_mqc_list = []
    ls_counts_list = []
    ls_euc_list = []
    ibu_mqc_list = []
    ibu_counts_list = []
    ibu_euc_list = []
    neu_mqc_list = []
    neu_counts_list = []
    neu_euc_list = []

    #######################################################################################################################
    # Set the quantum hardware for estimating the multiple quantum coherence.
    #######################################################################################################################
    # For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
    qc = QCompute.BackendName.LocalBaiduSim2

    # For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian
    # qc = QCompute.BackendName.CloudBaiduQPUQian

    # For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago
    # qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())

    # Generate the dataset
    for phi in phi_list:
        theo_val, theo_counts = theo_mqc(n, phi)
        theo_mqc_list.append(theo_val)
        theo_counts_list.append(theo_counts)

        noisy_val, noisy_counts = mqc(n, phi=phi, qc=qc)
        noisy_mqc_list.append(noisy_val)
        noisy_counts_list.append(noisy_counts)
        noisy_euc_list.append(np.linalg.norm((dict2vector(theo_counts) - dict2vector(noisy_counts))/NUMBER_OF_SHOTS))

        inv_val, inv_counts = corrected_mqc(n, phi=phi, qc=qc, method='inverse')
        inv_mqc_list.append(inv_val)
        inv_counts_list.append(inv_counts)
        inv_euc_list.append(np.linalg.norm((dict2vector(theo_counts) - dict2vector(inv_counts))/NUMBER_OF_SHOTS))

        ls_val, ls_counts = corrected_mqc(n, phi=phi, qc=qc, method='least square')
        ls_mqc_list.append(ls_val)
        ls_counts_list.append(ls_counts)
        ls_euc_list.append(np.linalg.norm((dict2vector(theo_counts) - dict2vector(ls_counts))/NUMBER_OF_SHOTS))

        ibu_val, ibu_counts = corrected_mqc(n, phi=phi, qc=qc, method='ibu')
        ibu_mqc_list.append(ibu_val)
        ibu_counts_list.append(ibu_counts)
        ibu_euc_list.append(np.linalg.norm((dict2vector(theo_counts) - dict2vector(ibu_counts))/NUMBER_OF_SHOTS))

        neu_val, neu_counts = corrected_mqc(n, phi=phi, qc=qc, method='neu')
        neu_mqc_list.append(neu_val)
        neu_counts_list.append(neu_counts)
        neu_euc_list.append(np.linalg.norm((dict2vector(theo_counts) - dict2vector(neu_counts))/NUMBER_OF_SHOTS))

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
            'simulator': cm(1.*0/NUM_COLORS),
            'hardware': cm(1.*1/NUM_COLORS),
            'inverse': cm(1.*2/NUM_COLORS),
            'least square': cm(1.*3/NUM_COLORS),
            'ibu': cm(1.*4/NUM_COLORS),
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
        plt.plot(phi_list, theo_mqc_list, '-', color='red', alpha=0.8, linewidth=1, label='Theoretical', zorder=1)

        # Plot the noisy result
        plt.scatter(phi_list, noisy_mqc_list,
                    marker=markers['hardware'],
                    color=colors['hardware'],
                    edgecolors='none',
                    alpha=0.8,
                    label="Santiago",
                    s=16,
                    zorder=2)

        # Plot the mitigated result using the inverse approach
        plt.scatter(phi_list, inv_mqc_list,
                    marker=markers['inverse'],
                    color=colors['inverse'],
                    edgecolors='none',
                    alpha=1.0,
                    label="Santiago Inverse",
                    s=18,
                    zorder=2)

        # Plot the mitigated result using the least square approach
        plt.scatter(phi_list, ls_mqc_list,
                    marker=markers['least square'],
                    color=colors['least square'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago LS",
                    s=18,
                    zorder=2)

        # Plot the mitigated result using the ibu approach
        plt.scatter(phi_list, ibu_mqc_list,
                    marker=markers['ibu'],
                    color=colors['ibu'],
                    edgecolors='none',
                    alpha=1,
                    s=18,
                    label="Santiago IBU",
                    zorder=2)

        # Plot the mitigated result using the Neumann approach
        plt.scatter(phi_list, neu_mqc_list,
                    marker=markers['neu'],
                    color=colors['neu'],
                    edgecolors='none',
                    alpha=1,
                    s=18,
                    label="Santiago Neumann",
                    zorder=2)

        # Define the xticklables
        # https://stackoverflow.com/questions/40642061/how-to-set-axis-ticks-in-multiples-of-pi-python-matplotlib
        ax = plt.gca()
        ax.set_xticks(np.arange(START, STOP+0.01, math.pi/16))
        labels = ['$0$', r'$\pi/16$', r'$\pi/8$', r'$3\pi/16$', r'$\pi/4$',
                  r'$5\pi/16$', r'$3\pi/8$', r'$7\pi/16$', r'$\pi/2$']
        ax.set_xticklabels(labels)

        # Add the y=0 reference line
        plt.hlines(y=0, xmin=START, xmax=STOP, color='black', linestyles='dotted', linewidth=1)

        # Give x and y axis labels
        plt.xlabel(r'Rotation Angle $\phi$', fontsize=14)
        plt.ylabel(r'Multiple Quantum Coherence', fontsize=14)
        # Legend
        plt.legend(loc='upper center')

        plt.savefig("GHZ_MQC_N={}.png".format(n),
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
        plt.scatter(x_range, sorted(noisy_euc_list),
                    marker=markers['hardware'],
                    color=colors['hardware'],
                    edgecolors='none',
                    alpha=0.75,
                    label="Santiago",
                    s=16,
                    zorder=2)

        # Plot the Inverse mitigated quantum computer result
        plt.scatter(x_range, sorted(inv_euc_list),
                    marker=markers['inverse'],
                    color=colors['inverse'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago Inverse",
                    s=18,
                    zorder=2)

        # Plot the MEM mitigated quantum computer result
        plt.scatter(x_range, sorted(ls_euc_list),
                    marker=markers['least square'],
                    color=colors['least square'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago LS",
                    s=18,
                    zorder=2)

        # Plot the IBU mitigated quantum computer result
        plt.scatter(x_range, sorted(ibu_euc_list),
                    marker=markers['ibu'],
                    color=colors['ibu'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago IBU",
                    s=18,
                    zorder=2)

        # Plot the Neumann mitigated quantum computer result
        plt.scatter(x_range, sorted(neu_euc_list),
                    marker=markers['neu'],
                    color=colors['neu'],
                    edgecolors='none',
                    alpha=1,
                    label="Santiago Neumann",
                    s=18,
                    zorder=2)

        # Give x and y axis labels
        plt.xlabel(r'Rotation Angle $\phi$', fontsize=14)
        plt.ylabel(r'Euclidean Distance', fontsize=14)
        # Legend
        plt.legend(loc='best')

        plt.savefig("GHZ_MQC2_N={}.png".format(n),
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
        vals_diff[0, :] = np.asarray(theo_mqc_list) - np.asarray(noisy_mqc_list)
        vals_diff[1, :] = np.asarray(theo_mqc_list) - np.asarray(inv_mqc_list)
        vals_diff[2, :] = np.asarray(theo_mqc_list) - np.asarray(ls_mqc_list)
        vals_diff[3, :] = np.asarray(theo_mqc_list) - np.asarray(ibu_mqc_list)
        vals_diff[4, :] = np.asarray(theo_mqc_list) - np.asarray(neu_mqc_list)

        legends = ['Theo - Noisy', 'Theo - INV', 'Theo - LSC', 'Theo - IBU', 'Theo - Neumann']
        plot_histograms(counts=vals_diff,
                        legends=['Theo - Noisy', 'Theo - INV', 'Theo - LSC', 'Theo - IBU', 'Theo - Neumann'],
                        binary_labels=False,
                        fig_name="GHZ_MQC_FakeSantiago_MEM2_N={}.png".format(n))

        print("Euclidean distance between theoretical and noisy values: {}".format(np.linalg.norm(vals_diff[0, :])))
        print("Euclidean distance between theoretical and inverse values: {}".format(np.linalg.norm(vals_diff[1, :])))
        print("Euclidean distance between theoretical and least square values: {}".format(
            np.linalg.norm(vals_diff[2, :])))
        print("Euclidean distance between theoretical and ibu values: {}".format(np.linalg.norm(vals_diff[3, :])))
        print("Euclidean distance between theoretical and neu values: {}".format(np.linalg.norm(vals_diff[4, :])))
    pass
