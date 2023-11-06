#!/usr/bin/python3
# -*- coding: utf8 -*-

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

r"""
This script contains two simple examples for showing Hamiltonian simulation.
"""

import numpy as np
from numpy import pi
from QCompute import X, H, S, SDG, RY, CZ, QEnv, BackendName, MeasureZ
from QCompute.Define import Settings as QC_Settings

from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation import (
    circ_HS_QSVT,
    func_HS_QSVT,
)

# not to draw the quantum circuit locally
QC_Settings.drawCircuitControl = []
QC_Settings.outputInfo = False
QC_Settings.autoClearOutputDirAfterFetchMeasure = True


def func_MS_test_QSVT():
    r"""Test MS gate as a Hamiltonian time evolution operator.

    Use the following identity

    .. math::

        (H\cdot S \otimes H \cdot S) \cdot CZ \cdot  (H \otimes H) \cdot e^{i\pi X\otimes X/4) = ID \otimes ID

    to test it.
    """
    list_str_Pauli_representation = [(1, "X0X1")]
    num_qubit_sys = 2
    float_tau = -np.pi / 4
    float_epsilon = 1e-6

    for idx in range(5):
        # create the quantum environment, can choose local backend
        env = QEnv()
        env.backend(BackendName.LocalBaiduSim2)

        # the other two ancilla qubit introduced from qubitization and QSP
        reg_ancilla = [env.Q[0], env.Q[1]]
        # compute the number of qubits needed in the block-encoding, and form a register
        num_qubit_blocking = max(1, int(np.ceil(np.log2(len(list_str_Pauli_representation)))))
        reg_blocking = list(env.Q[idx] for idx in range(2, 2 + num_qubit_blocking))
        # create the system register for the Hamiltonian
        reg_sys = list(env.Q[idx] for idx in range(2 + num_qubit_blocking, 2 + num_qubit_blocking + num_qubit_sys))

        # the following is the quantum circuit for the time evolution operator for the Hamiltonian
        if idx == 4:
            H(reg_sys[0])
            H(reg_sys[1])
        else:
            if idx // 2 == 1:
                X(reg_sys[0])
            if idx % 2 == 1:
                X(reg_sys[1])

        circ_HS_QSVT(reg_sys, reg_blocking, reg_ancilla, list_str_Pauli_representation, float_tau, float_epsilon)
        H(reg_sys[0])
        H(reg_sys[1])
        S(reg_sys[0])
        S(reg_sys[1])
        CZ(reg_sys[0], reg_sys[1])
        H(reg_sys[0])
        H(reg_sys[1])

        if idx == 4:
            H(reg_sys[0])
            H(reg_sys[1])
        else:
            if idx // 2 == 1:
                X(reg_sys[0])
            if idx % 2 == 1:
                X(reg_sys[1])

        # measure
        MeasureZ(*env.Q.toListPair())

        # commit
        assert env.commit(10000, downloadResult=True)["counts"]["00000"] == 10000
    print("MS test passed.")


def func_HH_test_QSVT():
    r"""Test MS gate as a Hamiltonian time evolution operator.

    Use the following identity

    .. math::

        (Ry(\pi/4)\otimes Ry(\pi/4)) \cdot CZ \cdot (SDG\cdot Ry(-\pi/4) \otimes SDG\cdot Ry(-\pi/4))
        \cdot e^{-i\pi H\otimes H/4) = ID \otimes ID

    to test it.
    """
    # create the quantum environment, can choose local backend
    env = QEnv()
    env.backend(BackendName.LocalBaiduSim2)

    list_str_Pauli_representation = [(0.5, "X0X1"), (0.5, "X0Z1"), (0.5, "Z0X1"), (0.5, "Z0Z1")]
    num_qubit_sys = 2
    float_tau = np.pi / 4
    float_epsilon = 1e-6

    for idx in range(5):
        # create the quantum environment, can choose local backend
        env = QEnv()
        env.backend(BackendName.LocalBaiduSim2)

        # the other two ancilla qubit introduced from qubitization and QSP
        reg_ancilla = [env.Q[0], env.Q[1]]
        # compute the number of qubits needed in the block-encoding, and form a register
        num_qubit_blocking = max(1, int(np.ceil(np.log2(len(list_str_Pauli_representation)))))
        reg_blocking = list(env.Q[idx] for idx in range(2, 2 + num_qubit_blocking))
        # create the system register for the Hamiltonian
        reg_sys = list(env.Q[idx] for idx in range(2 + num_qubit_blocking, 2 + num_qubit_blocking + num_qubit_sys))

        # the following is the quantum circuit for the time evolution operator for the Hamiltonian
        if idx == 4:
            H(reg_sys[0])
            H(reg_sys[1])
        else:
            if idx // 2 == 1:
                X(reg_sys[0])
            if idx % 2 == 1:
                X(reg_sys[1])

        circ_HS_QSVT(reg_sys, reg_blocking, reg_ancilla, list_str_Pauli_representation, float_tau, float_epsilon)
        RY(-pi / 4)(reg_sys[0])
        RY(-pi / 4)(reg_sys[1])
        SDG(reg_sys[0])
        SDG(reg_sys[1])
        CZ(reg_sys[0], reg_sys[1])
        RY(pi / 4)(reg_sys[0])
        RY(pi / 4)(reg_sys[1])

        if idx == 4:
            H(reg_sys[0])
            H(reg_sys[1])
        else:
            if idx // 2 == 1:
                X(reg_sys[0])
            if idx % 2 == 1:
                X(reg_sys[1])

        # measure
        MeasureZ(*env.Q.toListPair())

        # commit
        assert env.commit(10000, downloadResult=True)["counts"]["000000"] == 10000
    print("HH test passed.")


if __name__ == "__main__":
    func_MS_test_QSVT()
    func_HH_test_QSVT()
    print(
        func_HS_QSVT(
            list_str_Pauli_rep=[(1, "X0X1"), (1, "X0Z1"), (1, "Z0X1"), (1, "Z0Z1")],
            num_qubit_sys=2,
            float_tau=-pi / 8,
            float_epsilon=1e-6,
            circ_output=False,
        )
    )
    func_HS_QSVT(
        list_str_Pauli_rep=[(1, "X0X1"), (1, "X0Z1"), (1, "Z0X1"), (1, "Z0Z1")],
        num_qubit_sys=2,
        float_tau=-pi / 8,
        float_epsilon=1e-6,
        circ_output="HH_HS_Circuit.c",
    )
