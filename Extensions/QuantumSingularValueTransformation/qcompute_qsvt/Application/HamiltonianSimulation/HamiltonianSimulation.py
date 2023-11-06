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
This script implement the quantum circuit for Hamiltonian time evolution operator,
and a function to output such quantum circuit or measure a final state under such operator in **QCompute**.

Reader may refer to following reference for more insights.

.. [DMW+21] Dong, Yulong, et al. "Efficient phase-factor evaluation in quantum signal processing."
    Physical Review A 103.4 (2021): 042419.

.. [MRT+21] Martyn, John M., et al. "Grand unification of quantum algorithms." PRX Quantum 2.4 (2021): 040203.
"""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from QCompute import H, RY, CU, QEnv, BackendName, MeasureZ, UnrollCircuitModule, CompressGateModule, CircuitToQasm
from QCompute.QPlatform.QRegPool import QRegStorage

from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.QSVT import (
    circ_QSVT_from_BE,
    circ_QSVT_from_BE_inverse,
)
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation.SymmetricQSPHS import (
    func_LBFGS_QSP_HS,
)


def circ_HS_QSVT(
    reg_sys: List[QRegStorage],
    reg_blocking: List[QRegStorage],
    reg_ancilla: List[QRegStorage],
    list_str_Pauli_rep: List[Tuple[float, str]],
    float_tau: float,
    float_epsilon: float,
) -> None:
    r"""A quantum circuit implementing the Hamiltonian time evolution operator :math:`e^{-i\check H\tau}` on system
    register :math:`|s\rangle` with precision :math:`\epsilon`.

    It is proved that :math:`\operatorname{QSVT}` is a block-encoding of :math:`e^{-i\check H\tau}/4` defined in
    **qcompute_qsvt.QSVT.QSVT.circ_QSVT_from_BE**.
    We need to introduce an amplitude amplification to obtain a block-encoding of :math:`e^{-i\check H\tau}`,
    where we use QSVT algorithm one more time to implement a function mapping singular value :math:`1/4` into :math:`1`.

    :param reg_sys: :math:`|s\rangle`, `List[QRegStorage]`,
        the system register for block-encoding,
        indicates the quantum state to be operated Hamiltonian evolution operator
    :param reg_blocking: :math:`|a\rangle`, `List[QRegStorage]`,
        the ancilla register introduced in block-encoding step,
        should be at state :math:`|0\rangle` before this circuit is operated
    :param reg_ancilla: :math:`|c\rangle|b\rangle`, `List[QRegStorage]` of length :math:`2`,
        a register of two ancilla qubits introduced in QSVT,
        should be at state :math:`|0\rangle` before this circuit is operated
    :param list_str_Pauli_rep: :math:`\check H`, `List[Tuple[float, str]]`,
        a list of form :math:`(a_j, P_j)` such as
        `[(-0.09706, \'I\'), (-0.045302, \'X0X1Y2Y3\'), (0.17441, \'Z2Z3\'), (-0.2234, \'Z3\')]`,
        where :math:`a_j` is a float indicating the coefficient of :math:`P_j` in :math:`\check H`,
        and :math:`P_j` is a string indicating which multi-Pauli gate
    :param float_tau: :math:`\tau`, `float`, the simulation time in Hamiltonian simulation
    :param float_epsilon: :math:`\epsilon`, `float`, the simulation precision in Hamiltonian simulation
    :return: **None**
    """
    float_norm_square = sum(abs(idx_cor) for (idx_cor, idx_str) in list_str_Pauli_rep)
    list_float_target_state = list(
        np.sqrt(abs(idx_cor) / float_norm_square) for (idx_cor, idx_str) in list_str_Pauli_rep
    )
    # the evolution time should be multiplied by the normalised constant
    float_tau *= float_norm_square

    # Compute the processing parameters
    list_re, list_im = func_LBFGS_QSP_HS(float_tau, float_epsilon)
    # for tau = -np.pi / 4 in func_MS_gate_test_QSVT(), we have following approximating processing parameters
    # list_re = [2.921010576373295, 1.6524642686243183, 2.921010576373295]
    # list_im = [0.7803500265368747, 0.18564952439004778, 0.18564952439004778, 0.7803500265368747]

    # in QSVT convention, we need to modify those processing parameters and then fill them in QSVT circuit
    for idx in range(len(list_re)):
        list_re[idx] -= np.pi / 2
    for idx in range(len(list_im)):
        list_im[idx] -= np.pi / 2
    list_re[0] += np.pi / 4
    list_re[-1] += np.pi / 4
    list_im[0] += np.pi / 4
    list_im[-1] += np.pi / 4

    # the partial modified processing parameters used in amplitude amplification
    vec_Phi_AA = np.array((-np.arccos(1 / 3), -np.arccos(7 / 9), 0, 0, np.arccos(7 / 9), np.arccos(1 / 3))) - np.pi

    H(reg_ancilla[0])
    H(reg_ancilla[1])
    circ_QSVT_from_BE(
        reg_sys,
        reg_blocking,
        reg_ancilla[1],
        reg_ancilla[0],
        list_str_Pauli_rep,
        list_float_target_state,
        list_re,
        list_im,
    )

    for idx in range(3):
        RY(np.pi / 2)(reg_ancilla[0])
        RY(np.pi / 2)(reg_ancilla[1])
        CU(0, 0, vec_Phi_AA[-idx * 2 - 1])(reg_ancilla[0], reg_ancilla[1])
        RY(-np.pi / 2)(reg_ancilla[0])
        RY(-np.pi / 2)(reg_ancilla[1])

        circ_QSVT_from_BE_inverse(
            reg_sys,
            reg_blocking,
            reg_ancilla[1],
            reg_ancilla[0],
            list_str_Pauli_rep,
            list_float_target_state,
            list_re,
            list_im,
        )
        RY(np.pi / 2)(reg_ancilla[0])
        RY(np.pi / 2)(reg_ancilla[1])
        CU(0, 0, vec_Phi_AA[-idx * 2 - 2])(reg_ancilla[0], reg_ancilla[1])
        RY(-np.pi / 2)(reg_ancilla[0])
        RY(-np.pi / 2)(reg_ancilla[1])

        circ_QSVT_from_BE(
            reg_sys,
            reg_blocking,
            reg_ancilla[1],
            reg_ancilla[0],
            list_str_Pauli_rep,
            list_float_target_state,
            list_re,
            list_im,
        )

    H(reg_ancilla[0])
    H(reg_ancilla[1])


def func_HS_QSVT(
    list_str_Pauli_rep: List[Tuple[float, str]],
    num_qubit_sys: int,
    float_tau: float,
    float_epsilon: float,
    circ_output: Union[str, bool] = False,
) -> Optional[Dict[str, Union[str, Dict[str, int]]]]:
    r"""Operate the time evolution operator on initial state, and return the measurement of the final state.

    :param list_str_Pauli_rep: :math:`\check H`, `List[Tuple[float, str]]`,
        a list of form :math:`(a_j, P_j)` such as
        `[(-0.09706, \'I\'), (-0.045302, \'X0X1Y2Y3\'), (0.17441, \'Z2Z3\'), (-0.2234, \'Z3\')]`,
        where :math:`a_j` is a float indicating the coefficient of :math:`P_j` in :math:`\check H`,
        and :math:`P_j` is a string indicating which multi-Pauli gate
    :param num_qubit_sys: `int`,
        should be positive, the number of qubits needed in the Pauli representation :math:`\check H`
    :param float_tau: :math:`\tau`, `float`, the simulation time in Hamiltonian simulation
    :param float_epsilon: :math:`\epsilon`, `float`, the simulation precision in Hamiltonian simulation
    :param circ_output: `str` or the `bool` **False**, where

        + **False** indicating to commit but not to output the quantum circuit;

        + `str` as a file name indicating not to commit but output the quantum circuit in such file

    :return: the return for such commit return if **circ_output** is a `str`, and **None** else
    """
    # create the quantum environment, can choose a backend
    env = QEnv()
    # env.backend(BackendName.CloudBaiduSim2Water)
    env.backend(BackendName.LocalBaiduSim2)

    # such two ancilla qubits introduced in QSVT
    reg_ancilla = [env.Q[0], env.Q[1]]
    # compute the number of qubits needed in the block-encoding, and form a register
    num_qubit_blocking = max(1, int(np.ceil(np.log2(len(list_str_Pauli_rep)))))
    reg_blocking = list(env.Q[idx] for idx in range(2, 2 + num_qubit_blocking))
    # create the system register for the Hamiltonian
    reg_sys = list(env.Q[idx] for idx in range(2 + num_qubit_blocking, 2 + num_qubit_blocking + num_qubit_sys))

    # if you need to prepare the initial state of the Hamiltonian, insert it here

    # the following is the quantum circuit for the time evolution operator for the Hamiltonian
    circ_HS_QSVT(reg_sys, reg_blocking, reg_ancilla, list_str_Pauli_rep, float_tau, float_epsilon)
    # if you need other quantum operations, insert it here

    # measure
    MeasureZ(*env.Q.toListPair())

    # not to draw the quantum circuit locally
    from QCompute.Define import Settings

    Settings.drawCircuitControl = []

    if not circ_output:
        # commit
        return env.commit(8000, downloadResult=True)
    else:
        if not isinstance(circ_output, str):
            circ_output = "circ_output"

        env.module(UnrollCircuitModule({"disable": True}))
        env.module(CompressGateModule({"disable": True}))
        env.publish()
        with open(circ_output, "w") as file:
            file.write(CircuitToQasm().convert(env.program))
