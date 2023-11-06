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
This script contains a simple example for showing the preparation for quantum states remains correct.
"""

from typing import List

import numpy as np
from QCompute import QEnv, BackendName, MeasureZ
from QCompute.Define import Settings as QC_Settings

from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Oracle.StatePreparation import circ_state_pre

# not to draw the quantum circuit locally
QC_Settings.drawCircuitControl = []
QC_Settings.outputInfo = False
QC_Settings.autoClearOutputDirAfterFetchMeasure = True


def func_state_pre_test(list_float_target_state: List[float], int_shots: int) -> List[float]:
    r"""Prepare a quantum state and then measure it.

    We will use local quantum backend to prepare a quantum state which equals to the input real vector,
    and then measure such quantum state

    :param list_float_target_state: :math:`\vec t`, `List[float]`,
        a list of non-negative floats regarded as a quantum state
    :param int_shots: `int`, the number of shots we will operate to measure the prepared quantum state
    :return: `List[float]`, a list of populations for the state we have prepared
    """
    env = QEnv()

    # from QCompute import Define
    # Define.hubToken = ''
    # env.backend(BackendName.CloudBaiduSim2Wind)
    env.backend(BackendName.LocalBaiduSim2)

    int_dim = len(list_float_target_state)  # the dimension of the input vector
    num_qubit_sys = max(int(np.ceil(np.log2(int_dim))), 1)  # the number of qubits we need to encode the input vector
    reg_sys = list(env.Q[idx] for idx in range(num_qubit_sys))  # create the quantum register

    # call the quantum circuit to prepare quantum state
    circ_state_pre(reg_sys, [], list_float_target_state, reg_borrowed=[])

    # measure the quantum state we have prepared
    MeasureZ(reg_sys, list(reversed(range(num_qubit_sys))))

    task_result = env.commit(int_shots, fetchMeasure=True)["counts"]  # commit to the task

    list_population = [0 for _ in range(2**num_qubit_sys)]  # register for finial populations
    for idx_key in task_result.keys():
        list_population[int(idx_key, base=2)] = task_result[idx_key]
    return list_population


if __name__ == "__main__":
    print(func_state_pre_test([0.25, 0.25, -0.5, 0.5, 0.5, -0.25, 0.25], 16000))
