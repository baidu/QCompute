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
This script contains a simple example for showing the implement for CnX remains correct.
"""

from QCompute import QEnv, H, CX, BackendName, MeasureZ
from QCompute.Define import Settings as QC_Settings

from qcompute_qsvt.Gate.MultiCtrlGates import circ_multictrl_X

# not to draw the quantum circuit locally
QC_Settings.drawCircuitControl = []
QC_Settings.outputInfo = False
QC_Settings.autoClearOutputDirAfterFetchMeasure = True


def func_multictrl_X_test_local(num_qubit_test: int, int_shots: int = 10000):
    r"""Generate a test circuit, commit with local backend to test.

    The final state we will measure should be

    .. math::
                                      |0\rangle |00\cdots0\rangle |1\rangle |0\rangle |11\cdots1\rangle |0\rangle +
        sum_{|j\rangle\ne|11\cdots1>} |0\rangle |00\cdots0\rangle |0\rangle |0\rangle |j\rangle         |0\rangle
    where each of :math:`|j\rangle`, :math:`|00\cdots0\rangle` and :math:`|11\cdots1\rangle`
    is an :math:`n-2` qubits state, and others are single-qubit state.

    :param num_qubit_test: :math:`n`, `int`, the number of qubits on which we will operate a multictrl X gate
    :param int_shots: `int`, the number of shots for such quantum task
    :return: the task result for such commit, and print the test result (pass or fail).
    """
    # create quantum environment and choose local backend
    env = QEnv()
    env.backend(BackendName.LocalBaiduSim2)

    # create two quantum registers with n qubits
    reg_ancilla = list(env.Q[idx] for idx in range(num_qubit_test))
    reg_work = list(env.Q[idx] for idx in range(num_qubit_test, 2 * num_qubit_test))

    # the following is the test circuit
    for idx in range(num_qubit_test):
        H(reg_work[idx])
        CX(reg_work[idx], reg_ancilla[idx])

    circ_multictrl_X(reg_work[-1], reg_work[1:-1], reg_borrowed=reg_work[:1])

    for idx in range(num_qubit_test):
        CX(reg_work[idx], reg_ancilla[idx])

    H(reg_work[0])
    H(reg_work[-1])

    # The state should be sum_{|j>!=|11...1>} |0>|0...0>|0> |0>|j>|0>+ |0>|0...0>|1> |0>|11...1>|0>
    # and we will measure it.
    MeasureZ(reg_ancilla + reg_work, list(reversed(range(2 * num_qubit_test))))

    # not to draw the quantum circuit locally
    from QCompute.Define import Settings
    Settings.drawCircuitControl = []
    Settings.outputInfo = False
    Settings.autoClearOutputDirAfterFetchMeasure = True
    # Settings.drawCircuitControl = []

    taskResult = env.commit(int_shots, fetchMeasure=True)
    taskResult_part = {}
    for idx_key, idx_value in taskResult['counts'].items():
        bin_test = int(idx_key[: num_qubit_test - 1] + idx_key[-num_qubit_test] + idx_key[-1], 2)
        if bin_test != 0:
            taskResult_part[idx_key] = idx_value
    if taskResult_part == {}:
        print("\nCnX gate test passed.")
        print("Counts", taskResult['counts'])
        return taskResult
    else:
        print("\nCnX gate test failed.")
        print("Counts", taskResult_part)
        return taskResult_part, taskResult


if __name__ == "__main__":
    func_multictrl_X_test_local(6, 10000)
