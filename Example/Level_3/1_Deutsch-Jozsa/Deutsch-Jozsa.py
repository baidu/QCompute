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

"""
Deutsch-Jozsa Algorithm.
"""
import sys

sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

matchSdkVersion('Python 3.0.0')

# In this example we use 10 qubits as the main register,
# and also an ancillary qubit else
MainReg_num = 10


def main():
    """
    main
    """
    # Create two environment separately, and choose backend
    # We will execute D-J algorithm for f1 and f2 simultaneously
    env1 = QEnv()
    env1.backend(BackendName.LocalBaiduSim2)
    env2 = QEnv()
    env2.backend(BackendName.LocalBaiduSim2)

    # Initialize two registers on 11 qubits respectively,
    # where the last qubit in each register refers to the ancillary qubit,
    # and q1 and q2 correspond to f1 and f2 respectively.
    q1 = env1.Q.createList(MainReg_num + 1)
    q2 = env2.Q.createList(MainReg_num + 1)

    # As a preparation for D-J algorithm, we flip the ancillary qubit from |0> to |1>
    X(q1[MainReg_num])
    X(q2[MainReg_num])

    # In D-J algorithm, we apply a Hadamard gate on each qubit
    # in main register and the ancillary qubit
    for i in range(MainReg_num + 1):
        H(q1[i])
        H(q2[i])

    # Then apply U_f:
    # for f1 = 0, we need to do nothing on q1;
    # for f2 = the value of first qubit,so if f2 = 0 do nothing,
    # else to flip the ancillary qubit in q2, which is exactly a CX gate
    CX(q2[0], q2[MainReg_num])

    # Then we apply a Hadamard gate on each qubit in main register again
    for i in range(MainReg_num):
        H(q1[i])
        H(q2[i])

    # Measure the main registers with the computational basis
    MeasureZ(q1[:-1], range(MainReg_num))
    MeasureZ(q2[:-1], range(MainReg_num))
    # Commit the quest, where we need only 1 shot to distinguish that
    # f1 is constant for the measurement result |0>,
    # and f2 is balanced for the measurement result unequal to |0>
    env1.commit(shots=1, downloadResult=False)
    env2.commit(shots=1, downloadResult=False)


if __name__ == '__main__':
    main()
