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
Phase
"""
import numpy as np

import sys
sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

matchSdkVersion('Python 3.0.0')

qubit_num = 4
shots = 1000
phase = 2 * np.pi / 5


def main():
    """
    main
    """
    # Create environment
    env = QEnv()
    # Choose backend Baidu Local Quantum Simulator-Sim2
    env.backend(BackendName.LocalBaiduSim2)

    # Initialize qubits
    q = env.Q.createList(qubit_num)

    # Prepare eigenstate |1> = X|0> on the ancillary qubit
    X(q[3])

    # Superposition
    H(q[0])
    H(q[1])
    H(q[2])

    # Control-U gates
    CU(0, 0, phase)(q[0], q[3])

    CU(0, 0, phase)(q[1], q[3])
    CU(0, 0, phase)(q[1], q[3])

    CU(0, 0, phase)(q[2], q[3])
    CU(0, 0, phase)(q[2], q[3])
    CU(0, 0, phase)(q[2], q[3])
    CU(0, 0, phase)(q[2], q[3])

    # 3-qubit inverse QFT
    SWAP(q[0], q[2])
    H(q[0])
    CU(0, 0, -np.pi / 2)(q[0], q[1])
    H(q[1])
    CU(0, 0, -np.pi / 4)(q[0], q[2])
    CU(0, 0, -np.pi / 2)(q[1], q[2])
    H(q[2])

    # Measurement result
    MeasureZ(*env.Q.toListPair())
    taskResult = env.commit(shots, fetchMeasure=True)
    return taskResult['counts']


if __name__ == '__main__':
    main()
