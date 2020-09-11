#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2020 Baidu, Inc. All Rights Reserved.
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
Grover's Algorithm Test
"""
import sys
sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

qubit_num = 3
shots = 1000


def main():
    """
    main
    """
    # Create environment
    env = QuantumEnvironment()
    # Choose backend
    env.backend(BackendName.LocalBaiduSim2)

    # Initialize a 3-qubit state
    q = [env.Q[i] for i in range(qubit_num)]

    # Superposition
    H(q[0])
    H(q[1])
    H(q[2])

    # Oracle for |101>
    X(q[1])
    H(q[2])
    CCX(q[0], q[1], q[2])
    X(q[1])
    H(q[2])

    # Diffusion Operator
    H(q[0])
    H(q[1])
    H(q[2])
    X(q[0])
    X(q[1])
    X(q[2])

    H(q[2])
    CCX(q[0], q[1], q[2])
    H(q[2])

    X(q[0])
    X(q[1])
    X(q[2])
    H(q[0])
    H(q[1])
    H(q[2])

    # Measurement result
    MeasureZ(q, range(qubit_num))
    taskResult = env.commit(shots, fetchMeasure=True)
    return taskResult['counts']


if __name__ == '__main__':
    main()
