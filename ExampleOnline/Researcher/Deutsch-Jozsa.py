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
Deutsch-Jozsa Algorithm.
Suppose: f1 = 0, f2 = first bit.
The result should be downloaded from quantum-hub.
"""

from QCompute import *

# Your token:
# Define.hubToken = ''

n = 10  # total qubit number

def main():
    """
    main
    """
    env1 = QuantumEnvironment()
    env1.backend(BackendName.CloudBaiduSim2)
    env2 = QuantumEnvironment()
    env2.backend(BackendName.CloudBaiduSim2)

    # Prepare the state:
    q1 = []
    q2 = []
    for i in range(n):
        q1.append(env1.Q[i])
        q2.append(env2.Q[i])
        H(q1[i])
        H(q2[i])
    q1.append(env1.Q[n])
    q2.append(env2.Q[n])
    X(q1[n])
    X(q2[n])
    H(q1[n])
    H(q2[n])

    # Apply U_f:
    # f1 = 0, so do nothing on q1.
    # f2 = first bit, so if the first bit is 0 do nothing, else swap q2[n].
    CX(q2[0], q2[n])

    # Measure:
    for i in range(n):
        H(q1[i])
        H(q2[i])
    MeasureZ(q1, range(n + 1))
    MeasureZ(q2, range(n + 1))
    env1.commit(shots=1, downloadResult=False)
    env2.commit(shots=1, downloadResult=False)


if __name__ == '__main__':
    main()
