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
Grover's Algorithm
This is an instance for Grover's search algorithm on 3 qubits.
More information refers to the tutorial.

Reference
[1] Grover, Lov K. "A fast quantum mechanical algorithm for database search." Proceedings of the 28th Annual ACM
    Symposium on Theory of Computing (https://dl.acm.org/doi/10.1145/237814.237866). 1996.
[2] 百度量子计算研究所, "格罗弗算法." 量易简 (https://qulearn.baidu.com/textbook/chapter3/格罗弗算法.html), 2022.
"""
import sys

sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

# matchSdkVersion('Python 2.0.4')

# Your token:
Define.hubToken = ''


def main():
    """
    main
    """
    # Create environment
    env = QEnv()
    # Choose backend Baidu Cloud Quantum Simulator-Sim2
    env.backend(BackendName.CloudBaiduSim2Water)

    # Initialize the three-qubit register
    q = env.Q.createList(3)

    # Prepare the superposition state
    H(q[0])
    H(q[1])
    H(q[2])

    # Alternate calling Grover oracles and the diffusion operators
    for _ in range(1):
        Barrier(*q)
        Barrier(*q)
        # Call Grover oracle
        # The first layer of X gates in Grover oracle
        X(q[2])

        Barrier(*q)
        # The CCZ gate in Grover oracle
        H(q[2])
        CCX(q[0], q[1], q[2])
        H(q[2])

        Barrier(*q)
        # The second layer of X gates in Grover oracle
        X(q[2])

        Barrier(*q)
        Barrier(*q)
        # Call the diffusion operator
        # The first layer of Hadamard gates in the diffusion operator
        H(q[0])
        H(q[1])
        H(q[2])

        Barrier(*q)
        # The first layer of X gates in the diffusion operator
        X(q[0])
        X(q[1])
        X(q[2])

        Barrier(*q)
        # The CCZ gate in the diffusion operator
        H(q[2])
        CCX(q[0], q[1], q[2])
        H(q[2])

        Barrier(*q)
        # The second layer of X gates in the diffusion operator
        X(q[0])
        X(q[1])
        X(q[2])

        Barrier(*q)
        # The second layer of Hadamard gates in the diffusion operator
        H(q[0])
        H(q[1])
        H(q[2])

    Barrier(*q)
    Barrier(*q)
    # Finally, we measure the quantum system
    MeasureZ(*env.Q.toListPair())
    # Commit the quest to the cloud
    env.commit(1000, downloadResult=False)


if __name__ == '__main__':
    main()
