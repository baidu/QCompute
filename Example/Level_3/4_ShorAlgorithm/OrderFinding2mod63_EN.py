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
This is a simple case of using Quantum Order Finding algorithm to solve the order of 2 mod 63.
"""


from QCompute import *
from numpy import pi

matchSdkVersion('Python 3.0.0')


def func_order_finding_2_mod_63():
    """
    This function will give an approximation related to the eigenphase s/6 for some s=0,1,2,3,4,5
    where 6 is the order of 2 mod 63.
    """
    env = QEnv()  # Create environment
    env.backend(BackendName.LocalBaiduSim2)  # Choose backend Baidu Local Quantum Simulator-Sim2

    L = 6  # The number of qubits to encode the gate U, also the number of qubits in the system register
    N = 3 * L + 1  # The total number of qubits in this algorithm
    # The number of ancilla qubits used in the quantum phase estimation algorithm (QPE), also the number of qubits in
    # the ancilla register
    t = 2 * L + 1

    # Create a register, the first t qubits of which form the ancilla register, and the others form the system register.
    q = env.Q.createList(N)

    X(q[N - 1])  # We prepare the state |1> in the system register, and will operate QPE

    for i in range(t):
        H(q[i])  # The first step in QPE, we prepare an average superposition state,

    # The following is the transfer step in QPE, we will operate several C(U^(2^j)) gates

    # The following is a decomposition of the gate C(U), the ctrlling qubit is the last qubit in the system register
    CSWAP(q[2 * L], q[t + 4], q[t + 5])
    CSWAP(q[2 * L], q[t + 3], q[t + 4])
    CSWAP(q[2 * L], q[t + 2], q[t + 3])
    CSWAP(q[2 * L], q[t + 1], q[t + 2])
    CSWAP(q[2 * L], q[t + 0], q[t + 1])

    s = 2 * L - 1  # For the other C(U^(2^j)) gates, where q[s] is just the ctrlling qubit
    while s >= 0:
        if s % 2 == 1:
            # The decomposition of C(U^2) under this condition
            CSWAP(q[s], q[t + 1], q[t + 3])
            CSWAP(q[s], q[t + 3], q[t + 5])
            CSWAP(q[s], q[t + 0], q[t + 2])
            CSWAP(q[s], q[t + 2], q[t + 4])
        else:
            # The decomposition of C(U^4) under this condition
            CSWAP(q[s], q[t + 3], q[t + 5])
            CSWAP(q[s], q[t + 1], q[t + 3])
            CSWAP(q[s], q[t + 2], q[t + 4])
            CSWAP(q[s], q[t + 0], q[t + 2])
        s -= 1  # Move the pointer to a higher ancilla qubit

    # We need to operate an inverse Quantum Fourier Transform (QFT) on the ancilla register in the last step of QPE
    # The SWAP step in inverse QFT
    for i in range(t // 2):
        SWAP(q[i], q[t - i - 1])

    # The ctrl-rotation step in inverse QFT
    for i in range(t - 1):
        H(q[t - i - 1])
        for j in range(i + 1):
            CU(0, 0, -pi / pow(2, (i - j + 1)))(q[t - j - 1], q[t - i - 2])
    H(q[0])

    # We have completed the inverse QFT and also QPE, and will measure the quantum state we have obtained
    MeasureZ(q[:t], range(t))  # Only the ancilla register (i.e. the first t qubits) need to be measured
    env.commit(8192, downloadResult=False)


if __name__ == "__main__":
    func_order_finding_2_mod_63()
