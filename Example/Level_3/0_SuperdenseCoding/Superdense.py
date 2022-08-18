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
Superdense Coding
"""
import sys

sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

matchSdkVersion('Python 3.0.0')

# Set the shot number for each quest
shots = 1024

# The message that Alice want to send to Bob.
# The user you can modify the massage to '00', '01' or '10' as you like
message = '10'


def main():
    """
    main
    """
    # Create environment
    env = QEnv()
    # Choose backend Baidu Local Quantum Simulator-Sim2
    env.backend(BackendName.LocalBaiduSim2)

    # Initialize the two-qubit circuit
    q = env.Q.createList(2)

    # Alice and Bob share a Bell state (|00>+|11>) / sqrt(2),
    # where Alice holds q[0] and Bob holds q[1]
    H(q[0])
    CX(q[0], q[1])

    # For different messages, Alice applies different gates on q[0],
    # where she applies nothing to indicate the message '00'
    if message == '01':
        X(q[0])
    elif message == '10':
        Z(q[0])
    elif message == '11':
        # Here ZX = iY
        X(q[0])
        Z(q[0])

    # Alice sends her qubit q[0] to Bob,
    # and then Bob decodes the two qubits.
    # Bob needs to measure the two qubits with Bell basis,
    # or transforms the state to computational basis
    # and measures it with the computational basis
    CX(q[0], q[1])
    H(q[0])

    # Measure with the computational basis
    MeasureZ(*env.Q.toListPair())
    # Commit the quest
    taskResult = env.commit(shots, fetchMeasure=True)
    print(taskResult['counts'])


if __name__ == '__main__':
    main()
