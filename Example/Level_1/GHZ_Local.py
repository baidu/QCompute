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
This is a simple case of using Hadamard gate and CNOT gate to generate a GHZ state.
Results will be fetched from a local program.
"""

import sys
from pprint import pprint

sys.path.append('../..')  # "from QCompute import *" requires this
from QCompute import *

matchSdkVersion('Python 2.0.4')

# Create environment
env = QEnv()
# Choose backend Baidu Local Quantum Simulator-Sim2
env.backend(BackendName.LocalBaiduSim2)

# Initialize the five-qubit circuit
q = env.Q.createList(5)

# Apply a Hadamard gate on the 0th qubit
H(q[3])

# Apply some CX gates where the 0th qubit controls flipping each other qubit
CX(q[3], q[1])
CX(q[3], q[2])

# Measure with the computational basis
# You can adjust the order of QRegs to measure, and specify the CRegs you want
MeasureZ([env.Q[3], env.Q[4], env.Q[1]], [9, 1, 2])

# Commit the quest with 1024 shots
taskResult = env.commit(1024, fetchMeasure=True)

pprint(taskResult)
