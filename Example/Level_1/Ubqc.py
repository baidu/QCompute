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
Universal blind quantum computation (UBQC) test: two-qubits Grover algorithm
"""

# Import packages
import sys
sys.path.append('../..')  # "from QCompute import *" requires this line
from QCompute import *
# Set the interpreter
matchSdkVersion('Python 2.0.6')

# Set you Token here
# Define.hubToken = 'your Token'
# Define.hubToken = ''

# Build the environment for quantum programming
env = QEnv()
# Choose the ``ServiceUbqc`` backend
env.backend(BackendName.ServiceUbqc)

# Initialize two qubits
q = env.Q.createList(2)
# Input the Grover circuit
H(q[0])
H(q[1])
CZ(q[0], q[1])
H(q[0])
H(q[1])
Z(q[0])
Z(q[1])
CZ(q[0], q[1])
H(q[0])
H(q[1])
# Measure all qubits in the Z basis
MeasureZ(*env.Q.toListPair())

# Submit the task to the server, sample 16 shots and obtain the counts
taskResult = env.commit(16, fetchMeasure=True)
print("The sample counts are:", taskResult['counts'])
