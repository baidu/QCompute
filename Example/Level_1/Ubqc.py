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
This is a simple case of two-qubits Grover algorithm.
Results will be fetched from universal blind quantum computation (UBQC) backend.
"""

import sys

sys.path.append('../..')
from QCompute import *

matchSdkVersion('Python 3.0.0')

# Your token:
Define.hubToken = ''

# Create environment
env = QEnv()
# Choose backend ServiceUbqc
env.backend(BackendName.ServiceUbqc)

# Initialize the two-qubit circuit
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

# Measure with the computational basis
MeasureZ(*env.Q.toListPair())

# Commit the task to with 16 shots and obtain the counts
taskResult = env.commit(16, fetchMeasure=True)

print("The sample counts are:", taskResult['counts'])
