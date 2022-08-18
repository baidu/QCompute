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
This is a simple case of using autoClearOutputDirAfterFetchMeasure
and cloudTaskDoNotWriteFile to manage the storage of process and result file.
"""

import sys
sys.path.append('../..')
from QCompute import *
from QCompute.Define import Settings

matchSdkVersion('Python 3.0.0')

# Your token:
Define.hubToken = ''

# Create environment
env = QEnv()
# Choose backend Baidu local simulator
env.backend(BackendName.CloudBaiduSim2Water)

# Initialize the three-qubit circuit
q = env.Q.createList(3)

# Apply a Hadamard gate on the 0th qubit
H(q[0])

# Apply some CX gates where the 0th qubit controls flipping each other qubit
CX(q[0], q[1])
CX(q[0], q[2])

# Measure with the computational basis
MeasureZ(*env.Q.toListPair())

# Commit the quest with 1024 shots
Settings.autoClearOutputDirAfterFetchMeasure = False
env.commit(1024, fetchMeasure=True)

# Commit the quest with 1024 shots and auto clear file
Settings.autoClearOutputDirAfterFetchMeasure = True
env.commit(1024, fetchMeasure=True)

# Commit the quest with 1024 shots and don't write file
Settings.cloudTaskDoNotWriteFile = True
env.commit(1024, fetchMeasure=True)
