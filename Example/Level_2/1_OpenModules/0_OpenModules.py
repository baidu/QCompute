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
This is a simple case of output circuit code with UnrollCircuitModule and CompressGateModule disabled.
"""

import sys
sys.path.append('../..')
from QCompute import *

# Create a quantum environment env and initialize the 1-qubits circuit
env = QEnv()
env.backend(BackendName.LocalBaiduSim2)
q = env.Q.createList(1)

# Apply gates
S(q[0])
H(q[0])
MeasureZ(*env.Q.toListPair())

# Disable circuit drawing
# Disable UnrollCircuitModule and CompressGateModule
from QCompute.Define import Settings
Settings.drawCircuitControl = []
env.module(UnrollCircuitModule({'disable': True}))
env.module(CompressGateModule({'disable': True}))

# Output circuit code
env.publish()
print(env.program)
