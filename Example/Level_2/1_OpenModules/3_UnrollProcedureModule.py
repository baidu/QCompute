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
This is a simple case of using UnrollProcedureModule.
"""

import sys
sys.path.append('../..')
from QCompute import *

# Create a quantum environment env and initialize the 1-qubits circuit
env = QEnv()
env.backend(BackendName.LocalBaiduSim2)
q = env.Q.createList(3)

# Define subroutine procedure0
procedure0Env = QEnv()
H(procedure0Env.Q[0])
CX(procedure0Env.Q[0], procedure0Env.Q[1])
procedure0 = procedure0Env.convertToProcedure('procedure0', env)

# Define subroutine procedure1 and call procedure0
procedure1Env = QEnv()
RX(procedure1Env.Parameter[0])(procedure1Env.Q[0])
procedure0()(procedure1Env.Q[1], procedure1Env.Q[0])
procedure1 = procedure1Env.convertToProcedure('procedure1', env)

# Define subroutine procedure2
procedure2Env = QEnv()
RX(3.2)(procedure2Env.Q[0])
CX(procedure2Env.Q[0], procedure2Env.Q[1])
procedure2 = procedure2Env.convertToProcedure('procedure2', env)

# Generate the inverse and reverse of procedure1
procedure2__inversed, _ = env.inverseProcedure('procedure2')
procedure2__reversed, _ = env.reverseProcedure('procedure2')

# Call procedure0, procedure1 procedure1__inversed and procedure1__reversed in main procedure
procedure0()(q[0], q[1])
Barrier(*q)
procedure1(6.4)(q[1], q[2])
Barrier(*q)
procedure2()(q[0], q[1])
Barrier(*q)
procedure2__inversed()(q[0], q[1])
Barrier(*q)
procedure2__reversed()(q[0], q[1])
Barrier(*q)

# Apply gates
H(q[0])
MeasureZ(*env.Q.toListPair())

# Automatically call UnrollProcedureModule. Disable it with the following statement
# env.module(UnrollProcedureModule({'disable': True}))

# Disable circuit drawing
# Disable UnrollCircuitModule and CompressGateModule
from QCompute.Define import Settings
Settings.drawCircuitControl = []
env.module(UnrollCircuitModule({'disable': True}))
env.module(CompressGateModule({'disable': True}))

# Output circuit code
env.publish()
print(env.program)
