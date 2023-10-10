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
This is a smoke test for BackendName.LocalCuQuantum
"""
import math
import sys
import random
from pprint import pprint

sys.path.append('../..')
from QCompute import *

matchSdkVersion('Python 3.3.5')

# Create environment
env = QEnv()
# Choose backend Baidu local simulator
env.backend(BackendName.LocalCuQuantum)

# Initialize
N = 3
q = env.Q.createList(N)
env.module(UnrollCircuitModule({'disable': True}))
env.module(CompressGateModule({'disable': True}))

# Smoke test
X(q[0])
CX(q[0], q[1])
Y(q[0])
CY(q[0], q[1])
Z(q[0])
CZ(q[0], q[1])
H(q[0])
CH(q[0], q[1])
S(q[0])
SDG(q[0])
T(q[0])
TDG(q[0])
SWAP(q[0], q[1])
RX(1.1)(q[0])
RY(1.1)(q[0])
RZ(1.1)(q[0])

# Random gates
for _ in range(100):
    gate_rnd = random.randrange(16)
    bit0, bit1 = map(int, random.sample(range(N), 2))
    angle = random.uniform(0, math.pi)
    print(gate_rnd, bit0, bit1, angle)
    if gate_rnd == 0:
        X(q[bit0])
    elif gate_rnd == 1:
        CX(q[bit0], q[bit1])
    elif gate_rnd == 2:
        Y(q[bit0])
    elif gate_rnd == 3:
        CY(q[bit0], q[bit1])
    elif gate_rnd == 4:
        Z(q[bit0])
    elif gate_rnd == 5:
        CZ(q[bit0], q[bit1])
    elif gate_rnd == 6:
        H(q[bit0])
    elif gate_rnd == 7:
        CH(q[bit0], q[bit1])
    elif gate_rnd == 8:
        S(q[bit0])
    elif gate_rnd == 9:
        SDG(q[bit0])
    elif gate_rnd == 10:
        T(q[bit0])
    elif gate_rnd == 11:
        TDG(q[bit0])
    elif gate_rnd == 12:
        SWAP(q[bit0], q[bit1])
    elif gate_rnd == 13:
        RX(angle)(q[bit0])
    elif gate_rnd == 14:
        RY(angle)(q[bit0])
    elif gate_rnd == 15:
        RZ(angle)(q[bit0])

# Measure with the computational basis
MeasureZ(*env.Q.toListPair())

# Commit the task with 1024 shots
taskResult = env.commit(1024, fetchMeasure=True)

pprint(taskResult)
