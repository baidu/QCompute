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
Cloud Test
"""

import sys

sys.path.append('../..')  # "from QCompute import *" requires this
from QCompute import *

# Your token:
# Define.hubToken = ''

env = QuantumEnvironment()
# Baidu Cloud Quantum Simulator-Sim2
env.backend(BackendName.CloudBaiduSim2)


u1 = U(1.1)
u2 = U(1.1, 2.2)
u3 = U(1.1, 2.2, 3.3)
q = [env.Q[0], env.Q[1]]
ID(q[0])
H(q[0])
CX(q[0], q[1])
u3(q[0])
u2(q[0])
u1(q[0])
RZZ(1.1, 2.2, 3.3)(q[0], q[1])
Barrier(q[0], q[1])
MeasureZ(q, range(2))

env.module(CompositeGate())  # RZZ needed
env.commit(1024, downloadResult=False)

