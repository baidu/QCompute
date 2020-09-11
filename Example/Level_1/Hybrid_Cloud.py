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
This is a simple case of using Hybrid language.
Results will be fetched from a cloud program.
"""

from pprint import pprint
import sys

sys.path.append('../..')  # "from QCompute import *" requires this
from QCompute import *

# Your token:
Define.hubToken = ''

env = QuantumEnvironment()
# Baidu Cloud Quantum Simulator-Sim2
env.backend(BackendName.CloudBaiduSim2)

TotalNumQReg = 8

q = []
for index in range(TotalNumQReg):
    q.append(env.Q[index])

for index in range(TotalNumQReg):
    H(q[index])
    # X(q[index])

MeasureZ(q, range(TotalNumQReg))

taskResult = env.commit(1000, fetchMeasure=True)
pprint(taskResult)

