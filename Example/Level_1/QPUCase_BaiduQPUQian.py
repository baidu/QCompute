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
There are three simple cases of using the QPU CloudBaiduQPUQian and modules.
Results will be fetched from a cloud program.
"""

import sys
from pprint import pprint

sys.path.append('../..')
from QCompute import *

matchSdkVersion('Python 3.0.0')

# Your token:
Define.hubToken = ''

# Create environment
env = QEnv()
# Choose backend CloudBaiduQPUQian
env.backend(BackendName.CloudBaiduQPUQian)
# Initialize the three-qubit circuit
q = env.Q.createList(3)

# case 1:
# The case below demonstrating the function of
# UnrollCircuitToBaiduQPUQian Module and MappingToBaiduQPUQian Module added by default.
# UnrollCircuitToBaiduQPUQian Module decomposes H,CH into
# a combination of Rx, Ry, and CZ that QPU CloudBaiduQPUQian can handle,
# then MappingToBaiduQPUQian Module maps the two logical qubits in a two-qubit gate(CZ) to two coupled physical qubits.
H(q[0])
CH(q[0], q[1])
CH(q[0], q[2])

# case 2:
# The case below demonstrating what will happen when we close ServerModule.UnrollCircuitToBaiduQPUQian Module
# by setting the disable parameter true.
# H(q[0])
# CH(q[0], q[1])
# CH(q[0], q[2])
# env.serverModule(ServerModule.UnrollCircuitToBaiduQPUQian, {"disable": True})

# case 3:
# The case below demonstrating what will happen when we close ServerModule.MappingToBaiduQPUQian Module
# by setting the disable parameter true.
# H(q[0])
# CH(q[0], q[1])
# CH(q[0], q[2])
# env.serverModule(ServerModule.MappingToBaiduQPUQian, {"disable": True})

# Measure with the computational basis
MeasureZ(*env.Q.toListPair())

# Generates circuit in Protobuf format and print
env.publish()
print(env.program)

# Commit the quest with 1000 shots to the cloud
taskResult = env.commit(1000, fetchMeasure=True)
pprint(taskResult)
