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
There are two simple cases of using the QPU CloudIpnAPM and modules.
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
# Choose backend CloudIonAPM
env.backend(BackendName.CloudIonAPM)
# Initialize the three-qubit circuit
q = env.Q.createList(1)

# case 1:
# The case below demonstrating the function of UnrollCircuitToIonAPM Module added by default.
# UnrollCircuitToIonAPM Module decomposes H into a combination of X, Y, RX, RY that QPU CloudIonAPM can handle.
H(q[0])

# case 2:
# The case below demonstrating what will happen when we close ServerModule.UnrollCircuitToIonAPM Module
# by setting the disable parameter true.
# H(q[0])
# env.serverModule(ServerModule.UnrollCircuitToIonAPM, {"disable": True})

# Measure with the computational basis
MeasureZ(*env.Q.toListPair())

# Generates circuit in Protobuf format and print
env.publish()
print(env.program)

# Commit the quest with 1000 shots to the cloud
taskResult = env.commit(1000, fetchMeasure=True)
pprint(taskResult)
