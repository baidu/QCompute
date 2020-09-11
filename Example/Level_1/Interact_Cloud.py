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
An example of coding on Quantum-Leaf with classical-quantum information interacting features.
Adjust parameter of rotation gate to eliminate '0' state.
"""

from pprint import pprint
import sys

sys.path.append('../..')  # "from QCompute import *" requires this
from QCompute import *

# Your token:
Define.hubToken = ''

uValue = 1  # flag of intereactions
for _ in range(15):

    print("uValue is :", uValue)

    env = QuantumEnvironment()  # environment set-up

    env.backend(BackendName.CloudBaiduSim2)  # set a backend to execute the quantum program

    q = [env.Q[0]]  # define quantum registers in need

    # apply gates and measurement operations to construct the circuit:
    u = RX(uValue)
    u(q[0])

    MeasureZ(q, range(1))

    taskResult = env.commit(1024, fetchMeasure=True)  # submit the circuit, execute and get results

    # interaction: change parameters of unitary in next experiment according to current result
    CountsDict = taskResult['counts']
    if CountsDict['0'] > 5:
        uValue = uValue * 2
    else:
        print("When the parameter is %d, 0 is eliminated." % uValue)
        break