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
This is an example to calculate a specific noisy circuit.
"""
from QCompute import *
import sys

sys.path.append('../..')

# Create environment
env = QEnv()
# Choose backend Baidu local simulator
env.backend(BackendName.LocalBaiduSim2)

# Initialize a 2-qubit circuit
q = env.Q.createList(5)

# A self-defined circuit
H(q[4])
CX(q[1], q[2])
H(q[3])
RX(0.5)(q[1])
H(q[0])
CX(q[4], q[1])
H(q[1])
CX(q[0], q[4])

# Measure with the computational basis
MeasureZ(*env.Q.toListPair())

# Define 1-qubit noise objects
bfobj = BitFlip(0.1)

# Add noise
env.noise(gateNameList=['H'], noiseList=[bfobj], qRegList=[3], positionList=[0])

# Commit the task with 1024 shots
shots = 1000
res = env.commit(shots, fetchMeasure=True)

counts_str = ['00001', '00011', '11001', '10001', '01000', '10011', '10000', '11000', \
              '01001', '00000', '11010', '11011', '01010', '10010', '10010', '00010']

for str in counts_str:  

    assert abs(res['counts'][str] - 63) <= 20
