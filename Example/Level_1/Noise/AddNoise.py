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
This is a simple case shows different ways of adding noise.
Two 1-qubit noises, i.e. BitFlip and AmplitudeDamping, and a 2-qubit noise, i.e. Depolarizing, are taken as examples.
"""
from QCompute import *
import sys

sys.path.append('../..')

matchSdkVersion('Python 3.3.1')

# Create environment
env = QEnv()
# Choose backend Baidu local simulator
env.backend(BackendName.LocalBaiduSim2)

# Initialize a 2-qubit circuit
q = env.Q.createList(2)

# A self-defined circuit
H(q[0])
H(q[0])
X(q[0])
X(q[0])
CX(q[0], q[1])
CX(q[0], q[1])
H(q[1])
H(q[1])
X(q[1])
X(q[1])

# Measure with the computational basis
MeasureZ(*env.Q.toListPair())

# Define 1-qubit noise objects
bfobj_008 = BitFlip(probability=0.08)
adobj_01 = AmplitudeDamping(probability=0.1)

# Define 2-qubit noise object
dpobj_2_01 = Depolarizing(bits=2, probability=0.1)

# Case1: Add 1-qubit BitFlip noise on all H gates in q[0]
env.noise(gateNameList=['H'], noiseList=[bfobj_008], qRegList=[0])

# Case2: Add 1-qubit BitFlip noise on all H gates in all qubits
env.noise(gateNameList=['H'], noiseList=[bfobj_008])

# Case3: Add 1-qubit BitFlip noise on the 2-nd H gate in q[0]
env.noise(gateNameList=['H'], noiseList=[bfobj_008],
          qRegList=[0], positionList=[1])

# Case4: Add 1-qubit BitFlip and Amplitude noises on all H gates in q[0]
env.noise(gateNameList=['H'], noiseList=[bfobj_008,
          adobj_01], qRegList=[0], positionList=[0])

# Case5: Add 1-qubit BitFlip noise on all H and X gates in q[1]
env.noise(gateNameList=['H', 'X'], noiseList=[bfobj_008], qRegList=[1])

# Case6: Add 2-qubit Depolarizing noise on all CX gates in (q[0], q[1])
env.noise(gateNameList=['CX'], noiseList=[dpobj_2_01], qRegList=[0, 1])

# Commit the task with 1024 shots
taskResult = env.commit(1024, fetchMeasure=True)