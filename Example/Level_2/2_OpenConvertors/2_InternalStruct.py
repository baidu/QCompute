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
This is a simple case of bidirectional converters for implementing Circuit and InternalStruct.
"""

from QCompute import *
from pprint import pprint
from QCompute.Define import Settings
Settings.outputInfo = False

env = QEnv()

q = env.Q.createList(2)

X(q[0])
CX(q[0], q[1])
MeasureZ(*env.Q.toListPair())

# Generate PBProgram
env.publish()
pprint(env.program)

# Convert PBProgram to InternalStruct and output
circuitLineList = CircuitToInternalStruct().convert(env.program.body.circuit)
pprint(circuitLineList)

# Convert InternalStruct to PBProgram and output
circuit = InternalStructToCircuit().convert(circuitLineList)
pprint(circuit)
