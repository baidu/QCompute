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
This is a simple case of bidirectional conversion of QCompute circuit model to IonQ circuit model.
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

# Convert PBProgram to IonQ-JSON and output
ionq_program = CircuitToIonQ().convert(env.program)
pprint(ionq_program)

# Convert IonQ-JSON to PBProgram and output
circuit = IonQToCircuit().convert(ionq_program)
pprint(circuit)
