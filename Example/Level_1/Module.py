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
There are six simple cases demonstrating the function of OpenModule.
"""

from pprint import pprint
import sys
import numpy as np

sys.path.append('../..')  # "from QCompute import *" requires this
from QCompute import *

# Your token:
# Define.hubToken = ""

# Create environment
env = QEnv()

# Choose a backend as you want. Baidu Local Quantum Simulator-Sim2
env.backend(BackendName.LocalBaiduSim2)
# or Baidu Cloud Quantum Simulator-Sim2
# env.backend(BackendName.CloudBaiduSim2Water)

# case1:
# The case below demonstrating the function of the Compress Module added by default.
# CompressGate Module is used to compress one-qubit gates into two-qubit gates to accelerate the simulator process.
q = env.Q.createList(2)
H(q[0])
CX(q[0], q[1])
MeasureZ(*env.Q.toListPair())
env.publish()

# case2:
# The case below demonstrating the function of the UnrollCircuitModule Module added by default.
# UnrollCircuit Module supported gates are CX, U3, barrier, measure,
# supported fixed gates are ID, X, Y, Z, H, S, SDG, T, TDG, CY, CZ, CH, SWAP, CCX, CSWAP,
# supported rotation gates are U, RX, RY, RZ, CU, CRX, CRY, CRZ.
# q = env.Q.createList(4)
# H(q[0])
# X(q[0])
# Z(q[1])
# CRX(np.pi / 2)(q[1], q[2])
# CX(q[0], q[3])
# CSWAP(q[1], q[2], q[3])
# MeasureZ(*env.Q.toListPair())
# env.publish()

# case3:
# The case below demonstrating the function of the Subprocedure Module,
# which is used to unroll procedure.
# q = env.Q.createList(3)
#
# # Define a subprocedure0
# procedure0Env = QEnv()
# H(procedure0Env.Q[0])
# CX(procedure0Env.Q[0], procedure0Env.Q[1])
# Barrier(procedure0Env.Q[0], procedure0Env.Q[1])
# procedure0 = procedure0Env.convertToProcedure('procedure0', env)
#
# # Define a subprocedure1
# procedure1Env = QEnv()
# H(procedure1Env.Q[0])
# CX(procedure1Env.Q[0], procedure1Env.Q[1])
# RX(6.4)(procedure1Env.Q[0])
# procedure0()(procedure1Env.Q[1], procedure1Env.Q[0])
# procedure1 = procedure1Env.convertToProcedure('procedure1', env)
#
# # Main procedure calls subprocedure0
# procedure0()(env.Q[0], env.Q[1])
# # Main procedure calls subprocedure1
# procedure1()(env.Q[1], env.Q[2])
# H(q[0])
# MeasureZ(*env.Q.toListPair())
# env.module(UnrollProcedureModule())
# env.publish()

# case4:
# The case below demonstrating the function of the InverseCircuit Module.
# InverseCircuit Module is used to invert the circuit by changing the order of the gates,
# inverting the customized gate matrix, and modifying the angles of rotation gates specifically.
# q = env.Q.createList(1)
# MeasureZ(*env.Q.toListPair())
# H(q[0])
# S(q[0])
# env.module(InverseCircuitModule())
# env.publish()

# case5:
# The case below demonstrating the function of the ReverseCircuit Module.
# ReverseCircuit Module is used to reverse the circuit by changing the order of the gates.
# q = env.Q.createList(1)
# MeasureZ(*env.Q.toListPair())
# H(q[0])
# S(q[0])
# env.module(ReverseCircuitModule())
# env.publish()


# Commit the quest with 4000 shots to the cloud simulator
print(env.program)
taskResult = env.commit(4000, fetchMeasure=True)

pprint(taskResult)
