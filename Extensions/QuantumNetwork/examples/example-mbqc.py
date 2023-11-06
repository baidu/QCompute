#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
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

r"""
An example of MBQC.
"""

from numpy import pi, random

from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends.mbqc import MBQC
from Extensions.QuantumNetwork.qcompute_qnet.quantum.basis import Basis
from Extensions.QuantumNetwork.qcompute_qnet.quantum.gate import Gate
from Extensions.QuantumNetwork.qcompute_qnet.quantum.state import PureState

# Single-qubit gate
G = [["1", "2", "3", "4", "5"], [("1", "2"), ("2", "3"), ("3", "4"), ("4", "5")]]
input_vec = PureState.random_state_vector(1, is_real=False)

# Suppose the single-qubit gate is decomposed by Rx(gamma) Rz(beta) Rx(alpha)
alpha = pi * random.uniform()
beta = pi * random.uniform()
gamma = pi * random.uniform()

mbqc = MBQC()
mbqc.set_graph(G)
mbqc.set_input_state(PureState(input_vec, ["1"]))

# Measure qubit '1', with "theta = 0"
theta_1 = 0
mbqc.measure("1", Basis.Plane("XY", theta_1))

# Measure qubit '2', with "theta = (-1)^{s_1 + 1} * alpha"
theta_2 = (-1) ** mbqc.sum_outcomes(["1"], 1) * alpha
mbqc.measure("2", Basis.Plane("XY", theta_2))

# Measure qubit '3', with "theta = (-1)^{s_2 + 1} * alpha"
theta_3 = (-1) ** mbqc.sum_outcomes(["2"], 1) * beta
mbqc.measure("3", Basis.Plane("XY", theta_3))

# Measure qubit '4', with "theta = (-1)^{s_1 + s_3 + 1} * alpha"
theta_4 = (-1) ** mbqc.sum_outcomes(["1", "3"], 1) * gamma
mbqc.measure("4", Basis.Plane("XY", theta_4))

# Correct byproduct operators
mbqc.correct_byproduct("X", "5", mbqc.sum_outcomes(["2", "4"]))
mbqc.correct_byproduct("Z", "5", mbqc.sum_outcomes(["1", "3"]))

state_out = mbqc.get_quantum_output()

# Find the standard result
# Note: here we adopt the Rx Rz Rx decomposition form (instead of Rz Rx Rz) of unitary gate
vec_std = Gate.Rx(gamma) @ Gate.Rz(beta) @ Gate.Rx(alpha) @ input_vec
system_std = ["5"]
state_std = PureState(vec_std, system_std)
# Compare with the standard result
print(state_out.compare_by_vector(state_std))

# CNOT gate
X_basis = Basis.X()
Y_basis = Basis.Y()

# Construct the underlying graph of CNOT implementation in MBQC
V = [str(i) for i in range(1, 16)]
E = [
    ("1", "2"),
    ("2", "3"),
    ("3", "4"),
    ("4", "5"),
    ("5", "6"),
    ("6", "7"),
    ("4", "8"),
    ("8", "12"),
    ("9", "10"),
    ("10", "11"),
    ("11", "12"),
    ("12", "13"),
    ("13", "14"),
    ("14", "15"),
]
G = [V, E]

# Generate a random state vector
input_psi = PureState.random_state_vector(2, is_real=True)

# Instantiate a MBQC class
mbqc = MBQC()
# Set the underlying graph for computation
mbqc.set_graph(G)
# Set the input state
mbqc.set_input_state(PureState(input_psi, ["1", "9"]))

# Start measurement process
mbqc.measure("1", X_basis)
mbqc.measure("2", Y_basis)
mbqc.measure("3", Y_basis)
mbqc.measure("4", Y_basis)
mbqc.measure("5", Y_basis)
mbqc.measure("6", Y_basis)
mbqc.measure("8", Y_basis)
mbqc.measure("9", X_basis)
mbqc.measure("10", X_basis)
mbqc.measure("11", X_basis)
mbqc.measure("12", Y_basis)
mbqc.measure("13", X_basis)
mbqc.measure("14", X_basis)

# Obtain byproduct's exponents
cx = mbqc.sum_outcomes(["2", "3", "5", "6"])
tx = mbqc.sum_outcomes(["2", "3", "8", "10", "12", "14"])
cz = mbqc.sum_outcomes(["1", "3", "4", "5", "8", "9", "11"], 1)
tz = mbqc.sum_outcomes(["9", "11", "13"])

# Correct byproducts
mbqc.correct_byproduct("X", "7", cx)
mbqc.correct_byproduct("X", "15", tx)
mbqc.correct_byproduct("Z", "7", cz)
mbqc.correct_byproduct("Z", "15", tz)

# Obtain the output state
state_out = mbqc.get_quantum_output()

# Find the standard result
vec_std = Gate.CNOT() @ input_psi
system_std = ["7", "15"]
state_std = PureState(vec_std, system_std)
# Compare with the standard result
print(state_out.compare_by_vector(state_std))
