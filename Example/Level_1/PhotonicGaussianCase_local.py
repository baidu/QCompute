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
This is a simple case of photonic.
"""
import math
import sys
from pprint import pprint

sys.path.append('../..')
from QCompute.QPlatform.QOperation.Photonic.PhotonicGaussianGate import PhotonicDX, PhotonicDP, PhotonicDIS, \
    PhotonicPHA, PhotonicSQU, PhotonicCZ, PhotonicCX, PhotonicTSQU, PhotonicBS, PhotonicMZ
from QCompute.QPlatform.QOperation.Photonic.PhotonicGaussianMeasure import MeasureHeterodyne, MeasureHomodyne, \
    MeasurePhotonCount
from QCompute import *
from QCompute.Define import Settings

Settings.outputInfo = True

matchSdkVersion('Python 3.3.5')

# Create environment
env = QEnv()
# Choose backend Baidu local simulator
env.backend(BackendName.LocalBaiduSimPhotonic)

# num_mode = 3
# cutoff = 3
# depth = 1
# Initialize the three-qubit circuit
# q = env.Q.createList(num_mode)
# for index in range(num_mode):
#     PhotonicDIS(1, math.pi / 4)(q[index])
#     PhotonicSQU(1.5, math.pi / 3)(q[index])
#
# for _ in range(depth):
#     for index in range(num_mode - 1):
#         PhotonicBS(0.5)(q[index], q[index + 1])
#
# MeasurePhotonCount(cutoff)(*env.Q.toListPair())

q = env.Q.createList(3)
PhotonicDX(1.2)(q[0])
PhotonicDP(1.2)(q[0])
PhotonicDIS(1.2, math.pi / 3)(q[0])
PhotonicPHA(math.pi / 6)(q[0])
PhotonicSQU(1.2, math.pi / 4)(q[0])
PhotonicCZ(1.2)(q[0], q[2])
PhotonicCX(1.2)(q[0], q[2])
PhotonicTSQU(1.2, math.pi / 5)(q[1], q[2])
PhotonicBS(0.6)(q[0], q[1])
PhotonicMZ(math.pi / 3, math.pi / 4)(q[0], q[2])
# MeasurePhotonCount(2)(*env.Q.toListPair())

MeasureHeterodyne([(0.7, math.pi / 3), (1.1, math.pi)])([q[0], q[2]], [0, 2])
# MeasureHomodyne([q[0], q[2]], [0, 2])


# MeasureHomodyne(*env.Q.toListPair())
# PhotonicSQU(2, 0)(q[0])
# PhotonicSQU(2, 0)(q[1])
# PhotonicSQU(2, 0)(q[2])
# PhotonicBS(0.5)(q[0], q[1])
# PhotonicBS(0.5)(q[1], q[2])
# MeasureHeterodyne([(0.7, math.pi / 3), (1.1, math.pi)])([q[0], q[2]], [0, 2])
# We can also remove the 'import paths' of gates and measurements,
# and type the following code to run the quantum circuit.
# GaussianSQU(2, 0)(q[0])
# GaussianSQU(2, 0)(q[1])
# GaussianSQU(2, 0)(q[2])
# GaussianBS(0.5)(q[0], q[1])
# GaussianBS(0.5)(q[1], q[2])
# GaussianMeasureHeterodyne([(0.7, math.pi / 3), (0.9, math.pi / 4), (1.1, math.pi)])(*env.Q.toListPair())

# For Homodyne or Hetrodyne measurement, the argument 'shots' must be set to 1 in single 'env.commit'.
# In current version,
# the statistical results under these two measurements can be obtained by performing 'env.commit' multiple times.
taskResult = env.commit(1000, fetchMeasure=True)
pprint(taskResult)
