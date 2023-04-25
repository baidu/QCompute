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
import sys
from pprint import pprint

sys.path.append('../..')
from QCompute.QPlatform.QOperation.Photonic.PhotonicFockGate import PhotonicAP, PhotonicBS
from QCompute.QPlatform.QOperation.Photonic.PhotonicFockMeasure import MeasurePhotonCount
from QCompute import *
from QCompute.Define import Settings

Settings.outputInfo = True

matchSdkVersion('Python 3.3.1')

# Create environment
env = QEnv()
# Choose backend Baidu local simulator
env.backend(BackendName.LocalBaiduSimPhotonic)

# Initialize the three-qubit circuit
q = env.Q.createList(3)

PhotonicAP(1)(q[0])
PhotonicAP(2)(q[1])
PhotonicAP(1)(q[2])
PhotonicBS(0.5)(q[0], q[1])
PhotonicBS(0.5)(q[1], q[2])
MeasurePhotonCount(2)(*env.Q.toListPair())
# We can also remove the 'import paths' of gates and measurements,
# and type the following code to run the quantum circuit.
# FockAP(1)(q[0])
# FockAP(2)(q[1])
# FockAP(1)(q[2])
# FockBS(0.5)(q[0], q[1])
# FockBS(0.5)(q[1], q[2])
# FockMeasurePhotonCount(2)(*env.Q.toListPair())

taskResult = env.commit(1024, fetchMeasure=True)
pprint(taskResult)