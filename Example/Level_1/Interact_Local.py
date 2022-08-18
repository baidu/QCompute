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
An example of coding on Quantum Leaf with classical-quantum information interacting features.
Adjust parameter of rotation gates to eliminate '0' state.
"""

import sys

sys.path.append('../..')
from QCompute import *

matchSdkVersion('Python 3.0.0')

# Flag of interactions
uValue = 1
for _ in range(15):

    print('uValue is :', uValue)

    # Create environment
    env = QEnv()
    # Choose backend Baidu local simulator
    env.backend(BackendName.LocalBaiduSim2)

    # Apply gates and measurement operations to construct the circuit
    u = RX(uValue)
    u(env.Q[0])

    MeasureZ(*env.Q.toListPair())

    # Commit the quest with 1024 shots
    taskResult = env.commit(1024, fetchMeasure=True)

    # Interaction: change parameters of unitary in next experiment according to current result
    CountsDict = taskResult['counts']
    if CountsDict.get('0', 0) > 5:
        uValue = uValue * 2
    else:
        print('When the parameter is %d, 0 is eliminated.' % uValue)
        break
