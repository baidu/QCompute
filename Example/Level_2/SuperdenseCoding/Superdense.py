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
Superdense
"""
import sys
sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

# hyper-parameter
shots = 1024

# The message that Alice want to send to Bob
message = '11'


def main():
    """
    main
    """
    env = QuantumEnvironment()
    env.backend(BackendName.LocalBaiduSim2)
    q = [env.Q[0], env.Q[1]]
    H(q[0])
    CX(q[0], q[1])

    if message == '01':
        X(q[0])
    elif message == '10':
        Z(q[0])
    elif message == '11':
        Z(q[0])
        X(q[0])

    CX(q[0], q[1])
    H(q[0])

    MeasureZ(q, range(2))
    taskResult = env.commit(shots, fetchMeasure=True)
    print(taskResult['counts'])


if __name__ == '__main__':
    main()
