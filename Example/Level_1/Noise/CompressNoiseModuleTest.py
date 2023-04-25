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
This is a simple case of to use CompressGateModule.
"""

import sys
sys.path.append('../..')
from QCompute import *

from QCompute.QProtobuf import PBFixedGate


def self_defined_circuit() -> 'QEnv':
    """
    A self defined noisy circuit
    """

    env = QEnv()
    q = env.Q.createList(4)
    RY(1)(q[3])
    H(q[2])
    H(q[1])
    CX(q[2], q[0])
    cu = CU(2, 1, 0)
    cu(q[0], q[3])
    CSWAP(q[0], q[2], q[3])
    X(q[1])
    RX(1)(q[1])
    X(q[2])
    H(q[2])
    CX(q[0], q[1])
    CX(q[1], q[2])
    ID(q[0])
    H(q[0])
    X(q[0])
    H(q[3])
    CX(q[0], q[1])
    H(q[0])
    CX(q[0], q[1])
    Z(q[0])

    dpobj = Depolarizing(1, 0.1)
    env.noise(['H', 'X', 'RY'], [dpobj])
    
    return env


def test_main():
    """
    Main test
    """
    env = self_defined_circuit()
    env.module(UnrollNoiseModule())
    env.module(CompressNoiseModule())
    env.publish()

    assert env.program.sdkVersion == Define.sdkVersion
    head = env.program.head
    assert head.usingQRegList == [0, 1, 2, 3]
    assert head.usingCRegList == []
    body = env.program.body

    circuitLine = body.circuit[0]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[1].data.name)
    assert circuitLine.qRegList == env.circuit[1].qRegList
    circuitLine = body.circuit[1]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[3].data.name)
    assert circuitLine.qRegList == env.circuit[3].qRegList
    circuitLine = body.circuit[2]
    assert circuitLine.WhichOneof('op') == 'rotationGate'
    assert circuitLine.qRegList == [3]
    assert circuitLine.argumentValueList == env.circuit[0].data.argumentList
    assert circuitLine.argumentIdList == []
    circuitLine = body.circuit[3]
    assert circuitLine.WhichOneof('op') == 'rotationGate'
    assert circuitLine.qRegList == env.circuit[4].qRegList
    assert circuitLine.argumentValueList == env.circuit[4].data.argumentList
    assert circuitLine.argumentIdList == []
    circuitLine = body.circuit[4]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[5].data.name)
    assert circuitLine.qRegList == env.circuit[5].qRegList
    circuitLine = body.circuit[5]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[15].data.name)
    assert circuitLine.qRegList == env.circuit[15].qRegList
    circuitLine = body.circuit[6]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[2].data.name)
    assert circuitLine.qRegList == env.circuit[2].qRegList
    circuitLine = body.circuit[7]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[6].data.name)
    assert circuitLine.qRegList == env.circuit[6].qRegList
    circuitLine = body.circuit[8]
    assert circuitLine.WhichOneof('op') == 'customizedGate'
    assert circuitLine.qRegList == env.circuit[10].qRegList
    circuitLine = body.circuit[9]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[8].data.name)
    assert circuitLine.qRegList == env.circuit[8].qRegList
    circuitLine = body.circuit[10]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[9].data.name)
    assert circuitLine.qRegList == env.circuit[9].qRegList
    circuitLine = body.circuit[11]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[11].data.name)
    assert circuitLine.qRegList == env.circuit[11].qRegList
    circuitLine = body.circuit[12]
    assert circuitLine.WhichOneof('op') == 'customizedGate'
    assert circuitLine.qRegList == env.circuit[13].qRegList    
    circuitLine = body.circuit[13]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[14].data.name)
    assert circuitLine.qRegList == env.circuit[14].qRegList    
    circuitLine = body.circuit[14]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[16].data.name)
    assert circuitLine.qRegList == env.circuit[16].qRegList    
    circuitLine = body.circuit[15]
    assert circuitLine.WhichOneof('op') == 'fixedGate'
    assert circuitLine.fixedGate == PBFixedGate.Value(env.circuit[17].data.name)
    assert circuitLine.qRegList == env.circuit[17].qRegList
    circuitLine = body.circuit[16]
    assert circuitLine.WhichOneof('op') == 'customizedGate'
    assert circuitLine.qRegList == env.circuit[18].qRegList


if __name__ == '__main__':
    test_main()