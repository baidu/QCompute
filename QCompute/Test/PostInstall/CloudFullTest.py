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
Cloud Full Test
"""
from QCompute import *


def cloudFullTest():
    """
    Cloud Full Test
    """

    env = QEnv()
    env.backend(BackendName.CloudBaiduSim2Water, '-s 0')

    # procedure0 start
    procedure0Env = QEnv()
    U(1.1, 2.2, 3.3)(procedure0Env.Q[0])
    U(1.1, 2.2)(procedure0Env.Q[0])
    U(1.1)(procedure0Env.Q[0])
    RX(1.1)(procedure0Env.Q[0])
    RY(1.1)(procedure0Env.Q[0])
    RZ(1.1)(procedure0Env.Q[0])
    CU(1.1, 2.2, 3.3)(procedure0Env.Q[0], procedure0Env.Q[1])  # CU3
    CRX(1.1)(procedure0Env.Q[0], procedure0Env.Q[1])
    CRY(1.1)(procedure0Env.Q[0], procedure0Env.Q[1])
    CRZ(1.1)(procedure0Env.Q[0], procedure0Env.Q[1])

    ID(procedure0Env.Q[0])
    X(procedure0Env.Q[0])
    Y(procedure0Env.Q[0])
    Z(procedure0Env.Q[0])
    H(procedure0Env.Q[0])
    S(procedure0Env.Q[0])
    SDG(procedure0Env.Q[0])
    T(procedure0Env.Q[0])
    TDG(procedure0Env.Q[0])
    CX(procedure0Env.Q[0], procedure0Env.Q[1])
    CY(procedure0Env.Q[0], procedure0Env.Q[1])
    CZ(procedure0Env.Q[0], procedure0Env.Q[1])
    CH(procedure0Env.Q[0], procedure0Env.Q[1])
    SWAP(procedure0Env.Q[0], procedure0Env.Q[1])
    CCX(procedure0Env.Q[0], procedure0Env.Q[1], procedure0Env.Q[2])
    CSWAP(procedure0Env.Q[0], procedure0Env.Q[1], procedure0Env.Q[2])

    procedure0 = procedure0Env.convertToProcedure('procedure0', env)
    # procedure0 end

    # procedure1 start
    procedure1Env = QEnv()
    U(1.1, 2.2, 3.3)(procedure1Env.Q[0])
    U(1.1, 2.2)(procedure1Env.Q[0])
    U(1.1)(procedure1Env.Q[0])
    RX(1.1)(procedure1Env.Q[0])
    RY(1.1)(procedure1Env.Q[0])
    RZ(1.1)(procedure1Env.Q[0])
    CU(1.1, 2.2, 3.3)(procedure1Env.Q[0], procedure1Env.Q[1])  # CU3
    CRX(1.1)(procedure1Env.Q[0], procedure1Env.Q[1])
    CRY(1.1)(procedure1Env.Q[0], procedure1Env.Q[1])
    CRZ(1.1)(procedure1Env.Q[0], procedure1Env.Q[1])

    procedure0()(procedure1Env.Q[1], procedure1Env.Q[0], procedure1Env.Q[2])

    ID(procedure1Env.Q[0])
    X(procedure1Env.Q[0])
    Y(procedure1Env.Q[0])
    Z(procedure1Env.Q[0])
    H(procedure1Env.Q[0])
    S(procedure1Env.Q[0])
    SDG(procedure1Env.Q[0])
    T(procedure1Env.Q[0])
    TDG(procedure1Env.Q[0])
    procedure1 = procedure1Env.convertToProcedure('procedure1', env)
    # procedure1 end

    q = [env.Q[0], env.Q[1], env.Q[2]]

    ID(q[0])
    X(q[0])
    Y(q[0])
    Z(q[0])
    H(q[0])
    S(q[0])
    SDG(q[0])
    T(q[0])
    TDG(q[0])
    CX(q[0], q[1])
    CY(q[0], q[1])
    CZ(q[0], q[1])
    CH(q[0], q[1])
    SWAP(q[0], q[1])
    CCX(q[0], q[1], q[2])
    CSWAP(q[0], q[1], q[2])

    U(1.1, 2.2, 3.3)(q[0])
    U(1.1, 2.2)(q[0])
    U(1.1)(q[0])
    RX(1.1)(q[0])
    RY(1.1)(q[0])
    RZ(1.1)(q[0])
    CU(1.1, 2.2, 3.3)(q[0], q[1])  # CU3
    CRX(1.1)(q[0], q[1])
    CRY(1.1)(q[0], q[1])
    CRZ(1.1)(q[0], q[1])

    procedure0()(q[0], q[1], q[2])
    procedure1()(q[1], q[2], q[0])

    Barrier(q[0], q[1], q[2])

    MeasureZ(q, range(3))

    env.module(UnrollProcedureModule())
    
    ret = env.commit(1024, fetchMeasure=True)
    assert ret['counts']['000'] == 157
    assert ret['counts']['001'] == 128
    assert ret['counts']['010'] == 137
    assert ret['counts']['011'] == 288
    assert ret['counts']['100'] == 202
    assert ret['counts']['101'] == 2
    assert ret['counts']['110'] == 97
    assert ret['counts']['111'] == 13

