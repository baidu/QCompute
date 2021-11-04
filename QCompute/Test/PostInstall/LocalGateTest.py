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
Local Gate Test: test execution of various gates and modules
"""
from QCompute import *
from numpy import pi as PI

from QCompute.Define import Settings

normal_seed = '-s 68652'


def setup_environment(qRegs_number, backendname=None, seeds=normal_seed):
    """
    Setup execution environment
    """
    if backendname is None:
        backendname = BackendName.LocalBaiduSim2
    env = QEnv()
    if seeds is None:
        env.backend(backendname)
    else:
        env.backend(backendname, seeds)
    q = []
    for idex in range(qRegs_number):  # prepare quantum registers in need
        q.append(env.Q[idex])
    return env, q


def get_outcome(env, q, N, Module=None):
    """
    Return measurement value
    """
    MeasureZ(q, range(N))
    if Module is not None:
        env.module(Module)
    env.publish()
    out_come = env.commit(1024, fetchMeasure=True)
    return out_come


def core(Gate, bits=0, Module=None, N=None, backendname=BackendName.LocalBaiduSim2, seeds=normal_seed):
    """
    The core to construct circuits
    """
    if N is None:
        if type(bits) is int:
            N = 1
        elif type(bits) is list:
            N = len(bits)
    env, q = setup_environment(N, backendname=backendname, seeds=seeds)  # environment set-up
    if type(bits) is int:
        Gate(q[bits])  # circuit construction
    elif len(bits) == 2:
        Gate(q[bits[0]], q[bits[1]])
    return get_outcome(env, q, N, Module=Module)


def localGateTest():
    """
    Local Gate Test
    """
    ret = core(ID)
    assert ret['counts']['0'] == 1024
    ret = core(H)
    assert ret['counts']['0'] == ret['counts']['1']
    ret = core(X)
    assert ret['counts']['1'] == 1024

    # Test Y
    N = 2
    env, q = setup_environment(N)
    H(q[1])
    X(q[0])
    CX(q[0], q[1])
    Y(q[1])
    H(q[1])
    ret = get_outcome(env, q, N)
    assert ret['counts']['11'] == 1024

    # Test Z: under x-basis, Z is like X
    N = 1
    env, q = setup_environment(N)
    H(q[0])
    Z(q[0])
    H(q[0])
    ret = get_outcome(env, q, N)
    assert ret['counts']['1'] == 1024

    # Test U
    N = 1
    env, q = setup_environment(N)
    H(q[0])
    U(PI / 2, 0, PI)(q[0])  # This is like H
    ret = get_outcome(env, q, N)
    assert ret['counts']['0'] == 1024

    # Test RX
    N = 1
    env, q = setup_environment(N)
    Y(q[0])
    RX(PI)(q[0])  # This is like Y
    ret = get_outcome(env, q, N, Module=UnrollCircuitModule())
    assert ret['counts']['0'] == 1024

    # Test RY
    N = 1
    env, q = setup_environment(N)
    H(q[0])
    RY(PI / 2 * 3)(q[0])  # This is like H
    ret = get_outcome(env, q, N)
    assert ret['counts']['0'] == 1024

    # Test RZ
    N = 1
    env, q = setup_environment(N)
    H(q[0])
    RZ(PI / 2)(q[0])
    H(q[0])
    MeasureZ(q, range(N))
    env.publish()
    ret = env.commit(1024, fetchMeasure=True)
    assert ret['counts']['0'] == ret['counts']['1']

    # Test S
    N = 2
    env, q = setup_environment(N)
    H(q[0])
    CX(q[0], q[1])
    S(q[1])
    SDG(q[0])
    CX(q[0], q[1])
    H(q[0])
    ret = get_outcome(env, q, N)
    assert ret['counts']['00'] == 1024

    # Test T
    N = 2
    env, q = setup_environment(N)
    H(q[0])
    CX(q[0], q[1])
    T(q[1])
    RZ(-PI / 4)(q[1])
    TDG(q[0])
    RZ(-PI / 4)(q[0])
    CX(q[0], q[1])
    H(q[0])
    ret = get_outcome(env, q, N)
    assert ret['counts']['00'] == ret['counts']['01']

    # Test CX
    N = 3
    env, q = setup_environment(N)
    H(q[0])
    CX(q[0], q[1])
    CX(q[1], q[2])
    ret = get_outcome(env, q, N)
    assert ret['counts']['000'] == ret['counts']['111']

    # Test CY
    N = 2
    env, q = setup_environment(N)
    H(q[1])
    X(q[0])
    CY(q[0], q[1])
    X(q[1])
    H(q[1])
    ret = get_outcome(env, q, N)
    assert ret['counts']['11'] == 1024

    # Test CZ
    N = 2
    env, q = setup_environment(N)
    H(q[0])
    H(q[1])
    CZ(q[0], q[1])
    H(q[1])
    ret = get_outcome(env, q, N)
    assert ret['counts']['00'] == ret['counts']['11']

    # Test SWAP
    N = 2
    env, q = setup_environment(N, seeds=normal_seed)
    H(q[0])
    SWAP(q[0], q[1])
    ret = get_outcome(env, q, N)
    assert ret['counts']['00'] == ret['counts']['10']

    # Test CRX
    N = 2
    env, q = setup_environment(N, seeds=normal_seed)
    X(q[0])
    CRX(PI / 2)(q[0], q[1])
    ret = get_outcome(env, q, N)
    assert ret['counts']['01'] == ret['counts']['11']

    # Test CRY
    N = 2
    env, q = setup_environment(N, seeds=normal_seed)
    H(q[0])
    H(q[1])
    CRY(PI / 2 * 3)(q[0], q[1])
    ret = get_outcome(env, q, N)
    assert ret['counts']['00'] == ret['counts']['10'] == ret['counts']['01'] / 2

    # Test CRY
    N = 2
    env, q = setup_environment(2, seeds=normal_seed)
    H(q[0])
    H(q[1])
    CRZ(PI)(q[0], q[1])
    H(q[0])
    H(q[1])
    ret = get_outcome(env, q, N)
    assert ret['counts']['00'] == ret['counts']['01'] == ret['counts']['10'] == ret['counts']['11']

    # Test CCX
    N = 3
    env, q = setup_environment(N)
    X(q[0])
    Y(q[1])
    CCX(q[0], q[1], q[2])
    ret = get_outcome(env, q, N)
    assert ret['counts']['111'] == 1024

    # Test CSWAP
    N = 3
    env, q = setup_environment(N)
    X(q[0])
    Y(q[1])
    CSWAP(q[0], q[1], q[2])
    ret = get_outcome(env, q, N)
    assert ret['counts']['101'] == 1024

    # Test UnrollCircuit
    N = 2
    env, q = setup_environment(N)
    H(q[0])
    CX(q[0], q[1])
    X(q[1])
    ret = get_outcome(env, q, N, Module=UnrollCircuitModule())
    assert ret['counts']['01'] == ret['counts']['10']
