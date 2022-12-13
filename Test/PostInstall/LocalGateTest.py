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
Local Gate Test: test execution of various gates and modules
"""
from typing import TYPE_CHECKING, List, Tuple, Union, Dict
from numpy import pi as PI
from QCompute import *

if TYPE_CHECKING:  
    from QCompute.QPlatform.QRegPool import QRegStorage 
    from QCompute.OpenModule import ModuleImplement
    from QCompute.QPlatform.QOperation.FixedGate import FixedGateOP
    from QCompute.QPlatform.QOperation.RotationGate import RotationGateOP


normal_seed = '-s 68652'
    

def setup_environment(qRegs_number: int, backendname: 'BackendName' = None,
                      seeds: str = normal_seed) -> Tuple['QEnv', List['QRegStorage']]:
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
    q = [env.Q[idex] for idex in range(qRegs_number)]  # prepare quantum registers in need
    return env, q


def get_outcome(env: 'QEnv', q: List['QRegStorage'], num: int,
                module: 'ModuleImplement' = None) -> Dict[str, Union[str, Dict[str, int]]]:
    """
    Return measurement value
    """
    MeasureZ(q, range(num))
    if module is not None:
        env.module(module)
    env.publish()
    out_come = env.commit(1024, fetchMeasure=True)
    return out_come


def core(gate: Union['FixedGateOP', 'RotationGateOP'], bits: Union[List[int], int] = 0,
         module: 'ModuleImplement' = None, num: int = None, backendname: 'BackendName' = BackendName.LocalBaiduSim2,
         seeds: str = normal_seed) -> Dict[str, Union[str, Dict[str, int]]]:
    """
    The core to construct circuits
    """
    if num is None:
        if type(bits) is int:
            num = 1
        elif type(bits) is list:
            num = len(bits)
    env, q = setup_environment(num, backendname=backendname, seeds=seeds)  # environment set-up
    if type(bits) is int:
        gate(q[bits])  # circuit construction
    elif len(bits) == 2:
        gate(q[bits[0]], q[bits[1]])
    return get_outcome(env, q, num, module=module)


def localGateTest():
    """
    Local Gate Test
    """
    for backendname in [BackendName.LocalBaiduSim2, BackendName.LocalBaiduSim2WithNoise]:
        ret = core(ID, backendname=backendname)
        assert ret['counts']['0'] == 1024
        ret = core(H, backendname=backendname)
        assert ret['counts']['0'] == ret['counts']['1']
        ret = core(X, backendname=backendname)
        assert ret['counts']['1'] == 1024

        # Test Y
        n = 2
        env, q = setup_environment(n, backendname=backendname)
        H(q[1])
        X(q[0])
        CX(q[0], q[1])
        Y(q[1])
        H(q[1])
        ret = get_outcome(env, q, n)
        assert ret['counts']['11'] == 1024

        # Test Z: under x-basis, Z is like X
        n = 1
        env, q = setup_environment(n, backendname=backendname)
        H(q[0])
        Z(q[0])
        H(q[0])
        ret = get_outcome(env, q, n)
        assert ret['counts']['1'] == 1024

        # Test U
        n = 1
        env, q = setup_environment(n, backendname=backendname)
        H(q[0])
        U(PI / 2, 0, PI)(q[0])  # This is like H
        ret = get_outcome(env, q, n)
        assert ret['counts']['0'] == 1024

        # Test RX
        n = 1
        env, q = setup_environment(n, backendname=backendname)
        Y(q[0])
        RX(PI)(q[0])  # This is like Y
        ret = get_outcome(env, q, n, module=UnrollCircuitModule())
        assert ret['counts']['0'] == 1024

        # Test RY
        n = 1
        env, q = setup_environment(n, backendname=backendname)
        H(q[0])
        RY(PI / 2 * 3)(q[0])  # This is like H
        ret = get_outcome(env, q, n)
        assert ret['counts']['0'] == 1024

        # Test RZ
        n = 1
        env, q = setup_environment(n, backendname=backendname)
        H(q[0])
        RZ(PI / 2)(q[0])
        H(q[0])
        MeasureZ(q, range(n))
        env.publish()
        ret = env.commit(1024, fetchMeasure=True)
        assert ret['counts']['0'] == ret['counts']['1']

        # Test S
        n = 2
        env, q = setup_environment(n, backendname=backendname)
        H(q[0])
        CX(q[0], q[1])
        S(q[1])
        SDG(q[0])
        CX(q[0], q[1])
        H(q[0])
        ret = get_outcome(env, q, n)
        assert ret['counts']['00'] == 1024

        # Test T
        n = 2
        env, q = setup_environment(n, backendname=backendname)
        H(q[0])
        CX(q[0], q[1])
        T(q[1])
        RZ(-PI / 4)(q[1])
        TDG(q[0])
        RZ(-PI / 4)(q[0])
        CX(q[0], q[1])
        H(q[0])
        ret = get_outcome(env, q, n)
        assert ret['counts']['00'] == ret['counts']['01']

        # Test CX
        n = 3
        env, q = setup_environment(n, backendname=backendname)
        H(q[0])
        CX(q[0], q[1])
        CX(q[1], q[2])
        ret = get_outcome(env, q, n)
        assert ret['counts']['000'] == ret['counts']['111']

        # Test CY
        n = 2
        env, q = setup_environment(n, backendname=backendname)
        H(q[1])
        X(q[0])
        CY(q[0], q[1])
        X(q[1])
        H(q[1])
        ret = get_outcome(env, q, n)
        assert ret['counts']['11'] == 1024

        # Test CZ
        n = 2
        env, q = setup_environment(n, backendname=backendname)
        H(q[0])
        H(q[1])
        CZ(q[0], q[1])
        H(q[1])
        ret = get_outcome(env, q, n)
        assert ret['counts']['00'] == ret['counts']['11']

        # Test SWAP
        n = 2
        env, q = setup_environment(n, backendname=backendname, seeds=normal_seed)
        H(q[0])
        SWAP(q[0], q[1])
        ret = get_outcome(env, q, n)
        assert ret['counts']['00'] == ret['counts']['10']

        # Test CRX
        n = 2
        env, q = setup_environment(n, backendname=backendname, seeds=normal_seed)
        X(q[0])
        CRX(PI / 2)(q[0], q[1])
        ret = get_outcome(env, q, n)
        assert ret['counts']['01'] == ret['counts']['11']

        # Test CRY
        n = 2
        env, q = setup_environment(n, backendname=backendname, seeds=normal_seed)
        H(q[0])
        H(q[1])
        CRY(PI / 2 * 3)(q[0], q[1])
        ret = get_outcome(env, q, n)
        assert ret['counts']['00'] == ret['counts']['10'] == ret['counts']['01'] / 2

        # Test CRY
        n = 2
        env, q = setup_environment(2, backendname=backendname, seeds=normal_seed)
        H(q[0])
        H(q[1])
        CRZ(PI)(q[0], q[1])
        H(q[0])
        H(q[1])
        ret = get_outcome(env, q, n)
        assert ret['counts']['00'] == ret['counts']['01'] == ret['counts']['10'] == ret['counts']['11']

        # Test CCX
        n = 3
        env, q = setup_environment(n, backendname=backendname)
        X(q[0])
        Y(q[1])
        CCX(q[0], q[1], q[2])
        ret = get_outcome(env, q, n)
        assert ret['counts']['111'] == 1024

        # Test CSWAP
        n = 3
        env, q = setup_environment(n, backendname=backendname)
        X(q[0])
        Y(q[1])
        CSWAP(q[0], q[1], q[2])
        ret = get_outcome(env, q, n)
        assert ret['counts']['101'] == 1024

        # Test UnrollCircuit
        n = 2
        env, q = setup_environment(n, backendname=backendname)
        H(q[0])
        CX(q[0], q[1])
        X(q[1])
        ret = get_outcome(env, q, n, module=UnrollCircuitModule())
        assert ret['counts']['01'] == ret['counts']['10']