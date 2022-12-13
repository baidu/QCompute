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
Local Noise Test: test execution of various noises and noise-related modules
"""
from typing import TYPE_CHECKING, List, Tuple, Union, Dict
from numpy import pi as PI
import numpy as np

from QCompute import *
from QCompute.QPlatform.QNoise.Utilities import sigma, noiseTensor

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
        backendname = BackendName.LocalBaiduSim2WithNoise
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
    out_come = env.commit(1000, fetchMeasure=True)
    return out_come


operatorType = Union['FixedGateOP', 'RotationGateOP']


def pre_defined_circuit(env: 'QEnv', q: List['QRegStorage'], gate_list: List[operatorType]) -> 'QEnv':
    """
    Pre-defined circuit before core function
    """
    if gate_list:
        for gate in gate_list:
            if gate.bits == 1:
                gate(q[0])
            elif gate.bits == 2:
                gate(q[0], q[1])
    return env


noiseType = Union['AmplitudeDamping', 'BitFlip', 'CustomizedNoise', 'BitPhaseFlip',
                  'Depolarizing', 'PauliNoise', 'PhaseDamping', 'PhaseFlip', 'ResetNoise']


def core(noise_list: List[noiseType], pre_gate_list: List[operatorType] = None,
         post_gate_list: List[operatorType] = None, bits: Union[List[int], int] = 0, module: 'ModuleImplement' = None,
         num: int = None, backendname: 'BackendName' = BackendName.LocalBaiduSim2WithNoise,
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

    pre_defined_circuit(env, q, pre_gate_list)
    if type(bits) is int:
        ID(q[0])
        env.noise(['ID'], noise_list, [0], [0])  # circuit construction
    elif len(bits) == 2:
        CRX(PI)(q[0], q[1])
        CRX(PI)(q[0], q[1])
        env.noise(['CRX'], noise_list, [0, 1], [1])
    pre_defined_circuit(env, q, post_gate_list)
    return get_outcome(env, q, num, module=module)


def localNoiseTest():
    """
    Local Noise Test
    """
    # Noise instance
    # Define a BitFlip noise
    bfobj = BitFlip(0.1)
    # Define an AmplitudeDamping noise
    adobj = AmplitudeDamping(0.1)
    # Define a Customized noise
    kraus_list = [np.array([[0.0, 0.0], [np.sqrt(0.1), 0.0]]),
                  np.array([[np.sqrt(1 - 0.1), 0.0], [0.0, 1.0]])
                  ]
    cnobj = CustomizedNoise(kraus_list)
    # Define a BitPhaseFlip noise
    bpfobj = BitPhaseFlip(0.1)
    # Define a 1-qubit Depolarizing noise
    dpobj_1 = Depolarizing(1, 0.1)
    # Define a PauliNoise noise
    pnobj = PauliNoise(0.0, 0.3, 0.7)
    # Define a PhaseDamping noise
    pdobj = PhaseDamping(0.1)
    # Define a PhaseFlip noise
    pfobj = PhaseFlip(0.1)
    # Define a ResetNoise noise
    rnobj = ResetNoise(1, 0)
    # Define a 2-qubit Depolarizing noise
    dpobj_2 = Depolarizing(2, 0.1)
    # Define a 2-qubit Customized noise
    kraus_list_2 = [np.sqrt(1 - 0.1) * np.eye(2), np.sqrt(0.1) * sigma(3)]
    cnobj_2 = CustomizedNoise(noiseTensor(kraus_list_2, kraus_list_2))

    # Test 1-qubit noise
    ret = core([bfobj], [H], [H])
    assert ret['counts']['0'] == 1000
    ret = core([adobj])
    assert ret['counts']['0'] == 1000
    ret = core([cnobj], [X], [X])
    assert ret['counts']['0'] == 1000
    ret = core([bpfobj], [RX(PI / 2)], [RX(- PI / 2)])
    assert ret['counts']['0'] == 1000
    ret = core([dpobj_1], [H])
    # Theoretically, ret['counts']['0'] == ret['counts']['1'] in the above noisy circuit
    assert ret['counts']['0'] - ret['counts']['1'] <= 100
    ret = core([pnobj], [H], [RY(PI / 2)])
    assert ret['counts']['0'] == 1000
    ret = core([pdobj])
    assert ret['counts']['0'] == 1000
    ret = core([pfobj])
    assert ret['counts']['0'] == 1000
    ret = core([rnobj])
    assert ret['counts']['0'] == 1000

    # Test 2-qubit noise
    ret = core([dpobj_2], [H, CX], [], [0, 1])
    # Theoretically, ret['counts']['00'] ret['counts']['01']== ret['counts']['10'] + ret['counts']['11']
    # in the above noisy circuit
    assert (ret['counts']['00'] + ret['counts']['01']) - (ret['counts']['10'] + ret['counts']['11']) <= 100
    ret = core([cnobj_2], [X], [X], [0, 1])
    assert ret['counts']['00'] == 1000

    # Test multiple noises on one gate
    ret = core([adobj, pdobj, pfobj, rnobj])
    assert ret['counts']['0'] == 1000

    # Test nonadjacent noises   
    n = 3
    env, q = setup_environment(n)
    H(q[0])
    env.noise(['H'], [bfobj], [0], [0])
    H(q[0])
    CCX(q[0], q[1], q[2])
    CCX(q[0], q[1], q[2])
    X(q[1])
    env.noise(['X'], [cnobj], [1], [0])
    X(q[1])
    ret = get_outcome(env, q, n)
    assert ret['counts']['000'] == 1000

    # Test Rotation Gate noise
    n = 1
    env, q = setup_environment(n)
    RZ(1.1)(q[0])
    RZ(- 1.1)(q[0])
    env.noise(['RZ'], [adobj], [0], [1])
    ret = get_outcome(env, q, n)
    assert ret['counts']['0'] == 1000
