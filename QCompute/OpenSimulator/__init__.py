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
Quantum executor interface definition
"""
ModuleErrorCode = 8
FileErrorCode = 1

import json
from typing import TYPE_CHECKING, Dict, Union, List, Any

import numpy

from QCompute import Define
from QCompute.QPlatform.Utilities import numpyMatrixToDictMatrix, dictMatrixToNumpyMatrix

if TYPE_CHECKING:
    pass


class SimulatorVersion:
    """
    The version of quantum simulator.
    """

    gitHash: str = None
    """
    Git Hash
    """

    gitTime: str = None
    """
    Git Time
    """

    fileHash: str = None
    """
    File Hash
    """

    compileTime: str = None
    """
    Compile Time
    """

    def __init__(self):
        self.gitHash = None
        self.gitTime = None
        self.fileHash = None
        self.compileTime = None


class Ancilla:
    """
    Ancilla
    """
    usedQRegList: List[int] = None
    """
    used qReg list
    """

    usedCRegList: List[int] = None
    """
    used cReg list
    """

    compactedQRegDict: Dict[int, int] = None
    """
    computed qReg list
    """

    compactedCRegDict: Dict[int, int] = None
    """
    measured qReg list
    """

    def __init__(self):
        self.usedQRegList = None
        self.usedCRegList = None
        self.compactedQRegDict = None
        self.compactedCRegDict = None


class QResult:
    """
    The result of experiment.
    """

    sdkVersion = Define.sdkVersion
    """
    SDK Version from Define.sdkVersion
    """

    simulatorVersion: SimulatorVersion = None
    """
    The version of quantum simulator
    """

    code = 0
    """
    error code
    """

    vendor = None
    """
    vendor code for universal error
    """

    output = ''
    """
    output results
    """

    log = ''
    """
    output results
    """

    ancilla: Ancilla = Ancilla()
    """
    ancilla
    """

    moduleList: Dict[str, Any] = None
    """
    moduleList
    """

    shots = 0
    """
    number of shots
    """

    counts: Union[Dict[str, int], Dict[str, float]] = None
    """
    counts for results
    """

    state: numpy.ndarray = None
    """
    counts for results
    """

    seed = 0
    """
    random seed
    """

    startTimeUtc = ''
    """
    start utc time
    """

    endTimeUtc = ''
    """
    end utc time
    """

    def __init__(self):
        self.simulatorVersion = None
        self.code = 0
        self.vendor = None
        self.output = ''
        self.log = ''
        self.ancilla = Ancilla()
        self.moduleList = None
        self.shots = 0
        self.counts = None
        self.state = None
        self.seed = 0
        self.startTimeUtc = ''
        self.endTimeUtc = ''

    def fromJson(self, text: str) -> Dict[str, Any]:
        """
        fromJson
        """
        data = json.loads(text)
        if 'sdkVersion' in data:
            self.sdkVersion = data['sdkVersion']  # 否则在模拟器提交前写好
        if 'simulatorVersion' in data:
            simulatorVersion = data['simulatorVersion']
            self.simulatorVersion.gitHash = simulatorVersion['gitHash']
            self.simulatorVersion.gitTime = simulatorVersion['gitTime']
            self.simulatorVersion.fileHash = simulatorVersion['fileHash']
            self.simulatorVersion.compileTime = simulatorVersion['compileTime']
        # self.code = data['code']
        # self.output = data['output']
        # self.log = data['log']
        if 'ancilla' in data:
            if 'usedQRegList' in data['ancilla'] and data['ancilla']['usedQRegList'] is not None:
                self.ancilla.usedQRegList = data['ancilla']['usedQRegList']
            if 'usedCRegList' in data['ancilla'] and data['ancilla']['usedCRegList'] is not None:
                self.ancilla.usedCRegList = data['ancilla']['usedCRegList']
            if 'compactedQRegDict' in data['ancilla'] and data['ancilla']['compactedQRegDict'] is not None:
                self.ancilla.compactedQRegDict = data['ancilla']['compactedQRegDict']
            if 'compactedCRegDict' in data['ancilla'] and data['ancilla']['compactedCRegDict'] is not None:
                self.ancilla.compactedCRegDict = data['ancilla']['compactedCRegDict']
        if 'moduleList' in data:
            self.moduleList = data['moduleList']
        self.shots = data['shots']
        if 'counts' in data:
            self.counts = data['counts']
        if 'state' in data and data['state'] is not None:
            self.state = dictMatrixToNumpyMatrix(data['state'], complex)
        self.seed = data['seed']
        self.startTimeUtc = data['startTimeUtc']
        self.endTimeUtc = data['endTimeUtc']
        return data

    def toJson(self, inside: bool = None) -> str:
        """
        toJson
        """
        ret = {
            'sdkVersion': self.sdkVersion,
            'ancilla': {
                'usedQRegList': self.ancilla.usedQRegList,
                'usedCRegList': self.ancilla.usedCRegList,
                'compactedQRegDict': self.ancilla.compactedQRegDict,
                'compactedCRegDict': self.ancilla.compactedCRegDict,
            },
            'moduleList': self.moduleList,
            'shots': self.shots,
            'counts': self.counts,
            'seed': self.seed,
            'startTimeUtc': self.startTimeUtc,
            'endTimeUtc': self.endTimeUtc,
        }
        if self.state is not None:
            ret['state'] = numpyMatrixToDictMatrix(self.state)
        if inside is None:
            inside = Define.taskInside
        if inside:
            ret['output'] = self.output
            ret['log'] = self.log
        if self.simulatorVersion is not None:
            simulatorVersion = {
                'gitHash': self.simulatorVersion.gitHash,
                'gitTime': self.simulatorVersion.gitTime,
                'fileHash': self.simulatorVersion.fileHash,
                'compileTime': self.simulatorVersion.compileTime,
            }
            ret['simulatorVersion'] = simulatorVersion
        return json.dumps(ret)


class QPhotonicResult(QResult):
    """
    The result of photonic experiment.

    For homodyne measure and heterodyne measure.
    """

    value: Union[Dict[str, int], Dict[str, float]] = None
    """
    value for results
    """

    def __int__(self):
        self.value = None

    def fromJson(self, text: str):
        data = super().fromJson(text)
        self.value = data['value']

    def toJson(self, inside: bool = None):
        ret = super().toJson(inside)
        data = json.loads(ret)
        data['value'] = self.value
        return json.dumps(data)


class QImplement:
    """
    Implement params for quantum execution.

    Send to the simulator when submitting a task.
    """

    program = None
    """
    Protobuf format of the circuit
    """

    shots = 0
    """
    Number of shots
    """

    backendArgument = None
    """
    The arguments of backend
    """

    result: Union[QResult, QPhotonicResult] = None
    """
    The final result
    """

    def commit(self):
        """
        Commit task
        """
        pass


def withNoise(program: 'PBProgram') -> bool:
    """
    With noise
    """
    return any(len(circLine.noiseList) >= 1 for circLine in program.body.circuit)