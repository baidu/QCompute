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
Quantum executor interface definition
"""
import json
from typing import TYPE_CHECKING, Dict, Union, List, Any

import numpy

from QCompute import Define
from QCompute.QPlatform.Utilities import numpyMatrixToDictMatrix, dictMatrixToNumpyMatrix

ModuleErrorCode = 7

if TYPE_CHECKING:
    pass


class SimulatorVersion:
    """
    The version of quantum simulator.
    """

    gitHash = None  # type: str
    """
    Git Hash
    """

    gitTime = None  # type: str
    """
    Git Time
    """

    fileHash = None  # type: str
    """
    File Hash
    """

    compileTime = None  # type: str
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
    usedQRegList = None  # type: List[int]
    """
    used qReg list
    """

    usedCRegList = None  # type: List[int]
    """
    used cReg list
    """

    compactedQRegDict = None  # type: Dict[int, int]
    """
    computed qReg list
    """

    compactedCRegDict = None  # type: Dict[int, int]
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

    simulatorVersion = None  # type: SimulatorVersion
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

    ancilla = Ancilla()  # type: Ancilla
    """
    ancilla
    """

    moduleList = None  # type:Dict[str,Any]
    """
    moduleList
    """

    shots = 0
    """
    number of shots
    """

    counts = None  # type: Union[Dict[str, int], Dict[str, float]]
    """
    counts for results
    """

    state = None  # type: numpy.ndarray
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
        self.simulatorVersionq = None
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

    def fromJson(self, text: str):
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
            if 'usedCRegList' in ['ancilla'] and data['ancilla']['usedCRegList'] is not None:
                self.ancilla.usedCRegList = data['ancilla']['usedCRegList']
            if 'compactedQRegDict' in data['ancilla'] and data['ancilla']['compactedQRegDict'] is not None:
                self.ancilla.compactedQRegDict = data['ancilla']['compactedQRegDict']
            if 'compactedCRegDict' in data['ancilla'] and data['ancilla']['compactedCRegDict'] is not None:
                self.ancilla.compactedCRegDict = data['ancilla']['compactedCRegDict']
        if 'moduleList' in data:
            self.moduleList = data['moduleList']
        self.shots = data['shots']
        self.counts = data['counts']
        if 'state' in data and data['state'] is not None:
            self.state = dictMatrixToNumpyMatrix(data['state'], complex)
        self.seed = data['seed']
        self.startTimeUtc = data['startTimeUtc']
        self.endTimeUtc = data['endTimeUtc']

    def toJson(self, inside: bool = None):
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


class QImplement:
    """
    Implement params for quantum execution.

    Send to the simulator when submitting a task.
    """

    program = None  # type: 'PBProgram'
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

    result = QResult()
    """
    The final result
    """

    def commit(self):
        """
        Commit task
        """
        pass
