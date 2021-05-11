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
Quantum Environment
"""
import importlib
import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Set, List, Dict, TYPE_CHECKING, Union, Optional, Tuple

from QCompute.Define import sdkVersion, noLocalTask, circuitPackageFile, outputPath, noWaitTask
from QCompute.OpenModule import ModuleImplement
from QCompute.QPlatform import Error
from QCompute.QPlatform.CircuitTools import QEnvToProtobuf
from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterPool
from QCompute.QPlatform.Processor.PostProcessor import formatMeasure
from QCompute.QPlatform.QOperation.QProcedure import QProcedure, QProcedureOP
from QCompute.QPlatform.QRegPool import QRegPool
from QCompute.QPlatform.QTask import QTask
from QCompute.QPlatform.Utilities import destoryObject
from QCompute.QProtobuf import PBProgram

if TYPE_CHECKING:
    from QCompute.QPlatform import BackendName

BackendArgumentType = Union[str,
                            'Sim2Argument',
]


class QEnv:
    """
    Quantum Environment class

    The member variable is the static saved state of the circuit.

    Quantum registers in use, classical registers, circuits in each line, and modules for sequential process.

    The backend function selects the process types, local or cloud.

    Configurations can be passed to the backend according to alternative parameters.

    The commitXXX function series call local or cloud subroutine according to prefix of backend names

    The publish function generates circuit definition in Protobuf format from the statical circuit data.

    The module function inserts the processing module to the end of module list.
    """

    def __init__(self) -> None:
        self.Q = QRegPool(self)
        self.Parameter = ProcedureParameterPool()

        self.measuredQRegSet = set()  # type: Set[int]
        self.measuredCRegSet = set()  # type: Set[int]
        self.circuit = []  # type: List['CircuitLine']
        self.procedureMap = {}  # type: Dict[str, 'QProcedure']
        self.program = None  # type: Optional['PBProgram']

        self.backendName = None  # type: Optional['BackendName']
        self.backendArgument = None  # type: Optional[List['BackendArgumentType']]

        self.usedModuleList = []  # type: List['ModuleImplement']

    def backend(self, backendName: 'BackendName', *backendArgument: BackendArgumentType) -> None:
        if type(backendName) is not str:
            backendName = backendName.value
        self.backendName = backendName
        self.backendArgument = list(backendArgument)

    def convertToProcedure(self, name: str, env: 'QEnv') -> 'QProcedure':
        if name in env.procedureMap:
            raise Error.ArgumentError(f'Duplicate procedure name: {name}!')
        procedure = QProcedure(name, self.Q, self.Parameter, self.circuit)
        env.procedureMap[name] = procedure
        destoryObject(self)
        return procedure

    def inverseProcedure(self, name: str) -> Tuple['QProcedure', str]:
        procedure = self.procedureMap.get(name)
        if procedure is None:
            raise Error.ArgumentError(f"Don't have procedure name: {name}!")
        inversedProcedure = None  # type: 'QProcedure'
        inversedProcedureName = None  # type: str
        if name.endswith('__inversed'):
            inversedProcedureName = name[:-10]
            inversedProcedure = self.procedureMap.get(inversedProcedureName)
            if inversedProcedure is not None:
                return inversedProcedure, inversedProcedureName
        else:
            inversedProcedureName = name + '__inversed'
            inversedProcedure = self.procedureMap.get(inversedProcedureName)
            if inversedProcedure is not None:
                return inversedProcedure, inversedProcedureName
        inversedProcedure = deepcopy(procedure)
        inversedProcedure.name = inversedProcedureName
        self.procedureMap[inversedProcedureName] = inversedProcedure
        for index, circuitLine in enumerate(reversed(procedure.circuit)):
            newLine = deepcopy(circuitLine)
            if isinstance(newLine.data, QProcedureOP):
                data, name = self.inverseProcedure(circuitLine.data.name)
                op = deepcopy(circuitLine.data)  # type: QProcedureOP
                op.procedureData = data
                op.name = name
                newLine.data = op
            else:
                newLine.data = circuitLine.data.getInverse()
            inversedProcedure.circuit[index] = newLine
        return inversedProcedure, inversedProcedureName

    def reverseProcedure(self, name: str) -> Tuple['QProcedure', str]:
        procedure = self.procedureMap.get(name)
        if procedure is None:
            raise Error.ArgumentError(f"Don't have procedure name: {name}!")
        reversedProcedure = None  # type: 'QProcedure'
        reversedProcedureName = None  # type: str
        if name.endswith('__reversed'):
            reversedProcedureName = name[:-10]
            reversedProcedure = self.procedureMap.get(reversedProcedureName)
            if reversedProcedure is not None:
                return reversedProcedure, reversedProcedureName
        else:
            reversedProcedureName = name + '__reversed'
            reversedProcedure = self.procedureMap.get(reversedProcedureName)
            if reversedProcedure is not None:
                return reversedProcedure, reversedProcedureName
        reversedProcedure = deepcopy(procedure)
        reversedProcedure.name = reversedProcedureName
        self.procedureMap[reversedProcedureName] = reversedProcedure
        for index, circuitLine in enumerate(reversed(procedure.circuit)):
            newLine = deepcopy(circuitLine)
            if isinstance(newLine.data, QProcedureOP):
                data, name = self.reverseProcedure(circuitLine.data.name)
                op = deepcopy(circuitLine.data)  # type: QProcedureOP
                op.procedureData = data
                op.name = name
                newLine.data = op
            reversedProcedure.circuit[index] = newLine
        return reversedProcedure, reversedProcedureName

    def publish(self, applyModule=True) -> None:
        program = PBProgram()
        self.program = program
        program.sdkVersion = sdkVersion
        QEnvToProtobuf(self.program, self)

        if applyModule:
            # filter the circuit by Modules
            for module in self.usedModuleList:
                self.program = module(self.program)

    def commit(self, shots: int, fetchMeasure=True, downloadResult=True, debug: Optional[str] = None) -> Dict[
        str, Union[str, Dict[str, int]]]:
        """
        Switch local/cloud commitment by prefix of backend name

        Example:

        env.commit(1024)

        env.commit(1024, fetchMeasure=True)

        env.commit(1024, downloadResult=False)

        :param shots: experiment counts
        :param fetchMeasure: named param, default is False, means 'Extract data from measurement results', downloadResult must be True
        :param downloadResult: named param, default is True, means 'Download experiment results from the server'
        :return: local or cloud commit result

        Successful:

        {status: 'success', origin: resultFilePath, measure: measureDict}  # fetchMeasure=True

        {status: 'success', origin: resultFilePath, counts: 1024}  # fetchMeasure=False

        {status: 'success'}  # downloadResult=False

        failed:

        {status: 'error', reason: ''}

        {status: 'failed', reason: ''}
        """

        self.publish()  # circuit in Protobuf format

        if self.backendName.startswith('local_'):
            return self._localCommit(shots, fetchMeasure)
        elif self.backendName.startswith('cloud_'):
            return self._cloudCommit(shots, fetchMeasure, downloadResult, debug)
        else:
            raise Error.ArgumentError(f"Invalid backendName => {self.backendName}")

    def _localCommit(self, shots: int, fetchMeasure: bool) -> Dict[str, Union[str, Dict[str, int]]]:
        """
        Local commitment

        :return: task result
        """

        if noLocalTask is not None:
            raise Error.RuntimeError('Local tasks are not allowed in the online environment!')

        # import the backend plugin according to the backend name
        module = _loadPythonModule(
            f'QCompute.OpenSimulator.{self.backendName}')
        if module is None:
            module = _loadPythonModule(
                f'QCompute.Simulator.{self.backendName}')
        if module is None:
            raise Error.ArgumentError(f"Invalid local backend => {self.backendName}!")
        backendClass = getattr(module, 'Backend')

        # configure the parameters
        backend = backendClass()  # type: 'QImplement'
        backend.program = self.program
        backend.shots = shots
        backend.backendArgument = self.backendArgument
        # execution
        backend.commit()

        # wrap taskResult
        if backend.result.code != 0:
            if backend.result.log != '':
                logFd, logFn = tempfile.mkstemp(prefix="local.", suffix=".log", dir=outputPath)
                with os.fdopen(logFd, "wt") as file:
                    file.write(backend.result.log)
            return {"status": "error", "reason": backend.result.log}

        if backend.result.counts is not None or backend.result.state is not None:
            cRegCount = max(self.program.head.usingCRegList) + 1

            if backend.result.counts is not None:
                backend.result.counts = formatMeasure(backend.result.counts, cRegCount)

            originFd, originFn = tempfile.mkstemp(prefix="local.", suffix=".origin.json", dir=outputPath)
            rsplitedFn = originFn.rsplit('.', 4)
            taskResult = {'taskId': rsplitedFn[-3], 'status': 'success'}
            if backend.result.output != '':
                with os.fdopen(originFd, "wt") as fObj:
                    fObj.write(backend.result.output)
                taskResult["origin"] = originFn
            else:
                os.close(originFd)

            if backend.result.counts is not None:
                measureFn = Path(originFn[:-12] + '.measure.json')
                with open(measureFn, 'wt') as file:
                    file.write(json.dumps(backend.result.counts))
            elif backend.result.state is not None:
                measureFn = Path(originFn[:-12] + '.measure.txt')
                with open(measureFn, 'wt') as file:
                    file.write(str(backend.result.state))
            taskResult["measure"] = str(measureFn)

            if backend.result.log != '':
                logFn = Path(originFn[:-12] + '.log')
                with open(logFn, 'wt') as file:
                    file.write(backend.result.log)
                taskResult["log"] = str(logFn)

            if fetchMeasure:
                taskResult['ancilla'] = {}
                taskResult['ancilla']['usedQRegList'] = backend.result.ancilla.usedQRegList
                taskResult['ancilla']['usedCRegList'] = backend.result.ancilla.usedCRegList
                taskResult['ancilla']['compactedQRegDict'] = backend.result.ancilla.compactedQRegDict
                taskResult['ancilla']['compactedCRegDict'] = backend.result.ancilla.compactedCRegDict
                taskResult['shots'] = backend.result.shots
                taskResult['counts'] = backend.result.counts
                taskResult['state'] = backend.result.state

            return taskResult

        return {"status": "failed", "reason": backend.result.log}

    def _cloudCommit(self, shots: int, fetchMeasure: bool, downloadResult: bool, debug: Optional[str]) -> Dict[
        str, Union[str, Dict[str, int]]]:
        """
        Cloud Commitment

        :return: task result
        """

        self.publish(False)  # circuit in Protobuf format
        programBuf = self.program.SerializeToString()  # the sequential bytes of the circuit which is already in PB
        with open(circuitPackageFile, 'wb') as file:
            file.write(programBuf)

        modules = []
        for module in self.usedModuleList:
            modules.append((module.__class__.__name__, module.arguments))

        # todo process the file and upload failed case
        task = QTask()
        task.uploadCircuit(circuitPackageFile)
        backend = self.backendName[6:]  # omit the prefix `cloud_`
        task.create(len(self.program.head.usingQRegList), shots, backend, None, modules, debug)

        outputInfo = True
        if outputInfo:
            print(f'Circuit upload successful, circuitId => {task.circuitId} taskId => {task.taskId}')

        # skip waiting when the relevant variable exists
        if noWaitTask is not None:
            return {"taskId": task.taskId}

        taskResult = task.wait(fetchMeasure=fetchMeasure, downloadResult=downloadResult)
        if type(taskResult) == str:
            print(taskResult)
        elif taskResult.get('counts') is not None:
            cRegCount = max(self.program.head.usingCRegList) + 1
            taskResult['counts'] = formatMeasure(taskResult['counts'], cRegCount)

        return taskResult

    def module(self, moduleObj: 'ModuleImplement') -> None:
        """
        Add processing Modules, register module object and params

        Example:

        env.module(CompositeGate())

        env.module(UnrollCircuit({'errorOnUnsupported': True, 'targetGates': [CX, U]}))

        :param moduleObj: module object
        """

        self.usedModuleList.append(moduleObj)


def _loadPythonModule(moduleName: str):
    """
    Load module from file system.

    :param moduleName: Module name
    :return: Module object
    """

    moduleSpec = importlib.util.find_spec(moduleName)
    if moduleSpec is None:
        return None
    module = importlib.util.module_from_spec(moduleSpec)
    moduleSpec.loader.exec_module(module)
    return module
