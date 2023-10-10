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
Quantum Environment
"""
from QCompute.QPlatform.Processor.BackendFilter import filterCloudBackend

FileErrorCode = 8

import json
import os
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Set, List, Dict, TYPE_CHECKING, Union, Optional, Tuple, Any

from QCompute import Define
from QCompute.Define import Settings
from QCompute.Define.Utils import loadPythonModule, clearOutputDir
from QCompute.OpenConvertor.CircuitToDrawConsole import CircuitToDrawConsole
from QCompute.OpenModule import ModuleImplement
from QCompute.OpenSimulator import QResult, QImplement, QPhotonicResult
from QCompute.QPlatform import Error, ModuleErrorCode, BackendName, getBackendFromName
from QCompute.QPlatform.CircuitTools import QEnvToProtobuf
from QCompute.QPlatform.InteractiveModule import InteractiveModule
from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterPool
from QCompute.QPlatform.Processor.ModuleFilter import filterModule, printModuleListDescription
from QCompute.QPlatform.Processor.PostProcessor import formatMeasure
from QCompute.QPlatform.QNoise import QNoise, QNoiseDefine
from QCompute.QPlatform.QOperation import getGateBits
from QCompute.QPlatform.QOperation.QProcedure import QProcedure, QProcedureOP
from QCompute.QPlatform.QRegPool import QRegPool
from QCompute.QPlatform.QTask import QTask
from QCompute.QPlatform.Utilities import destoryObject
from QCompute.QProtobuf import PBProgram
from QCompute.Utilize.ControlledCircuit import getControlledCircuit

if TYPE_CHECKING:
    from QCompute.QPlatform import ServerModule


class QEnv:
    """
    Quantum Environment class.

    The member variable is the static saved state of the circuit.

    Quantum registers in use, classical registers, circuits in each line, and modules for sequential process.

    The backend function selects the process types, local or cloud.

    Configurations can be passed to the backend according to alternative parameters.

    The commit function series call local or cloud subroutine according to prefix of backend names

    The publish function generates circuit definition in Protobuf format from the statical circuit data.

    The module function inserts the processing module to the end of module list.
    """

    def __init__(self) -> None:
        self.Q = QRegPool(self)
        self.Parameter = ProcedureParameterPool()

        self.measuredQRegSet: Set[int] = set()
        self.measuredCRegSet: Set[int] = set()
        self.circuit: List['CircuitLine'] = []
        self.procedureMap: Dict[str, 'QProcedure'] = {}
        self.program: Optional['PBProgram'] = None

        self.backendName: Optional[BackendName] = None
        self.backendArgument: Optional[List] = None
        self.shots = 0

        self.usingModuleList: List['ModuleImplement'] = []
        self.usedServerModuleList: List[Tuple[str, Dict]] = []

        self.noiseDefineMap: Dict[str, List[QNoiseDefine]] = {}

    def backend(self, backendName: Union['BackendName', str], *backendArgument: Any) -> None:
        """
        Set backend.
        """
        if type(backendName) is str:
            self.backendName = getBackendFromName(backendName)
        else:
            self.backendName = backendName
        self.backendArgument = list(backendArgument)

    def convertToProcedure(self, name: str, env: 'QEnv') -> 'QProcedure':
        if name in env.procedureMap:
            raise Error.ArgumentError(f'Duplicate procedure name: {name}!', ModuleErrorCode, FileErrorCode, 1)

        if len(self.procedureMap) > 0:
            for subName, subProc in self.procedureMap.items():
                if subName not in env.procedureMap:
                    env.procedureMap[subName] = subProc

        procedure = QProcedure(name, self.Q, self.Parameter, self.circuit)
        env.procedureMap[name] = procedure
        destoryObject(self)
        return procedure

    def inverseProcedure(self, name: str) -> Tuple['QProcedure', str]:
        procedure = self.procedureMap.get(name)
        if procedure is None:
            raise Error.ArgumentError(f"Don't have procedure name: {name}!", ModuleErrorCode, FileErrorCode, 2)
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
        inversedProcedure.Q.changeEnv(inversedProcedure)
        self.procedureMap[inversedProcedureName] = inversedProcedure
        for index, circuitLine in enumerate(reversed(procedure.circuit)):
            newLine = deepcopy(circuitLine)
            if isinstance(newLine.data, QProcedureOP):
                data, name = self.inverseProcedure(circuitLine.data.name)
                op: QProcedureOP = deepcopy(circuitLine.data)
                op.procedureData = data
                op.name = name
                newLine.data = op
            else:
                newLine.data = circuitLine.data.getInversed()
            inversedProcedure.circuit[index] = newLine
        return inversedProcedure, inversedProcedureName

    def reverseProcedure(self, name: str) -> Tuple['QProcedure', str]:
        procedure = self.procedureMap.get(name)
        if procedure is None:
            raise Error.ArgumentError(f"Don't have procedure name: {name}!", ModuleErrorCode, FileErrorCode, 3)
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
        reversedProcedure.Q.changeEnv(reversedProcedure)
        self.procedureMap[reversedProcedureName] = reversedProcedure
        for index, circuitLine in enumerate(reversed(procedure.circuit)):
            newLine = deepcopy(circuitLine)
            if isinstance(newLine.data, QProcedureOP):
                data, name = self.reverseProcedure(circuitLine.data.name)
                op: QProcedureOP = deepcopy(circuitLine.data)
                op.procedureData = data
                op.name = name
                newLine.data = op
            reversedProcedure.circuit[index] = newLine
        return reversedProcedure, reversedProcedureName

    def inverseCircuit(self) -> 'QEnv':
        inversedEnv = deepcopy(self)
        inversedEnv.Q.changeEnv(inversedEnv)
        for index, circuitLine in enumerate(reversed(self.circuit)):
            newLine = deepcopy(circuitLine)
            if isinstance(newLine.data, QProcedureOP):
                data, name = inversedEnv.inverseProcedure(circuitLine.data.name)
                op: QProcedureOP = deepcopy(circuitLine.data)
                op.procedureData = data
                op.name = name
                newLine.data = op
            else:
                newLine.data = circuitLine.data.getInversed()
            inversedEnv.circuit[index] = newLine
        return inversedEnv

    def reverseCircuit(self) -> 'QEnv':
        reversedEnv = deepcopy(self)
        reversedEnv.Q.changeEnv(reversedEnv)
        for index, circuitLine in enumerate(reversed(self.circuit)):
            newLine = deepcopy(circuitLine)
            if isinstance(newLine.data, QProcedureOP):
                data, name = self.reverseProcedure(circuitLine.data.name)
                op: QProcedureOP = deepcopy(circuitLine.data)
                op.procedureData = data
                op.name = name
                newLine.data = op
            reversedEnv.circuit[index] = newLine
        return reversedEnv

    def controlProcedure(self, name: str, cuFirst: bool = False) -> Tuple[str, 'QProcedure']:
        """
        Control Procedure
        """

        procedure = self.procedureMap.get(name)
        if procedure is None:
            raise Error.ArgumentError(f"Don't have procedure name: {name}!", ModuleErrorCode, FileErrorCode, 4)

        controlledProcedureName = name + '__controlled'
        controlledProcedure = self.procedureMap.get(controlledProcedureName)
        if controlledProcedure is not None:
            return controlledProcedureName, controlledProcedure

        controlledProcedure = deepcopy(procedure)
        controlledProcedure.name = controlledProcedureName
        controlledProcedure.Q.changeEnv(controlledProcedure)
        controlQRegIndex = max(controlledProcedure.Q.registerMap.keys()) + 1
        controlledProcedure.Q(controlQRegIndex)
        controlledProcedure.circuit.clear()
        self.procedureMap[controlledProcedureName] = controlledProcedure
        for circuitLine in procedure.circuit:
            newCircuitLineList = getControlledCircuit(self, circuitLine, controlQRegIndex, cuFirst)
            controlledProcedure.circuit.extend(newCircuitLineList)
        return controlledProcedureName, controlledProcedure

    def publish(self, applyModule=True, program: PBProgram = None) -> List['ModuleImplement']:
        """
        To protobuf.
        """
        if program is None:
            program = PBProgram()
            program.sdkVersion = Define.sdkVersion
            QEnvToProtobuf(program, self)
        self.program = program

        moduleStep = 0
        circuitToDrawTerminal = CircuitToDrawConsole()
        if Settings.outputInfo and (
                Settings.drawCircuitControl is None or moduleStep in Settings.drawCircuitControl):
            asciiPic = circuitToDrawTerminal.convert(self.program)
            print('Origin circuit:')
            print(asciiPic)

        if applyModule:
            # filter the circuit by Modules
            usedModuleList = filterModule(program, self.backendName, self.usingModuleList)
            for module in usedModuleList:
                moduleStep += 1
                self.program = module(self.program)

                if Settings.outputInfo and (
                        Settings.drawCircuitControl is None or moduleStep in Settings.drawCircuitControl):
                    asciiPic = circuitToDrawTerminal.convert(self.program)
                    print(f'{module.__class__.__name__} pass...')
                    print(asciiPic)

            return usedModuleList
        else:
            return self.usingModuleList

    def commit(self, shots: int, fetchMeasure=True, downloadResult=True, program: PBProgram = None,
               notes: Optional[str] = None
               
               ) -> Dict[str, Union[str, Dict[str, int]]]:
        """
        Switch local/cloud commitment by prefix of backend name

        Example:

        env.commit(1024)

        env.commit(1024, fetchMeasure=True)

        env.commit(1024, downloadResult=False)

        env.commit(1024, fetchMeasure=True, notes='Task 001')

        env.commit(1024, fetchMeasure=True, program=program)

        :param shots: experiment counts

        :param fetchMeasure: named param, default is True, means 'Extract data from measurement results', downloadResult must be True

        :param downloadResult: named param, default is True, means 'Download experiment results from the server'

        :return: local or cloud commit result

        Successful:

        {status: 'success', origin: resultFilePath, counts: measureDict}  # fetchMeasure=True

        {status: 'success', origin: resultFilePath, measure: measurePath}  # fetchMeasure=False

        {status: 'success'}  # downloadResult=False

        failed:

        {status: 'error', reason: ''}

        {status: 'failed', reason: ''}
        """

        self.shots = shots

        ret: Dict[str, Union[str, Dict[str, int]]] = None
        if self.backendName.value.startswith('local_'):
            usedModuleList = self.publish(program=program)  # circuit in Protobuf format
            moduleList = [{
                'module': module.__class__.__name__,
                'arguments': module.arguments
            } for module in usedModuleList]
            ret = self._localCommit(fetchMeasure, moduleList)
        elif self.backendName.value.startswith('cloud_'):
            self.publish(False)  # circuit in Protobuf format
            moduleList = [(module.__class__.__name__, module.arguments) for module in self.usingModuleList]
            moduleList.extend(self.usedServerModuleList)

            ret = self._cloudCommit(fetchMeasure, downloadResult, moduleList, notes
                                    
                                    )
        elif self.backendName.value.startswith('service_'):
            usedModuleList = self.publish(program=program)  # circuit in Protobuf format
            moduleList = [{
                'module': module.__class__.__name__,
                'arguments': module.arguments
            } for module in usedModuleList]
            ret = self._serviceCommit(fetchMeasure, moduleList)
        else:
            raise Error.ArgumentError(f'Invalid backendName => {self.backendName.value}',
                ModuleErrorCode, FileErrorCode, 5)

        if Settings.outputInfo and 'moduleList' in ret:
            moduleList: List[str] = [moduleSetting['module'] for moduleSetting in ret['moduleList']]
            interactiveModule = InteractiveModule(self)
            print('Modules called sequentially')
            interactiveModule.printModuleList(moduleList)
            printModuleListDescription(moduleList)

        if Settings.autoClearOutputDirAfterFetchMeasure:
            clearOutputDir()

        return ret

    def _localCommit(self, fetchMeasure: bool, moduleList: []) -> Dict[str, Union[str, Dict[str, int]]]:
        """
        Local commitment

        :return: task result
        """

        if Define.noLocalTask is not None:
            raise Error.RuntimeError(
                'Local tasks are not allowed in the online environment!', ModuleErrorCode, FileErrorCode, 6)

        # import the backend plugin according to the backend name
        module = loadPythonModule(f'QCompute.OpenSimulator.{self.backendName.value}')
        if module is None:
            module = loadPythonModule(f'QCompute.Simulator.{self.backendName.value}')
        if module is None:
            raise Error.ArgumentError(
                f'Invalid local backend => {self.backendName.value}!', ModuleErrorCode, FileErrorCode, 7)

        backendClass = getattr(module, 'Backend')

        # configure the parameters
        backend: QImplement = backendClass()
        backend.program = self.program
        backend.shots = self.shots
        backend.backendArgument = self.backendArgument
        # execution
        backend.commit()

        # wrap taskResult
        if backend.result.code != 0:
            if backend.result.log != '':
                logFd, logFn = tempfile.mkstemp(prefix="local.", suffix=".log", dir=Define.outputDirPath)
                with os.fdopen(logFd, "wt") as file:
                    file.write(backend.result.log)
            return {"status": "error", "reason": backend.result.log}

        if 'photonic' not in self.backendName.value:
            if backend.result.counts is not None or backend.result.state is not None:
                cRegCount = max(self.program.head.usingCRegList) + 1
                if backend.result.counts is not None:
                    backend.result.counts = formatMeasure(backend.result.counts, cRegCount)

                try:
                    ret = QResult()
                    ret.fromJson(backend.result.output)
                    ret.moduleList = moduleList
                    backend.result.output = ret.toJson()
                except Exception as ex:
                    print(ex)

                originFd, originFn = tempfile.mkstemp(prefix="local.", suffix=".origin.json", dir=Define.outputDirPath)
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
                    measureFn.write_text(json.dumps(backend.result.counts), encoding='utf-8')
                elif backend.result.state is not None:
                    measureFn = Path(originFn[:-12] + '.measure.txt')
                    measureFn.write_text(str(backend.result.state), encoding='utf-8')
                taskResult["measure"] = str(measureFn)

                if backend.result.log != '':
                    logFn = Path(originFn[:-12] + '.log')
                    logFn.write_text(backend.result.log, encoding='utf-8')
                    taskResult["log"] = str(logFn)

                taskResult["moduleList"] = moduleList

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
        else:
            if backend.result.value is not None or backend.result.counts is not None:
                try:
                    ret = QPhotonicResult()
                    ret.fromJson(backend.result.output)
                    ret.moduleList = moduleList
                    backend.result.output = ret.toJson()
                except Exception as ex:
                    print(ex)

                originFd, originFn = tempfile.mkstemp(prefix="local.", suffix=".origin.json", dir=Define.outputDirPath)
                rsplitedFn = originFn.rsplit('.', 4)
                taskResult = {'taskId': rsplitedFn[-3], 'status': 'success'}
                if backend.result.output != '':
                    with os.fdopen(originFd, "wt") as fObj:
                        fObj.write(backend.result.output)
                    taskResult["origin"] = originFn
                else:
                    os.close(originFd)

                if backend.result.value is not None:
                    measureFn = Path(originFn[:-12] + '.measure.json')
                    measureFn.write_text(json.dumps(backend.result.value), encoding='utf-8')
                elif backend.result.state is not None:
                    measureFn = Path(originFn[:-12] + '.measure.txt')
                    measureFn.write_text(str(backend.result.state), encoding='utf-8')
                taskResult["measure"] = str(measureFn)

                if backend.result.log != '':
                    logFn = Path(originFn[:-12] + '.log')
                    logFn.write_text(backend.result.log, encoding='utf-8')
                    taskResult["log"] = str(logFn)

                taskResult["moduleList"] = moduleList

                if fetchMeasure:
                    taskResult['ancilla'] = {}
                    taskResult['ancilla']['usedQRegList'] = backend.result.ancilla.usedQRegList
                    taskResult['ancilla']['usedCRegList'] = backend.result.ancilla.usedCRegList
                    taskResult['ancilla']['compactedQRegDict'] = backend.result.ancilla.compactedQRegDict
                    taskResult['ancilla']['compactedCRegDict'] = backend.result.ancilla.compactedCRegDict
                    taskResult['shots'] = backend.result.shots
                    taskResult['value'] = backend.result.value
                    taskResult['state'] = backend.result.state

                return taskResult

        return {"status": "failed", "reason": backend.result.log}

    def _cloudCommit(self, fetchMeasure: bool, downloadResult: bool, moduleList: List[Tuple[str, Any]],
                     notes: str
                     
                     ) -> Dict[
        str, Union[str, Dict[str, int]]]:
        """
        Cloud Commitment

        :return: task result
        """

        filterCloudBackend(self)

        # the sequential bytes of the circuit which is already in PB
        programBuf: bytes = self.program.SerializeToString()

        if not Settings.cloudTaskDoNotWriteFile:
            circuitPackageFd, circuitPackageFn = tempfile.mkstemp(prefix="circuit.", suffix=".pb",
                                                                  dir=Define.outputDirPath)
            with os.fdopen(circuitPackageFd, "wb") as file:
                file.write(programBuf)
            if Settings.outputInfo:
                print(f'CircuitPackageFile: {circuitPackageFn}')

        while True:
            try:
                # todo process the file and upload failed case
                task = QTask()
                task.uploadCircuit(programBuf)
                backend = self.backendName.value[6:]  # omit the prefix `cloud_`
                task.createCircuitTask(self.shots, backend, len(self.program.head.usingQRegList), self.backendArgument,
                                       moduleList, notes
                                       
                                       )

                if Settings.outputInfo:
                    print(f'Circuit upload successful, circuitId => {task.circuitId} taskId => {task.taskId}')

                # skip waiting when the relevant variable exists
                if Define.noWaitTask is not None:
                    return {"taskId": task.taskId}

                taskResult = task.waitCircuitTask(fetchMeasure=fetchMeasure, downloadResult=downloadResult)
                if type(taskResult) == str:
                    print(taskResult)
                elif taskResult.get('counts') is not None:
                    cRegCount = max(self.program.head.usingCRegList) + 1
                    taskResult['counts'] = formatMeasure(taskResult['counts'], cRegCount)

                return taskResult
            except Exception as e:
                if Settings.alwaysRetryTask:
                    print('Always Retry Task!')
                else:
                    raise

    def _serviceCommit(self, fetchMeasure: bool, moduleList: []) -> Dict[str, Union[str, Dict[str, int]]]:
        """
        Service commitment

        :return: task result
        """

        # import the service plugin according to the backend name
        module = loadPythonModule(f'QCompute.OpenService.{self.backendName.value}')
        # if module is None:
        #     module = loadPythonModule(f'QCompute.Service.{self.backendName.value}')
        if module is None:
            raise Error.ArgumentError(
                f'Invalid service backend => {self.backendName.value}!', ModuleErrorCode, FileErrorCode, 8)

        backendClass = getattr(module, 'Backend')

        # configure the parameters
        backend: QImplement = backendClass()
        backend.program = self.program
        backend.shots = self.shots
        backend.backendArgument = self.backendArgument
        # execution
        backend.commit()

        taskResult = {'status': 'success'}
        taskResult['shots'] = backend.result.shots
        taskResult['counts'] = backend.result.counts
        return taskResult

    def module(self, moduleObj: 'ModuleImplement') -> None:
        """
        Add processing Modules, register module object and params

        Example:

        env.module(CompositeGate())

        env.module(UnrollCircuit({'errorOnUnsupported': True, 'targetGates': [CX, U]}))

        :param moduleObj: module object
        """

        self.usingModuleList.append(moduleObj)

    def serverModule(self, module: 'ServerModule', arguments: Dict) -> None:
        """
        Add server processing Modules, register module object and params

        Example:

        env.serverModule(ServerModule.MappingToBaiduQPUQian, {"disable": True})

        :param module: module enum

        :param arguments: param
        """

        self.usedServerModuleList.append((module.value, arguments))

    def noise(self, gateNameList: List[str], noiseList: List[QNoise], qRegList: List[int] = None,
              positionList: List[int] = None) -> None:
        """
        Add noise

        :param gateNameList: a list of gate names
        :param noiseList: a list of noises
        :param qRegList: a list of qubits. When it's None, noises act on gates in all qubits; otherwise, noises act on gates in the specified qubits.
        :param positionList: a list of noise inserting locations. When it's None, noises act on all gates; otherwise, noises act on the specified gates in specified position.
        """
        for gateName in gateNameList:
            gateBits = getGateBits(gateName)
            if qRegList:
                if gateBits != len(set(qRegList)):
                    raise Error.ArgumentError(
                        f'Invalid qRegList({qRegList}) in noise {gateName}!', ModuleErrorCode, FileErrorCode, 9)

            for noise in noiseList:
                if 0 < noise.bits != gateBits:
                    raise Error.ArgumentError(
                        f'Invalid bits({gateBits}/{noise.bits}) in noise {gateName}!',
                        ModuleErrorCode, FileErrorCode, 10)

            noiseDefine = QNoiseDefine(noiseList, qRegList, positionList)
            defineList = self.noiseDefineMap.get(gateName)
            if defineList is None:
                defineList = []
                self.noiseDefineMap[gateName] = defineList
            defineList.append(noiseDefine)

    def join(self, env: 'QEnv', startQRegIndex: int = 0) -> None:
        """
        Join quantum environment

        :param env: a target quantum environment, which is joined in the current environment.
        :param startQRegIndex: a index of QRegister is used to indicate which qubit the connection starts from.
        """

        if len(env.noiseDefineMap) > 0:
            raise Error.ArgumentError(
                f'Unsupported noise in QEnv.join and concatEnv!', ModuleErrorCode, FileErrorCode, 11)

        env = deepcopy(env)
        nameMap: Dict[str, str] = {}
        for procedureName, procedure in env.procedureMap.items():
            newProcedureName = procedureName + '_joined'
            num = 0
            while newProcedureName + str(num) in self.procedureMap:
                num += 1
            nameMap[procedureName] = newProcedureName + str(num)
        for procedureName, procedure in env.procedureMap.items():
            for circuitLine in procedure.circuit:
                if isinstance(circuitLine.data, QProcedureOP):
                    circuitLine.data.name = nameMap[circuitLine.data.name]
                    circuitLine.data.procedureData.name = circuitLine.data.name
            self.procedureMap[nameMap[procedureName]] = procedure
        for circuitLine in env.circuit:
            self.circuit.append(circuitLine)
            for index, qreg in enumerate(circuitLine.qRegList):
                qreg += startQRegIndex
                circuitLine.qRegList[index] = qreg
                self.Q(qreg)
            if isinstance(circuitLine.data, QProcedureOP):
                circuitLine.data.name = nameMap[circuitLine.data.name]
                circuitLine.data.procedureData.name = circuitLine.data.name


def concatQEnv(*envList: QEnv) -> QEnv:
    """
    Concatenate quantum environments

    :param envList: a list of quantum environments

    :return: a new quantum environment with all environments concatenated
    """

    ret = deepcopy(envList[0])
    for env in envList[1:]:
        ret.join(env)
    return ret
