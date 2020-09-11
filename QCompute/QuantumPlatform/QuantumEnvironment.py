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
import traceback

from QCompute.Define import circuitPackageFile, sdkVersion, noWaitTask, noLocalTask
from QCompute.Define.Settings import outputInfo
from QCompute.QuantumPlatform import Error
from QCompute.QuantumPlatform.ProcedureParams import ProcedureParams
from QCompute.QuantumPlatform.QuantumOperation.QuantumProcedure import QuantumProcedure
from QCompute.QuantumPlatform.QuantumRegisters import QuantumRegisters
from QCompute.QuantumPlatform.QuantumTask import _uploadCircuit, _createTask, _waitTask
from QCompute.QuantumPlatform.Utilities import _mergePBList, _formatMeasure
from QCompute.QuantumProtobuf.Library.PlatformStruct_pb2 import Program


class QuantumEnvironment:
    """
    QuantumEnvironment class

    The member variable is the static saved state of the circuit.

    Quantum registers in use, classical registers, circuits in each line, and modules for sequential process.

    The backend function selects the process types, local or cloud.

    Configurations can be passed to the backend according to alternative parameters.

    The commitXXX function series call local or cloud subroutine according to prefix of backend names

    The publish function generates circuit definition in Protobuf format from the statical circuit data.

    The module function inserts the processing module to the end of module list.
    """

    circuitSerialNumber = 0
    """
    Circuit serial number in one single project
    """

    def __init__(self, status=None):
        # quantum registers
        self.Q = QuantumRegisters(self)
        # classical registers
        self.ClassicRegister = set()
        # procedure parameters
        self.params = ProcedureParams()

        # circuit
        self.circuit = []
        # processing module
        self.usedModule = []
        # procedure
        self.procedureMap = {}

        # status
        self.circuitSerialNumber = QuantumEnvironment.circuitSerialNumber
        QuantumEnvironment.circuitSerialNumber += 1
        self.status = status

    def backend(self, backendEnum, *backendParam):
        """
        Setup for backends(simulator/physical machines)

        Example:

        env = QuantumEnvironment()

        env.backend(BackendName.LocalBaiduSim2)

        or

        env.backend(BackendName.LocalBaiduSim2, Sim2Param.Dense_Matmul_Probability)

        :param backendEnum: enum of the backend
        :param backendParam: parameters of the backend(enum or string or number)
        """

        self.backendName = backendEnum.value
        self.backendParam = backendParam

    def convertToProcedure(self, name, env):
        """
        Convert to sub-procedure

        Self env will be destroyed

        Example:

        procedure0 = env.makeProcedure('procedure0')

        :param name: name of sub procedure(not allow duplication)
        :param env: env of sub procedure
        :return: QuantumProcedure
        """

        procedure = QuantumProcedure(name, self.params, self.Q, self.circuit)

        # insert it into quantum programming environment
        if env.procedureMap.get(name) != None:
            raise Error.ParamError(f'same procedure name "{name}"')
        env.procedureMap[name] = procedure

        # self destroy
        self.__dict__.clear()

        return procedure

    def commit(self, shots, **kwargs):
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

        if self.backendName.startswith('local_'):
            return self._localCommit(shots, **kwargs)
        elif self.backendName.startswith('cloud_'):
            return self._cloudCommit(shots, **kwargs)
        elif self.backendName.startswith('agent_'):
            return self._agentCommit(shots, **kwargs)
        else:
            raise Error.ParamError(f"invalid backendName => {self.backendName}")

    def _localCommit(self, shots, **kwargs):
        """
        Local commitment

        :return: task result
        """

        if noLocalTask is not None:
            raise Error.RuntimeError('Local tasks are not allowed in the online environment');

        self.publish()  # circuit in Protobuf format

        # import the simulator plugin according to the backend name
        module = _loadPythonModule(
            f'QCompute.OpenSimulator.{self.backendName}')
        if module is None:
            module = _loadPythonModule(
                f'QCompute.Simulator.{self.backendName}')
        if module is None:
            raise Error.ParamError(f"invalid local backend => {self.backendName}")
        simulatorClass = getattr(module, 'Backend')

        # configure the parameters
        simulator = simulatorClass()
        simulator.program = self.program
        simulator.shots = shots
        simulator.backendParam = self.backendParam
        # execution
        simulator.commit()

        # wrap taskResult
        output = simulator.result.output
        if simulator.result.code != 0:
            return {"status": "error", "reason": output}

        if simulator.result.counts:
            cRegCount = max(self.program.head.usingCRegs) + 1
            simulator.result.counts = _formatMeasure(simulator.result.counts, cRegCount)

            originFd, originFn = tempfile.mkstemp(suffix=".json", prefix="local.", dir="./Output")
            with os.fdopen(originFd, "wt") as fObj:
                fObj.write(output)
            taskResult = {"status": "success", "origin": originFn}

            if kwargs.get("fetchMeasure", False):
                taskResult["counts"] = simulator.result.counts
            else:
                measureFd, measureFn = tempfile.mkstemp(suffix=".json", prefix="local.", dir="./Output")
                with os.fdopen(measureFd, "wt") as fObj:
                    fObj.write(json.dumps(simulator.result.counts))
                taskResult["measure"] = measureFn

            return taskResult

        return {"status": "failed", "reason": output}

    def _cloudCommit(self, shots, **kwargs):
        """
        Cloud Commitment

        :return: task result
        """

        circuitId = None
        taskId = None
        if self.status is not None:
            circuitId, taskId = self.status.getTask(self.circuitSerialNumber)
        if circuitId is not None and taskId is not None and outputInfo:
            print(f"Circuit already uploaded, circuitId => {circuitId} taskId => {taskId}")
        else:
            self.publish(False)  # circuit in Protobuf format
            programBuf = self.program.SerializeToString()  # the sequential bytes of the circuit which is already in PB
            with open(circuitPackageFile, 'wb') as file:
                file.write(programBuf)

            modules = []
            for module in self.usedModule:
                modules.append((module.__class__.__name__, module.params))

            # todo process the file and upload failed case
            token, circuitId = _uploadCircuit(circuitPackageFile)
            backend = self.backendName[6:]  # omit the prefix `cloud_`
            taskId = _createTask(token, circuitId, shots, backend, self.backendParam, modules)
            if self.status is not None:
                self.status.addTask(self.circuitSerialNumber, circuitId, taskId)

            if outputInfo:
                print(f"Circuit upload successful, circuitId => {circuitId} taskId => {taskId}")

        # skip waiting when the relevant variable exists
        if noWaitTask is not None:
            return {"taskId": taskId}

        taskResult = _waitTask(token, taskId, **kwargs)
        if type(taskResult) == str:
            print(taskResult)
        elif taskResult.get('counts') is not None:
            cRegCount = max(self.program.head.usingCRegs) + 1
            taskResult['counts'] = _formatMeasure(taskResult['counts'], cRegCount)

        return taskResult

    def _agentCommit(self, shots, **kwargs):
        """
        Agent commitment

        :return: task result
        """

        if noLocalTask is not None:
            raise Error.RuntimeError('Agent tasks are not allowed in the online environment');

        try:
            self.publish()  # circuit in Protobuf format

            # import the agent plugin according to the backend name
            module = _loadPythonModule(
                f'QCompute.Agent.{self.backendName}')
            if module is None:
                raise Error.ParamError(f"invalid agent backend => {self.backendName}")
            simulatorClass = getattr(module, 'Backend')

            # configure the parameters
            agent = simulatorClass()
            agent.program = self.program
            agent.shots = shots
            agent.backendParam = self.backendParam
            # execution
            agent.commit()

            # wrap taskResult
            output = agent.result.output
            if agent.result.code != 0:
                return {"status": "error", "reason": output}

            if agent.result.counts:
                cRegCount = max(self.program.head.usingCRegs) + 1
                agent.result.counts = _formatMeasure(agent.result.counts, cRegCount)

                originFd, originFn = tempfile.mkstemp(suffix=".json", prefix="local.", dir="./Output")
                with os.fdopen(originFd, "wt") as fObj:
                    fObj.write(output)
                taskResult = {"status": "success", "origin": originFn}

                if kwargs.get("fetchMeasure", False):
                    taskResult["counts"] = agent.result.counts
                else:
                    measureFd, measureFn = tempfile.mkstemp(suffix=".json", prefix="local.", dir="./Output")
                    with os.fdopen(measureFd, "wt") as fObj:
                        fObj.write(json.dumps(agent.result.counts))
                    taskResult["measure"] = measureFn

                return taskResult

            return {"status": "failed", "reason": output}
        except Exception:
            raise Error.RuntimeError(traceback.format_exc())

    def publish(self, applyModule=True):
        """
        Generate the circuit data

        Save to memory for local simulator instead of writing to file

        Example:

        env.publish()

        pprint(env.circuit)

        :param applyModule: make module running and effecting
        """

        self.program = Program()
        self.program.sdkVersion = sdkVersion

        # fill in the head
        _mergePBList(self.program.head.usingQRegs, sorted(
            self.Q.registerDict.keys()))  # all the index of the used q registers are put into usingQRegs
        _mergePBList(self.program.head.usingCRegs, sorted(
            self.ClassicRegister))  # all the index of the used classical registers are put into usingCRegs
        for name, procedure in self.procedureMap.items():
            self._makeProcedure(name, procedure)
        # fill in the circuit
        for circuitLine in self.circuit:
            self.program.body.circuit.append(circuitLine._toPB())

        if applyModule:
            # filter the circuit by Modules
            for module in self.usedModule:
                self.program = module(self.program)

    def _makeProcedure(self, name, procedure):
        pbProcedure = self.program.body.procedureMap[name]
        if len(procedure.params.paramsDict) == 0:
            pbProcedure.paramCount = 0
        else:
            pbProcedure.paramCount = max(param for param in procedure.params.paramsDict.keys()) + 1
        _mergePBList(pbProcedure.usingQRegs, sorted(
            procedure.Q.registerDict.keys()))  # all the index of q regs from sub procedures are storing into usingQRegs
        for circuitLine in procedure.circuit:
            pbProcedure.circuit.append(circuitLine._toPB())

    def module(self, moduleObj):
        """
        Add processing Modules, register module object and params

        Example:

        env.module(CompositeGate())

        env.module(UnrollCircuit({'errorOnUnsupported': True, 'targetGates': [CX, U]}))

        :param moduleObj: module object
        """

        self.usedModule.append(moduleObj)


def _loadPythonModule(moduleName):
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
