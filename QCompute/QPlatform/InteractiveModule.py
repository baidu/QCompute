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
Interactive Module
"""
import copy
import json
import sys
from typing import List, Dict, Type

from QCompute import Define



from QCompute.OpenConvertor.CircuitToDrawConsole import CircuitToDrawConsole
from QCompute.OpenModule import ModuleImplement
from QCompute.OpenModule.CompositeGateModule import CompositeGateModule
from QCompute.OpenModule.CompressGateModule import CompressGateModule
from QCompute.OpenModule.InverseCircuitModule import InverseCircuitModule
from QCompute.OpenModule.ReverseCircuitModule import ReverseCircuitModule
from QCompute.OpenModule.UnrollCircuitModule import UnrollCircuitModule
from QCompute.OpenModule.UnrollProcedureModule import UnrollProcedureModule
from QCompute.QPlatform import QEnv, BackendName
from QCompute.QPlatform.CircuitTools import QEnvToProtobuf
from QCompute.QPlatform.Processor.ModuleFilter import filterModule
from QCompute.QProtobuf import PBProgram

_SimulatorModuleList = [
    [UnrollProcedureModule, CompositeGateModule, UnrollCircuitModule, CompressGateModule],
    [InverseCircuitModule, ReverseCircuitModule]
]
_HardwareOptionalModuleList = [UnrollProcedureModule, CompositeGateModule, InverseCircuitModule, ReverseCircuitModule]
_AerSimulatorModuleList = [
    [UnrollProcedureModule, CompositeGateModule, UnrollCircuitModule],
    [InverseCircuitModule, ReverseCircuitModule]
]

BackendModuleDict = {
    BackendName.LocalBaiduSim2: _SimulatorModuleList,
    
    BackendName.CloudBaiduSim2Water: _SimulatorModuleList,
    BackendName.CloudBaiduSim2Earth: _SimulatorModuleList,
    BackendName.CloudBaiduSim2Thunder: _SimulatorModuleList,
    BackendName.CloudBaiduSim2Heaven: _SimulatorModuleList,
    BackendName.CloudBaiduSim2Wind: _SimulatorModuleList,
    BackendName.CloudBaiduSim2Lake: _SimulatorModuleList,
    BackendName.CloudAerAtBD: _AerSimulatorModuleList,

    
}

BackendModuleNameDict = {
    BackendName.CloudBaiduQPUQian: ['MappingToBaiduQPUQianModule', 'UnrollCircuitToBaiduQPUQianModule'],
    BackendName.CloudIoPCAS: ['MappingToIoPCASModule', 'UnrollCircuitToIoPCASModule'],
    BackendName.CloudIonAPM: ['UnrollCircuitToIonAPMModule'],
    
}


class InteractiveModule:
    """
    Interactive Module
    """

    def __init__(self, env: QEnv):
        self.env = env
        self.circuitToDrawTerminal = CircuitToDrawConsole()
        self.originProgram = PBProgram()
        self.originProgram.sdkVersion = Define.sdkVersion
        QEnvToProtobuf(self.originProgram, env)
        self.backendName = env.backendName  # type: BackendName
        self.moduleSetting = BackendModuleDict.get(self.backendName)
        if self.moduleSetting:
            self.moduleDict = {}  # type: Dict[str, Type[ModuleImplement]]
            self.usingModuleList = []  # type: List[ModuleImplement]
            self.necessaryModuleList = self.moduleSetting[0]
            self.optionalModuleList = self.moduleSetting[1]
            self.necessaryModuleNameList = []  # type: List[str]
            self.optionalModuleNameList = []  # type: List[str]
            for module in self.necessaryModuleList:
                self.necessaryModuleNameList.append(module.__name__)
                self.moduleDict[module.__name__] = module
            for module in self.optionalModuleList:
                self.optionalModuleNameList.append(module.__name__)
                self.moduleDict[module.__name__] = module
        else:
            self.necessaryModuleNameList = BackendModuleNameDict[self.backendName]

    def interactive(self):
        if not self.moduleSetting:
            print(f'Interactive module [Unsupport backend: {self.backendName.name}]')
            exit(0)

        print(f'Interactive module [With backend: {self.backendName.name}]')
        print('(*): Module required\n')

        asciiPic = self.circuitToDrawTerminal.convert(self.originProgram)
        print('Origin circuit:')
        print(asciiPic)
        self.refurbishStatus()
        while True:
            try:
                cmd = input('(add, remove, move, draw, commit, exit) $')
                cmdList = cmd.split()
                verb = cmdList[0] if len(cmdList) >= 1 else None
                arguments = cmdList[1:] if len(cmdList) >= 2 else None
                self.do(verb, arguments)
            except Exception as ex:
                print(ex)

    def refurbishStatus(self):
        usingModuleList = filterModule(self.backendName, self.usingModuleList)
        usingModuleNameList = []  # type: List[str]
        reorderUsingModuleList = []  # type: List[ModuleImplement]
        for module in usingModuleList:
            if module in self.usingModuleList:
                usingModuleNameList.append(module.__class__.__name__)
                reorderUsingModuleList.append(module)
        self.usingModuleList = reorderUsingModuleList

        canBeUsedModuleNameList = []  # type: List[str]
        for moduleName in self.optionalModuleNameList:
            if moduleName not in usingModuleNameList:
                canBeUsedModuleNameList.append(moduleName)
        for moduleName in self.necessaryModuleNameList:
            if moduleName not in usingModuleNameList:
                canBeUsedModuleNameList.append(moduleName)
        print(f'Modules can be used: {self._markModuleNameList(canBeUsedModuleNameList)}')
        usingModuleListFormArrow = self._markModuleNameList(usingModuleNameList).replace(',', ' ->')
        print(f'Modules used by order: {usingModuleListFormArrow}')

    def _markModuleNameList(self, moduleNameList: List[str]) -> str:
        nameList = []  # type: List[str]
        for moduleName in moduleNameList:
            if moduleName in self.necessaryModuleNameList:
                nameList.append('(*)' + moduleName)
            else:
                nameList.append(moduleName)
        return json.dumps(nameList).replace('"', '')

    def do(self, verb: str, arguments: List[str]):
        if verb == 'add':
            self._add(arguments)
        elif verb == 'remove':
            self._remove(arguments)
        elif verb == 'move':
            self._move(arguments)
        elif verb == 'draw':
            self._draw()
        elif verb == 'commit':
            self._commit(arguments)
        elif verb == 'exit':
            self._exit()
        else:
            return
        self.refurbishStatus()

    def _add(self, arguments: List[str]):
        if arguments is None:
            print('add moduleName0 [moduleName1 moduleName2 moduleName3 ...]')
            return
        for moduleName in arguments:
            found = False
            for module in self.usingModuleList:
                if module.__class__.__name__ == moduleName:
                    print(f'Module {moduleName} already exists.')
                    found = True
                    break
            if found:
                continue
            module = self.moduleDict[moduleName]
            self.usingModuleList.append(module())

    def _remove(self, arguments: List[str]):
        if arguments is None:
            print('remove moduleName0 [moduleName1 moduleName2 moduleName3 ...]')
            return
        for moduleName in arguments:
            for index, module in enumerate(self.usingModuleList):
                if module.__class__.__name__ == moduleName:
                    del self.usingModuleList[index]

    def _move(self, arguments: List[str]):
        if arguments is None:
            print('move moduleName pos(-2/-1/1/2 ...)')
            return
        moduleName = arguments[0] if len(arguments) >= 1 else None
        movePos = int(arguments[1]) if len(arguments) >= 2 else None
        if not movePos:
            return
        for index, module in enumerate(self.usingModuleList):
            if module.__class__.__name__ == moduleName:
                del self.usingModuleList[index]
                if movePos < 0:
                    pos = index + movePos
                else:
                    pos = index + movePos
                if pos < 0:
                    pos = 0
                self.usingModuleList.insert(pos, module)
                return

    def _draw(self):
        program = self.originProgram
        for module in self.usingModuleList:
            program = module(program)
        asciiPic = self.circuitToDrawTerminal.convert(program)
        print('Circuit:')
        print(asciiPic)

    def _commit(self, arguments: List[str]):
        if arguments is None:
            print('commit shots')
            return
        shots = int(arguments[0]) if len(arguments) >= 1 else None
        self.env.usingModuleList = copy.copy(self.usingModuleList)
        for necessaryModule in BackendModuleDict[self.backendName][0]:
            found = False
            for usingModule in self.usingModuleList:
                if usingModule.__class__ is necessaryModule:
                    found = True
                    break
            if found:
                continue
            else:
                self.env.usingModuleList.append(necessaryModule({'disable': True}))

        self.env.commit(shots)

    def _exit(self):
        sys.exit()

    def printModuleList(self, moduleList: List[str]) -> None:
        nameList = []  # type: List[str]
        for moduleName in moduleList:
            if moduleName in self.necessaryModuleNameList:
                nameList.append('(*)' + moduleName)
            else:
                nameList.append(moduleName)
        print('-> ' + ' -> '.join(nameList))
