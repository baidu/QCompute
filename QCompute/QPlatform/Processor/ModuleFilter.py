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
Module Filter
"""
from typing import List, Optional

from QCompute import UnrollCircuitModule, CompressGateModule, UnrollProcedureModule, CompositeGateModule, \
    InverseCircuitModule

from QCompute.OpenModule import ModuleImplement
from QCompute.QPlatform import BackendName, Error, ModuleErrorCode

FileErrorCode = 15


def filterModule(backendName: Optional['BackendName'], moduleList: List['ModuleImplement']) \
        -> List['ModuleImplement']:
    if backendName is None:
        return moduleList

    if backendName in [
        BackendName.LocalBaiduSim2,
        
        BackendName.CloudBaiduSim2Water,
        BackendName.CloudBaiduSim2Earth,
        BackendName.CloudBaiduSim2Thunder,
        BackendName.CloudBaiduSim2Heaven,
        BackendName.CloudBaiduSim2Wind,
        BackendName.CloudBaiduSim2Lake,
        BackendName.CloudAerAtBD,
    ]:
        return _filterSimulator(backendName, moduleList)
    
    else:
        return moduleList


def _filterSimulator(backendName: BackendName, moduleList: List['ModuleImplement']) -> List['ModuleImplement']:
    unrollProcedureModule = None  # type: Optional[UnrollProcedureModule]
    compositeGateModule = None  # type: Optional[CompositeGateModule]
    inverseCircuitModule = None  # type: Optional[InverseCircuitModule]
    unrollCircuitModule = None  # type: Optional[UnrollCircuitModule]
    compressGateModule = None  # type: Optional[CompressGateModule]
    ret = []  # type: List['ModuleImplement']
    for module in moduleList:
        
        if module.__class__.__name__ == 'UnrollProcedureModule':
            unrollProcedureModule = module
        elif module.__class__.__name__ == 'CompositeGateModule':
            compositeGateModule = module
        elif module.__class__.__name__ == 'InverseCircuitModule':
            inverseCircuitModule = module
        elif module.__class__.__name__ == 'UnrollCircuitModule':
            unrollCircuitModule = module
        elif module.__class__.__name__ == 'CompressGateModule':
            compressGateModule = module
        elif not module.disable:
            ret.append(module)

    if unrollProcedureModule is not None:
        if not unrollProcedureModule.disable:
            ret.append(unrollProcedureModule)
    else:
        ret.append(UnrollProcedureModule())

    if compositeGateModule is not None:
        if not compositeGateModule.disable:
            ret.append(compositeGateModule)
    else:
        ret.append(CompositeGateModule())

    if inverseCircuitModule is not None:
        if not inverseCircuitModule.disable:
            ret.append(inverseCircuitModule)

    if unrollCircuitModule is not None:
        if not unrollCircuitModule.disable:
            ret.append(unrollCircuitModule)
    else:
        ret.append(UnrollCircuitModule())

    if backendName not in [
        
    ]:
        if compressGateModule is not None:
            if not compressGateModule.disable:
                ret.append(compressGateModule)
        else:
            ret.append(CompressGateModule())
    return ret




def printModuleListDescription(moduleList: List[str]):
    for moduleName in moduleList:
        if moduleName in [
            'MappingToBaiduQPUQianModule',
            'MappingToIoPCASModule',
            'MappingToIonAPMModule',
            
        ]:
            print(
                f'- {moduleName}: The qubit mapping module reconstructs the mapping from quantum gates to quantum registers, \n  and adds SWAP gates if necessary to ensure the two-qubit gates in the circuit can be run on hardware devices.'
            )
        elif moduleName in [
            'UnrollCircuitToBaiduQPUQianModule',
            'UnrollCircuitToIoPCASModule',
            'UnrollCircuitToIonAPMModule',
            
        ]:
            print(
                f'- {moduleName}: The circuit decomposition module decomposes general quantum circuits into native gate circuits \n  supported by hardware devices.'
            )
        elif moduleName == 'UnrollCircuitModule':
            print(
                f'- {moduleName}: The circuit decomposition module decomposes generic quantum circuits into native gates circuits \n  supported by the simulator. (Gates supported by the simulator are CX, U, barrier, measure)'
            )
        elif moduleName == 'CompressGateModule':
            print(
                f'- {moduleName}: The compression gate module compress single qubit gates into two-qubit gates.'
            )
        elif moduleName == 'CompositeGateModule':
            print(
                f'- {moduleName}: The composite gate module decomposes composite gates in a circuit into native gates that can be \n  executed by a simulator or hardware device. When different backends are chosen, modules are decomposed differently.'
            )
        elif moduleName == 'InverseCircuitModule':
            print(
                f'- {moduleName}: The inverse circuit module return inverse quantum circuit.'
            )
        elif moduleName == 'ReverseCircuitModule':
            print(
                f'- {moduleName}: The reverse circuit module return reverse quantum circuit.'
            )
        elif moduleName == 'UnrollProcedureModule':
            print(
                f'- {moduleName}: The subprocedure decomposition module expands all the subprocedures in the quantum circuit.'
            )
