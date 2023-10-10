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
FileErrorCode = 18

from typing import List, Optional

from QCompute.Define import Settings
from QCompute.QProtobuf import PBProgram
from QCompute.OpenModule import ModuleImplement
from QCompute.OpenModule.ReverseCircuitModule import ReverseCircuitModule
from QCompute.OpenModule.InverseCircuitModule import InverseCircuitModule
from QCompute.OpenModule.CompositeGateModule import CompositeGateModule
from QCompute.OpenModule.UnrollProcedureModule import UnrollProcedureModule
from QCompute.OpenModule.UnrollCircuitModule import UnrollCircuitModule
from QCompute.OpenModule.CompressGateModule import CompressGateModule
from QCompute.OpenModule.CompressNoiseModule import CompressNoiseModule
from QCompute.OpenModule.UnrollNoiseModule import UnrollNoiseModule
from QCompute.QPlatform import BackendName, Error, ModuleErrorCode



def filterModule(programe: PBProgram, backendName: Optional['BackendName'], moduleList: List['ModuleImplement']) \
        -> List['ModuleImplement']:
    if backendName is None:
        return moduleList

    if backendName in [
        BackendName.LocalBaiduSim2,
        BackendName.LocalCuQuantum,
        BackendName.LocalBaiduSimPhotonic,
        
        BackendName.CloudBaiduSim2Water,
        BackendName.CloudBaiduSim2Earth,
        BackendName.CloudBaiduSim2Thunder,
        BackendName.CloudBaiduSim2Heaven,
        BackendName.CloudBaiduSim2Wind,
        BackendName.CloudBaiduSim2Lake,
        BackendName.CloudAerAtBD,
    ]:
        return _filterSimulator(programe, backendName, moduleList)
    
    else:
        return moduleList


def _filterSimulator(programe: PBProgram, backendName: BackendName, moduleList: List['ModuleImplement']) -> List[
    'ModuleImplement']:
    reverseCircuitModule: Optional[ReverseCircuitModule] = None
    inverseCircuitModule: Optional[InverseCircuitModule] = None
    compositeGateModule: Optional[CompositeGateModule] = None
    unrollProcedureModule: Optional[UnrollProcedureModule] = None

    unrollCircuitModule: Optional[UnrollCircuitModule] = None
    compressGateModule: Optional[CompressGateModule] = None

    unrollNoiseModule: Optional[UnrollNoiseModule] = None
    compressNoiseModule: Optional[CompressNoiseModule] = None
    ret: List['ModuleImplement'] = []
    for module in moduleList:
        
        if backendName in [
            
        ] and module.__class__.__name__ == 'CompressGateModule' \
                and not module.disable:
            raise Error.ArgumentError(f'Unsupported {module.__class__.__name__} in {backendName.name}', ModuleErrorCode, FileErrorCode, 2)

        elif module.__class__.__name__ == 'ReverseCircuitModule':
            reverseCircuitModule = module
            ret.append(module)
        elif module.__class__.__name__ == 'InverseCircuitModule':
            inverseCircuitModule = module
            ret.append(module)
        elif module.__class__.__name__ == 'CompositeGateModule':
            compositeGateModule = module
            ret.append(module)
        elif module.__class__.__name__ == 'UnrollProcedureModule':
            unrollProcedureModule = module
            ret.append(module)

        elif module.__class__.__name__ == 'UnrollCircuitModule':
            unrollCircuitModule = module
        elif module.__class__.__name__ == 'CompressGateModule':
            compressGateModule = module

        elif module.__class__.__name__ == 'UnrollNoiseModule':
            unrollNoiseModule = module
        elif module.__class__.__name__ == 'CompressNoiseModule':
            compressNoiseModule = module

        elif not module.disable:
            ret.append(module)

    if backendName == BackendName.LocalBaiduSimPhotonic:
        return []

    if len(programe.body.noiseMap) > 0:
        if unrollProcedureModule is not None:
            if not unrollProcedureModule.disable:
                ret.append(unrollProcedureModule)
        else:
            ret.append(UnrollProcedureModule())

        if unrollNoiseModule is not None:
            if not unrollNoiseModule.disable:
                ret.append(unrollNoiseModule)
        else:
            ret.append(UnrollNoiseModule())

        if compressNoiseModule is not None:
            if not compressNoiseModule.disable:
                ret.append(compressNoiseModule)
        else:
            ret.append(CompressNoiseModule())

        return ret

    if compositeGateModule is None:
        ret.append(CompositeGateModule())

    if unrollProcedureModule is None:
        ret.append(UnrollProcedureModule())

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
    if not Settings.outputInfo:
        return
    for moduleName in set(moduleList):
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
        elif moduleName == 'UnrollNoiseModule':
            print(
                f'- {moduleName}: The noise unrolling module assigns all noises to the quantum circuit by user-defined rules.'
            )
        elif moduleName == 'CompressNoiseModule':
            print(
                f'- {moduleName}: The compress noise module compress single qubit noiseless gates into two-qubit gates and reorder gates to construct a more dense circuit.'
            )
    if len(moduleList) > 0:
        print(
            '*Tips: to close the output info, you can insert `QCompute.Define.Settings.outputInfo = False` at the beginning of your code.')