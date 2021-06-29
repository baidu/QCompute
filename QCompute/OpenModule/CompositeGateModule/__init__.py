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
Composite Gate
"""
from copy import deepcopy
from typing import List, Optional, Union, Dict

from QCompute.OpenModule import ModuleImplement
from QCompute.QPlatform.CircuitTools import gateToProtobuf
from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterStorage
from QCompute.QPlatform.QOperation.FixedGate import CX
from QCompute.QPlatform.QOperation.RotationGate import U
from QCompute.QProtobuf import PBCircuitLine, PBProgram, PBCompositeGate


class CompositeGateModule(ModuleImplement):
    """
    The decomposition of composite gate

    Example:

    env.module(CompositeGateModule())

    env.module(CompositeGateModule({'disable': True}))  # Disable

    env.module(CompositeGateModule({'compositeGateList': ['RZZ']}))
    """
    arguments = None  # type: Optional[Dict[str, Union[List[str], bool]]]
    compositeGateList = None  # type: List[str]

    def __init__(self, arguments: Optional[Dict[str, Union[List[str], bool]]] = None) -> None:
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.
        """
        self.arguments = arguments
        if arguments is not None and type(arguments) is dict:
            if 'disable' in arguments:
                self.disable = arguments['disable']

            if 'compositeGateList' in arguments:
                self.compositeGateList = arguments['compositeGateList']

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program
        :return: decomposed circuit
        """

        ret = deepcopy(program)

        for name, procedure in program.body.procedureMap.items():
            procedureOut = ret.body.procedureMap[name]
            del procedureOut.circuit[:]
            self._decompose(procedure.circuit, procedureOut.circuit)
        del ret.body.circuit[:]
        self._decompose(program.body.circuit, ret.body.circuit)
        return ret

    def _decompose(self, circuitIn: List['PBCircuitLine'], circuitOut: List['PBCircuitLine']) -> None:
        """
        Decompose circuit

        :param circuitIn: CircuitLine list
        """

        for index, circuitLine in enumerate(circuitIn):
            compositeGate = circuitLine.compositeGate  # type: 'PBCompositeGate'
            if circuitLine.HasField('compositeGate') and (
                    self.compositeGateList is None or compositeGate.name in self.compositeGateList):
                if compositeGate == PBCompositeGate.RZZ:
                    # Insert the decomposed circuit
                    circuitOut.extend(_RZZ(circuitLine))
                    continue
            circuitOut.append(circuitLine)


def _RZZ(rzzGate: PBCircuitLine) -> List[PBCircuitLine]:
    """
    RZZ(xyz)(Q0, Q1)

    =

    CX(Q0, Q1)

    U(xyz)(Q1)

    CX(Q0, Q1)
    """
    ret = []  # type: List['PBCircuitLine']

    ret.append(gateToProtobuf(CX, rzzGate.qRegList))

    argumentList = []
    if len(rzzGate.argumentIdList) > 0:
        for index, argumentId in enumerate(rzzGate.argumentIdList):
            if argumentId >= 0:
                argumentList.append(ProcedureParameterStorage(argumentId))
            else:
                argumentList.append(rzzGate.argumentValueList[index])
    else:
        argumentList = rzzGate.argumentValueList
    ret.append(gateToProtobuf(U(*argumentList), [rzzGate.qRegList[1]]))

    ret.append(gateToProtobuf(CX, rzzGate.qRegList))

    return ret
