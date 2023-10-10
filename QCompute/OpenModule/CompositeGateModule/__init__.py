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
Composite Gate
"""
FileErrorCode = 1

from copy import deepcopy
from typing import List, Optional, Union, Dict

from QCompute.OpenModule import ModuleImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QPlatform.CircuitTools import gateToProtobuf
from QCompute.QPlatform.QOperation.FixedGate import H, CZ, S
from QCompute.QPlatform.QOperation.RotationGate import CRX, RX, CU
from QCompute.QProtobuf import PBCircuitLine, PBProgram, PBCompositeGate


class CompositeGateModule(ModuleImplement):
    """
    The decomposition of composite gate

    Example:

    env.module(CompositeGateModule())

    env.module(CompositeGateModule({'disable': True}))  # Disable

    env.module(CompositeGateModule({'compositeGateList': ['MS']}))
    """
    compositeGateList: List[str] = None

    def __init__(self, arguments: Optional[Dict[str, Union[List[str], bool]]] = None) -> None:
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.
        """
        super().__init__(arguments)
        if arguments is not None and type(arguments) is dict:
            if 'compositeGateList' in arguments:
                self.compositeGateList = arguments['compositeGateList']

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program
        :return: decomposed circuit
        """
        if self.disable:
            return program

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

        for circuitLine in circuitIn:
            compositeGate: 'PBCompositeGate' = circuitLine.compositeGate
            if circuitLine.HasField('compositeGate') and (
                    self.compositeGateList is None or PBCompositeGate.Name(compositeGate) in self.compositeGateList):
                if compositeGate == PBCompositeGate.MS:
                    # Insert the decomposed circuit
                    circuitOut.extend(_MS(circuitLine))
                    continue
                elif compositeGate == PBCompositeGate.CK:
                    # Insert the decomposed circuit
                    circuitOut.append(_CK(circuitLine))
                    continue
                else:
                    raise Error.ArgumentError(
                        f'Unsupported composite gate {compositeGate.name}!', ModuleErrorCode, FileErrorCode, 1)

            circuitOut.append(circuitLine)


def _MS(msGate: PBCircuitLine) -> List[PBCircuitLine]:
    """
    MS()(Q0, Q1)
    MS(theta)(Q0, Q1)
    """
    if len(msGate.argumentIdList) > 0:
        raise Error.ArgumentError(
            'MS gate in procedure, must unroll procedure first!', ModuleErrorCode, FileErrorCode, 2)

    ret: List['PBCircuitLine'] = []
    if len(msGate.argumentValueList) > 0:
        # CRX
        theta: 'RotationArgument' = msGate.argumentValueList[0]
        ret.append(gateToProtobuf(H, [msGate.qRegList[0]]))
        ret.append(gateToProtobuf(CRX(-2 * theta), msGate.qRegList))
        ret.append(gateToProtobuf(H, [msGate.qRegList[0]]))
        ret.append(gateToProtobuf(RX(theta), [msGate.qRegList[1]]))
    else:
        # CZ
        ret.append(gateToProtobuf(H, [msGate.qRegList[0]]))
        ret.append(gateToProtobuf(H, [msGate.qRegList[1]]))
        ret.append(gateToProtobuf(CZ, msGate.qRegList))
        ret.append(gateToProtobuf(S, [msGate.qRegList[0]]))
        ret.append(gateToProtobuf(S, [msGate.qRegList[1]]))
        ret.append(gateToProtobuf(H, [msGate.qRegList[0]]))
        ret.append(gateToProtobuf(H, [msGate.qRegList[1]]))
    return ret


def _CK(ckGate: PBCircuitLine) -> PBCircuitLine:
    """
    CK(kappa)(Q0, Q1)
    """
    kappa: 'RotationArgument' = ckGate.argumentValueList[0]
    return gateToProtobuf(CU(0, kappa, 0), ckGate.qRegList)

# removed. only example
# def _RZZ(rzzGate: PBCircuitLine) -> List[PBCircuitLine]:
#     """
#     RZZ(xyz)(Q0, Q1)
#
#     =
#
#     CX(Q0, Q1)
#
#     U(xyz)(Q1)
#
#     CX(Q0, Q1)
#     """
#     ret: List['PBCircuitLine'] = []
#
#     ret.append(gateToProtobuf(CX, rzzGate.qRegList))
#
#     if len(rzzGate.argumentIdList) > 0:
#         argumentList: List['RotationArgument'] = []
#         for index, argumentId in enumerate(rzzGate.argumentIdList):
#             if argumentId >= 0:
#                 argumentList.append(ProcedureParameterStorage(argumentId))
#             else:
#                 argumentList.append(rzzGate.argumentValueList[index])
#     else:
#         argumentList: List['RotationArgument'] = rzzGate.argumentValueList
#     ret.append(gateToProtobuf(U(*argumentList), [rzzGate.qRegList[1]]))
#
#     ret.append(gateToProtobuf(CX, rzzGate.qRegList))
#
#     return ret