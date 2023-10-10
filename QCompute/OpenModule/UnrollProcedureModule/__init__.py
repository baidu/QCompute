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
Unroll Procedure
"""
FileErrorCode = 8

import math
from copy import deepcopy
from typing import Dict, List, Optional

from QCompute.Define import Settings
from QCompute.OpenModule import ModuleImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBParameterExpression, PBMathOperator, PBQProcedure, \
    PBExpressionList


class UnrollProcedureModule(ModuleImplement):
    """
    Unroll Procedure

    Example:

    env.module(UnrollProcedureModule())

    env.module(UnrollProcedureModule({'disable': True}))  # Disable
    """
    _procedureMap: Dict[str, 'PBQProcedure'] = None
    _circuitOut: List['PBCircuitLine'] = None

    def __init__(self, arguments: Optional[Dict[str, bool]] = None):
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.
        """
        super().__init__(arguments)

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program
        :return: unrolled procedure
        """
        if self.disable:
            return program

        ret = deepcopy(program)

        ret.body.procedureMap.clear()
        del ret.body.circuit[:]
        self._circuitOut = ret.body.circuit

        # List all the quantum registers
        qRegMap = {qReg: qReg for qReg in program.head.usingQRegList}

        self._procedureMap = program.body.procedureMap
        self._unrollProcedure(None, program.body.circuit, qRegMap, None, 0)

        return ret

    def _unrollProcedure(self, procedureName: str, circuit: List['PBCircuitLine'], qRegMap: Dict[int, int],
                         argumentValueList: Optional[List[float]], indentationNum: int) -> None:
        if Settings.outputInfo:
            if procedureName:
                print(' ' * indentationNum * 2 + 'Procedure', procedureName, argumentValueList)
            else:
                print('Base circuit')

        # Fill in the circuit
        for circuitLine in circuit:
            op = circuitLine.WhichOneof('op')
            if op in ['fixedGate', 'rotationGate', 'compositeGate', 'measure', 'barrier']:
                ret = deepcopy(circuitLine)
                if len(ret.argumentExpressionList) > 0:
                    ret.argumentValueList[:] = self._unrollArgumentExpression(ret.argumentExpressionList,
                                                                              argumentValueList)
                    del ret.argumentExpressionList[:]
                    del ret.argumentIdList[:]
                elif len(ret.argumentIdList) > 0:
                    if len(ret.argumentValueList) == 0:
                        ret.argumentValueList[:] = [0.0] * len(ret.argumentIdList)
                    for index, argumentId in enumerate(ret.argumentIdList):
                        if argumentId >= 0:
                            ret.argumentValueList[index] = argumentValueList[argumentId]
                    del ret.argumentIdList[:]
                for index, qReg in enumerate(ret.qRegList):
                    if qReg not in qRegMap:
                        raise Error.ArgumentError(
                            'QReg argument is not in procedure!', ModuleErrorCode, FileErrorCode, 1)

                    ret.qRegList[index] = qRegMap[qReg]
                self._circuitOut.append(ret)
            elif op == 'customizedGate':  # Customized gate
                raise Error.ArgumentError('Unsupported operation customizedGate!', ModuleErrorCode, FileErrorCode, 2)
            elif op == 'procedureName':  # procedure
                qProcedureRegMap = {index: qRegMap[qReg] for index, qReg in enumerate(circuitLine.qRegList)}

                if len(circuitLine.argumentExpressionList) > 0:
                    circuitLine.argumentValueList[:] = self._unrollArgumentExpression(
                        circuitLine.argumentExpressionList,
                        argumentValueList)
                    del circuitLine.argumentExpressionList[:]
                    del circuitLine.argumentIdList[:]
                elif len(circuitLine.argumentIdList) > 0:
                    if len(circuitLine.argumentValueList) == 0:
                        circuitLine.argumentValueList[:] = [0.0] * len(circuitLine.argumentIdList)
                    for index, argumentId in enumerate(circuitLine.argumentIdList):
                        if argumentId >= 0:
                            circuitLine.argumentValueList[index] = argumentValueList[argumentId]
                    del circuitLine.argumentIdList[:]

                procedure = self._procedureMap[circuitLine.procedureName]
                self._unrollProcedure(circuitLine.procedureName, procedure.circuit, qProcedureRegMap,
                                      circuitLine.argumentValueList, indentationNum + 1)
            else:  # Unsupported operation
                raise Error.ArgumentError(
                    f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 3)

    def _unrollArgumentExpression(self,
                                  argumentExpressionList: List['PBExpressionList'],
                                  argumentValueList: Optional[List[float]]) -> List[float]:
        ret: List[float] = []
        for expressionList in argumentExpressionList:
            valueQueue: List[PBParameterExpression] = []
            for parameterExpression in expressionList.list:
                expressionType = parameterExpression.WhichOneof('expression')
                if expressionType == 'argumentId':
                    valueQueue.append(argumentValueList[parameterExpression.argumentId])
                elif expressionType == 'argumentValue':
                    valueQueue.append(parameterExpression.argumentValue)
                elif expressionType == 'operator':
                    self._computeOperator(parameterExpression.operator, valueQueue)
                else:
                    raise Error.ArgumentError("Wrong expressionType!")
            ret.append(valueQueue[0])
        return ret

    def _computeOperator(self, operator: PBMathOperator, valueQueue: List[PBParameterExpression]) -> None:
        if operator is PBMathOperator.NEG:
            valueQueue[-1] = -valueQueue[-1]
        elif operator is PBMathOperator.POS:
            pass
        elif operator is PBMathOperator.ABS:
            valueQueue[-1] = math.fabs(valueQueue[-1])

        elif operator is PBMathOperator.ADD:
            valueQueue[-2] = valueQueue[-2] + valueQueue[-1]
            del valueQueue[-1]
        elif operator is PBMathOperator.SUB:
            valueQueue[-2] = valueQueue[-2] - valueQueue[-1]
            del valueQueue[-1]
        elif operator is PBMathOperator.MUL:
            valueQueue[-2] = valueQueue[-2] * valueQueue[-1]
            del valueQueue[-1]
        elif operator is PBMathOperator.TRUEDIV:
            valueQueue[-2] = valueQueue[-2] / valueQueue[-1]
            del valueQueue[-1]
        elif operator is PBMathOperator.FLOORDIV:
            valueQueue[-2] = valueQueue[-2] // valueQueue[-1]
            del valueQueue[-1]
        elif operator is PBMathOperator.MOD:
            valueQueue[-2] = valueQueue[-2] % valueQueue[-1]
            del valueQueue[-1]
        elif operator is PBMathOperator.POW:
            valueQueue[-2] = valueQueue[-2] ** valueQueue[-1]
            del valueQueue[-1]

        elif operator is PBMathOperator.SIN:
            valueQueue[-1] = math.sin(valueQueue[-1])
        elif operator is PBMathOperator.COS:
            valueQueue[-1] = math.cos(valueQueue[-1])
        elif operator is PBMathOperator.TAN:
            valueQueue[-1] = math.tan(valueQueue[-1])

        else:
            raise Error.ArgumentError("Wrong operator!")