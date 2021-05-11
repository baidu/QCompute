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
from typing import TYPE_CHECKING, Dict, List, Optional

from QCompute.OpenModule import ModuleImplement
from QCompute.QPlatform import Error
from QCompute.QProtobuf import PBProgram, PBCircuitLine

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBProcedure


class UnrollProcedureModule(ModuleImplement):
    """
    Unroll Procedure

    Example:

    env.module(UnrollProcedure())
    """
    arguments = None

    _procedureMap = None  # type: Dict[str, 'PBProcedure']
    _circuitOut = None  # type: List['PBCircuitLine']

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program
        :return: unrolled procedure
        """

        ret = deepcopy(program)

        ret.body.procedureMap.clear()
        del ret.body.circuit[:]
        self._circuitOut = ret.body.circuit

        # list all the quantum registers
        qRegMap = {qReg: qReg for qReg in program.head.usingQRegList}

        self._procedureMap = program.body.procedureMap
        self._unrollProcedure(program.body.circuit, qRegMap, None)

        return ret

    def _unrollProcedure(self, circuit: List['PBCircuitLine'], qRegMap: Dict[int, int],
                         argumentValueList: Optional[List[float]]) -> None:
        # fill in the circuit
        for circuitLine in circuit:
            op = circuitLine.WhichOneof('op')
            if op in ['fixedGate', 'rotationGate', 'compositeGate', 'measure', 'barrier']:
                ret = deepcopy(circuitLine)  # type: 'PBCircuitLine'
                if len(ret.argumentIdList) > 0:
                    if len(ret.argumentValueList) == 0:
                        ret.argumentValueList[:] = [0.0] * len(ret.argumentIdList)
                    for index, argumentId in enumerate(ret.argumentIdList):
                        if argumentId >= 0:
                            ret.argumentValueList[index] = argumentValueList[argumentId]
                    del ret.argumentIdList[:]
                for index, qReg in enumerate(ret.qRegList):
                    if qReg not in qRegMap:
                        raise Error.ArgumentError('QReg argument is not in procedure!')
                    ret.qRegList[index] = qRegMap[qReg]
                self._circuitOut.append(ret)
            elif op == 'customizedGate':  # customized gate
                raise Error.ArgumentError('Unsupported operation customizedGate!')
            elif op == 'procedureName':  # procedure
                qProcedureRegMap = {index: qRegMap[qReg] for index, qReg in enumerate(circuitLine.qRegList)}

                argumentIdLen = len(circuitLine.argumentIdList)
                argumentValueLen = len(circuitLine.argumentValueList)
                argumentLen = argumentIdLen if argumentIdLen > argumentValueLen else argumentValueLen
                argumentList = [0.0] * argumentLen
                for i in range(argumentLen):
                    if i < argumentIdLen:
                        argumentId = circuitLine.argumentIdList[i]
                        if argumentId != -1:
                            argumentList[i] = argumentValueList[argumentId]
                            continue
                    argumentList[i] = circuitLine.argumentValueList[i]

                procedure = self._procedureMap[circuitLine.procedureName]
                self._unrollProcedure(procedure.circuit, qProcedureRegMap, argumentList)
            else:  # unsupported operation
                raise Error.ArgumentError(f'Unsupported operation {op}!')
