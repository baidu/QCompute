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

import copy

from QCompute.QuantumPlatform import Error
from QCompute.QuantumPlatform.QuantumOperation import FixedGate, RotationGate
from QCompute.QuantumPlatform.QuantumOperation.Barrier import Barrier
from QCompute.QuantumPlatform.QuantumOperation.Measure import MeasureZ
from QCompute.QuantumProtobuf.Library.QuantumOperation_pb2 import FixedGate as FixedGateEnum, \
    RotationGate as RotationGateEnum, Measure as PBMeasure


class UnrollProcedure:
    """
    Unroll Procedure

    Example:

    env.module(UnrollProcedure())
    """

    def __call__(self, program):
        """
        Process the Module

        :param program: the program
        :return: unrolled procedure
        """

        # list all the quantum registers
        qRegsMap = {qReg: qReg for qReg in program.head.usingQRegs}

        ret = copy.deepcopy(program)
        ret.body.ClearField('circuit')

        self._procedureMap = program.body.procedureMap
        self._circuitOut = ret.body.circuit
        self._unrollProcedure(program.body.circuit, qRegsMap, None)

        ret.body.ClearField('procedureMap')
        return ret

    def _unrollProcedure(self, circuit, qRegsMap, paramValues):
        # fill in the circuit
        for circuitLine in circuit:
            if circuitLine.HasField('fixedGate'):  # fixed gate
                fixedGateClass = getattr(FixedGate, FixedGateEnum.Name(circuitLine.fixedGate))  # get gate class
                qRegList = []
                for qReg in circuitLine.qRegs:  # quantum register lists
                    qRegList.append(qRegsMap[qReg])
                self._circuitOut.append(fixedGateClass._toPB(*qRegList))
            elif circuitLine.HasField('rotationGate'):  # rotation gate
                rotationGateClass = getattr(RotationGate, RotationGateEnum.Name(
                    circuitLine.rotationGate))  # get gate class
                qRegList = []
                for qReg in circuitLine.qRegs:  # quantum register lists
                    qRegList.append(qRegsMap[qReg])
                params = []
                if len(circuitLine.paramIds) > 0:
                    for index, paramId in enumerate(circuitLine.paramIds):  # check procedure params
                        if paramId == -1:
                            params.append(circuitLine.paramValues[index])  # from angles
                        else:
                            params.append(paramValues[paramId])  # from procedure params
                else:
                    params = circuitLine.paramValues

                self._circuitOut.append(rotationGateClass(*params)._toPB(*qRegList))
            elif circuitLine.HasField('customizedGate'):  # customized gate
                raise Error.ParamError('unsupported operation customizedGate')
                # todo it is not implemented
            elif circuitLine.HasField('procedureName'):  # procedure
                qProcedureRegsMap = {index: qRegsMap[qReg] for index, qReg in enumerate(circuitLine.qRegs)}

                paramIdsLen = len(circuitLine.paramIds)
                paramValuesLen = len(circuitLine.paramValues)
                procedureParamLen = paramIdsLen if paramIdsLen > paramValuesLen else paramValuesLen
                procedureParamValues = [None] * procedureParamLen
                for i in range(procedureParamLen):
                    if i < paramIdsLen:
                        paramId = circuitLine.paramIds[i]
                        if paramId != -1:
                            procedureParamValues[i] = paramValues[paramId]
                            continue
                    procedureParamValues[i] = circuitLine.paramValues[i]

                procedure = self._procedureMap[circuitLine.procedureName]
                self._unrollProcedure(procedure.circuit, qProcedureRegsMap, procedureParamValues)
            elif circuitLine.HasField('measure'):  # measure
                if circuitLine.measure.type == PBMeasure.Type.Z:  # only Z measure is supported
                    pass
                else:  # unsupported measure types
                    raise Error.ParamError(
                        f'unsupported operation measure {PBMeasure.Type.Name(circuitLine.measure.type)}')
                qRegList = []
                for qReg in circuitLine.qRegs:  # quantum register list
                    qRegList.append(qRegsMap[qReg])
                self._circuitOut.append(MeasureZ._toPB(qRegList, circuitLine.measure.cRegs))
            elif circuitLine.HasField('barrier'):  # barrier
                qRegList = []
                for qReg in circuitLine.qRegs:  # quantum register list
                    qRegList.append(qRegsMap[qReg])
                self._circuitOut.append(Barrier()._toPB(*qRegList))
            else:  # unsupported operation
                raise Error.ParamError('unsupported operation')
