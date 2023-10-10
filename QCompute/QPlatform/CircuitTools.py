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
Circuit tools
"""
FileErrorCode = 2

from typing import List, TYPE_CHECKING, Tuple, Optional

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.ProcedureParameterExpression import ProcedureParameterExpression, MathOpEnum
from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterStorage
from QCompute.QPlatform.QNoise import QNoiseDefine
from QCompute.QPlatform.QNoise.AmplitudeDamping import AmplitudeDamping
from QCompute.QPlatform.QNoise.BitFlip import BitFlip
from QCompute.QPlatform.QNoise.BitPhaseFlip import BitPhaseFlip
from QCompute.QPlatform.QNoise.CustomizedNoise import CustomizedNoise
from QCompute.QPlatform.QNoise.Depolarizing import Depolarizing
from QCompute.QPlatform.QNoise.PauliNoise import PauliNoise
from QCompute.QPlatform.QNoise.PhaseDamping import PhaseDamping
from QCompute.QPlatform.QNoise.PhaseFlip import PhaseFlip
from QCompute.QPlatform.QNoise.ResetNoise import ResetNoise
from QCompute.QPlatform.QOperation.Barrier import BarrierOP
from QCompute.QPlatform.QOperation.CompositeGate import CompositeGateOP
from QCompute.QPlatform.QOperation.CustomizedGate import CustomizedGateOP
from QCompute.QPlatform.QOperation.FixedGate import FixedGateOP
from QCompute.QPlatform.QOperation.Measure import MeasureOP
from QCompute.QPlatform.QOperation.Photonic.PhotonicGaussianGate import PhotonicGaussianGateOP
from QCompute.QPlatform.QOperation.Photonic.PhotonicGaussianMeasure import PhotonicGaussianMeasureOP
from QCompute.QPlatform.QOperation.Photonic.PhotonicFockGate import PhotonicFockGateOP
from QCompute.QPlatform.QOperation.Photonic.PhotonicFockMeasure import PhotonicFockMeasureOP
from QCompute.QPlatform.QOperation.QProcedure import QProcedureOP
from QCompute.QPlatform.QOperation.RotationGate import RotationGateOP
from QCompute.QPlatform.Utilities import numpyMatrixToProtobufMatrix
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBFixedGate, PBRotationGate, PBCompositeGate, \
    PBPhotonicGaussianGate, PBPhotonicGaussianMeasure, PBPhotonicFockGate, PBPhotonicFockMeasure, \
    PBMeasure, PBQNoise, PBQNoiseDefine, PBExpressionList, PBParameterExpression

if TYPE_CHECKING:
    from QCompute.QPlatform.QEnv import QEnv
    from QCompute.QPlatform.QOperation import CircuitLine, QOperation


def QEnvToProtobuf(program: PBProgram, env: 'QEnv') -> None:
    head = program.head
    head.usingQRegList[:] = sorted([qReg for qReg in env.Q.registerMap.keys()])
    head.usingCRegList[:] = sorted([cReg for cReg in env.measuredCRegSet])

    body = program.body

    for circuitLine in env.circuit:
        body.circuit.append(circuitLineToProtobuf(circuitLine))

    for name, procedure in env.procedureMap.items():
        pbProcedure = body.procedureMap[name]
        pbProcedure.parameterCount = len(procedure.Parameter.parameterMap)
        pbProcedure.usingQRegList[:] = sorted([qReg for qReg in procedure.Q.registerMap.keys()])
        for circuitLine in procedure.circuit:
            pbProcedure.circuit.append(circuitLineToProtobuf(circuitLine))

    for name, noiseDefineList in env.noiseDefineMap.items():
        for noiseDefine in noiseDefineList:
            body.noiseMap[name].noiseDefineList.append(noiseDefineToProtobuf(noiseDefine))


def circuitLineToProtobuf(circuitLine: 'CircuitLine') -> 'PBCircuitLine':
    pbCircuitLine = PBCircuitLine()
    if isinstance(circuitLine.data, FixedGateOP):
        pbCircuitLine.fixedGate = PBFixedGate.Value(circuitLine.data.name)
    elif isinstance(circuitLine.data, RotationGateOP):
        pbCircuitLine.rotationGate = PBRotationGate.Value(circuitLine.data.name)
        argumentIdList, argumentValueList, argumentExpressionList = getRotationArgumentList(circuitLine.data)
        pbCircuitLine.argumentIdList[:] = argumentIdList
        pbCircuitLine.argumentValueList[:] = argumentValueList
        for argumentExpression in argumentExpressionList:
            pbCircuitLine.argumentExpressionList.append(argumentExpression)
    elif isinstance(circuitLine.data, CustomizedGateOP):
        pbCircuitLine.customizedGate.matrix.CopyFrom(numpyMatrixToProtobufMatrix(circuitLine.data.getMatrix()))
    elif isinstance(circuitLine.data, CompositeGateOP):
        pbCircuitLine.compositeGate = PBCompositeGate.Value(circuitLine.data.name)
        argumentIdList, argumentValueList, argumentExpressionList = getRotationArgumentList(circuitLine.data)
        pbCircuitLine.argumentIdList[:] = argumentIdList
        pbCircuitLine.argumentValueList[:] = argumentValueList
        for argumentExpression in argumentExpressionList:
            pbCircuitLine.argumentExpressionList.append(argumentExpression)
    elif isinstance(circuitLine.data, QProcedureOP):
        pbCircuitLine.procedureName = circuitLine.data.name
        argumentIdList, argumentValueList, argumentExpressionList = getRotationArgumentList(circuitLine.data)
        pbCircuitLine.argumentIdList[:] = argumentIdList
        pbCircuitLine.argumentValueList[:] = argumentValueList
        for argumentExpression in argumentExpressionList:
            pbCircuitLine.argumentExpressionList.append(argumentExpression)
    elif isinstance(circuitLine.data, MeasureOP):
        pbCircuitLine.measure.type = PBMeasure.Type.Value(circuitLine.data.name)
        pbCircuitLine.measure.cRegList[:] = circuitLine.cRegList
    elif isinstance(circuitLine.data, BarrierOP):
        pbCircuitLine.barrier = True
    elif isinstance(circuitLine.data, PhotonicGaussianGateOP):
        pbCircuitLine.photonicGaussianGate = PBPhotonicGaussianGate.Value(circuitLine.data.name)
        pbCircuitLine.argumentValueList[:] = circuitLine.data.argumentList
    elif isinstance(circuitLine.data, PhotonicGaussianMeasureOP):
        pbCircuitLine.photonicGaussianMeasure.type = PBPhotonicGaussianMeasure.Type.Value(circuitLine.data.name)
        pbCircuitLine.photonicGaussianMeasure.cRegList[:] = circuitLine.cRegList
        if pbCircuitLine.photonicGaussianMeasure.type == PBPhotonicGaussianMeasure.Type.Heterodyne:
            for argument in circuitLine.data.heterodyneArgument:
                arg = pbCircuitLine.photonicGaussianMeasure.heterodyne.add()
                arg.r = argument[0]
                arg.phi = argument[1]
        elif pbCircuitLine.photonicGaussianMeasure.type == PBPhotonicGaussianMeasure.Type.PhotonCount:
            pbCircuitLine.photonicGaussianMeasure.photonCount.cutoff = circuitLine.data.cutoff
    elif isinstance(circuitLine.data, PhotonicFockGateOP):
        pbCircuitLine.photonicFockGate = PBPhotonicFockGate.Value(circuitLine.data.name)
        pbCircuitLine.argumentValueList[:] = circuitLine.data.argumentList
    elif isinstance(circuitLine.data, PhotonicFockMeasureOP):
        pbCircuitLine.photonicFockMeasure.cRegList[:] = circuitLine.cRegList
        pbCircuitLine.photonicFockMeasure.cutoff = circuitLine.data.cutoff

    pbCircuitLine.qRegList[:] = circuitLine.qRegList
    return pbCircuitLine


def getRotationArgumentList(data: 'QOperation') -> Tuple[List[int], List[float], List[PBExpressionList]]:
    if not (isinstance(data, RotationGateOP) or
            isinstance(data, CompositeGateOP) or
            isinstance(data, QProcedureOP)):
        raise Error.ArgumentError('Wrong operation type!', ModuleErrorCode, FileErrorCode, 1)

    needExpression = False
    argumentExpressionList: List[PBExpressionList] = []
    for argument in data.argumentList:
        expressionList = PBExpressionList()
        argumentExpressionList.append(expressionList)
        if isinstance(argument, ProcedureParameterExpression):
            needExpression = True
            for param in argument.expressionList:
                expression = PBParameterExpression()
                if isinstance(param, MathOpEnum):
                    expression.operator = param.value
                elif isinstance(param, ProcedureParameterStorage):
                    expression.argumentId = param.index
                else:
                    expression.argumentValue = param
                expressionList.list.append(expression)
        elif isinstance(argument, ProcedureParameterStorage):
            expression = PBParameterExpression()
            expression.argumentId = argument.index
            expressionList.list.append(expression)
        else:
            expression = PBParameterExpression()
            expression.argumentValue = argument
            expressionList.list.append(expression)
    if needExpression:
        return [], [], argumentExpressionList

    argumentIdList: List[int] = []
    argumentValueList: List[float] = []
    validArgumentIdList = False
    validArgumentValueList = False
    for argument in data.argumentList:
        if isinstance(argument, ProcedureParameterStorage):
            argumentIdList.append(argument.index)
            argumentValueList.append(0)
            validArgumentIdList = True
        else:
            argumentIdList.append(-1)
            argumentValueList.append(argument)
            validArgumentValueList = True
    if not validArgumentIdList:
        argumentIdList.clear()
    elif not validArgumentValueList:
        argumentValueList.clear()
    return argumentIdList, argumentValueList, []


def gateToProtobuf(data: 'QOperation', qRegList: List[int], cRegList: Optional[List[int]] = None) -> 'PBCircuitLine':
    ret = PBCircuitLine()
    ret.qRegList[:] = qRegList
    if data.__class__.__name__ == 'FixedGateOP':
        ret.fixedGate = PBFixedGate.Value(data.name)
    elif data.__class__.__name__ == 'RotationGateOP':
        ret.rotationGate = PBRotationGate.Value(data.name)
        argumentIdList, argumentValueList, argumentExpressionList = getRotationArgumentList(data)
        ret.argumentIdList[:] = argumentIdList
        ret.argumentValueList[:] = argumentValueList
        for argumentExpression in argumentExpressionList:
            ret.argumentExpressionList.append(argumentExpression)
    elif data.__class__.__name__ == 'CustomizedGateOP':
        ret.customizedGate.matrix.CopyFrom(numpyMatrixToProtobufMatrix(data.getMatrix()))
    elif data.__class__.__name__ == 'CompositeGateOP':
        ret.compositeGate = PBCompositeGate.Value(data.name)
        argumentIdList, argumentValueList, argumentExpressionList = getRotationArgumentList(data)
        ret.argumentIdList[:] = argumentIdList
        ret.argumentValueList[:] = argumentValueList
        for argumentExpression in argumentExpressionList:
            ret.argumentExpressionList.append(argumentExpression)
    elif data.__class__.__name__ == 'QProcedureOP':
        ret.procedure_name = data.name
        argumentIdList, argumentValueList, argumentExpressionList = getRotationArgumentList(data)
        ret.argumentIdList[:] = argumentIdList
        ret.argumentValueList[:] = argumentValueList
        for argumentExpression in argumentExpressionList:
            ret.argumentExpressionList.append(argumentExpression)
    elif data.__class__.__name__ == 'MeasureOP':
        ret.measure.type = PBMeasure.Type.Value(data.name)
        ret.measure.cRegList[:] = cRegList
    elif data.__class__.__name__ == 'BarrierOP':
        ret.barrier = True
    return ret


def noiseDefineToProtobuf(noiseDefine: QNoiseDefine) -> PBQNoiseDefine:
    pbQNoiseDefine = PBQNoiseDefine()
    for noise in noiseDefine.noiseList:
        pbNoise = PBQNoise()
        if isinstance(noise, BitFlip):
            pbNoise.bitFlip.probability = noise.probability
        if isinstance(noise, PhaseFlip):
            pbNoise.phaseFlip.probability = noise.probability
        if isinstance(noise, BitPhaseFlip):
            pbNoise.bitPhaseFlip.probability = noise.probability
        if isinstance(noise, PauliNoise):
            pbNoise.pauliNoise.probability1 = noise.probability1
            pbNoise.pauliNoise.probability2 = noise.probability2
            pbNoise.pauliNoise.probability3 = noise.probability3
        if isinstance(noise, Depolarizing):
            pbNoise.depolarizing.bits = noise.bits
            pbNoise.depolarizing.probability = noise.probability
        if isinstance(noise, AmplitudeDamping):
            pbNoise.amplitudeDamping.probability = noise.probability
        if isinstance(noise, ResetNoise):
            pbNoise.resetNoise.probability1 = noise.probability1
            pbNoise.resetNoise.probability2 = noise.probability2
        if isinstance(noise, PhaseDamping):
            pbNoise.phaseDamping.probability = noise.probability
        if isinstance(noise, CustomizedNoise):
            for pbMatrix in [numpyMatrixToProtobufMatrix(matrix) for matrix in noise.krauses]:
                pbNoise.customizedNoise.krauses.append(pbMatrix)
        pbQNoiseDefine.noiseList.append(pbNoise)
    if noiseDefine.qRegList:
        pbQNoiseDefine.qRegList[:] = noiseDefine.qRegList
    if noiseDefine.positionList:
        pbQNoiseDefine.positionList[:] = noiseDefine.positionList
    return pbQNoiseDefine