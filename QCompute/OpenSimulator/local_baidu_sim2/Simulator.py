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
Simulator
The simulator uses statevector (a rank-n tensor representing an n-qubit state) to simulate quantum behaviors.
Basically, the core of the algorithm is tensor contraction with one-way calculation. 
The initial state and gates are converted to tensors and gate implementation is simulated as contraction of tensors. 
"""

import argparse
from datetime import datetime
from typing import List, TYPE_CHECKING, Union, Dict, Optional

import numpy


from QCompute.Define import outputPath
from QCompute.OpenConvertor.JsonToCircuit import JsonToCircuit
from QCompute.OpenSimulator import QResult, ModuleErrorCode
from QCompute.OpenSimulator.local_baidu_sim2.InitState import MatrixType, initState_1_0
from QCompute.OpenSimulator.local_baidu_sim2.Measure import MeasureMethod, Measurer
from QCompute.OpenSimulator.local_baidu_sim2.Transfer import Algorithm, TransferProcessor
from QCompute.QPlatform import Error
from QCompute.QPlatform.Processor.PostProcessor import filterMeasure
from QCompute.QPlatform.Processor.PreProcess import preProcess
from QCompute.QPlatform.QOperation.FixedGate import CX, X, Y, Z, H, SWAP
from QCompute.QPlatform.QOperation.RotationGate import U
from QCompute.QPlatform.Utilities import protobufMatrixToNumpyMatrix, normalizeNdarrayOrderForTranspose
from QCompute.QProtobuf import PBFixedGate, PBRotationGate, PBCustomizedGate, PBMeasure

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBProgram

FileErrorCode = 2


def runSimulator(args: Optional[List[str]], program: Optional['PBProgram']) -> 'QResult':
    """
    Initialization process
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-mt', default='dense', type=str)
    parser.add_argument('-a', default='matmul', type=str)
    parser.add_argument('-mm', default='probability', type=str)
    parser.add_argument('-s', default=None, type=int)
    parser.add_argument('-shots', default=None, type=int)
    parser.add_argument('-inputFile', default=None, type=str)

    args = parser.parse_args(args=args)
    matrixType = args.mt.lower()  # type: str
    algorithm = args.a.lower()  # type: str
    measureMethod = args.mm.lower()  # type: str
    seed = args.s  # type: int
    shots = args.shots  # type: int
    inputFile = args.inputFile  # type: str

    matrixTypeValue = None  # type: Optional[MatrixType]
    if matrixType == 'dense':
        matrixTypeValue = MatrixType.Dense
    
    else:
        raise Error.ArgumentError(f'Invalid MatrixType {matrixTypeValue}', ModuleErrorCode, FileErrorCode, 1)

    algorithmValue = None  # type: Optional[Algorithm]
    if algorithm == 'matmul':
        algorithmValue = Algorithm.Matmul
    elif algorithm == 'einsum':
        algorithmValue = Algorithm.Einsum
    else:
        raise Error.ArgumentError(f'Invalid Algorithm {algorithmValue}', ModuleErrorCode, FileErrorCode, 2)

    measureMethodValue = None  # type: Optional[MeasureMethod]
    if measureMethod == 'probability':
        measureMethodValue = MeasureMethod.Probability
    elif measureMethod == 'output_probability':
        measureMethodValue = MeasureMethod.OutputProbability
    elif measureMethod == 'output_state':
        measureMethodValue = MeasureMethod.OutputState
    elif measureMethod == 'accumulation':
        measureMethodValue = MeasureMethod.Accumulation
    else:
        raise Error.ArgumentError(f'Invalid MeasureMethod {measureMethodValue}', ModuleErrorCode, FileErrorCode, 3)

    if seed is not None:
        if seed < 0 or seed > 2147483647:
            raise Error.ArgumentError(f'Invalid Seed {seed}', ModuleErrorCode, FileErrorCode, 4)

    if shots <= 0:
        raise Error.ArgumentError(f'Invalid shots {shots}', ModuleErrorCode, FileErrorCode, 5)

    if inputFile is not None:
        with open(inputFile, "rt") as fObj:
            jsonStr = fObj.read()
        program = JsonToCircuit().convert(jsonStr)

    return core(program, matrixTypeValue, algorithmValue, measureMethodValue, shots, seed)


def core(program: 'PBProgram', matrixType: 'MatrixType', algorithm: 'Algorithm', measureMethod: 'MeasureMethod',
         shots: int,
         seed: int) -> 'QResult':
    """
    Simulaton process
        Check if the argument is available. The accepted ones are:

        1)DENSE-EINSUM-SINGLE

        2)DENSE-EINSUM-PROB

        3)DENSE-MATMUL-SINGLE

        4)DENSE-MATMUL-PROB

        
    """

    usedQRegSet, usedCRegSet, compactedQRegDict, compactedCRegDict = preProcess(program, True)

    

    qRegMap = {qReg: index for index, qReg in enumerate(program.head.usingQRegList)}
    qRegCount = len(qRegMap)

    operationDict = loadGates(matrixType)

    if seed is None:
        seed = numpy.random.randint(0, 2147483647 + 1)
    numpy.random.seed(seed)

    state = initState_1_0(matrixType, qRegCount)
    transfer = TransferProcessor(matrixType, algorithm)
    measurer = Measurer(matrixType, algorithm, measureMethod)

    # collect the result to simulator for the subsequent invoking
    result = QResult()
    result.startTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'

    measured = False
    for circuitLine in program.body.circuit:  # Traverse the circuit
        op = circuitLine.WhichOneof('op')

        qRegList = []  # type List[int]
        for qReg in circuitLine.qRegList:
            qRegList.append(qRegMap[qReg])

        if op == 'fixedGate':  # fixed gate
            fixedGate = circuitLine.fixedGate  # type: PBFixedGate
            matrix = operationDict.get(fixedGate)
            if matrix is None:
                raise Error.ArgumentError(f'Unsupported operation {PBFixedGate.Name(fixedGate)}!', ModuleErrorCode,
                                          FileErrorCode, 7)
            state = transfer(state, matrix, qRegList)
        elif op == 'rotationGate':  # rotation gate
            rotationGate = circuitLine.rotationGate  # type: PBRotationGate
            if rotationGate != PBRotationGate.U:
                raise Error.ArgumentError(
                    f'Unsupported operation {PBRotationGate.Name(rotationGate)}!', ModuleErrorCode, FileErrorCode, 8)
            uGate = U(*circuitLine.argumentValueList)
            if matrixType == MatrixType.Dense:
                matrix = uGate.getMatrix()
            else:
                raise Error.RuntimeError('Not implemented')
            state = transfer(state, matrix, qRegList)
        elif op == 'customizedGate':  # customized gate
            customizedGate = circuitLine.customizedGate  # type: PBCustomizedGate
            if matrixType == MatrixType.Dense:
                matrix = protobufMatrixToNumpyMatrix(customizedGate.matrix)
            else:
                raise Error.RuntimeError('Not implemented')
            state = transfer(state, matrix, qRegList)
        elif op == 'procedureName':  # procedure
            raise Error.ArgumentError('Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                                      ModuleErrorCode, FileErrorCode, 9)
            # it is not implemented, flattened by UnrollProcedureModule
        elif op == 'measure':  # measure
            measure = circuitLine.measure  # type: PBMeasure
            if measure.type != PBMeasure.Type.Z:  # only Z measure is supported
                raise Error.ArgumentError(
                    f'Unsupported operation measure {PBMeasure.Type.Name(measure.type)}!', ModuleErrorCode,
                    FileErrorCode, 10)
            if not measured:
                result.counts = measurer(state, shots)
                measured = True
        elif op == 'barrier':  # barrier
            pass
            # unimplemented operation
        else:  # unsupported operation
            raise Error.ArgumentError(f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 11)

    result.endTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'
    result.shots = shots
    if measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation]:
        result.counts = filterMeasure(result.counts, compactedCRegDict)
    result.ancilla.usedQRegList = list(usedQRegSet)
    result.ancilla.usedCRegList = list(usedCRegSet)
    result.ancilla.compactedQRegDict = compactedQRegDict
    result.ancilla.compactedCRegDict = compactedCRegDict
    if measureMethod == MeasureMethod.OutputState:
        if matrixType == MatrixType.Dense:
            result.state = normalizeNdarrayOrderForTranspose(state)
        else:
            result.state = normalizeNdarrayOrderForTranspose(state.todense())
    result.seed = int(seed)
    return result


def loadGates(matrixType: 'MatrixType') -> Dict['PBFixedGate', Union[numpy.ndarray, 'COO']]:
    """
    Load the matrix of the gate
    """
    if matrixType == MatrixType.Dense:
        operationDict = {
            PBFixedGate.X: X.getMatrix(),
            PBFixedGate.Y: Y.getMatrix(),
            PBFixedGate.Z: Z.getMatrix(),
            PBFixedGate.H: H.getMatrix(),
            PBFixedGate.CX: CX.getMatrix().reshape(2, 2, 2, 2),
            PBFixedGate.SWAP: SWAP.getMatrix()
        }  # type: Dict['PBFixedGate', Union[numpy.ndarray, 'COO']]
    else:
        raise Error.RuntimeError('Not implemented')
    return operationDict


if __name__ == '__main__':
    result = runSimulator(None, None)
    countsFilePath = outputPath / 'counts.json'
    with open(countsFilePath, 'wt', encoding='utf-8') as file:
        file.write(result.toJson(True))
