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
import json
from datetime import datetime

import numpy



from bidict import bidict
from google.protobuf.json_format import Parse

from QCompute.Define.Settings import doCompressGate
from QCompute.OpenModule.CompositeGateModule import CompositeGate
from QCompute.OpenModule.CompressGateModule import CompressGate
from QCompute.OpenModule.UnrollCircuitModule import UnrollCircuit
from QCompute.OpenModule.UnrollProcedureModule import UnrollProcedure
from QCompute.OpenSimulator import QuantumResult
from QCompute.OpenSimulator.local_baidu_sim2.InitState import MatrixType, initState_1_0
from QCompute.OpenSimulator.local_baidu_sim2.Measure import MeasureMethod, Measurer
from QCompute.OpenSimulator.local_baidu_sim2.Transfer import Algorithm, TransferProcessor
from QCompute.QuantumPlatform import Error
from QCompute.QuantumPlatform.QuantumOperation.FixedGate import CX, X, Y, Z, H
from QCompute.QuantumPlatform.QuantumOperation.RotationGate import U
from QCompute.QuantumPlatform.Utilities import _protobufMatrixToNumpyMatrix, _filterMeasure
from QCompute.QuantumProtobuf.Library.PlatformStruct_pb2 import Program
from QCompute.QuantumProtobuf.Library.QuantumOperation_pb2 import FixedGate as FixedGateEnum, \
    RotationGate as RotationGateEnum, Measure as PBMeasure
from QCompute.QuantumPlatform import Error


def runSimulator(args, program):
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
    matrixType = args.mt.lower()
    algorithm = args.a.lower()
    measureMethod = args.mm.lower()
    seed = args.s
    shots = args.shots
    inputFile = args.inputFile

    if matrixType == 'dense':
        matrixType = MatrixType.Dense
    
    else:
        raise Error.ParamError(f'Invalid MatrixType {matrixType}')

    if algorithm == 'matmul':
        algorithm = Algorithm.Matmul
    elif algorithm == 'einsum':
        algorithm = Algorithm.Einsum
    else:
        raise Error.ParamError(f'Invalid Algorithm {algorithm}')

    if measureMethod == 'probability':
        measureMethod = MeasureMethod.Probability
    elif measureMethod == 'accumulation':
        measureMethod = MeasureMethod.Accumulation
    else:
        raise Error.ParamError(f'Invalid MeasureMethod {measureMethod}')

    if seed is not None:
        if isinstance(seed, int):
            if seed < 0 or seed > 2147483647:
                raise Error.ParamError(f'Invalid Seed {seed}')
        else:
            raise Error.ParamError(f'Invalid Seed {seed}')

    if shots <= 0:
        raise Error.ParamError(f'invalid shots {shots}')

    if inputFile is not None:
        with open(inputFile, "rt") as fObj:
            jsonStr = fObj.read()
        program = Parse(jsonStr, Program())

    return core(program, matrixType, algorithm, measureMethod, shots, seed)


def core(program, matrixType, algorithm, measureMethod, shots, seed):
    """
    Simulaton process
        Check if the argument is available. The accepted ones are:

        1)DENSE-EINSUM-SINGLE

        2)DENSE-EINSUM-PROB

        3)DENSE-MATMUL-SINGLE

        4)DENSE-MATMUL-PROB

        
    """

    compositeGate = CompositeGate()
    program = compositeGate(program)

    unrollProcedure = UnrollProcedure()
    program = unrollProcedure(program)

    unrollCircuit = UnrollCircuit()
    program = unrollCircuit(program)  # must unrollProcedure before, because of paramIds to paramValues

    if doCompressGate:
        compressGate = CompressGate()
        program = compressGate(program)

    

    qRegsMap = {qReg: index for index, qReg in enumerate(program.head.usingQRegs)}
    qRegCount = len(qRegsMap)

    operationDict = {}
    loadGates(matrixType, operationDict)

    if seed is None:
        seed = numpy.random.randint(0, 2147483647 + 1)
    numpy.random.seed(seed)

    state = initState_1_0(matrixType, qRegCount)
    transfer = TransferProcessor(matrixType, algorithm)
    measurer = Measurer(matrixType, algorithm, measureMethod)

    # collect the result to simulator for the subsequent invoking
    result = QuantumResult()
    result.startTimeUtc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')

    measured = False
    counts = {}
    measuredQRegsToCRegsBidict = bidict()
    for circuitLine in program.body.circuit:  # Traverse the circuit
        if measured and not circuitLine.HasField('measure'):
            raise Error.ParamError('measure must be the last operation')

        qRegs = []
        for qReg in circuitLine.qRegs:
            qRegs.append(qRegsMap[qReg])

        if circuitLine.HasField('fixedGate'):  # fixed gate
            matrix = operationDict.get(circuitLine.fixedGate)
            if matrix is None:
                raise Error.ParamError(f'unsupported operation {FixedGateEnum.Name(circuitLine.fixedGate)}')
            state = transfer(state, matrix, qRegs)
        elif circuitLine.HasField('rotationGate'):  # rotation gate
            if circuitLine.rotationGate != RotationGateEnum.U:
                raise Error.ParamError(
                    f'unsupported operation {RotationGateEnum.Name(circuitLine.rotationGate)}')
            uGate = U(*circuitLine.paramValues)
            if matrixType == MatrixType.Dense:
                matrix = uGate.matrix
            else:
                raise Error.RuntimeError('Not implemented')
            state = transfer(state, matrix, qRegs)
        elif circuitLine.HasField('customizedGate'):  # customized gate
            if matrixType == MatrixType.Dense:
                matrix = _protobufMatrixToNumpyMatrix(circuitLine.customizedGate.matrix)
            else:
                raise Error.RuntimeError('Not implemented')
            state = transfer(state, matrix, qRegs)
        elif circuitLine.HasField('procedureName'):  # procedure
            Error.ParamError('unsupported operation procedure, please flatten by UnrollProcedureModule')
            # it is not implemented, flattened by UnrollProcedureModule
        elif circuitLine.HasField('measure'):  # measure
            if circuitLine.measure.type != PBMeasure.Type.Z:  # only Z measure is supported
                raise Error.ParamError(
                    f'unsupported operation measure {PBMeasure.Type.Name(circuitLine.measure.type)}')
            if not measured:
                counts = measurer(state, shots)
                measured = True
            cRegs = []
            for cReg in circuitLine.measure.cRegs:
                cRegs.append(cReg)
            for i in range(len(cRegs)):
                if measuredQRegsToCRegsBidict.get(qRegs[i]) is not None:
                    raise Error.ParamError('measure must be once on a QReg')
                if measuredQRegsToCRegsBidict.inverse.get(cRegs[i]) is not None:
                    raise Error.ParamError('measure must be once on a CReg')
                measuredQRegsToCRegsBidict[qRegs[i]] = cRegs[i]
        elif circuitLine.HasField('barrier'):  # barrier
            pass
            # unimplemented operation
        else:  # unsupported operation
            raise Error.ParamError('unsupported operation')
    measuredCRegsList = list(measuredQRegsToCRegsBidict.keys())

    result.endTimeUtc = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')
    result.shots = shots
    result.counts = _filterMeasure(counts, measuredCRegsList)
    result.seed = int(seed)
    return result


def loadGates(matrixType, operationDict):
    """
    Load the matrix of the gate
    """

    if matrixType == MatrixType.Dense:
        operationDict[FixedGateEnum.X] = X.matrix
        operationDict[FixedGateEnum.Y] = Y.matrix
        operationDict[FixedGateEnum.Z] = Z.matrix
        operationDict[FixedGateEnum.H] = H.matrix
        operationDict[FixedGateEnum.CX] = CX.matrix.reshape(2, 2, 2, 2)
    else:
        raise Error.RuntimeError('Not implemented')


if __name__ == '__main__':
    result = runSimulator(None, None)
    print(json.dumps({
        'shots': result.shots,
        'counts': result.counts,
        'seed': result.seed,
        'startTimeUtc': result.startTimeUtc,
        'endTimeUtc': result.endTimeUtc,
    }))
