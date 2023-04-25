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
NoNoiseSimulator
"""

from datetime import datetime
from typing import List, Union, Dict

import numpy
from google.protobuf.json_format import ParseDict

from QCompute.OpenSimulator import QResult, ModuleErrorCode
from QCompute.OpenSimulator.local_baidu_sim2.BaseSimulator import BaseSimulator
from QCompute.OpenSimulator.local_baidu_sim2.InitState import MatrixType, initState_1_0
from QCompute.OpenSimulator.local_baidu_sim2.Measure import MeasureMethod, Measurer
from QCompute.OpenSimulator.local_baidu_sim2.Transfer import Algorithm, TransferProcessor
from QCompute.QPlatform import Error
from QCompute.QPlatform.Processor.PostProcessor import filterMeasure
from QCompute.QPlatform.Processor.PreProcess import preProcess
from QCompute.QPlatform.QOperation.RotationGate import U
from QCompute.QPlatform.Utilities import protobufMatrixToNumpyMatrix, normalizeNdarrayOrderForTranspose
from QCompute.QProtobuf import PBProgram, PBFixedGate, PBRotationGate, PBCustomizedGate, PBMeasure



FileErrorCode = 5


class NoNoiseSimulator(BaseSimulator):
    def core(self) -> 'QResult':
        program = self.program
        matrixType = self.matrixType
        algorithm = self.algorithm
        measureMethod = self.measureMethod
        shots = self.shots
        seed = self.seed

        usedQRegSet, usedCRegSet, compactedQRegDict, compactedCRegDict = preProcess(program, True, True)

        self.compactedCRegDict = compactedCRegDict

        

        if seed is None:
            seed = numpy.random.randint(0, 2147483647 + 1)
        numpy.random.seed(seed)

        # collect the result to simulator for the subsequent invoking
        result = QResult()
        result.startTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'

        if measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation,
                             MeasureMethod.OutputProbability]:
            result.counts = self.coreOnceWithoutNoise()
        elif measureMethod == MeasureMethod.OutputState:
            result.state = self.coreOnceWithoutNoise()

        result.endTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'
        result.shots = shots
        result.ancilla.usedQRegList = list(usedQRegSet)
        result.ancilla.usedCRegList = list(usedCRegSet)
        result.ancilla.compactedQRegDict = compactedQRegDict
        result.ancilla.compactedCRegDict = compactedCRegDict
        result.seed = int(seed)

        return result

    def coreOnceWithoutNoise(self) -> Union[Dict[str, int], Dict[str, float], numpy.ndarray]:
        """
        Simulation process for ideal circuit
        """

        if self.program is None:
            program = PBProgram()
            ParseDict(self.programDict, program)
        else:
            program = self.program
        matrixType = self.matrixType
        algorithm = self.algorithm
        measureMethod = self.measureMethod
        shots = self.shots
        seed = self.seed
        operationDict = self.operationDict

        qRegMap = {qReg: index for index,
                                   qReg in enumerate(program.head.usingQRegList)}
        qRegCount = len(qRegMap)

        if seed is None:
            seed = numpy.random.randint(0, 2147483647 + 1)
        numpy.random.seed(seed)

        state = initState_1_0(matrixType, qRegCount)
        transfer = TransferProcessor(matrixType, algorithm)
        measurer = Measurer(matrixType, algorithm, measureMethod)

        measured = False
        for circuitLine in program.body.circuit:  # Traverse the circuit
            op = circuitLine.WhichOneof('op')

            qRegList: List[int] = [qRegMap[qReg] for qReg in circuitLine.qRegList]

            if op == 'fixedGate':  # fixed gate
                fixedGate: PBFixedGate = circuitLine.fixedGate
                matrix = operationDict.get(fixedGate)
                if matrix is None:
                    raise Error.ArgumentError(f'Unsupported operation {PBFixedGate.Name(fixedGate)}!', ModuleErrorCode,
                                              FileErrorCode, 7)
                state = transfer(state, matrix, qRegList)
            elif op == 'rotationGate':  # rotation gate
                rotationGate: PBRotationGate = circuitLine.rotationGate
                if rotationGate != PBRotationGate.U:
                    raise Error.ArgumentError(
                        f'Unsupported operation {PBRotationGate.Name(rotationGate)}!', ModuleErrorCode, FileErrorCode,
                        8)
                uGate = U(*circuitLine.argumentValueList)
                if matrixType == MatrixType.Dense:
                    matrix = uGate.getMatrix()
                else:
                    from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                state = transfer(state, matrix, qRegList)
            elif op == 'customizedGate':  # customized gate
                customizedGate: PBCustomizedGate = circuitLine.customizedGate
                if matrixType == MatrixType.Dense:
                    matrix = protobufMatrixToNumpyMatrix(customizedGate.matrix)
                else:
                    from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                state = transfer(state, matrix, qRegList)
            elif op == 'procedureName':  # procedure
                raise Error.ArgumentError('Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                                          ModuleErrorCode, FileErrorCode, 9)
                # it is not implemented, flattened by UnrollProcedureModule
            elif op == 'measure':  # measure
                measure: PBMeasure = circuitLine.measure
                if measure.type != PBMeasure.Type.Z:  # only Z measure is supported
                    raise Error.ArgumentError(
                        f'Unsupported operation measure {PBMeasure.Type.Name(measure.type)}!', ModuleErrorCode,
                        FileErrorCode, 10)
                if not measured:
                    counts = measurer(state, shots)
                    measured = True
            elif op == 'barrier':  # barrier
                pass
                # unimplemented operation
            else:  # unsupported operation
                raise Error.ArgumentError(f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 11)

        if measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation]:
            return filterMeasure(counts, self.compactedCRegDict)
        elif measureMethod == MeasureMethod.OutputProbability:
            return counts
        if measureMethod == MeasureMethod.OutputState:
            if matrixType == MatrixType.Dense:
                return normalizeNdarrayOrderForTranspose(state)
            else:
                return normalizeNdarrayOrderForTranspose(state.todense())
