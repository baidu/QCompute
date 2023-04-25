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
GeneralNoiseSimulator
"""

from datetime import datetime
from typing import List, Union, Dict

import multiprocess
import numpy
from google.protobuf.json_format import MessageToDict, ParseDict

from QCompute.Define import Settings
from QCompute.OpenSimulator import QResult, ModuleErrorCode
from QCompute.OpenSimulator.local_baidu_sim2.BaseSimulator import BaseSimulator
from QCompute.OpenSimulator.local_baidu_sim2.InitState import MatrixType, initState_1_0
from QCompute.OpenSimulator.local_baidu_sim2.Measure import MeasureMethod, Measurer
from QCompute.OpenSimulator.local_baidu_sim2.Transfer import Algorithm, TransferProcessor
from QCompute.QPlatform import Error
from QCompute.QPlatform.Processor.PostProcessor import filterMeasure
from QCompute.QPlatform.Processor.PreProcess import preProcess
from QCompute.QPlatform.Utilities import protobufQNoiseToQNoise, contract_in_3
from QCompute.QPlatform.Utilities import normalizeNdarrayOrderForTranspose
from QCompute.QProtobuf import PBProgram, PBMeasure



FileErrorCode = 5


class GeneralNoiseSimulator(BaseSimulator):
    def core(self) -> 'QResult':

        usedQRegSet, usedCRegSet, compactedQRegDict, compactedCRegDict = preProcess(self.program, True, True)

        self.compactedCRegDict = compactedCRegDict

        

        if self.seed is None:
            self.seed = numpy.random.randint(0, 2147483647 + 1)
        numpy.random.seed(self.seed)

        # collect the result to simulator for the subsequent invoking
        result = QResult()
        result.startTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'

        pool = multiprocess.Pool()

        maxRunNum = self.maxRunNum

        # accumulate
        if self.measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation,
                             MeasureMethod.OutputProbability]:
            counts = {}

            if Settings.noiseMultiprocessingSimulator:
                self.programDict = MessageToDict(self.program)
                self.program = None
                countsList = list(pool.map(self.coreOnceGeneralNoise, range(maxRunNum)))
            else:
                countsList = list(map(self.coreOnceGeneralNoise, range(maxRunNum)))

            for counts_once in countsList:
                # counts accumulate
                for key, value in counts_once.items():
                    if key not in counts:
                        counts[key] = 0
                    counts[key] += counts_once[key]

            # average
            for key in counts:
                counts[key] /= maxRunNum
                if self.measureMethod == MeasureMethod.OutputProbability:
                    counts[key] = counts[key].round(6)

            # safe round
            if self.measureMethod != MeasureMethod.OutputProbability:
                counts = self.safeRoundCounts(counts, self.shots)

            # write counts back to result
            result.counts = counts
        elif self.measureMethod == MeasureMethod.OutputState:
            state = 0.0 + 0.0j
            if Settings.noiseMultiprocessingSimulator:
                self.programDict = MessageToDict(self.program)
                self.program = None
                stateList = list(pool.map(self.coreOnceGeneralNoise, range(maxRunNum)))
            else:
                stateList = list(map(self.coreOnceGeneralNoise, range(maxRunNum)))

            # state accumulate
            for state_once in stateList:
                state_temp = state_once.reshape(-1)
                state += numpy.outer(state_temp, state_temp.conjugate())

            # Average
            state /= maxRunNum

            # write state back to result, restrict precision to e-03
            result.state = state.round(3)

        result.endTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'
        result.shots = self.shots
        result.ancilla.usedQRegList = list(usedQRegSet)
        result.ancilla.usedCRegList = list(usedCRegSet)
        result.ancilla.compactedQRegDict = compactedQRegDict
        result.ancilla.compactedCRegDict = compactedCRegDict
        result.seed = int(self.seed)

        return result

    def coreOnceGeneralNoise(self, voidParam) -> Union[Dict[str, int], Dict[str, float], numpy.ndarray]:
        """
        Sample noise simulation process
        """
        if self.program is None:
            program = PBProgram()
            ParseDict(self.programDict, program)
        else:
            program = self.program

        qRegMap = {qReg: index for index, qReg in enumerate(program.head.usingQRegList)}
        qRegCount = len(qRegMap)

        state = initState_1_0(self.matrixType, qRegCount)
        transfer = TransferProcessor(self.matrixType, self.algorithm)
        measurer = Measurer(self.matrixType, self.algorithm, self.measureMethod)

        # local code reuse
        def transferNoiseList() -> None:            
            nonlocal state, matrixCache, qRegListCache, matrix, qRegList
            # matrix multiplication for maximum 3 qubits
            if len(set(qRegListCache)|set(qRegList)) <= 3: 
                matrixCache, qRegListCache = contract_in_3(matrixCache, qRegListCache, matrix, qRegList)
            else:
                if self.matrixType == MatrixType.Sparse:
                    matrixCache = sparse.COO(matrixCache)
                state = transfer(state, matrixCache, qRegListCache)
                matrixCache = matrix
                qRegListCache = qRegList
            for noise in circuitLine.noiseList:
                # create noise_instance
                noise_type = noise.WhichOneof('noise')
                noise_instance = protobufQNoiseToQNoise(noise)

                # calc noise matrix
                if noise_instance.noiseClass == 'non_mixed_unitary_noise': 
                    if self.matrixType == MatrixType.Sparse:
                        matrixCache = sparse.COO(matrixCache)
                    state = transfer(state, matrixCache, qRegListCache)
 
                noise_matrix = noise_instance.calc_noise_matrix(
                    transfer, state, qRegList)
                
                if len(qRegList) == 1 and noise_matrix.shape != (2, 2):
                    raise Error.ArgumentError(
                        f'Single-qubit noise {noise_type} must be applied after single-qubit gate!',
                        ModuleErrorCode, FileErrorCode, 7)
                if len(qRegList) == 2 and noise_matrix.shape != (2, 2, 2, 2):
                    raise Error.ArgumentError(
                        f'Double-qubit noise {noise_type} must be applied after double-qubit gate!',
                        ModuleErrorCode, FileErrorCode, 8)

                # update cache matrix and qRegList
                if noise_instance.noiseClass == 'non_mixed_unitary_noise': 
                    matrixCache = noise_matrix
                    qRegListCache = qRegList
                else:
                    matrixCache, qRegListCache = contract_in_3(matrixCache, qRegListCache, noise_matrix, sorted(qRegList))
                

        measured = False
        matrixCache = numpy.eye(2)  # Initialize an identity matrix
        qRegListCache = [0]  # Initialzie a qReg list  

        for circuitLine in program.body.circuit:  # Traverse the circuit
            op = circuitLine.WhichOneof('op')

            qRegList: List[int] = [qRegMap[_] for _ in circuitLine.qRegList]

            if op in ['fixedGate', 'rotationGate', 'customizedGate']:
                matrix = self.getGateMatrix(circuitLine, MatrixType.Dense, self.operationDict)
                transferNoiseList()
            elif op == 'procedureName':  # procedure
                raise Error.ArgumentError('Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                                          ModuleErrorCode, FileErrorCode, 12)
                # it is not implemented, flattened by UnrollProcedureModule
            elif op == 'measure':  # measure
                # transfer cache matrix
                if self.matrixType == MatrixType.Sparse:
                    matrixCache = sparse.COO(matrixCache)
                state = transfer(state, matrixCache, qRegListCache)

                measure: PBMeasure = circuitLine.measure
                if measure.type != PBMeasure.Type.Z:  # only Z measure is supported
                    raise Error.ArgumentError(
                        f'Unsupported operation measure {PBMeasure.Type.Name(measure.type)}!', ModuleErrorCode,
                        FileErrorCode, 13)
                if not measured:
                    counts = measurer(state, self.shots)
                    measured = True
            elif op == 'barrier':  # barrier
                pass
            else:  # unsupported operation
                raise Error.ArgumentError(
                    f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 14)

        if self.measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation]:
            counts = filterMeasure(counts, self.compactedCRegDict)
            return counts
        elif self.measureMethod == MeasureMethod.OutputProbability:
            return counts
        elif self.measureMethod == MeasureMethod.OutputState:
            if self.matrixType == MatrixType.Dense:
                state = normalizeNdarrayOrderForTranspose(state)
            else:
                state = normalizeNdarrayOrderForTranspose(state.todense())
            return state