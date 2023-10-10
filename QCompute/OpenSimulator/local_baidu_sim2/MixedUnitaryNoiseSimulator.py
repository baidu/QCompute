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
MixedUnitaryNoiseSimulator
"""
FileErrorCode = 7

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
from QCompute.QPlatform.Utilities import normalizeNdarrayOrderForTranspose
from QCompute.QPlatform.Utilities import protobufQNoiseToQNoise, contract_in_3
from QCompute.QProtobuf import PBProgram, PBMeasure



class MixedUnitaryNoiseSimulator(BaseSimulator):
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

            noise_rngList = self.batchedNoiseRng(maxRunNum)

            if Settings.noiseMultiprocessingSimulator:
                self.programDict = MessageToDict(self.program)
                self.program = None
                noise_rng_array = []
                for i in range(len(noise_rngList[0])):
                    array = []
                    for j in range(len(noise_rngList)):
                        array.append(noise_rngList[j][i])
                    noise_rng_array.append(array)
                countsList = list(pool.map(self.coreOnceMixedUnitaryNoise, noise_rng_array))
            else:
                countsList = list(map(self.coreOnceMixedUnitaryNoise, *noise_rngList))

            # counts accumulate
            for element in countsList:
                for key, value in element.items():
                    if key in counts.keys():
                        counts[key] += value
                    else:
                        counts[key] = value

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
            noise_rngList = self.batchedNoiseRng(maxRunNum)

            if Settings.noiseMultiprocessingSimulator:
                self.programDict = MessageToDict(self.program)
                self.program = None
                noise_rng_array = []
                for i in range(len(noise_rngList[0])):
                    array = []
                    for j in range(len(noise_rngList)):
                        array.append(noise_rngList[j][i])
                    noise_rng_array.append(array)
                countsList = list(pool.map(self.coreOnceMixedUnitaryNoise, noise_rng_array))
                stateList = list(pool.map(self.coreOnceMixedUnitaryNoise, noise_rng_array))
            else:
                stateList = list(map(self.coreOnceMixedUnitaryNoise, *noise_rngList))

            for v in stateList:
                state_temp = v.reshape(-1)
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

    def coreOnceMixedUnitaryNoise(self, *noise_rng: int) -> Union[Dict[str, int], Dict[str, float], numpy.ndarray]:
        """
        Sample noise simulation process
        """
        if isinstance(noise_rng[0], list):
            noise_rng = noise_rng[0]

        if self.program is None:
            program = PBProgram()
            ParseDict(self.programDict, program)
        else:
            program = self.program

        qRegMap = {qReg: index for index,
        qReg in enumerate(program.head.usingQRegList)}
        qRegCount = len(qRegMap)

        state = initState_1_0(self.matrixType, qRegCount)
        transfer = TransferProcessor(self.matrixType, self.algorithm)
        measurer = Measurer(self.matrixType, self.algorithm, self.measureMethod)

        # local code reuse
        def transferNoiseMatrix(rng: List[int]) -> None:
            nonlocal state, pointer, matrix, qRegList, matrixCache, qRegListCache

            # matrix multiplication for maximum 3 qubits
            if len(set(qRegListCache) | set(qRegList)) <= 3:
                matrixCache, qRegListCache = contract_in_3(matrixCache, qRegListCache, matrix, qRegList)
            else:
                
                state = transfer(state, matrixCache, qRegListCache)
                matrixCache = matrix
                qRegListCache = qRegList

            for noise in circuitLine.noiseList:

                # create noise_instance
                noise_type = noise.WhichOneof('noise')
                noise_instance = protobufQNoiseToQNoise(noise)

                # calc noise matrix
                noise_matrix = noise_instance.krauses[rng[pointer]]

                if len(qRegList) == 1 and noise_matrix.shape != (2, 2):
                    raise Error.ArgumentError(f'Single-qubit noise {noise_type} must be applied after single-qubit gate!', ModuleErrorCode, FileErrorCode, 2)

                if len(qRegList) == 2 and noise_matrix.shape != (2, 2, 2, 2):
                    raise Error.ArgumentError(f'Double-qubit noise {noise_type} must be applied after double-qubit gate!', ModuleErrorCode, FileErrorCode, 3)

                # update cache matrix and qRegList
                matrixCache, qRegListCache = contract_in_3(matrixCache, qRegListCache, noise_matrix, sorted(qRegList))

                pointer += 1

        measured = False
        pointer = 0
        matrixCache = numpy.eye(2)  # Initialize an identity matrix
        qRegListCache = [0]  # Initialzie a qReg list  

        for circuitLine in program.body.circuit:  # Traverse the circuit
            op = circuitLine.WhichOneof('op')

            qRegList: List[int] = [qRegMap[_] for _ in circuitLine.qRegList]

            if op in ['fixedGate', 'rotationGate', 'customizedGate']:
                matrix = self.getGateMatrix(circuitLine, MatrixType.Dense, self.operationDict)
                transferNoiseMatrix(noise_rng)
            elif op == 'procedureName':  # procedure
                raise Error.ArgumentError(
                    'Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                    ModuleErrorCode, FileErrorCode, 4)

                # it is not implemented, flattened by UnrollProcedureModule
            elif op == 'measure':  # measure
                # transfer cache matrix
                
                state = transfer(state, matrixCache, qRegListCache)

                measure: PBMeasure = circuitLine.measure
                if measure.type != PBMeasure.Type.Z:  # only Z measure is supported
                    raise Error.ArgumentError(
                        f'Unsupported operation measure {PBMeasure.Type.Name(measure.type)}!',
                        ModuleErrorCode, FileErrorCode, 5)

                if not measured:
                    counts = measurer(state, self.shots)
                    measured = True
            elif op == 'barrier':  # barrier
                pass
            else:  # unsupported operation
                raise Error.ArgumentError(
                    f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 6)

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

    def batchedNoiseRng(self, maxRunNum: int) -> List[int]:
        """
        Return a batch of samples of noises.
        """
        batchedNoiseRng = []

        # local code reuse
        def transferNoiseRng(num: int) -> List[int]:
            noiseRng = []
            for noise in circuitLine.noiseList:
                # create noise_instance
                noise_instance = protobufQNoiseToQNoise(noise)

                assert noise_instance.noiseClass == 'mixed_unitary_noise', 'Please use this method for mixed-unitarty noises'

                noiseRng.append(
                    noise_instance.calc_batched_noise_rng(num))

            return noiseRng

        for circuitLine in self.program.body.circuit:  # Traverse the circuit
            op = circuitLine.WhichOneof('op')

            if op in ['fixedGate', 'rotationGate', 'customizedGate']:  # QCompute gate
                rngList = transferNoiseRng(maxRunNum)
                if rngList:
                    batchedNoiseRng.append(rngList)
            elif op == 'procedureName':  # procedure
                raise Error.ArgumentError(
                    'Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                    ModuleErrorCode, FileErrorCode, 7)

                # it is not implemented, flattened by UnrollProcedureModule
            elif op == 'measure' or op == 'barrier':
                pass
            else:  # unsupported operation
                raise Error.ArgumentError(
                    f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 8)

        # Reshape list
        batchedNoiseRng = [out_rng for in_rng in batchedNoiseRng for out_rng in in_rng]

        return batchedNoiseRng
