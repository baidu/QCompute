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
LowNoiseCircuitSimulator
"""
FileErrorCode = 5

from cmath import isclose
from datetime import datetime
from typing import List, Union, Dict, Optional

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
from QCompute.QPlatform.Utilities import protobufQNoiseToQNoise
from QCompute.QProtobuf import PBProgram, PBMeasure




class LowNoiseCircuitSimulator(BaseSimulator):
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

            # todo: adaptive each_num
            if maxRunNum < 10:
                each_num = 1
            else:
                each_num = 10
            batch_num = round(maxRunNum / each_num)
            maxRunNum = each_num * batch_num

            if Settings.noiseMultiprocessingSimulator:
                self.programDict = MessageToDict(self.program)
                self.program = None
                countsList = list(pool.map(self.coreOnceLowNoiseCircuit, [each_num] * batch_num))
            else:
                countsList = list(map(self.coreOnceLowNoiseCircuit, [each_num] * batch_num))

            for counts_batch in countsList:
                # counts accumulate
                for key, value in counts_batch.items():
                    if key not in counts:
                        counts[key] = 0
                    counts[key] += counts_batch[key]

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

            # Todo: adaptive
            if maxRunNum < 10:
                each_num = 1
            else:
                each_num = 10
            batch_num = round(maxRunNum / each_num)

            if Settings.noiseMultiprocessingSimulator:
                self.programDict = MessageToDict(self.program)
                self.program = None
                stateList = list(pool.map(self.coreOnceLowNoiseCircuit, [each_num] * batch_num))
            else:
                stateList = list(map(self.coreOnceLowNoiseCircuit, [each_num] * batch_num))

            # state accumulate
            num = 0
            for stateDict in stateList:
                # maxRunNum accumulate, in case of 0 state output
                for key, value in stateDict.items():
                    state_temp = value.reshape(-1)
                    state += numpy.outer(state_temp,
                                         state_temp.conjugate()) * len(key)
                    num += len(key)
            maxRunNum = num

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

    def coreOnceLowNoiseCircuit(self, num: int) -> Union[
        Dict[str, int], Dict[str, float], Dict[str, numpy.ndarray]]:
        """
        Batched sample noise simulation process
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

        noiseType = Union['AmplitudeDamping', 'BitFlip', 'CustomizedNoise', 'BitPhaseFlip',
                          'Depolarizing', 'PauliNoise', 'PhaseDamping', 'PhaseFlip', 'ResetNoise']

        # Initialize a dict of state
        stateDict = {''.join([str(index) for index in range(num)]): state for _ in range(num)}

        # local code use
        def transferBatchNoisyGate(gate_matrix: numpy.ndarray, run_num: int) -> None:
            nonlocal stateDict
            for key, value in stateDict.items():  # transfer gate
                stateDict[key] = transfer(value, gate_matrix, qRegList)

            for noise in circuitLine.noiseList:
                # create noise_instance
                noise_instance = protobufQNoiseToQNoise(noise)

                # simulate batched samples of noise
                if noise_instance.noiseClass == 'mixed_unitary_noise':
                    rngList = noise_instance.calc_batched_noise_rng(run_num)
                    transferBatchNoises(noise_instance, rngList)
                else:
                    rngList = noise_instance.calc_batched_noise_rng_non_mixed(
                        transfer, stateDict, qRegList)
                    extendRngList(rngList, run_num)
                    transferBatchNoises(noise_instance, rngList)
                    normStateDict()

        # local code use
        def extendRngList(randomList: List[int], bound: int) -> None:
            """
            Extend a rngList up to a bound number
            """
            nonlocal stateDict
            index_current = [int(_) for key in stateDict.keys() for _ in key]
            index_complement = [_ for _ in range(bound) if _ not in index_current]
            for index in index_complement:  # complement rngList
                randomList.insert(index + 1, 0)

        # local code use
        def transferBatchNoises(noise_instance: noiseType, rngList: List[int]) -> None:
            """
            Transfer noise according to a batch of samples
            """
            nonlocal stateDict
            state_dict_temp = {}
            if len(set(rngList)) == 1:  # all samples give the same value
                for key_state_dict, value_state_dict in stateDict.items():
                    if self.matrixType == MatrixType.Dense:
                        noise_matrix = noise_instance.krauses[rngList[0]]
                    else:
                        from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                    state_dict_temp[key_state_dict] = transfer(
                        value_state_dict, noise_matrix, qRegList)  # noise transfer
            elif len(set(rngList)) == num:  # each sample gives a different value
                for key_state_dict, value_state_dict in stateDict.items():
                    for string in key_state_dict:  # transfer noise for all samples
                        if self.matrixType == MatrixType.Dense:
                            noise_matrix = noise_instance.krauses[rngList[int(  # get noise matrix according to samples
                                string)]]
                        else:
                            from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                        state_dict_temp[string] = transfer(
                            value_state_dict, noise_matrix, qRegList)  # noise transfer
            else:
                for key_state_dict, value_state_dict in stateDict.items():
                    # get samples for each key in the stateDict
                    rngSlice = [rngList[int(string)] for string in key_state_dict]

                    if len(set(rngSlice)) == 1:  # all samples in the key give the same value
                        if self.matrixType == MatrixType.Dense:
                            noise_matrix = noise_instance.krauses[rngList[int(
                                key_state_dict[0])]]
                        else:
                            from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                        state_dict_temp[key_state_dict] = transfer(
                            value_state_dict, noise_matrix, qRegList)  # noise transfer
                    # each sample in the key gives a different value
                    elif len(set(rngSlice)) == len(rngSlice):
                        for string in key_state_dict:
                            if self.matrixType == MatrixType.Dense:
                                noise_matrix = noise_instance.krauses[rngList[int(
                                    string)]]
                            else:
                                from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                            state_dict_temp[string] = transfer(
                                value_state_dict, noise_matrix, qRegList)  # noise transfer
                    else:
                        # split branch of stateDict according to noise samples
                        strDict = splitBranch(str(key_state_dict), rngList)

                        for key_str_dict in strDict.keys():
                            if self.matrixType == MatrixType.Dense:
                                noise_matrix = noise_instance.krauses[rngList[int(
                                    key_str_dict[0])]]
                            else:
                                from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                            state_dict_temp[key_str_dict] = transfer(
                                value_state_dict, noise_matrix, qRegList)  # noise transfer

            stateDict = state_dict_temp  # update the state dict

        # local code use
        def splitBranch(strList: str, rngList: List[int]) -> Dict[str, int]:
            """
            Split a str according to noise samples
            """
            str_dict = {}
            dict_temp = {}

            for string in strList:  # split branch for each string in strList
                rng_temp = rngList[int(string)]
                if rng_temp not in dict_temp.keys():
                    dict_temp[rng_temp] = string
                else:
                    dict_temp[rng_temp] += string

            # reverse value and key of a dict
            dict_temp = {value: key for key, value in dict_temp.items()}

            str_dict.update(dict_temp)
            return str_dict

        # local code use
        def normStateDict() -> None:
            """
            Return the norm of any unnormalized state in a state dict
            """
            nonlocal stateDict
            for key in list(stateDict.keys()):
                pro_temp = sum(numpy.reshape(
                    numpy.abs(stateDict[key]) ** 2, -1))

                if isclose(1.0, pro_temp):
                    pass
                elif isclose(0.0, pro_temp):
                    stateDict.pop(key)
                else:
                    stateDict[key] = stateDict[key] / numpy.sqrt(pro_temp)

        # local code use
        def batchMeasure(stateDic: Optional[Dict[str, numpy.ndarray]], measurerCore: 'Measurer',
                         countsShots: int) -> Optional[Union[Dict[str, int], Dict[str, float]]]:
            """
            Measure a batch of states
            """
            nonlocal stateDict

            if self.measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation,
                                 MeasureMethod.OutputProbability]:
                sumCounts = {}

                for key, stateKey in stateDic.items():
                    # clear 0 state
                    if type(stateKey) == numpy.ndarray:
                        counts_temp = measurerCore(stateKey, countsShots)
                    else:
                        counts_temp = {}

                    # counts accumulation
                    for key_counts_temp, value_counts_temp in counts_temp.items():
                        if key_counts_temp not in sumCounts:
                            sumCounts[key_counts_temp] = 0
                        sumCounts[key_counts_temp] += value_counts_temp * \
                                                      len(key)
                return sumCounts
            elif self.measureMethod in [MeasureMethod.OutputState, MeasureMethod.OutputProbability]:
                pass

        measured = False
        for circuitLine in program.body.circuit:  # Traverse the circuit
            op = circuitLine.WhichOneof('op')

            qRegList: List[int] = [qRegMap[_] for _ in circuitLine.qRegList]

            if not stateDict:  # Verify the output stateDict is not None
                return {}
            else:
                if op in ['fixedGate', 'rotationGate', 'customizedGate']:
                    matrix = self.getGateMatrix(circuitLine, self.matrixType, self.operationDict)
                    transferBatchNoisyGate(matrix, num)
                elif op == 'proceduresName':  # procedure
                    raise Error.ArgumentError(
                        'Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                        ModuleErrorCode, FileErrorCode, 2)

                    # it is not implemented, flattened by UnrollProcedureModule
                elif op == 'measure':  # measure
                    measure: PBMeasure = circuitLine.measure
                    if measure.type != PBMeasure.Type.Z:  # only Z measure is supported
                        raise Error.ArgumentError(
                            f'Unsupported operation measure {PBMeasure.Type.Name(measure.type)}!',
                            ModuleErrorCode, FileErrorCode, 3)

                    if not measured:
                        # measure a batch of states
                        counts = batchMeasure(stateDict, measurer, self.shots)
                        measured = True
                elif op == 'barrier':  # barrier
                    pass
                    # unimplemented operation
                else:  # unsupported operation
                    raise Error.ArgumentError(
                        f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 4)

        if self.measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation]:
            counts = filterMeasure(counts, self.compactedCRegDict)
            return counts
        elif self.measureMethod == MeasureMethod.OutputProbability:
            return counts
        elif self.measureMethod == MeasureMethod.OutputState:
            if self.matrixType == MatrixType.Dense:
                stateDict = {key: normalizeNdarrayOrderForTranspose(value) for key, value in stateDict.items()}
            else:
                stateDict = {key: normalizeNdarrayOrderForTranspose(value.todense()) for key, value in
                             stateDict.items()}
            return stateDict
