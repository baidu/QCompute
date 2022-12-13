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
Simulator
The simulator uses statevector (a rank-n tensor representing an n-qubit state) to simulate quantum behaviors.
Basically, the core of the algorithm is tensor contraction with one-way calculation.
The initial state and gates are converted to tensors and gate implementation is simulated as contraction of tensors.
"""

import argparse
from cmath import isclose
from datetime import datetime
from pathlib import Path
from typing import List, TYPE_CHECKING, Union, Dict, Optional

import numpy

from QCompute import Define
from QCompute.OpenConvertor.JsonToCircuit import JsonToCircuit
from QCompute.OpenSimulator import QResult, ModuleErrorCode
from QCompute.OpenSimulator.local_baidu_sim2_with_noise.InitState import MatrixType, initState_1_0
from QCompute.OpenSimulator.local_baidu_sim2_with_noise.Measure import MeasureMethod, Measurer
from QCompute.OpenSimulator.local_baidu_sim2_with_noise.Transfer import Algorithm, TransferProcessor
from QCompute.QPlatform import Error
from QCompute.QPlatform.Processor.PostProcessor import filterMeasure
from QCompute.QPlatform.Processor.PreProcess import preProcess
from QCompute.QPlatform.QNoise.AmplitudeDamping import AmplitudeDamping
from QCompute.QPlatform.QNoise.BitFlip import BitFlip
from QCompute.QPlatform.QNoise.BitPhaseFlip import BitPhaseFlip
from QCompute.QPlatform.QNoise.CustomizedNoise import CustomizedNoise
from QCompute.QPlatform.QNoise.Depolarizing import Depolarizing
from QCompute.QPlatform.QNoise.PauliNoise import PauliNoise
from QCompute.QPlatform.QNoise.PhaseDamping import PhaseDamping
from QCompute.QPlatform.QNoise.PhaseFlip import PhaseFlip
from QCompute.QPlatform.QNoise.ResetNoise import ResetNoise
from QCompute.QPlatform.QOperation.FixedGate import ID, X, Y, Z, H, S, SDG, T, TDG, CX, CY, CZ, CH, SWAP, CCX, CSWAP
from QCompute.QPlatform.QOperation.RotationGate import RotationGateOP, U, RX, RY, RZ, CU, CRX, CRY, CRZ
from QCompute.QPlatform.Utilities import protobufMatrixToNumpyMatrix, normalizeNdarrayOrderForTranspose
from QCompute.QProtobuf import PBCircuitLine, PBFixedGate, PBRotationGate, PBCustomizedGate, PBMeasure

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBProgram

FileErrorCode = 5


class StateVector:
    def __init__(self, program: 'PBProgram', matrixType: 'MatrixType', algorithm: 'Algorithm',
                 measureMethod: 'MeasureMethod', shots: int, seed: int) -> None:
        self.program = program
        self.matrixType = matrixType
        self.algorithm = algorithm
        self.measureMethod = measureMethod
        self.shots = shots
        self.seed = seed
        self.num_qubits = len(program.head.usingQRegList)
        self.operationDict = self.loadGates(matrixType)
        self.compactedCRegDict = None

        # adaptive max_run_num
        epsilon = 0.05
        delta = 0.01
        self.max_run_num = round(
            numpy.log(2 ** (1 + self.num_qubits) / delta) / (2 * epsilon * epsilon))

        # different methods 0, 1, 2,
        # 0 for mixed unitary noise
        # 1 for general noise
        # 2 for low noise circuit, speed up
        self.method = 2

    def core(self) -> 'QResult':
        """
        Simulation process
            Check if the argument is available. The accepted ones are:

            1)DENSE-MATMUL-PROB

            2)DENSE-MATMUL-STATE

            3)DENSE-MATMUL-SINGLE

            4)DENSE-MATMUL-OUTPROB

            
        """
        program = self.program
        matrixType = self.matrixType
        algorithm = self.algorithm
        measureMethod = self.measureMethod
        shots = self.shots
        seed = self.seed

        usedQRegSet, usedCRegSet, compactedQRegDict, compactedCRegDict = preProcess(
            program, True, True)

        self.compactedCRegDict = compactedCRegDict

        

        if seed is None:
            seed = numpy.random.randint(0, 2147483647 + 1)
        numpy.random.seed(seed)

        # collect the result to simulator for the subsequent invoking
        result = QResult()
        result.startTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'

        if all(len(circLine.noiseList) < 1 for circLine in program.body.circuit):
            self.max_run_num = 1

        max_run_num = self.max_run_num

        # accumulate
        if measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation,
                             MeasureMethod.OutputProbability]:

            counts = {}

            if self.method == 0:
                noise_rngList = self.batched_noise_rng(max_run_num)

                if max_run_num == 1:
                    counts = self.core_once(noise_rngList)
                else:
                    countsList = list(map(self.core_once, *noise_rngList))

                    # counts accumulate
                    for element in countsList:
                        for key, value in element.items():
                            if key in counts.keys():
                                counts[key] += value
                            else:
                                counts[key] = value
            elif self.method == 1:

                for _ in range(max_run_num):
                    counts_once = self.core_once_v1()

                    # counts accumulate
                    for key, value in counts_once.items():
                        if key not in counts:
                            counts[key] = 0
                        counts[key] += counts_once[key]
            elif self.method == 2:
                # todo: adaptive each_num
                if max_run_num < 10:
                    each_num = 1
                else:
                    each_num = 10
                batch_num = round(max_run_num / each_num)
                max_run_num = each_num * batch_num

                for _ in range(batch_num):
                    counts_batch = self.core_once_v2(each_num)
                    # counts accumulate
                    for key, value in counts_batch.items():
                        if key not in counts:
                            counts[key] = 0
                        counts[key] += counts_batch[key]

            # average
            for key in counts:
                counts[key] /= max_run_num
                if measureMethod == MeasureMethod.OutputProbability:
                    counts[key] = counts[key].round(6)

            # safe round
            if measureMethod != MeasureMethod.OutputProbability:
                counts = safe_round_counts(counts, shots)

            # write counts back to result
            result.counts = counts
        elif measureMethod == MeasureMethod.OutputState:

            state = 0.0 + 0.0j
            if self.method == 0:
                noise_rngList = self.batched_noise_rng(max_run_num)
                stateList = list(map(self.core_once, *noise_rngList))
                for v in stateList:
                    state_temp = v.reshape(-1)
                    state += numpy.outer(state_temp, state_temp.conjugate())
            elif self.method == 1:
                # state accumulate
                for _ in range(max_run_num):
                    state_once = self.core_once_v1()
                    state_temp = state_once.reshape(-1)
                    state += numpy.outer(state_temp, state_temp.conjugate())
            elif self.method == 2:
                # Todo: adaptive
                if max_run_num < 10:
                    each_num = 1
                else:
                    each_num = 10
                batch_num = round(max_run_num / each_num)

                # state accumulate
                num = 0
                for _ in range(batch_num):
                    # max_run_num accumulate, in case of 0 state output
                    stateDict = self.core_once_v2(each_num)
                    for key, value in stateDict.items():
                        state_temp = value.reshape(-1)
                        state += numpy.outer(state_temp,
                                             state_temp.conjugate()) * len(key)
                        num += len(key)
                max_run_num = num

            # Average
            state /= max_run_num

            # write state back to result, restrict precision to e-03
            result.state = state.round(3)

        result.endTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'
        result.shots = shots
        result.ancilla.usedQRegList = list(usedQRegSet)
        result.ancilla.usedCRegList = list(usedCRegSet)
        result.ancilla.compactedQRegDict = compactedQRegDict
        result.ancilla.compactedCRegDict = compactedCRegDict
        result.seed = int(seed)

        return result

    def core_once(self, *noise_rng: int) -> Union[Dict[str, int], Dict[str, float], numpy.ndarray]:
        """
        Sample noise simulation process
        """
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

        # local code reuse
        def transfer_noise_matrix(rng: int) -> None:
            nonlocal state, pointer
            for noise in circuitLine.noiseList:

                # create noise_instance
                noise_type = noise.WhichOneof('noise')
                if noise_type == 'amplitudeDamping':
                    noise_instance = AmplitudeDamping(
                        noise.amplitudeDamping.probability)
                elif noise_type == 'bitFlip':
                    noise_instance = BitFlip(noise.bitFlip.probability)
                elif noise_type == 'bitPhaseFlip':
                    noise_instance = BitPhaseFlip(
                        noise.bitPhaseFlip.probability)
                elif noise_type == 'customizedNoise':
                    noise_instance = CustomizedNoise(
                        list(map(protobufMatrixToNumpyMatrix, noise.customizedNoise.krauses)))
                elif noise_type == 'depolarizing':
                    noise_instance = Depolarizing(noise.depolarizing.bits,
                                                  noise.depolarizing.probability)
                elif noise_type == 'pauliNoise':
                    noise_instance = PauliNoise(noise.pauliNoise.probability1, noise.pauliNoise.probability2,
                                                noise.pauliNoise.probability3)
                elif noise_type == 'phaseDamping':
                    noise_instance = PhaseDamping(
                        noise.phaseDamping.probability)
                elif noise_type == 'phaseFlip':
                    noise_instance = PhaseFlip(noise.phaseFlip.probability)
                elif noise_type == 'resetNoise':
                    noise_instance = ResetNoise(
                        noise.resetNoise.probability1, noise.resetNoise.probability2)
                else:
                    raise Error.ArgumentError(
                        f'Unsupported noise type {noise_type}!', ModuleErrorCode, FileErrorCode, 9)

                # calc noise matrix
                if matrixType == MatrixType.Dense:
                    noise_matrix = noise_instance.krauses[rng[pointer]]
                else:
                    noise_matrix = sparse.COO(
                        noise_instance.krauses[rng[pointer]])

                # noise transfer
                if len(qRegList) == 1 and noise_matrix.shape != (2, 2):
                    raise Error.ArgumentError(
                        f'Single-qubit noise {noise_type} must be applied after single-qubit gate!',
                        ModuleErrorCode, FileErrorCode, 7)
                if len(qRegList) == 2 and noise_matrix.shape != (2, 2, 2, 2):
                    raise Error.ArgumentError(
                        f'Double-qubit noise {noise_type} must be applied after double-qubit gate!',
                        ModuleErrorCode, FileErrorCode, 8)
                state = transfer(state, noise_matrix, qRegList)

                pointer += 1

        measured = False
        pointer = 0
        for circuitLine in program.body.circuit:  # Traverse the circuit
            op = circuitLine.WhichOneof('op')

            qRegList = [qRegMap[_]
                        for _ in circuitLine.qRegList]  # type List[int]

            if op == 'fixedGate':  # fixed gate
                fixedGate = circuitLine.fixedGate  # type: PBFixedGate
                matrix = operationDict.get(fixedGate)
                if matrix is None:
                    raise Error.ArgumentError(f'Unsupported operation {PBFixedGate.Name(fixedGate)}!', ModuleErrorCode,
                                              FileErrorCode, 10)
                state = transfer(state, matrix, qRegList)
                transfer_noise_matrix(noise_rng)

            elif op == 'rotationGate':  # rotation gate
                uGate = getRotationGate(circuitLine) # type RotationGateOp

                if matrixType == MatrixType.Dense:
                    matrix = uGate.getMatrix()
                else:
                    from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                state = transfer(state, matrix, qRegList)
                transfer_noise_matrix(noise_rng)
            elif op == 'customizedGate':  # customized gate
                customizedGate = circuitLine.customizedGate  # type: PBCustomizedGate
                if matrixType == MatrixType.Dense:
                    matrix = protobufMatrixToNumpyMatrix(customizedGate.matrix)
                else:
                    from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                state = transfer(state, matrix, qRegList)
                transfer_noise_matrix(noise_rng)
            elif op == 'procedureName':  # procedure
                raise Error.ArgumentError('Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                                          ModuleErrorCode, FileErrorCode, 12)
                # it is not implemented, flattened by UnrollProcedureModule
            elif op == 'measure':  # measure
                measure = circuitLine.measure  # type: PBMeasure
                if measure.type != PBMeasure.Type.Z:  # only Z measure is supported
                    raise Error.ArgumentError(
                        f'Unsupported operation measure {PBMeasure.Type.Name(measure.type)}!', ModuleErrorCode,
                        FileErrorCode, 13)
                if not measured:
                    counts = measurer(state, shots)
                    measured = True
            elif op == 'barrier':  # barrier
                pass
                # unimplemented operation
            else:  # unsupported operation
                raise Error.ArgumentError(
                    f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 14)

        if measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation]:
            counts = filterMeasure(counts, self.compactedCRegDict)
            return counts
        elif measureMethod == MeasureMethod.OutputProbability:
            return counts
        elif measureMethod == MeasureMethod.OutputState:
            if matrixType == MatrixType.Dense:
                state = normalizeNdarrayOrderForTranspose(state)
            else:
                state = normalizeNdarrayOrderForTranspose(state.todense())
            return state

    def batched_noise_rng(self, max_run_num: int) -> List[int]:
        """
        Return a batch of samples of noises.
        """
        batchedNoiseRng = []
        program = self.program
        matrixType = self.matrixType
        operationDict = self.operationDict

        # local code reuse
        def transfer_noise_rng(num: int) -> List[int]:
            noiseRng = []
            for noise in circuitLine.noiseList:
                # create noise_instance
                noise_type = noise.WhichOneof('noise')
                if noise_type == 'amplitudeDamping':
                    noise_instance = AmplitudeDamping(
                        noise.amplitudeDamping.probability)
                elif noise_type == 'bitFlip':
                    noise_instance = BitFlip(noise.bitFlip.probability)
                elif noise_type == 'bitPhaseFlip':
                    noise_instance = BitPhaseFlip(
                        noise.bitPhaseFlip.probability)
                elif noise_type == 'customizedNoise':
                    noise_instance = CustomizedNoise(
                        list(map(protobufMatrixToNumpyMatrix, noise.customizedNoise.krauses)))
                elif noise_type == 'depolarizing':
                    noise_instance = Depolarizing(noise.depolarizing.bits,
                                                  noise.depolarizing.probability)
                elif noise_type == 'pauliNoise':
                    noise_instance = PauliNoise(noise.pauliNoise.probability1, noise.pauliNoise.probability2,
                                                noise.pauliNoise.probability3)
                elif noise_type == 'phaseDamping':
                    noise_instance = PhaseDamping(
                        noise.phaseDamping.probability)
                elif noise_type == 'phaseFlip':
                    noise_instance = PhaseFlip(noise.phaseFlip.probability)
                elif noise_type == 'resetNoise':
                    noise_instance = ResetNoise(
                        noise.resetNoise.probability1, noise.resetNoise.probability2)
                else:
                    raise Error.ArgumentError(
                        f'Unsupported noise type {noise_type}!', ModuleErrorCode, FileErrorCode, 9)

                assert noise_instance.noiseClass == 'mixed_unitary_noise', 'Please use this method for mixed-unitarty noises'

                noiseRng.append(
                        noise_instance.calc_batched_noise_rng(num))

            return noiseRng

        for circuitLine in program.body.circuit:  # Traverse the circuit
            op = circuitLine.WhichOneof('op')

            if op == 'fixedGate':  # fixed gate
                fixedGate = circuitLine.fixedGate  # type: PBFixedGate
                matrix = operationDict.get(fixedGate)
                if matrix is None:
                    raise Error.ArgumentError(f'Unsupported operation {PBFixedGate.Name(fixedGate)}!', ModuleErrorCode,
                                              FileErrorCode, 10)

                rngList = transfer_noise_rng(max_run_num)
                if rngList:
                    batchedNoiseRng.append(rngList)
            elif op == 'rotationGate':  # rotation gate
                rotationGate = circuitLine.rotationGate  # type: PBRotationGate
                if rotationGate - PBRotationGate.U > 7:  # current support 8 rotation gates
                    raise Error.ArgumentError(
                    f'Unsupported operation {PBRotationGate.Name(rotationGate)}!', ModuleErrorCode,
                    FileErrorCode, 11)

                rngList = transfer_noise_rng(max_run_num)
                if rngList:
                    batchedNoiseRng.append(rngList)
            elif op == 'customizedGate':  # customized gate
                rngList = transfer_noise_rng(max_run_num)
                if rngList:
                    batchedNoiseRng.append(rngList)
            elif op == 'procedureName':  # procedure
                raise Error.ArgumentError('Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                                          ModuleErrorCode, FileErrorCode, 12)
                # it is not implemented, flattened by UnrollProcedureModule
            elif op == 'measure':  # measure
                pass
            elif op == 'barrier':  # barrier
                pass
                # unimplemented operation
            else:  # unsupported operation
                raise Error.ArgumentError(
                    f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 14)

        # Reshape list
        batchedNoiseRng = [out_rng for in_rng in batchedNoiseRng for out_rng in in_rng]

        return batchedNoiseRng

    def core_once_v1(self) -> Union[Dict[str, int], Dict[str, float], numpy.ndarray]:
        """
        Sample noise simulation process
        """
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

        # local code reuse
        def transfer_noise_list() -> None:
            nonlocal state
            state = transfer(state, matrix, qRegList)  # transfer gate

            for noise in circuitLine.noiseList:
                # create noise_instance
                noise_type = noise.WhichOneof('noise')
                if noise_type == 'amplitudeDamping':
                    noise_instance = AmplitudeDamping(
                        noise.amplitudeDamping.probability)
                elif noise_type == 'bitFlip':
                    noise_instance = BitFlip(noise.bitFlip.probability)
                elif noise_type == 'bitPhaseFlip':
                    noise_instance = BitPhaseFlip(
                        noise.bitPhaseFlip.probability)
                elif noise_type == 'customizedNoise':
                    noise_instance = CustomizedNoise(
                        list(map(protobufMatrixToNumpyMatrix, noise.customizedNoise.krauses)))
                elif noise_type == 'depolarizing':
                    noise_instance = Depolarizing(noise.depolarizing.bits,
                                                  noise.depolarizing.probability)
                elif noise_type == 'pauliNoise':
                    noise_instance = PauliNoise(noise.pauliNoise.probability1, noise.pauliNoise.probability2,
                                                noise.pauliNoise.probability3)
                elif noise_type == 'phaseDamping':
                    noise_instance = PhaseDamping(
                        noise.phaseDamping.probability)
                elif noise_type == 'phaseFlip':
                    noise_instance = PhaseFlip(noise.phaseFlip.probability)
                elif noise_type == 'resetNoise':
                    noise_instance = ResetNoise(
                        noise.resetNoise.probability1, noise.resetNoise.probability2)
                else:
                    raise Error.ArgumentError(
                        f'Unsupported noise type {noise_type}!', ModuleErrorCode, FileErrorCode, 9)

                # calc noise matrix
                if matrixType == MatrixType.Dense:
                    noise_matrix = noise_instance.calc_noise_matrix(
                        transfer, state, qRegList)
                else:
                    from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')

                # noise transfer
                if len(qRegList) == 1 and noise_matrix.shape != (2, 2):
                    raise Error.ArgumentError(
                        f'Single-qubit noise {noise_type} must be applied after single-qubit gate!',
                        ModuleErrorCode, FileErrorCode, 7)
                if len(qRegList) == 2 and noise_matrix.shape != (2, 2, 2, 2):
                    raise Error.ArgumentError(
                        f'Double-qubit noise {noise_type} must be applied after double-qubit gate!',
                        ModuleErrorCode, FileErrorCode, 8)
                state = transfer(state, noise_matrix, qRegList)

        measured = False
        for circuitLine in program.body.circuit:  # Traverse the circuit
            op = circuitLine.WhichOneof('op')

            qRegList = [qRegMap[_]
                        for _ in circuitLine.qRegList]  # type List[int]

            if op == 'fixedGate':  # fixed gate
                fixedGate = circuitLine.fixedGate  # type: PBFixedGate
                matrix = operationDict.get(fixedGate)
                if matrix is None:
                    raise Error.ArgumentError(f'Unsupported operation {PBFixedGate.Name(fixedGate)}!', ModuleErrorCode,
                                              FileErrorCode, 10)
                transfer_noise_list()
            elif op == 'rotationGate':  # rotation gate
                uGate = getRotationGate(circuitLine) # type RotationGateOp

                if matrixType == MatrixType.Dense:
                    matrix = uGate.getMatrix()
                else:
                    from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                transfer_noise_list()
            elif op == 'customizedGate':  # customized gate
                customizedGate = circuitLine.customizedGate  # type: PBCustomizedGate
                if matrixType == MatrixType.Dense:
                    matrix = protobufMatrixToNumpyMatrix(customizedGate.matrix)
                else:
                    from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                transfer_noise_list()
            elif op == 'procedureName':  # procedure
                raise Error.ArgumentError('Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                                          ModuleErrorCode, FileErrorCode, 12)
                # it is not implemented, flattened by UnrollProcedureModule
            elif op == 'measure':  # measure
                measure = circuitLine.measure  # type: PBMeasure
                if measure.type != PBMeasure.Type.Z:  # only Z measure is supported
                    raise Error.ArgumentError(
                        f'Unsupported operation measure {PBMeasure.Type.Name(measure.type)}!', ModuleErrorCode,
                        FileErrorCode, 13)
                if not measured:
                    counts = measurer(state, shots)
                    measured = True
            elif op == 'barrier':  # barrier
                pass
                # unimplemented operation
            else:  # unsupported operation
                raise Error.ArgumentError(
                    f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 14)

        if measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation]:
            counts = filterMeasure(counts, self.compactedCRegDict)
            return counts
        elif measureMethod == MeasureMethod.OutputProbability:
            return counts
        elif measureMethod == MeasureMethod.OutputState:
            if matrixType == MatrixType.Dense:
                state = normalizeNdarrayOrderForTranspose(state)
            else:
                state = normalizeNdarrayOrderForTranspose(state.todense())
            return state

    def core_once_v2(self, num: int) -> Union[Dict[str, int], Dict[str, float], Dict[str, numpy.ndarray]]:
        """
        Batched Samples noise simulation process
        """
        program = self.program
        matrixType = self.matrixType
        algorithm = self.algorithm
        measureMethod = self.measureMethod
        shots = self.shots
        seed = self.seed
        operationDict = self.operationDict

        if seed is None:
            seed = numpy.random.randint(0, 2147483647 + 1)
        numpy.random.seed(seed)

        qRegMap = {qReg: index for index,
                                   qReg in enumerate(program.head.usingQRegList)}
        qRegCount = len(qRegMap)

        state = initState_1_0(matrixType, qRegCount)
        transfer = TransferProcessor(matrixType, algorithm)
        measurer = Measurer(matrixType, algorithm, measureMethod)

        noiseType = Union['AmplitudeDamping', 'BitFlip', 'CustomizedNoise', 'BitPhaseFlip',
                          'Depolarizing', 'PauliNoise', 'PhaseDamping', 'PhaseFlip', 'ResetNoise']

        # Initialize a dict of state
        stateDict = {
            ''.join([str(index) for index in range(num)]): state for _ in range(num)}

        # local code use
        def transfer_batch_noisy_gate(gate_matrix: numpy.ndarray, run_num: int) -> None:
            nonlocal stateDict
            for key, value in stateDict.items():  # transfer gate
                stateDict[key] = transfer(value, gate_matrix, qRegList)

            for noise in circuitLine.noiseList:
                # create noise_instance
                noise_type = noise.WhichOneof('noise')
                if noise_type == 'amplitudeDamping':
                    noise_instance = AmplitudeDamping(
                        noise.amplitudeDamping.probability)
                elif noise_type == 'bitFlip':
                    noise_instance = BitFlip(noise.bitFlip.probability)
                elif noise_type == 'bitPhaseFlip':
                    noise_instance = BitPhaseFlip(
                        noise.bitPhaseFlip.probability)
                elif noise_type == 'customizedNoise':
                    noise_instance = CustomizedNoise(
                        list(map(protobufMatrixToNumpyMatrix, noise.customizedNoise.krauses)))
                elif noise_type == 'depolarizing':
                    noise_instance = Depolarizing(noise.depolarizing.bits,
                                                  noise.depolarizing.probability)
                elif noise_type == 'pauliNoise':
                    noise_instance = PauliNoise(noise.pauliNoise.probability1, noise.pauliNoise.probability2,
                                                noise.pauliNoise.probability3)
                elif noise_type == 'phaseDamping':
                    noise_instance = PhaseDamping(
                        noise.phaseDamping.probability)
                elif noise_type == 'phaseFlip':
                    noise_instance = PhaseFlip(noise.phaseFlip.probability)
                elif noise_type == 'resetNoise':
                    noise_instance = ResetNoise(
                        noise.resetNoise.probability1, noise.resetNoise.probability2)
                else:
                    raise Error.ArgumentError(
                        f'Unsupported noise type {noise_type}!', ModuleErrorCode, FileErrorCode, 9)

                # simulate batched samples of noise
                if noise_instance.noiseClass == 'mixed_unitary_noise':
                    rngList = noise_instance.calc_batched_noise_rng(run_num)
                    transfer_batch_noises(noise_instance, rngList)
                else:
                    rngList = noise_instance.calc_batched_noise_rng_non_mixed(
                        transfer, stateDict, qRegList)
                    extend_rng_list(rngList, run_num)
                    transfer_batch_noises(noise_instance, rngList)
                    norm_state_dict()

        # local code use
        def extend_rng_list(randomList: List[int], bound: int) -> None:
            """
            Extend a rngList up to a bound number
            """
            nonlocal stateDict
            index_current = [int(_)
                             for key in stateDict.keys() for _ in key]
            index_complement = [_ for _ in range(
                bound) if _ not in index_current]
            for index in index_complement:  # complement rngList
                randomList.insert(index + 1, 0)

        # local code use
        def transfer_batch_noises(noise_instance: noiseType, rngList: List[int]) -> None:
            """
            Transfer noise according to a batch of samples
            """
            nonlocal stateDict
            state_dict_temp = {}
            if len(set(rngList)) == 1:  # all samples give the same value
                for key_state_dict, value_state_dict in stateDict.items():
                    if matrixType == MatrixType.Dense:
                        noise_matrix = noise_instance.krauses[rngList[0]]
                    else:
                        noise_matrix = sparse.COO(
                            noise_instance.krauses[rngList[0]])
                    state_dict_temp[key_state_dict] = transfer(
                        value_state_dict, noise_matrix, qRegList)  # noise transfer
            elif len(set(rngList)) == num:  # each sample gives a different value
                for key_state_dict, value_state_dict in stateDict.items():
                    for string in key_state_dict:  # transfer noise for all samples
                        if matrixType == MatrixType.Dense:
                            noise_matrix = noise_instance.krauses[rngList[int(  # get noise matrix according to samples
                                string)]]
                        else:
                            noise_matrix = sparse.COO(
                                noise_instance.krauses[rngList[int(string)]])
                        state_dict_temp[string] = transfer(
                            value_state_dict, noise_matrix, qRegList)  # noise transfer
            else:
                for key_state_dict, value_state_dict in stateDict.items():
                    # get samples for each key in the stateDict
                    rngSlice = [rngList[int(string)]
                                for string in key_state_dict]

                    if len(set(rngSlice)) == 1:  # all samples in the key give the same value
                        if matrixType == MatrixType.Dense:
                            noise_matrix = noise_instance.krauses[rngList[int(
                                key_state_dict[0])]]
                        else:
                            noise_matrix = sparse.COO(
                                noise_instance.krauses[rngList[int(key_state_dict[0])]])
                        state_dict_temp[key_state_dict] = transfer(
                            value_state_dict, noise_matrix, qRegList)  # noise transfer
                    # each sample in the key gives a different value
                    elif len(set(rngSlice)) == len(rngSlice):
                        for string in key_state_dict:
                            if matrixType == MatrixType.Dense:
                                noise_matrix = noise_instance.krauses[rngList[int(
                                    string)]]
                            else:
                                noise_matrix = sparse.COO(
                                    noise_instance.krauses[rngList[int(string)]])
                            state_dict_temp[string] = transfer(
                                value_state_dict, noise_matrix, qRegList)  # noise transfer
                    else:
                        # split branch of stateDict according to noise samples
                        strDict = split_branch(str(key_state_dict), rngList)

                        for key_str_dict in strDict.keys():
                            if matrixType == MatrixType.Dense:
                                noise_matrix = noise_instance.krauses[rngList[int(
                                    key_str_dict[0])]]
                            else:
                                noise_matrix = sparse.COO(
                                    noise_instance.krauses[rngList[int(key_str_dict[0])]])
                            state_dict_temp[key_str_dict] = transfer(
                                value_state_dict, noise_matrix, qRegList)  # noise transfer

            stateDict = state_dict_temp  # update the state dict

        # local code use
        def split_branch(strList: str, rngList: List[int]) -> Dict[str, int]:
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
        def norm_state_dict() -> None:
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
        def batch_measure(stateDic: Optional[Dict[str, numpy.ndarray]], measurerCore: 'Measurer',
                          countsShots: int) -> Optional[Union[Dict[str, int], Dict[str, float]]]:
            """
            Measure a batch of states
            """
            nonlocal stateDict

            if measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation, 
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
            elif measureMethod in[MeasureMethod.OutputState, MeasureMethod.OutputProbability]:
                pass

        measured = False
        for circuitLine in program.body.circuit:  # Traverse the circuit
            op = circuitLine.WhichOneof('op')

            qRegList = [qRegMap[_]
                        for _ in circuitLine.qRegList]  # type List[int]

            if not stateDict:  # Verify the output stateDict is not None
                return {}
            else:
                if op == 'fixedGate':  # fixed gate
                    fixedGate = circuitLine.fixedGate  # type: PBFixedGate
                    matrix = operationDict.get(fixedGate)
                    if matrix is None:
                        raise Error.ArgumentError(f'Unsupported operation {PBFixedGate.Name(fixedGate)}!',
                                                  ModuleErrorCode, FileErrorCode, 10)
                    transfer_batch_noisy_gate(matrix, num)
                elif op == 'rotationGate':  # rotation gate
                    uGate = getRotationGate(circuitLine) # type RotationGateOp

                    if matrixType == MatrixType.Dense:
                        matrix = uGate.getMatrix()
                    else:
                        from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                    transfer_batch_noisy_gate(matrix, num)
                elif op == 'customizedGate':  # customized gate
                    customizedGate = circuitLine.customizedGate  # type: PBCustomizedGate
                    if matrixType == MatrixType.Dense:
                        matrix = protobufMatrixToNumpyMatrix(
                            customizedGate.matrix)
                    else:
                        from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
                    transfer_batch_noisy_gate(matrix, num)
                elif op == 'proceduresName':  # procedure
                    raise Error.ArgumentError('Unsupported operation procedure, please flatten by '
                                              'UnrollProcedureModule!', ModuleErrorCode, FileErrorCode, 12)
                    # it is not implemented, flattened by UnrollProcedureModule
                elif op == 'measure':  # measure
                    measure = circuitLine.measure  # type: PBMeasure
                    if measure.type != PBMeasure.Type.Z:  # only Z measure is supported
                        raise Error.ArgumentError(
                            f'Unsupported operation measure {PBMeasure.Type.Name(measure.type)}!', ModuleErrorCode,
                            FileErrorCode, 13)
                    if not measured:
                        # measure a batch of states
                        counts = batch_measure(stateDict, measurer, shots)
                        measured = True
                elif op == 'barrier':  # barrier
                    pass
                    # unimplemented operation
                else:  # unsupported operation
                    raise Error.ArgumentError(
                        f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 14)

        if measureMethod in [MeasureMethod.Probability, MeasureMethod.Accumulation]:
            counts = filterMeasure(counts, self.compactedCRegDict)
            return counts
        elif measureMethod == MeasureMethod.OutputProbability:
            return counts
        elif measureMethod == MeasureMethod.OutputState:
            if matrixType == MatrixType.Dense:
                stateDict = {key: normalizeNdarrayOrderForTranspose(
                    value) for key, value in stateDict.items()}
            else:
                stateDict = {key: normalizeNdarrayOrderForTranspose(
                    value.todense()) for key, value in stateDict.items()}
            return stateDict

    @staticmethod
    def loadGates(matrixType: 'MatrixType') -> Dict['PBFixedGate', Union[numpy.ndarray, 'COO']]:
        """
        Load the matrix of the gate
        """
        if matrixType == MatrixType.Dense:
            operationDict = {
                PBFixedGate.ID: ID.getMatrix(),
                PBFixedGate.X: X.getMatrix(),
                PBFixedGate.Y: Y.getMatrix(),
                PBFixedGate.Z: Z.getMatrix(),
                PBFixedGate.H: H.getMatrix(),
                PBFixedGate.S: S.getMatrix(),
                PBFixedGate.SDG: SDG.getMatrix(),
                PBFixedGate.T: T.getMatrix(),
                PBFixedGate.TDG: TDG.getMatrix(),
                PBFixedGate.CX: CX.getMatrix().reshape(2, 2, 2, 2),
                PBFixedGate.CY: CY.getMatrix().reshape(2, 2, 2, 2),
                PBFixedGate.CZ: CZ.getMatrix().reshape(2, 2, 2, 2),
                PBFixedGate.CH: CH.getMatrix().reshape(2, 2, 2, 2),
                PBFixedGate.SWAP: SWAP.getMatrix(),
                PBFixedGate.CCX: CCX.getMatrix(),
                PBFixedGate.CSWAP: CSWAP.getMatrix()
            }  # type: Dict['PBFixedGate', Union[numpy.ndarray, 'COO']]
        else:
            from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
        return operationDict


def getRotationGate(circuitLine: 'PBCircuitLine') -> 'RotationGateOP':
    """
    Get the rotation gate instance form PBCircuitLine
    """
    rotationGate = circuitLine.rotationGate  # type: PBRotationGate
    if rotationGate - PBRotationGate.U == 0:  
        ugate = U(*circuitLine.argumentValueList)
    elif rotationGate - PBRotationGate.U == 1:  
        ugate = RX(*circuitLine.argumentValueList)
    elif rotationGate - PBRotationGate.U == 2: 
        ugate = RY(*circuitLine.argumentValueList)
    elif rotationGate - PBRotationGate.U == 3:  
        ugate = RZ(*circuitLine.argumentValueList)
    elif rotationGate - PBRotationGate.U == 4: 
        ugate = CU(*circuitLine.argumentValueList)
    elif rotationGate - PBRotationGate.U == 5:
        ugate = CRX(*circuitLine.argumentValueList)
    elif rotationGate - PBRotationGate.U == 6:
        ugate = CRY(*circuitLine.argumentValueList)
    elif rotationGate - PBRotationGate.U == 7:
        ugate = CRZ(*circuitLine.argumentValueList)
    else:
        raise Error.ArgumentError(
            f'Unsupported operation {PBRotationGate.Name(rotationGate)}!', ModuleErrorCode,
            FileErrorCode, 11)

    return ugate


def safe_round_counts(counts: Union[Dict[str, int], Dict[str, float]], shots: int) -> Dict[str, int]:
    # round
    counts_round = dict(
        map(lambda kv: (kv[0], round(kv[1])), counts.items()))

    # fix counts after round
    fix_value = shots - sum(counts_round.values())
    if fix_value > 0:
        for _ in range(fix_value):
            # calc current shift
            shift = {}
            for key in counts:
                shift[key] = counts_round[key] - counts[key]
            # add 1 for current min shift
            min_shift_key = min(shift, key=shift.get)
            counts_round[min_shift_key] += 1
    elif fix_value < 0:
        for _ in range(-fix_value):
            # calc current shift
            shift = {}
            for key in counts:
                shift[key] = counts_round[key] - counts[key]
            # dec 1 for current max shift
            max_shift_key = max(shift, key=shift.get)
            counts_round[max_shift_key] -= 1

    # remove zeros values
    counts_round = dict(
        filter(lambda kv: (round(kv[1]) != 0), counts_round.items()))

    # sum check
    assert sum(counts_round.values()) == shots

    return counts_round


"""
print(safe_round_counts(
    {'0': 0.8, '1': 0.8, '2': 0.8, '3': 0.8, '4': 0.8, '5': 0.8, '6': 0.8, '7': 0.8, '8': 0.8, '9': 0.8}, 8))
print(safe_round_counts(
    {'0': 0.2, '1': 0.2, '2': 0.2, '3': 0.2, '4': 0.2, '5': 0.2, '6': 0.2, '7': 0.2, '8': 0.2, '9': 0.2}, 2))
exit(0)
"""


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

    if shots < 1 or shots > Define.maxShots:
        raise Error.ArgumentError(f'Invalid shots {shots}, should in [0, {Define.maxShots}]', ModuleErrorCode,
                                  FileErrorCode, 1)
    if seed is not None:
        if seed < 0 or seed > Define.maxSeed:
            raise Error.ArgumentError(f'Invalid seed {seed}, should in [0, {Define.maxSeed}]', ModuleErrorCode,
                                      FileErrorCode, 2)

    matrixTypeValue = None  # type: Optional[MatrixType]
    if matrixType == 'dense':
        matrixTypeValue = MatrixType.Dense
    
    else:
        raise Error.ArgumentError(
            f'Invalid MatrixType {matrixTypeValue}', ModuleErrorCode, FileErrorCode, 2)

    algorithmValue = None  # type: Optional[Algorithm]
    if algorithm == 'matmul':
        algorithmValue = Algorithm.Matmul
    elif algorithm == 'einsum':
        algorithmValue = Algorithm.Einsum
    else:
        raise Error.ArgumentError(
            f'Invalid Algorithm {algorithmValue}', ModuleErrorCode, FileErrorCode, 4)

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
        raise Error.ArgumentError(
            f'Invalid MeasureMethod {measureMethodValue}', ModuleErrorCode, FileErrorCode, 5)

    if inputFile is not None:
        jsonStr = Path(inputFile).read_text()
        program = JsonToCircuit().convert(jsonStr)

    SV = StateVector(program, matrixTypeValue, algorithmValue,
                     measureMethodValue, shots, seed)
    return SV.core()


if __name__ == '__main__':
    result = runSimulator(None, None)
    countsFilePath = Define.outputDirPath / 'counts.json'
    countsFilePath.write_text(result.toJson(True), encoding='utf-8')
