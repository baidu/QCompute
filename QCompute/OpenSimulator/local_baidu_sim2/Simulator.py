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
from pathlib import Path
from typing import List, Optional

from QCompute import Define
from QCompute.Define import Settings
from QCompute.Define.Settings import NoiseMethod
from QCompute.OpenConvertor.JsonToCircuit import JsonToCircuit
from QCompute.OpenSimulator import QResult, ModuleErrorCode, withNoise
from QCompute.OpenSimulator.local_baidu_sim2.GeneralNoiseSimulator import GeneralNoiseSimulator
from QCompute.OpenSimulator.local_baidu_sim2.InitState import MatrixType
from QCompute.OpenSimulator.local_baidu_sim2.LowNoiseCircuitSimulator import LowNoiseCircuitSimulator
from QCompute.OpenSimulator.local_baidu_sim2.Measure import MeasureMethod
from QCompute.OpenSimulator.local_baidu_sim2.MixedUnitaryNoiseSimulator import MixedUnitaryNoiseSimulator
from QCompute.OpenSimulator.local_baidu_sim2.NoNoiseSimulator import NoNoiseSimulator
from QCompute.OpenSimulator.local_baidu_sim2.Transfer import Algorithm
from QCompute.QPlatform import Error
from QCompute.QProtobuf import PBProgram


FileErrorCode = 5


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
    matrixType: str = args.mt.lower()
    algorithm: str = args.a.lower()
    measureMethod: str = args.mm.lower()
    seed: int = args.s
    shots: int = args.shots
    inputFile: str = args.inputFile

    if shots < 1 or shots > Define.maxShots:
        raise Error.ArgumentError(f'Invalid shots {shots}, should in [0, {Define.maxShots}]', ModuleErrorCode,
                                  FileErrorCode, 1)
    if seed is not None:
        if seed < 0 or seed > Define.maxSeed:
            raise Error.ArgumentError(f'Invalid seed {seed}, should in [0, {Define.maxSeed}]', ModuleErrorCode,
                                      FileErrorCode, 2)

    matrixTypeValue: Optional[MatrixType] = None
    if matrixType == 'dense':
        matrixTypeValue = MatrixType.Dense
    
    else:
        raise Error.ArgumentError(
            f'Invalid MatrixType {matrixTypeValue}', ModuleErrorCode, FileErrorCode, 2)

    algorithmValue: Optional[Algorithm] = None
    if algorithm == 'matmul':
        algorithmValue = Algorithm.Matmul
    elif algorithm == 'einsum':
        algorithmValue = Algorithm.Einsum
    else:
        raise Error.ArgumentError(
            f'Invalid Algorithm {algorithmValue}', ModuleErrorCode, FileErrorCode, 4)

    measureMethodValue: Optional[MeasureMethod] = None
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

    if withNoise(program):
        if Settings.noiseMethod.value == NoiseMethod.MixedUnitaryNoise.value:
            SV = MixedUnitaryNoiseSimulator(program, matrixTypeValue, algorithmValue, measureMethodValue, shots, seed)
        elif Settings.noiseMethod.value == NoiseMethod.GeneralNoise.value:
            SV = GeneralNoiseSimulator(program, matrixTypeValue, algorithmValue, measureMethodValue, shots, seed)
        elif Settings.noiseMethod.value == NoiseMethod.LowNoiseCircuit.value:
            SV = LowNoiseCircuitSimulator(program, matrixTypeValue, algorithmValue, measureMethodValue, shots, seed)
        else:
            assert False
    else:
        SV = NoNoiseSimulator(program, matrixTypeValue, algorithmValue, measureMethodValue, shots, seed)
    return SV.core()


if __name__ == '__main__':
    result = runSimulator(None, None)
    countsFilePath = Define.outputDirPath / 'counts.json'
    countsFilePath.write_text(result.toJson(True), encoding='utf-8')
