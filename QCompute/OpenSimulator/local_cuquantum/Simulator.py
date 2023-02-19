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
from datetime import datetime
from pathlib import Path
from typing import List, TYPE_CHECKING, Union, Dict, Optional

import numpy

from QCompute import Define
from QCompute.OpenConvertor.JsonToCircuit import JsonToCircuit
from QCompute.OpenSimulator import QResult, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QPlatform.Processor.PostProcessor import filterMeasure
from QCompute.QPlatform.Processor.PreProcess import preProcess
from QCompute.QPlatform.QOperation.FixedGate import ID, X, Y, Z, H, S, SDG, T, TDG, CX, CY, CZ, CH, SWAP, CCX, CSWAP
from QCompute.QPlatform.QOperation.RotationGate import U
from QCompute.QPlatform.Utilities import protobufMatrixToNumpyMatrix
from QCompute.QProtobuf import PBFixedGate, PBRotationGate, PBCustomizedGate, PBMeasure

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBProgram

FileErrorCode = 2


def runSimulator(args: Optional[List[str]], program: Optional['PBProgram']) -> 'QResult':
    """
    Initialization process
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('-s', default=None, type=int)
    parser.add_argument('-shots', default=None, type=int)
    parser.add_argument('-inputFile', default=None, type=str)

    args = parser.parse_args(args=args)
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

    if inputFile is not None:
        jsonStr = Path(inputFile).read_text()
        program = JsonToCircuit().convert(jsonStr)

    return core(program, shots, seed)


def core(program: 'PBProgram', shots: int, seed: int) -> 'QResult':
    """
    Simulation process
    """

    # lazy import. These need to run on linux OS with GPU
    import cupy
    try:
        from . import Kernel
    except ImportError:
        import Kernel

    usedQRegSet, usedCRegSet, compactedQRegDict, compactedCRegDict = preProcess(program, True, True)

    qRegMap = {qReg: index for index, qReg in enumerate(program.head.usingQRegList)}
    qRegCount = len(qRegMap)

    operationDict = loadGates()

    if seed is None:
        seed = int(cupy.random.randint(0, 2147483647 + 1))
    cupy.random.seed(seed)

    # init state
    state = Kernel.init_state_10(qRegCount)

    # make buffer
    expr_buffer = []
    gate_buffer = []

    result = QResult()
    result.startTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'
    measured = False
    for circuitLine in program.body.circuit:  # Traverse the circuit
        op = circuitLine.WhichOneof('op')

        qRegList: List[int] = [qRegMap[qReg] for qReg in circuitLine.qRegList]

        if op == 'fixedGate':  # fixed gate
            fixedGate: PBFixedGate = circuitLine.fixedGate
            matrix = operationDict.get(fixedGate)
            matrix = Kernel.NP_DATA_TYPE(matrix)
            if matrix is None:
                raise Error.ArgumentError(f'Unsupported operation {PBFixedGate.Name(fixedGate)}!', ModuleErrorCode,
                                          FileErrorCode, 3)
            Kernel.transfer_state_in_buffer(qRegCount, cupy.array(matrix), qRegList,
                                            expr_buffer, gate_buffer)
        elif op == 'rotationGate':  # rotation gate
            rotationGate: PBRotationGate = circuitLine.rotationGate
            if rotationGate != PBRotationGate.U:
                raise Error.ArgumentError(
                    f'Unsupported operation {PBRotationGate.Name(rotationGate)}!', ModuleErrorCode, FileErrorCode, 4)
            uGate = U(*circuitLine.argumentValueList)
            matrix = uGate.getMatrix()
            matrix = Kernel.NP_DATA_TYPE(matrix)
            if matrix.shape == (4, 4):
                matrix = matrix.reshape((2, 2, 2, 2))
            Kernel.transfer_state_in_buffer(qRegCount, cupy.array(matrix), qRegList,
                                            expr_buffer, gate_buffer)
        elif op == 'customizedGate':  # customized gate
            customizedGate: PBCustomizedGate = circuitLine.customizedGate
            matrix = protobufMatrixToNumpyMatrix(customizedGate.matrix)
            matrix = Kernel.NP_DATA_TYPE(matrix)
            if matrix.shape == (4, 4):
                matrix = matrix.reshape((2, 2, 2, 2))
            Kernel.transfer_state_in_buffer(qRegCount, cupy.array(matrix), qRegList,
                                            expr_buffer, gate_buffer)
        elif op == 'procedureName':  # procedure
            raise Error.ArgumentError('Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                                      ModuleErrorCode, FileErrorCode, 5)
            # it is not implemented, flattened by UnrollProcedureModule
        elif op == 'measure':  # measure
            measure: PBMeasure = circuitLine.measure
            if measure.type != PBMeasure.Type.Z:  # only Z measure is supported
                raise Error.ArgumentError(
                    f'Unsupported operation measure {PBMeasure.Type.Name(measure.type)}!', ModuleErrorCode,
                    FileErrorCode, 6)
            if not measured:
                # flush buffer
                state = Kernel.transfer_state_flush(state, expr_buffer, gate_buffer)

                # measure
                # collect the result to simulator for the subsequent invoking
                result.counts = {}
                for i in range(shots):
                    outs = Kernel.measure_all_2(qRegCount, state)
                    if outs not in result.counts:
                        result.counts[outs] = 0
                    result.counts[outs] += 1
                measured = True
        elif op == 'barrier':  # barrier
            pass
            # unimplemented operation
        else:  # unsupported operation
            raise Error.ArgumentError(f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 7)

    result.endTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'
    result.shots = shots
    result.counts = filterMeasure(result.counts, compactedCRegDict)
    result.ancilla.usedQRegList = list(usedQRegSet)
    result.ancilla.usedCRegList = list(usedCRegSet)
    result.ancilla.compactedQRegDict = compactedQRegDict
    result.ancilla.compactedCRegDict = compactedCRegDict
    result.seed = int(seed)

    return result


def loadGates() -> Dict['PBFixedGate', numpy.ndarray]:
    """
    Load the matrix of the gate
    """
    # lazy import. These need to run on linux OS with GPU
    try:
        from . import Kernel
    except ImportError:
        import Kernel

    operationDict: Dict['PBFixedGate', Union[numpy.ndarray, 'COO']] = {
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
    }
    for k, v in operationDict.items():
        operationDict[k] = Kernel.NP_DATA_TYPE(v)
    return operationDict


if __name__ == '__main__':
    result = runSimulator(None, None)
    countsFilePath = Define.outputDirPath / 'counts.json'
    countsFilePath.write_text(result.toJson(True), encoding='utf-8')
