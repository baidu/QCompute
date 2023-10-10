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
import random
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
from QCompute.QProtobuf import PBFixedGate, PBRotationGate, PBCustomizedGate, PBMeasure

FileErrorCode = 28

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBProgram


def runSimulator(args: List[str], program: 'PBProgram') -> 'QResult':
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
        raise Error.ArgumentError(
            f'Invalid shots {shots}, should in [0, {Define.maxShots}]', ModuleErrorCode, FileErrorCode, 1)

    if seed is not None:
        if seed < 0 or seed > Define.maxSeed:
            raise Error.ArgumentError(f'Invalid random seed {seed}, should in [0, {Define.maxSeed}]',
            ModuleErrorCode, FileErrorCode, 2)

    if inputFile is not None:
        jsonStr = Path(inputFile).read_text()
        program = JsonToCircuit().convert(jsonStr)

    return core(program, shots, seed)


def core(program: 'PBProgram', shots: int, seed: int) -> 'QResult':
    """
    Simulation process
    """

    # lazy import. These need to run on linux OS with GPU
    import cudaq
    # cudaq.initialize_cudaq()
    assert cudaq.num_available_gpus() >= 1
    cudaq.set_target('nvidia')

    usedQRegSet, usedCRegSet, compactedQRegDict, compactedCRegDict = preProcess(program, True, True)

    qRegMap = {qReg: index for index, qReg in enumerate(program.head.usingQRegList)}
    qRegCount = len(qRegMap)

    if seed is None:
        seed = int(random.randint(0, 2147483647 + 1))
    random.seed(seed)
    numpy.random.seed(seed)
    # cudaq.set_random_seed(seed)  # new API only valid in docker? not in pip wheel?

    # init kernel + qubits = state
    kernel = cudaq.make_kernel()
    qubits = kernel.qalloc(qRegCount)

    result = QResult()
    result.startTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'
    measured = False
    for circuitLine in program.body.circuit:  # Traverse the circuit
        op = circuitLine.WhichOneof('op')
        qRegList: List[int] = [qRegMap[qReg] for qReg in circuitLine.qRegList]
        qRegList = list(map(lambda x: qubits[x], qRegList))
        if op == 'fixedGate':  # fixed gate
            fixedGate: PBFixedGate = circuitLine.fixedGate
            gateName = PBFixedGate.Name(fixedGate)
            if fixedGate == PBFixedGate.X:
                kernel.x(*qRegList)
            elif fixedGate == PBFixedGate.CX:
                kernel.cx(*qRegList)
            elif fixedGate == PBFixedGate.Y:
                kernel.y(*qRegList)
            elif fixedGate == PBFixedGate.CY:
                kernel.cy(*qRegList)
            elif fixedGate == PBFixedGate.Z:
                kernel.z(*qRegList)
            elif fixedGate == PBFixedGate.CZ:
                kernel.cz(*qRegList)
            elif fixedGate == PBFixedGate.H:
                kernel.h(*qRegList)
            elif fixedGate == PBFixedGate.CH:
                kernel.ch(*qRegList)
            elif fixedGate == PBFixedGate.S:
                kernel.s(*qRegList)
            elif fixedGate == PBFixedGate.SDG:
                kernel.sdg(*qRegList)
            elif fixedGate == PBFixedGate.T:
                kernel.t(*qRegList)
            elif fixedGate == PBFixedGate.TDG:
                kernel.tdg(*qRegList)
            elif fixedGate == PBFixedGate.SWAP:
                kernel.swap(*qRegList)
            else:
                raise Error.ArgumentError(
                    f'Invalid gate: ({gateName}) for cuquantum simulator!', ModuleErrorCode, FileErrorCode, 3)

        elif op == 'rotationGate':  # rotation gate
            rotationGate: PBRotationGate = circuitLine.rotationGate
            gateName = PBRotationGate.Name(rotationGate)
            angles = circuitLine.argumentValueList
            if rotationGate != PBRotationGate.RX:
                assert len(angles) == 1
                assert len(qRegList) == 1
                kernel.rx(angles[0], qRegList[0])
            elif rotationGate != PBRotationGate.RY:
                assert len(angles) == 1
                assert len(qRegList) == 1
                kernel.ry(angles[0], qRegList[0])
            elif rotationGate != PBRotationGate.RZ:
                assert len(angles) == 1
                assert len(qRegList) == 1
                kernel.rz(angles[0], qRegList[0])
            else:
                raise Error.ArgumentError(
                    f'Invalid gate: ({gateName}) for cuquantum simulator!', ModuleErrorCode, FileErrorCode, 4)

        elif op == 'customizedGate':  # customized gate
            raise Error.ArgumentError(
                f'Invalid gate: (customizedGate) for cuquantum simulator!', ModuleErrorCode, FileErrorCode, 5)

        elif op == 'procedureName':  # procedure
            raise Error.ArgumentError(
                'Unsupported operation procedure, please flatten by UnrollProcedureModule!',
                ModuleErrorCode, FileErrorCode, 6)

            # it is not implemented, flattened by UnrollProcedureModule
        elif op == 'measure':  # measure
            measure: PBMeasure = circuitLine.measure
            if measure.type != PBMeasure.Type.Z:  # only Z measure is supported
                raise Error.ArgumentError(
                    f'Unsupported operation measure {PBMeasure.Type.Name(measure.type)}!',
                    ModuleErrorCode, FileErrorCode, 7)

            if not measured:
                # measure
                # collect the result to simulator for the subsequent invoking
                kernel.mz(qubits)
                result.counts = cudaq.sample(kernel, shots_count=shots)
                result.counts = dict(result.counts.items())
                measured = True
        elif op == 'barrier':  # barrier
            pass
            # unimplemented operation
        else:  # unsupported operation
            raise Error.ArgumentError(
                f'Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 8)

    result.endTimeUtc = datetime.utcnow().isoformat()[:-3] + 'Z'
    result.shots = shots
    result.counts = filterMeasure(result.counts, compactedCRegDict)
    result.ancilla.usedQRegList = list(usedQRegSet)
    result.ancilla.usedCRegList = list(usedCRegSet)
    result.ancilla.compactedQRegDict = compactedQRegDict
    result.ancilla.compactedCRegDict = compactedCRegDict
    result.seed = int(seed)

    return result


if __name__ == '__main__':
    result = runSimulator(None, None)
    countsFilePath = Define.outputDirPath / 'counts.json'
    countsFilePath.write_text(result.toJson(True), encoding='utf-8')
