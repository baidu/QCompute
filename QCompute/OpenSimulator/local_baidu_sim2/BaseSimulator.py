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
BaseSimulator
"""
FileErrorCode = 2

from typing import Union, Dict

import numpy

from QCompute.OpenSimulator import QResult, ModuleErrorCode
from QCompute.OpenSimulator.local_baidu_sim2.InitState import MatrixType
from QCompute.OpenSimulator.local_baidu_sim2.Measure import MeasureMethod
from QCompute.OpenSimulator.local_baidu_sim2.Transfer import Algorithm
from QCompute.QPlatform import Error
from QCompute.QPlatform.QOperation.FixedGate import ID, X, Y, Z, H, S, SDG, T, TDG, CX, CY, CZ, CH, SWAP, CCX, CSWAP
from QCompute.QPlatform.QOperation.RotationGate import RotationGateOP, U, RX, RY, RZ, CU, CRX, CRY, CRZ
from QCompute.QPlatform.Utilities import protobufMatrixToNumpyMatrix
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBFixedGate, PBRotationGate, PBCustomizedGate




class BaseSimulator:
    def __init__(self, program: 'PBProgram', matrixType: 'MatrixType', algorithm: 'Algorithm',
                 measureMethod: 'MeasureMethod', shots: int, seed: int) -> None:
        """
        Simulation process
            The combinations are not 2**3=8 cases. We only implement several combination checks:

        case 1), for ideal circuit:

            1)DENSE-EINSUM-SINGLE

            2)DENSE-EINSUM-PROB

            3)DENSE-MATMUL-SINGLE

            4)DENSE-MATMUL-PROB

            

        case 2), for noisy circuit:

            1)DENSE-MATMUL-PROB

            2)DENSE-MATMUL-STATE

            3)DENSE-MATMUL-SINGLE

            4)DENSE-MATMUL-OUTPROB

            
        """
        self.program = program
        self.matrixType = matrixType
        self.algorithm = algorithm
        self.measureMethod = measureMethod
        self.shots = shots
        self.seed = seed
        self.operationDict = self.__loadGates(matrixType)
        self.compactedCRegDict = None

        # adaptive maxRunNum
        maxRunNumEpsilon = 0.05
        maxRunNumDelta = 0.01
        self.maxRunNum = round(numpy.log(2 / maxRunNumDelta) / (2 * maxRunNumEpsilon ** 2))

    def core(self) -> 'QResult':
        assert False

    @staticmethod
    def __loadGates(matrixType: 'MatrixType') -> Dict['PBFixedGate', Union[numpy.ndarray, 'COO']]:
        """
        Load the matrix of the gate
        """
        if matrixType == MatrixType.Dense:
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
        else:
            from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
        return operationDict

    @staticmethod
    def getGateMatrix(circuitLine: 'PBCircuitLine', matrixType: 'MatrixType',
                      operationDict: Dict['PBFixedGate', Union[numpy.ndarray, 'COO']]) -> numpy.ndarray:
        """
        Get the matrix formation from circuitLine
        """
        op = circuitLine.WhichOneof('op')
        if op == 'fixedGate':  # fixed gate
            fixedGate: PBFixedGate = circuitLine.fixedGate
            # matrix = operationDict.get(fixedGate)
            matrix = BaseSimulator.__loadGates(matrixType).get(fixedGate)
            if matrix is None:
                raise Error.ArgumentError(
                    f'Unsupported operation {PBFixedGate.Name(fixedGate)}!', ModuleErrorCode, FileErrorCode, 1)

        elif op == 'rotationGate':  # rotation gate
            uGate: PBRotationGate = BaseSimulator.__getRotationGate(circuitLine)

            if matrixType == MatrixType.Dense:
                matrix = uGate.getMatrix()
            else:
                from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
        elif op == 'customizedGate':  # customized gate
            customizedGate: PBCustomizedGate = circuitLine.customizedGate
            if matrixType == MatrixType.Dense:
                matrix = protobufMatrixToNumpyMatrix(customizedGate.matrix)
            else:
                from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')
        return matrix

    @staticmethod
    def __getRotationGate(circuitLine: 'PBCircuitLine') -> 'RotationGateOP':
        """
        Get the rotation gate instance form PBCircuitLine
        """
        rotationGate: PBRotationGate = circuitLine.rotationGate
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
                f'Unsupported operation {PBRotationGate.Name(rotationGate)}!',
                ModuleErrorCode, FileErrorCode, 2)

        return ugate

    @staticmethod
    def safeRoundCounts(counts: Union[Dict[str, int], Dict[str, float]], shots: int) -> Dict[str, int]:
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
            filter(lambda kv: (round(kv[1]
                                     ) != 0), counts_round.items()))

        # sum check
        assert sum(counts_round.values()) == shots

        return counts_round