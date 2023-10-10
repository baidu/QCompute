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
Convert the circuit to draw console
"""
FileErrorCode = 2

from hashlib import blake2b
from io import StringIO
from typing import List

from QCompute.Define import Settings
from QCompute.OpenConvertor import ConvertorImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QPlatform.Utilities import protobufMatrixToNumpyMatrix
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBFixedGate, PBRotationGate, PBCompositeGate, PBMeasure, \
    PBPhotonicGaussianGate, PBPhotonicGaussianMeasure, PBPhotonicFockGate, PBPhotonicFockMeasure

gapNode = '-'
gap = gapNode * 2
gapLen = len(gap)
numPos = gapLen + 1


class CircuitToDrawConsole(ConvertorImplement):
    """
    Circuit to draw console
    """

    def __init__(self):
        self.stringIO: StringIO = None

    def convert(self, pbProgram: PBProgram) -> str:
        """
        Convert the circuit to drawing in terminal.

        Example:

        env.publish()  # need to generate protobuf circuit data

        asciiPic = CircuitToDrawConsole().convert(env.program)

        :param pbProgram: Protobuf format of the program
        :return: string
        """
        self.stringIO = StringIO()

        for name, procedure in pbProgram.body.procedureMap.items():
            self.stringIO.write(f'Procedure {name}\n')
            self._drawCircuit(name, procedure.circuit)
        self.stringIO.write(f'main\n')
        self._drawCircuit('main', pbProgram.body.circuit)
        return self.stringIO.getvalue()

    def _drawCircuit(self, name: str, pbCircuitLineList: List[PBCircuitLine]) -> str:
        qRegSet = set()
        for pbCircuitLine in pbCircuitLineList:
            qRegSet.update(pbCircuitLine.qRegList)
        qRegCount = max(qRegSet) + 1

        if qRegCount <= 0:
            return 'Enpty circuit.'

        circuitArray: List[List[str]] = []

        maxQRegNameLen = len(str(qRegCount - 1)) + 1
        for i in range(qRegCount):
            qRegName = f'Q{i}'
            if len(qRegName) < maxQRegNameLen:
                qRegName += ' ' * (maxQRegNameLen - len(qRegName))
            circuitArray.append([f'Q{i}'])
            if i < qRegCount - 1:
                circuitArray.append([' ' * maxQRegNameLen])

        for pbCircuitLine in pbCircuitLineList:
            op = pbCircuitLine.WhichOneof('op')
            if op == 'fixedGate':
                fixedGate: PBFixedGate = pbCircuitLine.fixedGate
                gateName = PBFixedGate.Name(fixedGate)
                gateName = self._appendNoise(gateName, pbCircuitLine)
                self._draw(qRegCount, f'|{gateName}|', pbCircuitLine.qRegList, circuitArray)
            elif op == 'rotationGate':
                rotationGate: PBRotationGate = pbCircuitLine.rotationGate
                gateName = PBRotationGate.Name(rotationGate)
                gateName = self._appendNoise(gateName, pbCircuitLine)
                argumentList = []
                if pbCircuitLine.argumentIdList:
                    for index, argumentId in enumerate(pbCircuitLine.argumentIdList):
                        if argumentId == -1:
                            argumentList.append(f'{pbCircuitLine.argumentValueList[index]}')
                        else:
                            argumentList.append(f'P{argumentId}')
                else:
                    for argument in pbCircuitLine.argumentValueList:
                        argumentList.append(f'{argument}')
                self._draw(qRegCount, f'|{gateName}|', pbCircuitLine.qRegList, circuitArray)
            elif op == 'compositeGate':
                compositeGate: PBCompositeGate = pbCircuitLine.compositeGate
                gateName = PBCompositeGate.Name(compositeGate)
                gateName = self._appendNoise(gateName, pbCircuitLine)
                argumentList = []
                if pbCircuitLine.argumentIdList:
                    for index, argumentId in enumerate(pbCircuitLine.argumentIdList):
                        if argumentId == -1:
                            argumentList.append(f'{pbCircuitLine.argumentValueList[index]}')
                        else:
                            argumentList.append(f'P{argumentId}')
                else:
                    for argument in pbCircuitLine.argumentValueList:
                        argumentList.append(f'{argument}')
                self._draw(qRegCount, f'|{gateName}|', pbCircuitLine.qRegList, circuitArray)
            elif op == 'customizedGate':
                customizedGate: PBCustomizedGate = pbCircuitLine.customizedGate
                matrixBytes = protobufMatrixToNumpyMatrix(customizedGate.matrix).tobytes()
                hash = blake2b(matrixBytes, digest_size=Settings.drawCircuitCustomizedGateHashLength)
                gateName = f'Custom[{hash.hexdigest()}]'
                gateName = self._appendNoise(gateName, pbCircuitLine)
                self._draw(qRegCount, f'|{gateName}]|', pbCircuitLine.qRegList, circuitArray)
            elif op == 'procedureName':
                gateName = pbCircuitLine.procedureName
                argumentList = []
                if pbCircuitLine.argumentIdList:
                    for index, argumentId in enumerate(pbCircuitLine.argumentIdList):
                        if argumentId == -1:
                            argumentList.append(f'{pbCircuitLine.argumentValueList[index]}')
                        else:
                            argumentList.append(f'P{argumentId}')
                else:
                    for argument in pbCircuitLine.argumentValueList:
                        argumentList.append(f'{argument}')
                self._draw(qRegCount, f'|{gateName}|', pbCircuitLine.qRegList, circuitArray)
            elif op == 'measure':
                for qReg in pbCircuitLine.qRegList:
                    measure: PBMeasure = pbCircuitLine.measure
                    typeName = PBMeasure.Type.Name(measure.type)
                    self._draw(qRegCount, f'|M[{typeName.lower()}]|', [qReg], circuitArray)
            elif op == 'barrier':
                for qReg in pbCircuitLine.qRegList:
                    self._draw(qRegCount, '|', [qReg], circuitArray)
            elif op == 'photonicGaussianGate':
                photonicGaussianGate: PBPhotonicGaussianGate = pbCircuitLine.photonicGaussianGate
                gateName = PBPhotonicGaussianGate.Name(photonicGaussianGate)
                # argumentList = []
                # if pbCircuitLine.argumentIdList:
                #     for index, argumentId in enumerate(pbCircuitLine.argumentIdList):
                #         if argumentId == -1:
                #             argumentList.append(f'{pbCircuitLine.argumentValueList[index]}')
                #         else:
                #             argumentList.append(f'P{argumentId}')
                # else:
                #     for argument in pbCircuitLine.argumentValueList:
                #         argumentList.append(f'{argument}')
                self._draw(qRegCount, f'|{gateName}|', pbCircuitLine.qRegList, circuitArray)
            elif op == 'photonicGaussianMeasure':
                for qReg in pbCircuitLine.qRegList:
                    photonicGaussianMeasure: PBPhotonicGaussianMeasure = pbCircuitLine.photonicGaussianMeasure
                    typeName = PBPhotonicGaussianMeasure.Type.Name(photonicGaussianMeasure.type)
                    self._draw(qRegCount, f'|M[{typeName.lower()}]|', [qReg], circuitArray)
            elif op == 'photonicFockGate':
                photonicFockGate: PBPhotonicFockGate = pbCircuitLine.photonicFockGate
                gateName = PBPhotonicFockGate.Name(photonicFockGate)
                # argumentList = []
                # if pbCircuitLine.argumentIdList:
                #     for index, argumentId in enumerate(pbCircuitLine.argumentIdList):
                #         if argumentId == -1:
                #             argumentList.append(f'{pbCircuitLine.argumentValueList[index]}')
                #         else:
                #             argumentList.append(f'P{argumentId}')
                # else:
                #     for argument in pbCircuitLine.argumentValueList:
                #         argumentList.append(f'{argument}')
                self._draw(qRegCount, f'|{gateName}|', pbCircuitLine.qRegList, circuitArray)
            elif op == 'photonicFockMeasure':
                for qReg in pbCircuitLine.qRegList:
                    self._draw(qRegCount, f'|M[{"PhotonCount".lower()}]|', [qReg], circuitArray)
            else:
                raise Error.ArgumentError(f'DrawConsole Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 1)

        for array in circuitArray:
            for text in array:
                self.stringIO.write(text)
            self.stringIO.write('\n')
        self.stringIO.write('\n')

    def _draw(self, qRegCount: int, operation: str, qRegList: List[int], circuitArray: List[List[str]]) -> None:
        maxQRegAsciiLen = len(str(max(qRegList)))
        maxLen = gapLen + max(maxQRegAsciiLen, len(operation))
        op = gap + operation
        if len(op) < maxLen:
            op += gapNode * (maxLen - len(op))
        circuitArray[qRegList[-1] * 2].append(op)

        for i in range(len(qRegList) - 1):
            line = gapNode * numPos + str(i)
            if len(line) < maxLen:
                line += gapNode * (maxLen - len(line))
            circuitArray[qRegList[i] * 2].append(line)

        line = gapNode * maxLen
        for i in range(qRegCount):
            if i not in qRegList:
                circuitArray[i * 2].append(line)

        qRegMin = min(qRegList) * 2
        qRegMax = max(qRegList) * 2
        line = ' ' * maxLen
        bridgeLine = ' ' * numPos + '|'
        if len(bridgeLine) < maxLen:
            bridgeLine += ' ' * (maxLen - len(bridgeLine))
        for i, array in enumerate(circuitArray):
            if i % 2 == 1:
                if i > qRegMin and i < qRegMax:
                    circuitArray[i].append(bridgeLine)
                else:
                    circuitArray[i].append(line)

    def _appendNoise(self, gateName: str, pbCircuitLine: PBCircuitLine):
        if len(pbCircuitLine.noiseList) == 0:
            return gateName

        gateName += '_{'
        for pbNoise in pbCircuitLine.noiseList:
            noise = pbNoise.WhichOneof('noise')
            if noise == 'bitFlip':
                gateName += 'BF-'
            elif noise == 'phaseFlip':
                gateName += 'PF-'
            elif noise == 'bitPhaseFlip':
                gateName += 'BPF-'
            elif noise == 'pauliNoise':
                gateName += 'PN-'
            elif noise == 'amplitudeDamping':
                gateName += 'AD-'
            elif noise == 'resetNoise':
                gateName += 'RN-'
            elif noise == 'phaseDamping':
                gateName += 'PD-'
            elif noise == 'depolarizing':
                gateName += 'DP-'
            elif noise == 'customizedNoise':
                gateName += 'CN-'
        gateName = gateName[:-1]
        gateName += '}'
        return gateName