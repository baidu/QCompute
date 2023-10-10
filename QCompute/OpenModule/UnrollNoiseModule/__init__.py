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
Unroll Noise
"""
FileErrorCode = 7

from copy import deepcopy
from typing import Dict, Optional, List, Any

from QCompute.OpenModule import ModuleImplement
from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBFixedGate, PBRotationGate


class UnrollNoiseModule(ModuleImplement):
    """
    Unroll Noise

    Example:

    env.module(UnrollNoiseModule())
    """

    def __init__(self, arguments: Optional[Dict[str, bool]] = None):
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.
        """
        super().__init__(arguments)

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program
        :return: unrolled noise
        """
        if self.disable:
            return program

        ret = deepcopy(program)

        self._unrollNoise(ret)

        ret.body.noiseMap.clear()

        return ret

    def _findGate(self, pbCircuit: List[PBCircuitLine], gateName: str, qRegList: List[int],
                  positionList: Optional[List[int]]) -> List[PBCircuitLine]:
        gateList: List[PBCircuitLine] = []
        for pbCircuitLine in pbCircuit:
            op = pbCircuitLine.WhichOneof('op')
            if op == 'fixedGate':
                fixedGate: PBFixedGate = pbCircuitLine.fixedGate
                opName = PBFixedGate.Name(fixedGate)
            elif op == 'rotationGate':
                rotationGate: PBRotationGate = pbCircuitLine.rotationGate
                opName = PBRotationGate.Name(rotationGate)
            else:
                continue
            if opName != gateName:
                continue

            if not qRegList or set(pbCircuitLine.qRegList) == set(qRegList):
                gateList.append(pbCircuitLine)

        if len(gateList) == 0:
            print(f'Noise gate {gateName}{qRegList} not found.')

        if positionList:
            gateList = [gateList[pos] for pos in positionList]

        return gateList

    def _unrollNoise(self, program: 'PBProgram') -> None:
        for gateName, noiseDefineList in program.body.noiseMap.items():
            for noiseDefine in noiseDefineList.noiseDefineList:
                diff = set(noiseDefine.qRegList) - set(program.head.usingQRegList)
                if len(diff) > 0:
                    raise Error.ArgumentError(f'Unnecessary QBit{diff} in noise {noiseDefine.qRegList}/{program.head.usingQRegList}!', ModuleErrorCode, FileErrorCode, 1)

                if noiseDefine.qRegList:
                    qRegList = noiseDefine.qRegList
                else:
                    qRegList = None

                gateList = self._findGate(program.body.circuit, gateName, qRegList, noiseDefine.positionList)
                for noise in noiseDefine.noiseList:
                    for gate in gateList:
                        gate.noiseList.append(noise)