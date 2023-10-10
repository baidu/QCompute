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
Convert the json to circuit
"""
FileErrorCode = 10

import json
from typing import Dict, Set, List

from QCompute.OpenConvertor import ConvertorImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBFixedGate, PBRotationGate


class IonQToCircuit(ConvertorImplement):
    """
    Json to circuit
    """

    def convert(self, ionQJsonStr: str) -> 'PBProgram':
        """
        Convert the json to circuit.

        Example:

        program = JsonToCircuit().convert(ionQJsonStr)

        :param ionQJsonStr: IonQ json str
        :return: Protobuf format of the circuit
        """

        ionQ: Dict = json.loads(ionQJsonStr)
        pbProgram = PBProgram()
        regSet = set()
        pbCircuit = pbProgram.body.circuit

        for circuitLine in ionQ['body']['circuit']:
            pbCircuitLineList: 'PBCircuitLine' = gateMapping(circuitLine, regSet)
            pbCircuit.extend(pbCircuitLineList)

        pbProgram.head.usingQRegList[:] = pbProgram.head.usingCRegList[:] = list(regSet)[:]
        return pbProgram


def unrollV(circuitLine: Dict, regSet: Set) -> List['PBCircuitLine']:
    ret: List[PBCircuitLine] = []

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.H
    pbCircuitLine.qRegList.append(circuitLine['target'])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.S
    pbCircuitLine.qRegList.append(circuitLine['target'])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.H
    pbCircuitLine.qRegList.append(circuitLine['target'])
    ret.append(pbCircuitLine)

    regSet.add(circuitLine['target'])
    return ret


def unrollVI(circuitLine: Dict, regSet: Set) -> List['PBCircuitLine']:
    ret: List[PBCircuitLine] = []

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.H
    pbCircuitLine.qRegList.append(circuitLine['target'])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.SDG
    pbCircuitLine.qRegList.append(circuitLine['target'])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.H
    pbCircuitLine.qRegList.append(circuitLine['target'])
    ret.append(pbCircuitLine)

    regSet.add(circuitLine['target'])
    return ret


def unrollXX(circuitLine: Dict, regSet: Set) -> List['PBCircuitLine']:
    ret: List[PBCircuitLine] = []

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.CX
    pbCircuitLine.qRegList.extend(circuitLine['targets'])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.rotationGate = PBRotationGate.RX
    pbCircuitLine.qRegList.append(circuitLine['targets'][0])
    pbCircuitLine.argumentValueList.append(circuitLine['rotation'])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.CX
    pbCircuitLine.qRegList.extend(circuitLine['targets'])
    ret.append(pbCircuitLine)

    regSet.update(circuitLine['targets'])
    return ret


def unrollYY(circuitLine: Dict, regSet: Set) -> List['PBCircuitLine']:
    ret: List[PBCircuitLine] = []

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.SDG
    pbCircuitLine.qRegList.append(circuitLine['targets'][1])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.CX
    pbCircuitLine.qRegList.extend(circuitLine['targets'])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.rotationGate = PBRotationGate.RY
    pbCircuitLine.qRegList.append(circuitLine['targets'][0])
    pbCircuitLine.argumentValueList.append(circuitLine['rotation'])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.CX
    pbCircuitLine.qRegList.extend(circuitLine['targets'])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.S
    pbCircuitLine.qRegList.append(circuitLine['targets'][1])
    ret.append(pbCircuitLine)

    regSet.update(circuitLine['targets'])
    return ret


def unrollZZ(circuitLine: Dict, regSet: Set) -> List['PBCircuitLine']:
    ret: List[PBCircuitLine] = []

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.CX
    pbCircuitLine.qRegList.extend(circuitLine['targets'])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.rotationGate = PBRotationGate.RZ
    pbCircuitLine.qRegList.append(circuitLine['targets'][1])
    pbCircuitLine.argumentValueList.append(circuitLine['rotation'])
    ret.append(pbCircuitLine)

    pbCircuitLine = PBCircuitLine()
    pbCircuitLine.fixedGate = PBFixedGate.CX
    pbCircuitLine.qRegList.extend(circuitLine['targets'])
    ret.append(pbCircuitLine)

    regSet.update(circuitLine['targets'])
    return ret


supportedFixedGate = {
    'x': 'X',
    'y': 'Y',
    'z': 'Z',
    'h': 'H',
    'not': 'X',
    'cnot': 'CX',  # cx cxx
    's': 'S',
    'si': 'SDG',
    't': 'T',
    'ti': 'TDG',
    'swap': 'SWAP',
}
supportedRotationGate = {
    'rx': 'RX',
    'ry': 'RY',
    'rz': 'RZ',
}
unrollFixedGate = {
    'v': unrollV,
    'vi': unrollVI,
    'xx': unrollXX,
    'yy': unrollYY,
    'zz': unrollZZ,
}


def gateMapping(circuitLine: Dict, regSet: Set) -> List['PBCircuitLine']:
    gateName = circuitLine['gate']
    if gateName in unrollFixedGate:
        return unrollFixedGate[gateName](circuitLine, regSet)

    pbCircuitLine = PBCircuitLine()
    if gateName in supportedFixedGate:
        gateName = supportedFixedGate[gateName]
        pbCircuitLine.fixedGate = PBFixedGate.Value(gateName)
    elif gateName in supportedRotationGate:
        gateName = supportedRotationGate[gateName]
        pbCircuitLine.rotationGate = PBRotationGate.Value(gateName)
        pbCircuitLine.argumentValueList[:] = [circuitLine['rotation']]
    else:
        raise Error.ArgumentError(f'Unsupporte gate {gateName} at IonQ!', ModuleErrorCode, FileErrorCode, 1)

    if 'control' in circuitLine:
        pbCircuitLine.qRegList.append(circuitLine['control'])
    elif 'controls' in circuitLine:
        if gateName == 'CX':
            pbCircuitLine.fixedGate = PBFixedGate.Value('CCX')
        pbCircuitLine.qRegList.extend(circuitLine['controls'])
    if 'target' in circuitLine:
        pbCircuitLine.qRegList.append(circuitLine['target'])
    elif 'targets' in circuitLine:
        pbCircuitLine.qRegList.extend(circuitLine['targets'])
    regSet.update(pbCircuitLine.qRegList)
    return [pbCircuitLine]