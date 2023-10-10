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
Convert the circuit to IonQ
"""
FileErrorCode = 4

import json
from typing import List, TYPE_CHECKING, Dict

from QCompute.OpenConvertor import ConvertorImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QProtobuf import PBFixedGate, PBRotationGate

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBProgram


class CircuitToIonQ(ConvertorImplement):
    """
    Circuit to IonQ
    """

    def convert(self, pbProgram: 'PBProgram', shots: int = None, target: str = None) -> str:
        """
        Convert the circuit to IonQ.
        """

        circuit = []
        body = {
            'qubits': len(pbProgram.head.usingQRegList),
            'circuit': circuit
        }
        ret = {
            'shots': shots,
            'target': target,
            'lang': 'json',
            'body': body
        }
        for pbCircuitLine in pbProgram.body.circuit:
            op = pbCircuitLine.WhichOneof('op')
            if op == 'fixedGate':
                fixedGate: PBFixedGate = pbCircuitLine.fixedGate
                gateName = PBFixedGate.Name(fixedGate)
                circuitLine = fixedGateMapping(gateName, pbCircuitLine.qRegList)
            elif op == 'rotationGate':
                rotationGate: PBRotationGate = pbCircuitLine.rotationGate
                gateName = PBRotationGate.Name(rotationGate)
                circuitLine = rotationGateMapping(gateName, pbCircuitLine.argumentValueList, pbCircuitLine.qRegList)
            elif op == 'measure':
                continue
            else:
                raise Error.ArgumentError(f'IonQ Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 1)
            circuit.append(circuitLine)

        return json.dumps(ret)


class IonQOperation:
    def __init__(self,
                 gate: str = None,
                 controlCount: int = 0,
                 targetCount: int = 0,
                 ):
        self.gate = gate
        self.controlCount = controlCount
        self.targetCount = targetCount


supportedFixedGate = {
    'X': IonQOperation(
        gate='x',
        targetCount=1,
    ),
    'Y': IonQOperation(
        gate='y',
        targetCount=1,
    ),
    'Z': IonQOperation(
        gate='z',
        targetCount=1,
    ),
    'H': IonQOperation(
        gate='h',
        targetCount=1,
    ),
    'CX': IonQOperation(  # cx ccx
        gate='cnot',
        controlCount=1,
        targetCount=1,
    ),
    'S': IonQOperation(
        gate='s',
        targetCount=1,
    ),
    'SDG': IonQOperation(
        gate='si',
        targetCount=1,
    ),
    'T': IonQOperation(
        gate='t',
        targetCount=1,
    ),
    'TDG': IonQOperation(
        gate='ti',
        targetCount=1,
    ),
    'SWAP': IonQOperation(
        gate='swap',
        targetCount=2,
    ),
    'CCX': IonQOperation(  # cx ccx
        gate='cnot',
        controlCount=2,
        targetCount=1,
    ),
}
supportedRotationGate = {
    'RX': IonQOperation(
        gate='rx',
        targetCount=1,
    ),
    'RY': IonQOperation(
        gate='ry',
        targetCount=1,
    ),
    'RZ': IonQOperation(
        gate='rz',
        targetCount=1,
    ),
}


def fixedGateMapping(gateName: str, qRegList: List[int]) -> Dict:
    if gateName not in supportedFixedGate:
        raise Error.ArgumentError(f'IonQ Unsupported fixedGate {gateName}!', ModuleErrorCode, FileErrorCode, 2)
    ionOp = supportedFixedGate[gateName]
    ret = {
        'gate': ionOp.gate
    }
    setQReg(ionOp, qRegList, ret)
    return ret


def rotationGateMapping(gateName: str, argumentValueList: List[float], qRegList: List[int]) -> Dict:
    if gateName not in supportedRotationGate:
        raise Error.ArgumentError(f'IonQ Unsupported rotationGate {gateName}!', ModuleErrorCode, FileErrorCode, 3)
    ionOp = supportedRotationGate[gateName]
    return {
        'gate': ionOp.gate,
        'rotation': argumentValueList[0],
        'target': qRegList[0]
    }


def setQReg(ionOp: IonQOperation, qRegList: List[int], op: Dict):
    currentQRegIndex = 0
    if ionOp.controlCount == 1:
        op['control'] = qRegList[currentQRegIndex]
        currentQRegIndex += 1
    elif ionOp.controlCount > 1:
        currentQRegIndex += ionOp.controlCount
        op['controls'] = qRegList[:currentQRegIndex]
    if ionOp.targetCount == 1:
        op['target'] = qRegList[currentQRegIndex]
        currentQRegIndex += 1
    elif ionOp.targetCount > 1:
        currentQRegIndex += ionOp.targetCount
        op['targets'] = qRegList[:currentQRegIndex]