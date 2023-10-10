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
Convert the circuit to XanaduSF
"""
FileErrorCode = 8

import math
from enum import Enum
from io import StringIO
from typing import List, TYPE_CHECKING, Dict

import numpy

from QCompute.OpenConvertor import ConvertorImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QProtobuf import PBFixedGate, PBRotationGate

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBProgram


class TwoQubitsGate(Enum):
    quarter = 'quarter'
    third = 'third'
    kerr = 'kerr'


class CircuitToXanaduSF(ConvertorImplement):
    """
    Circuit to XanaduSF
    """

    def convert(self, pbProgram: 'PBProgram', twoQubitsGate: TwoQubitsGate = TwoQubitsGate.quarter) -> str:
        """
        Convert the circuit to XanaduSF.
        """

        with StringIO() as ret:
            ret.write('name program\n')
            ret.write('version 1.0\n')

            if twoQubitsGate == TwoQubitsGate.quarter:
                quarterConvert(pbProgram, ret)
            if twoQubitsGate == TwoQubitsGate.third:
                thirdConvert(pbProgram, ret)
            if twoQubitsGate == TwoQubitsGate.kerr:
                kerrConvert(pbProgram, ret)
            return ret.getvalue()


ancillaQRegIndex = 0


def quarterConvert(pbProgram: 'PBProgram', ret: StringIO):
    qRegMap: Dict[int, int] = {}
    for qReg in pbProgram.head.usingQRegList:
        sfQReg = len(qRegMap)
        qRegMap[qReg] = sfQReg
        ret.write(f'Fock(1) | {sfQReg * 2}\n')
        ret.write(f'Fock(0) | {sfQReg * 2 + 1}\n')
    global ancillaQRegIndex
    ancillaQRegIndex = len(qRegMap)

    for pbCircuitLine in pbProgram.body.circuit:
        op = pbCircuitLine.WhichOneof('op')
        if op == 'fixedGate':
            fixedGate: PBFixedGate = pbCircuitLine.fixedGate
            quarterFixedGateMapping(fixedGate, pbCircuitLine.qRegList, qRegMap, ret)
        elif op == 'rotationGate':
            rotationGate: PBRotationGate = pbCircuitLine.rotationGate
            quarterRotationGateMapping(rotationGate, pbCircuitLine.argumentValueList, pbCircuitLine.qRegList, qRegMap,
                                       ret)
        elif op == 'measure':
            continue
        else:
            raise Error.ArgumentError(
                f'Unsupported operation {op} at XanaduSF quarter!', ModuleErrorCode, FileErrorCode, 1)


def quarterFixedGateMapping(pbFixedGate: PBFixedGate, qRegList: List[int], qRegMap: Dict[int, int], ret: StringIO):
    global ancillaQRegIndex
    if pbFixedGate == PBFixedGate.CX:
        ret.write(f'Fock(1) | {2 * ancillaQRegIndex}\n')
        ret.write(f'Fock(0) | {2 * ancillaQRegIndex + 1}\n')
        ret.write(f'Fock(1) | {2 * (ancillaQRegIndex + 1)}\n')
        ret.write(f'Fock(0) | {2 * (ancillaQRegIndex + 1) + 1}\n')

        # Gate operation
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[0]] + 1}, {2 * qRegMap[qRegList[1]]}]\n')

        # Ancilla for post-section control qubits
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[0]] + 1}\n')
        ret.write(f'BSgate({22.5000 / 180 * math.pi}, 0) | [{2 * ancillaQRegIndex}, {2 * ancillaQRegIndex + 1}]\n')
        ret.write(f'BSgate({65.5302 / 180 * math.pi}, 0) | [{2 * qRegMap[qRegList[0]] + 1}, {2 * ancillaQRegIndex}]\n')
        ret.write(f'BSgate({-22.5000 / 180 * math.pi}, 0) | [{2 * ancillaQRegIndex}, {2 * ancillaQRegIndex + 1}]\n')

        # Ancilla for post-section target qubits
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(
            f'BSgate({22.5000 / 180 * math.pi}, 0) | [{2 * (ancillaQRegIndex + 1)}, {2 * (ancillaQRegIndex + 1) + 1}]\n')
        ret.write(
            f'BSgate({65.5302 / 180 * math.pi}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * (ancillaQRegIndex + 1)}]\n')
        ret.write(
            f'BSgate({-22.5000 / 180 * math.pi}, 0) | [{2 * (ancillaQRegIndex + 1)}, {2 * (ancillaQRegIndex + 1) + 1}]\n')

        # Gate opration
        ret.write(f'BSgate({math.pi / 4}, 0).H | [{2 * qRegMap[qRegList[0]] + 1}, {2 * qRegMap[qRegList[1]]}]\n')
        ret.write(f'BSgate({math.pi / 4}, 0).H | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')

        ancillaQRegIndex += 2
    elif pbFixedGate == PBFixedGate.CZ:
        ret.write(f'Fock(1) | {2 * ancillaQRegIndex}\n')
        ret.write(f'Fock(0) | {2 * ancillaQRegIndex + 1}\n')
        ret.write(f'Fock(1) | {2 * (ancillaQRegIndex + 1)}\n')
        ret.write(f'Fock(0) | {2 * (ancillaQRegIndex + 1) + 1}\n')

        # H gate
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]] + 1}\n')

        # CX
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[0]] + 1}, {2 * qRegMap[qRegList[1]]}]\n')
        # Ancilla for post-section control qubits
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[0]] + 1}\n')
        ret.write(f'BSgate({22.5000 / 180 * math.pi}, 0) | [{2 * ancillaQRegIndex}, {2 * ancillaQRegIndex + 1}]\n')
        ret.write(f'BSgate({65.5302 / 180 * math.pi}, 0) | [{2 * qRegMap[qRegList[0]] + 1}, {2 * ancillaQRegIndex}]\n')
        ret.write(f'BSgate({-22.5000 / 180 * math.pi}, 0) | [{2 * ancillaQRegIndex}, {2 * ancillaQRegIndex + 1}]\n')
        # Ancilla for post-section target qubits
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(
            f'BSgate({22.5000 / 180 * math.pi}, 0) | [{2 * (ancillaQRegIndex + 1)}, {2 * (ancillaQRegIndex + 1) + 1}]\n')
        ret.write(
            f'BSgate({65.5302 / 180 * math.pi}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * (ancillaQRegIndex + 1)}]\n')
        ret.write(
            f'BSgate({-22.5000 / 180 * math.pi}, 0) | [{2 * (ancillaQRegIndex + 1)}, {2 * (ancillaQRegIndex + 1) + 1}]\n')
        ret.write(f'BSgate({math.pi / 4}, 0).H | [{2 * qRegMap[qRegList[0]] + 1}, {2 * qRegMap[qRegList[1]]}]\n')
        ret.write(f'BSgate({math.pi / 4}, 0).H | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')

        # H gate
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]] + 1}\n')

        ancillaQRegIndex += 2
    else:
        kerrFixedGateMapping(TwoQubitsGate.quarter, pbFixedGate, qRegList, qRegMap, ret)


def quarterRotationGateMapping(pbRotationGate: PBRotationGate, argumentValueList: List[float], qRegList: List[int],
                               qRegMap: Dict[int, int], ret: StringIO):
    kerrRotationGateMapping(TwoQubitsGate.quarter, pbRotationGate, argumentValueList, qRegList, qRegMap, ret)


twoQubitsSet = set()


def thirdConvert(pbProgram: 'PBProgram', ret: StringIO):
    qRegMap: Dict[int, int] = {}
    for qReg in pbProgram.head.usingQRegList:
        sfQReg = len(qRegMap)
        qRegMap[qReg] = sfQReg
        ret.write(f'Fock(1) | {sfQReg * 2}\n')
        ret.write(f'Fock(0) | {sfQReg * 2 + 1}\n')
    global ancillaQRegIndex
    ancillaQRegIndex = len(qRegMap)
    twoQubitsSet.clear()

    for pbCircuitLine in pbProgram.body.circuit:
        op = pbCircuitLine.WhichOneof('op')
        if op == 'fixedGate':
            fixedGate: PBFixedGate = pbCircuitLine.fixedGate
            thirdFixedGateMapping(fixedGate, pbCircuitLine.qRegList, qRegMap, ret)
        elif op == 'rotationGate':
            rotationGate: PBRotationGate = pbCircuitLine.rotationGate
            thirdRotationGateMapping(rotationGate, pbCircuitLine.argumentValueList, pbCircuitLine.qRegList, qRegMap,
                                     ret)
        elif op == 'measure':
            continue
        else:
            raise Error.ArgumentError(
                f'Unsupported operation {op} at XanaduSF third!', ModuleErrorCode, FileErrorCode, 2)


def thirdFixedGateMapping(pbFixedGate: PBFixedGate, qRegList: List[int], qRegMap: Dict[int, int], ret: StringIO):
    global ancillaQRegIndex
    if pbFixedGate == PBFixedGate.CX:
        if len(set.intersection(twoQubitsSet, set(qRegList))) > 0:
            raise Error.ArgumentError(f'Duplicate qubit {qRegList} at XanaduSF third!', ModuleErrorCode, FileErrorCode, 3)

        twoQubitsSet.update(qRegList)

        ret.write(f'Fock(0) | {2 * ancillaQRegIndex}\n')
        ret.write(f'Fock(0) | {2 * ancillaQRegIndex + 1}\n')

        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        ret.write(f'BSgate({numpy.arccos((1 / 3) ** 0.5)}, 0) | [{2 * qRegMap[qRegList[0]]}, {2 * ancillaQRegIndex}]\n')
        ret.write(
            f'BSgate({numpy.arccos((1 / 3) ** 0.5)}, 0) | [{2 * qRegMap[qRegList[0]] + 1}, {2 * qRegMap[qRegList[1]]}]\n')
        ret.write(
            f'BSgate({numpy.arccos((1 / 3) ** 0.5)}, 0) | [{2 * qRegMap[qRegList[1]] + 1}, {2 * ancillaQRegIndex + 1}]\n')
        ret.write(f'BSgate({math.pi / 4}, 0).H | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')

        ancillaQRegIndex += 1
    elif pbFixedGate == PBFixedGate.CZ:
        if len(set.intersection(twoQubitsSet, set(qRegList))) > 0:
            raise Error.ArgumentError(f'Duplicate qubit {qRegList} at XanaduSF third!', ModuleErrorCode, FileErrorCode, 4)

        twoQubitsSet.update(qRegList)

        ret.write(f'Fock(0) | {2 * ancillaQRegIndex}\n')
        ret.write(f'Fock(0) | {2 * ancillaQRegIndex + 1}\n')

        # H gate
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]] + 1}\n')
        # CX gate
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        ret.write(f'BSgate({numpy.arccos((1 / 3) ** 0.5)}, 0) | [{2 * qRegMap[qRegList[0]]}, {2 * ancillaQRegIndex}]\n')
        ret.write(
            f'BSgate({numpy.arccos((1 / 3) ** 0.5)}, 0) | [{2 * qRegMap[qRegList[0]] + 1}, {2 * qRegMap[qRegList[1]]}]\n')
        ret.write(
            f'BSgate({numpy.arccos((1 / 3) ** 0.5)}, 0) | [{2 * qRegMap[qRegList[1]] + 1}, {2 * ancillaQRegIndex + 1}]\n')
        ret.write(f'BSgate({math.pi / 4}, 0).H | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        # H gate
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]] + 1}\n')

        ancillaQRegIndex += 1
    else:
        kerrFixedGateMapping(TwoQubitsGate.third, pbFixedGate, qRegList, qRegMap, ret)


def thirdRotationGateMapping(pbRotationGate: PBRotationGate, argumentValueList: List[float], qRegList: List[int],
                             qRegMap: Dict[int, int], ret: StringIO):
    kerrRotationGateMapping(TwoQubitsGate.third, pbRotationGate, argumentValueList, qRegList, qRegMap, ret)


def kerrConvert(pbProgram: 'PBProgram', ret: StringIO):
    qRegMap: Dict[int, int] = {}
    for qReg in pbProgram.head.usingQRegList:
        sfQReg = len(qRegMap)
        qRegMap[qReg] = sfQReg
        ret.write(f'Fock(1) | {sfQReg * 2}\n')
        ret.write(f'Fock(0) | {sfQReg * 2 + 1}\n')

    for pbCircuitLine in pbProgram.body.circuit:
        op = pbCircuitLine.WhichOneof('op')
        if op == 'fixedGate':
            fixedGate: PBFixedGate = pbCircuitLine.fixedGate
            kerrFixedGateMapping(TwoQubitsGate.kerr, fixedGate, pbCircuitLine.qRegList, qRegMap, ret)
        elif op == 'rotationGate':
            rotationGate: PBRotationGate = pbCircuitLine.rotationGate
            kerrRotationGateMapping(TwoQubitsGate.kerr, rotationGate, pbCircuitLine.argumentValueList,
                                    pbCircuitLine.qRegList, qRegMap, ret)
        elif op == 'measure':
            continue
        else:
            raise Error.ArgumentError(
                f'Unsupported operation {op} at XanaduSF kerr!', ModuleErrorCode, FileErrorCode, 5)


def kerrFixedGateMapping(type: TwoQubitsGate, pbFixedGate: PBFixedGate, qRegList: List[int], qRegMap: Dict[int, int],
                         ret: StringIO):
    if pbFixedGate == PBFixedGate.X:
        ret.write(
            f'BSgate({math.pi / 2}, {-math.pi / 2}) | [{2 * qRegMap[qRegList[0]]}, {2 * qRegMap[qRegList[0]] + 1}]\n')
    elif pbFixedGate == PBFixedGate.H:
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[0]]}\n')
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[0]]}, {2 * qRegMap[qRegList[0]] + 1}]\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[0]]}\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[0]] + 1}\n')
    elif pbFixedGate == PBFixedGate.CZ:
        ret.write(f'CKgate({math.pi}) | [{2 * qRegMap[qRegList[0]] + 1}, {2 * qRegMap[qRegList[1]] + 1}]\n')
    elif pbFixedGate == PBFixedGate.CX:
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]] + 1}\n')
        ret.write(f'CKgate({math.pi}) | [{2 * qRegMap[qRegList[0]] + 1}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'BSgate({math.pi / 4}, 0) | [{2 * qRegMap[qRegList[1]]}, {2 * qRegMap[qRegList[1]] + 1}]\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]]}\n')
        ret.write(f'Rgate({math.pi}) | {2 * qRegMap[qRegList[1]] + 1}\n')
    else:
        raise Error.ArgumentError(f'Unsupported fixedGate {PBFixedGate.Name(pbFixedGate)} at XanaduSF {type.value}!', ModuleErrorCode, FileErrorCode, 6)


def kerrRotationGateMapping(type: TwoQubitsGate, pbRotationGate: PBRotationGate, argumentValueList: List[float],
                            qRegList: List[int],
                            qRegMap: Dict[int, int], ret: StringIO):
    if pbRotationGate == PBRotationGate.RX:
        ret.write(
            f'BSgate({argumentValueList[0] / 2}, {-math.pi / 2}) | [{2 * qRegMap[qRegList[0]]}, {2 * qRegMap[qRegList[0]] + 1}]\n')
    elif pbRotationGate == PBRotationGate.RY:
        ret.write(
            f'BSgate({argumentValueList[0] / 2}, 0) | [{2 * qRegMap[qRegList[0]]}, {2 * qRegMap[qRegList[0]] + 1}]\n')
    else:
        raise Error.ArgumentError(f'Unsupported rotationGate {PBRotationGate.Name(pbRotationGate)} at XanaduSF {type.value}!', ModuleErrorCode, FileErrorCode, 7)