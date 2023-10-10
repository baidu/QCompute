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
Controlled Circuit
"""
FileErrorCode = 2

import copy
from typing import List

import numpy

from QCompute.QPlatform import Error, QEnv
from QCompute.QPlatform.QOperation import CircuitLine
from QCompute.QPlatform.QOperation.FixedGate import CX, CY, CH, CCX, SDG, S, H, CSWAP, CZ
from QCompute.QPlatform.QOperation.RotationGate import RotationGateOP, CU, CRX, CRY, CRZ, U
from QCompute.Utilize import ModuleErrorCode


def getControlledCircuit(env: QEnv, circuitLine: CircuitLine, controlQRegIndex: int, cuFirst: bool) \
        -> List[CircuitLine]:
    ret: List['CircuitLine'] = []

    op = circuitLine.data

    if op.bits == 1:
        q0 = circuitLine.qRegList[0]
        q1 = controlQRegIndex
    elif op.bits == 2:
        q0 = circuitLine.qRegList[0]
        q1 = circuitLine.qRegList[1]
        q2 = controlQRegIndex
    elif op.bits == 3:
        q0 = circuitLine.qRegList[0]
        q1 = circuitLine.qRegList[1]
        q2 = circuitLine.qRegList[2]
        q3 = controlQRegIndex
    elif op.__class__.__name__ == 'QProcedureOP':
        pass
    else:
        raise Error.ArgumentError(
            f'Wrong bits count! {op.name} bits value: {op.bits}.', ModuleErrorCode, FileErrorCode, 1)

    if isinstance(op, RotationGateOP):
        nAngles = len(op.argumentList)
        if nAngles == 1:
            [theta] = op.argumentList
        elif nAngles == 2:
            [theta, phi] = op.argumentList
        elif nAngles == 3:
            [theta, phi, lamda] = op.argumentList
        else:
            raise Error.ArgumentError(
                f'Wrong angles count! {op.name} angles value: {op.argumentList}.', ModuleErrorCode, FileErrorCode, 2)

    # 1-qubit, FixedGate
    if op.name == 'ID':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(0, 0, 0)
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)
    elif op.name == 'X':
        newCircuitLine = CircuitLine()
        if cuFirst:
            newCircuitLine.data = CU(numpy.pi, 0, numpy.pi)
        else:
            newCircuitLine.data = CX
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)
    elif op.name == 'Y':
        newCircuitLine = CircuitLine()
        if cuFirst:
            newCircuitLine.data = CU(numpy.pi, numpy.pi / 2, numpy.pi / 2)
        else:
            newCircuitLine.data = CY
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)
    elif op.name == 'Z':
        newCircuitLine = CircuitLine()
        if cuFirst:
            newCircuitLine.data = CU(0, 0, numpy.pi)
        else:
            newCircuitLine.data = CZ
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)
    elif op.name == 'H':
        newCircuitLine = CircuitLine()
        if cuFirst:
            newCircuitLine.data = CU(numpy.pi / 2, 0, numpy.pi)
        else:
            newCircuitLine.data = CH
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)
    elif op.name == 'S':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(0, 0, numpy.pi / 2)
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)
    elif op.name == 'SDG':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(0, 0, -numpy.pi / 2)
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)
    elif op.name == 'T':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(0, 0, numpy.pi / 4)
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)
    elif op.name == 'TDG':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(0, 0, - numpy.pi / 4)
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)

    # 1-qubit, RotationGate
    elif op.name == 'U':
        newCircuitLine = CircuitLine()
        if nAngles == 1:
            newCircuitLine.data = CU(0, 0, theta)
        elif nAngles == 2:
            newCircuitLine.data = CU(numpy.pi / 2, theta, phi)
        elif nAngles == 3:
            newCircuitLine.data = CU(theta, phi, lamda)
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)
    elif op.name == 'RX':
        newCircuitLine = CircuitLine()
        if cuFirst:
            newCircuitLine.data = CU(theta, -numpy.pi / 2, numpy.pi / 2)
        else:
            newCircuitLine.data = CRX(theta)
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)
    elif op.name == 'RY':
        newCircuitLine = CircuitLine()
        if cuFirst:
            newCircuitLine.data = CU(theta, 0, 0)
        else:
            newCircuitLine.data = CRY(theta)
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)
    elif op.name == 'RZ':
        newCircuitLine = CircuitLine()
        if cuFirst:
            newCircuitLine.data = CU(0, 0, theta)
        else:
            newCircuitLine.data = CRZ(theta)
        newCircuitLine.qRegList = [q1, q0]
        ret.append(newCircuitLine)

    # 2-qubit, FixedGate
    elif op.name == 'CX':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)
    elif op.name == 'CY':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = SDG
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = S
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)
    elif op.name == 'CZ':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = H
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = H
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)
    elif op.name == 'CH':
        theta = numpy.pi / 2
        phi = 0
        lamda = numpy.pi

        newCircuitLine = CircuitLine()
        newCircuitLine.data = U((lamda + phi) / 2)
        newCircuitLine.qRegList = [q0]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = U((lamda - phi) / 2)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = U(-theta / 2, 0, -(lamda + phi) / 2)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = U(theta / 2, phi, 0)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)
    elif op.name == 'SWAP':
        if cuFirst:
            newCircuitLine = CircuitLine()
            newCircuitLine.data = CCX
            newCircuitLine.qRegList = [q2, q0, q1]
            ret.append(newCircuitLine)

            newCircuitLine = CircuitLine()
            newCircuitLine.data = CCX
            newCircuitLine.qRegList = [q2, q1, q0]
            ret.append(newCircuitLine)

            newCircuitLine = CircuitLine()
            newCircuitLine.data = CCX
            newCircuitLine.qRegList = [q2, q0, q1]
            ret.append(newCircuitLine)
        else:
            newCircuitLine = CircuitLine()
            newCircuitLine.data = CSWAP
            newCircuitLine.qRegList = [q2, q0, q1]
            ret.append(newCircuitLine)

    # 2-qubit, RotationGate
    elif op.name == 'CU':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = U((lamda + phi) / 2)
        newCircuitLine.qRegList = [q0]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = U((lamda - phi) / 2)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = U(-theta / 2, 0, -(lamda + phi) / 2)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = U(theta / 2, phi, 0)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)
    elif op.name == 'CRX':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = U(numpy.pi / 2)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = U(-theta / 2, 0, 0)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = U(theta / 2, -numpy.pi / 2, 0)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)
    elif op.name == 'CRY':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = U(theta / 2, 0, 0)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = U(-theta / 2, 0, 0)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)
    elif op.name == 'CRZ':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = U(theta / 2)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = U(-theta / 2)
        newCircuitLine.qRegList = [q1]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q2, q0, q1]
        ret.append(newCircuitLine)

    # 3-qubit, FixedGate
    elif op.name == 'CCX':
        newCircuitLine = CircuitLine()
        newCircuitLine.data = CRZ(numpy.pi / 2)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(numpy.pi / 2, 0, 0)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q3, q0, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(-numpy.pi / 2, 0, 0)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q3, q0, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CRZ(-numpy.pi / 2)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)
    elif op.name == 'CSWAP':
        originQ1 = q1
        originQ2 = q2

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CRZ(numpy.pi / 2)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(numpy.pi / 2, 0, 0)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q3, q0, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(-numpy.pi / 2, 0, 0)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q3, q0, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CRZ(-numpy.pi / 2)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        q2 = originQ1
        q1 = originQ2

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CRZ(numpy.pi / 2)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(numpy.pi / 2, 0, 0)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q3, q0, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(-numpy.pi / 2, 0, 0)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q3, q0, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CRZ(-numpy.pi / 2)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        q1 = originQ1
        q2 = originQ2

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CRZ(numpy.pi / 2)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(numpy.pi / 2, 0, 0)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q3, q0, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CU(-numpy.pi / 2, 0, 0)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CCX
        newCircuitLine.qRegList = [q3, q0, q2]
        ret.append(newCircuitLine)

        newCircuitLine = CircuitLine()
        newCircuitLine.data = CRZ(-numpy.pi / 2)
        newCircuitLine.qRegList = [q1, q2]
        ret.append(newCircuitLine)

    elif op.__class__.__name__ == 'QProcedureOP':
        newProcedureName, newProcedure = env.controlProcedure(op.name, cuFirst)
        env.procedureMap[newProcedureName] = newProcedure

        newCircuitLine = CircuitLine()
        newCircuitLine.data = newProcedure(*op.argumentList)
        newCircuitLine.qRegList = copy.copy(circuitLine.qRegList)
        newCircuitLine.qRegList.append(controlQRegIndex)
        ret.append(newCircuitLine)

    # Unsupported
    else:
        raise Error.ArgumentError(
            f"{op.__class__.__name__} {op.name} can't be controlled in procedure!", ModuleErrorCode, FileErrorCode, 3)

    return ret