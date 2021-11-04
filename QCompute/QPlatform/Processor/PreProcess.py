#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2020 Baidu, Inc. All Rights Reserved.
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
PreProcess
"""
from typing import TYPE_CHECKING, Tuple, Set, Dict

from QCompute.QPlatform import Error, ModuleErrorCode

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBProgram

FileErrorCode = 14


def preProcess(program: 'PBProgram', strictUsingCReg: bool) -> \
        Tuple[Set[int], Set[int], Dict[int, int], Dict[int, int]]:
    usedQRegSet = set(program.head.usingQRegList)
    usedCRegSet = set(program.head.usingCRegList)
    compactedQRegDict = {}
    compactedCRegDict = {}

    qRegSet = set()
    for circuitLine in program.body.circuit:
        for qReg in circuitLine.qRegList:
            qRegSet.add(qReg)
    for index, qReg in enumerate(sorted(list(qRegSet))):
        compactedQRegDict[qReg] = index

    measured = False
    for circuitLine in program.body.circuit:
        op = circuitLine.WhichOneof('op')
        if measured and op != 'measure':
            raise Error.ArgumentError('Measure must be the last operation!', ModuleErrorCode, FileErrorCode, 1)

        qRegList = []
        for qReg in circuitLine.qRegList:
            qRegList.append(compactedQRegDict[qReg])
        circuitLine.qRegList[:] = qRegList

        if op == 'measure':
            measured = True
            for index, cReg in enumerate(circuitLine.measure.cRegList):
                qReg = circuitLine.qRegList[index]
                if qReg in compactedCRegDict:
                    raise Error.ArgumentError('Measure must be once on a QReg', ModuleErrorCode, FileErrorCode, 2)
                if cReg in compactedCRegDict.values():
                    raise Error.ArgumentError('Measure must be once on a CReg', ModuleErrorCode, FileErrorCode, 3)
                compactedCRegDict[qReg] = cReg
                circuitLine.measure.cRegList[index] = qReg

    program.head.usingQRegList[:] = sorted(compactedQRegDict.values())
    if strictUsingCReg:
        program.head.usingCRegList[:] = sorted(compactedCRegDict.keys())
    else:
        program.head.usingCRegList[:] = program.head.usingQRegList[:]
    return usedQRegSet, usedCRegSet, compactedQRegDict, compactedCRegDict
