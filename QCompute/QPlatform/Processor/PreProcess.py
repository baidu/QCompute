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
PreProcess
"""
FileErrorCode = 20

from typing import TYPE_CHECKING, Tuple, Set, Dict

from QCompute.QPlatform import Error, ModuleErrorCode

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBProgram


def preProcess(program: 'PBProgram', rerangeQReg: bool, strictUsingCReg: bool) -> \
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
        if rerangeQReg:
            compactedQRegDict[qReg] = index
        else:
            compactedQRegDict[qReg] = qReg

    measured = False
    for circuitLine in program.body.circuit:
        op: str = circuitLine.WhichOneof('op')
        if measured and op not in ['measure', 'photonicGaussianMeasure', 'photonicFockMeasure']:
            raise Error.ArgumentError('Measure must be the last operation!', ModuleErrorCode, FileErrorCode, 1)

        circuitLine.qRegList[:] = [compactedQRegDict[qReg] for qReg in circuitLine.qRegList]

        if op in ['measure', 'photonicGaussianMeasure', 'photonicFockMeasure']:
            measured = True
            measure = getattr(circuitLine, op)
            for index, cReg in enumerate(measure.cRegList):
                qReg = circuitLine.qRegList[index]
                if qReg in compactedCRegDict:
                    raise Error.ArgumentError('Measure must be once on a QReg!', ModuleErrorCode, FileErrorCode, 2)
                if cReg in compactedCRegDict.values():
                    raise Error.ArgumentError('Measure must be once on a CReg!', ModuleErrorCode, FileErrorCode, 3)
                compactedCRegDict[qReg] = cReg
                measure.cRegList[index] = qReg

    if measured is False:
        raise Error.ArgumentError(
            'At least one measurement operation is required in a quantum circuit.', ModuleErrorCode, FileErrorCode, 4)

    program.head.usingQRegList[:] = sorted(compactedQRegDict.values())
    if strictUsingCReg:
        program.head.usingCRegList[:] = sorted(compactedCRegDict.keys())
    else:
        program.head.usingCRegList[:] = program.head.usingQRegList[:]
    return usedQRegSet, usedCRegSet, compactedQRegDict, compactedCRegDict