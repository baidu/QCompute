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
Condition
"""
FileErrorCode = 16

from typing import Tuple, Set

from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBPhotonicGaussianMeasure


def checkRealCondition(program: 'PBProgram') -> Tuple[int, int, int]:
    qRegSet: Set[int] = set()
    cRegSet: Set[int] = set()
    for circuitLine in program.body.circuit:
        qRegSet.update(circuitLine.qRegList)
        if circuitLine.WhichOneof('op') == 'measure':
            cRegSet.update(circuitLine.measure.cRegList)
    return len(qRegSet), len(cRegSet), len(program.body.circuit)


def checkPhotonicRealCondition(program: 'PBProgram') -> Tuple[int, int, int, int]:
    qRegSet: Set[int] = set()
    cRegSet: Set[int] = set()
    cutoff = 0
    for circuitLine in program.body.circuit:
        qRegSet.update(circuitLine.qRegList)
        if circuitLine.WhichOneof('op') == 'photonicGaussianMeasure':
            cRegSet.update(circuitLine.photonicGaussianMeasure.cRegList)
            if circuitLine.photonicGaussianMeasure.type == PBPhotonicGaussianMeasure.PhotonCount:
                cutoff = circuitLine.photonicGaussianMeasure.photonCount.cutoff
        if circuitLine.WhichOneof('op') == 'photonicFockMeasure':
            cRegSet.update(circuitLine.photonicFockMeasure.cRegList)
            cutoff = circuitLine.photonicFockMeasure.cutoff
    return len(qRegSet), len(cRegSet), len(program.body.circuit), cutoff