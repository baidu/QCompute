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
PostProcess
"""
FileErrorCode = 19

from typing import Dict, Union

from QCompute.Define import MeasureFormat
from QCompute.Define import Settings


def filterMeasure(counts: Dict[str, int], measuredQRegsToCRegsDict: Dict[int, int]
                  , reverse: bool = False) -> Dict[str, int]:
    # return counts

    sourceQRegCount = 0
    for key in counts.keys():
        sourceQRegCount = len(key)
        break

    assert sourceQRegCount > 0

    neededQRegCount = max(measuredQRegsToCRegsDict.keys()) + 1
    if neededQRegCount > sourceQRegCount:
        neededQRegList = sorted(measuredQRegsToCRegsDict.keys())
        newQRegsToCRegsDict = {}
        for index, key in enumerate(neededQRegList):
            newQRegsToCRegsDict[index] = measuredQRegsToCRegsDict[key]
        measuredQRegsToCRegsDict = newQRegsToCRegsDict

    qRegList = list(measuredQRegsToCRegsDict.keys())
    qRegCount = len(measuredQRegsToCRegsDict)
    targetList = sorted(measuredQRegsToCRegsDict.values())
    for key in measuredQRegsToCRegsDict.keys():
        measuredQRegsToCRegsDict[key] = targetList.index(measuredQRegsToCRegsDict[key])

    zeroKey = '0' * qRegCount
    binRet: Dict[str, int] = {}
    for k, v in counts.items():
        hit = False
        for qReg in qRegList:
            if k[sourceQRegCount - 1 - qReg] == '1':
                hit = True
                break
        if hit:
            keyList = ['0'] * qRegCount
            for qReg in qRegList:
                keyList[qRegCount - 1 - measuredQRegsToCRegsDict[qReg]] = k[sourceQRegCount - 1 - qReg]
            if reverse:
                key = ''.join(reversed(keyList))
            else:
                key = ''.join(keyList)
        else:
            key = zeroKey
        if binRet.get(key) is None:
            binRet[key] = v
        else:
            binRet[key] += v
    return binRet


def formatMeasure(counts: Dict[str, int], cRegCount: int, mFormat: MeasureFormat = Settings.measureFormat) -> Dict[
    str, int]:
    ret: Dict[str, int] = {}
    for (k, v) in counts.items():
        if mFormat == MeasureFormat.Bin and k.startswith('0x'):
            num = int(k, 16)
            ret[bin(num)[2:].zfill(cRegCount)] = v
        elif mFormat == MeasureFormat.Hex and not k.startswith('0x'):
            num = int(k, 2)
            ret[hex(num)] = v
        else:
            ret[k] = v
    return ret


def formatReverseMeasure(counts: Dict[str, int], cRegCount: int, reverse: bool,
                         mFormat: MeasureFormat = Settings.measureFormat) -> Dict[Union[str, int], int]:
    ret: Dict[Union[str, int], int] = {}
    for (k, v) in counts.items():
        if k.startswith('0x'):
            numKey = int(k, 16)
            binKey = bin(numKey)[2:].zfill(cRegCount)
        else:
            numKey = int(k, 2)
            binKey = bin(numKey)[2:].zfill(cRegCount)
        if reverse:
            binKey = binKey[::-1]
        if mFormat == MeasureFormat.Bin:
            ret[binKey] = v
        elif mFormat == MeasureFormat.Hex:
            numKey = int(binKey, 2)
            hexKey = hex(numKey)
            ret[hexKey] = v
        elif mFormat == MeasureFormat.Dec:
            numKey = int(binKey, 2)
            ret[numKey] = v
    return ret