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
PostProcess
"""
from typing import Dict

from QCompute.Define import MeasureFormat
from QCompute.Define.Settings import measureFormat


def filterMeasure(counts: Dict[str, int], measuredQRegsToCRegsDict: Dict[int, int]
                  , reverse: bool = False) -> Dict[str, int]:
    # return counts
    cRegList = list(measuredQRegsToCRegsDict.keys())
    cRegCount = len(measuredQRegsToCRegsDict)
    targetList = sorted(measuredQRegsToCRegsDict.values())
    for key in measuredQRegsToCRegsDict.keys():
        measuredQRegsToCRegsDict[key] = targetList.index(measuredQRegsToCRegsDict[key])

    sourceQRegCount = 0
    for key in counts.keys():
        sourceQRegCount = len(key)
        break

    assert sourceQRegCount > 0

    zeroKey = '0' * cRegCount
    binRet = {}  # type: Dict[str, int]
    for k, v in counts.items():
        hit = False
        for qReg in cRegList:
            if k[sourceQRegCount - 1 - qReg] == '1':
                hit = True
                break
        if hit:
            keyList = ['0'] * cRegCount
            for qReg in cRegList:
                keyList[cRegCount - 1 - measuredQRegsToCRegsDict[qReg]] = k[sourceQRegCount - 1 - qReg]
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


def formatMeasure(counts: Dict[str, int], cRegCount: int, mFormat: MeasureFormat = measureFormat) -> Dict[str, int]:
    ret = {}  # type: Dict[str, int]
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
