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
Photonic Condition
"""
FileErrorCode = 15

"""
所有均在2g内存下进行测试，量子门对量子态的变换本身对内存没影响，只会影响运行时间；Gaussian态和Fock态框架需要设置3点限制：
* qumode数最大为12；
* cutoff数最大为8；
* 运行时间最大为900秒；
"""

from QCompute.QProtobuf import PBPhotonicGaussianMeasure

_gaussianConditionDict = {
    1: 8,
    2: 8,
    3: 8,
    4: 8,
    5: 3,
    6: 3,
    7: 2,
    8: 1,
    9: 1,
    10: 1,
    11: 1,
    12: 1,
}


def gaussianCondition(measureType: PBPhotonicGaussianMeasure.Type, qumode: int, cutoff: int) -> bool:
    if measureType in [PBPhotonicGaussianMeasure.Homodyne, PBPhotonicGaussianMeasure.Heterodyne]:
        return True
    elif measureType == PBPhotonicGaussianMeasure.PhotonCount:
        if qumode in _gaussianConditionDict and _gaussianConditionDict[qumode] >= cutoff:
            return True
    return False


_fockConditionDict = {
    1: 8,
    2: 8,
    3: 8,
    4: 8,
    5: 8,
    6: 8,
    7: 8,
    8: 8,
    9: 7,
    10: 6,
    11: 5,
    12: 4,
}


def fockCondition(qumode: int, cutoff: int) -> bool:
    if qumode in _fockConditionDict and _fockConditionDict[qumode] >= cutoff:
        return True
    return False