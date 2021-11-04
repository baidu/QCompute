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
Customized Gate Operation
"""
from math import log2
from typing import TYPE_CHECKING

import numpy

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.QOperation import QOperation

if TYPE_CHECKING:
    from QCompute.QPlatform.QRegPool import QRegStorage

FileErrorCode = 11


class CustomizedGateOP(QOperation):
    """
    Customized gate

    The current version does not support arbitrary unitary as the lack of decomposition process.

    The user should not use this feature.
    """

    def __init__(self, matrix: numpy.ndarray) -> None:
        super().__init__(None, None, matrix)
        bits = log2(len(matrix))
        self.bits = int(bits)
        if bits != self.bits:
            raise Error.ArgumentError('bits must be an integer!', ModuleErrorCode, FileErrorCode, 1)

    def __call__(self, *qRegList: 'QRegStorage') -> None:
        self._op(list(qRegList))

    def getInverse(self) -> 'CustomizedGateOP':
        return CustomizedGateOP(numpy.linalg.pinv(self._matrix))
