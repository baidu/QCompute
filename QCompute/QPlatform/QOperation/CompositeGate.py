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
Composite Gate Operation
"""
from typing import List, Optional, TYPE_CHECKING

import numpy

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterStorage
from QCompute.QPlatform.QOperation import QOperation

if TYPE_CHECKING:
    from QCompute.QPlatform.QOperation import RotationArgument, OperationFunc
    from QCompute.QPlatform.QRegPool import QRegStorage

FileErrorCode = 12


class CompositeGateOP(QOperation):
    """
    The composite gate. Used for users' convenience.

    The programmer can define more composite gate in this file.

    The real implementation of Composite Gate can be given in the file OpenModule/CompositeGateModule/__init__.py

    An example "RZZ" has been given in this class.
    """

    argumentList = None  # type: List['RotationArgument']

    def __init__(self, gate: str, bits: int, angleList: List['RotationArgument']) -> None:
        super().__init__(gate, bits)
        self.argumentList = angleList

    def __call__(self, *qRegList: 'QRegStorage') -> None:
        self._op(list(qRegList))

    def getInverse(self) -> 'CompositeGateOP':
        for argument in self.argumentList:
            if isinstance(argument, ProcedureParameterStorage):
                raise Error.ArgumentError(f'Can not inverse argument id. angles id: {argument.index}!', ModuleErrorCode,
                                          FileErrorCode, 1)

        nAngles = len(self.argumentList)
        if nAngles == 1:
            [theta] = self.argumentList  # type: float
        elif nAngles == 2:
            [theta, phi] = self.argumentList  # type: float
        elif nAngles == 3:
            [theta, phi, lamda] = self.argumentList  # type: float
        else:
            raise Error.ArgumentError(f'Wrong angles count. angles value: {self.argumentList}!', ModuleErrorCode,
                                      FileErrorCode, 2)

        if self.name == 'RZZ':
            return RZZ(theta, numpy.pi - lamda, numpy.pi - phi)


def RZZ(theta: 'RotationArgument',
        phi: Optional['RotationArgument'] = None,
        lamda: Optional['RotationArgument'] = None) -> 'OperationFunc':
    """
    RZZ(xyz)(Q0, Q1)

    =

    CX(Q0, Q1)

    U(xyz)(Q1)

    CX(Q0, Q1)
    """
    angleList = [value for value in [theta, phi, lamda] if value is not None]
    gate = CompositeGateOP('RZZ', 2, angleList)
    return gate
