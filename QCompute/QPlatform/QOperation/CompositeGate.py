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
Composite Gate Operation
"""
import importlib
import math
from typing import List, Optional, TYPE_CHECKING

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

    An example "MS" has been given in this class.
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
        if nAngles == 0:
            pass
        elif nAngles == 1:
            [theta] = self.argumentList  # type: List[float]
        elif nAngles == 2:
            [theta, phi] = self.argumentList  # type: List[float]
        elif nAngles == 3:
            [theta, phi, lamda] = self.argumentList  # type: List[float]
        else:
            raise Error.ArgumentError(f'Wrong angles count. angles value: {self.argumentList}!', ModuleErrorCode,
                                      FileErrorCode, 2)

        if self.name == 'MS':
            if nAngles == 0:
                return CompositeGateOP('MS', 2, [-math.pi / 2])
            else:
                return CompositeGateOP('MS', 2, [-theta])
        elif self.name == 'CK':
            return CompositeGateOP('CK', 2, [-theta])
        else:
            raise Error.ArgumentError(f'Unsupported inverse {self.name}!', ModuleErrorCode,
                                      FileErrorCode, 3)


_MSNotified = False


def MS(theta: Optional['RotationArgument'] = None) -> 'OperationFunc':
    """
    MS()(Q0, Q1)
    MS(theta)(Q0, Q1)
    """
    global _MSNotified
    if not _MSNotified:
        _MSNotified = True
        print('**Attention** MS gate is a two qubit gate used in trapped ion quantum computing.\n'
              'We support MS as a native gate only when the target device type is an ion trap.\n'
              'In other cases, MS is a composite gate on platform.\n')

    angleList = [value for value in [theta] if value is not None]
    gate = CompositeGateOP('MS', 2, angleList)
    return gate


_CKNotified = False


def CK(kappa: 'RotationArgument') -> 'OperationFunc':
    """
    CK(kappa)(Q0, Q1)
    """
    global _CKNotified
    if not _CKNotified:
        _CKNotified = True
        print('**Attention** CK gate is a two qubit gate used in optical quantum computing.\n'
              'We support CK as a native gate only when the target device type is optical.\n'
              'In other cases, CK is a composite gate on platform.\n')

    gate = CompositeGateOP('CK', 2, [kappa])
    return gate


def createCompositeGateInstance(name: str, *angles: 'RotationArgument') -> 'OperationFunc':
    """
    Create a new gate according to name and angles

    :param name : rotation gate name
    :param angles: angle param list
    :return: new gate
    """

    currentModule = importlib.import_module(__name__)
    gateClass = getattr(currentModule, name)
    gate = gateClass(*angles)
    return gate

# removed. only example
# def RZZ(theta: 'RotationArgument',
#         phi: Optional['RotationArgument'] = None,
#         lamda: Optional['RotationArgument'] = None) -> 'OperationFunc':
#     """
#     RZZ(xyz)(Q0, Q1)
#
#     =
#
#     CX(Q0, Q1)
#
#     U(xyz)(Q1)
#
#     CX(Q0, Q1)
#     """
#     angleList = [value for value in [theta, phi, lamda] if value is not None]
#     gate = CompositeGateOP('RZZ', 2, angleList)
#     return gate
