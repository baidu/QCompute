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

    argumentList: List['RotationArgument'] = None

    def __init__(self, gate: str, bits: int, allowArgumentCounts: List[int],
                 angleList: List['RotationArgument']) -> None:
        super().__init__(gate, bits)
        if len(angleList) not in allowArgumentCounts:
            raise Error.ArgumentError(f'allowArgumentCounts is not len(angleList)!',
                                      ModuleErrorCode, FileErrorCode, 1)
        self.allowArgumentCounts = allowArgumentCounts
        self.argumentList = angleList

    def __call__(self, *qRegList: 'QRegStorage') -> None:
        self._op(list(qRegList))

    def getInversed(self) -> 'CompositeGateOP':
        for argument in self.argumentList:
            if isinstance(argument, ProcedureParameterStorage):
                raise Error.ArgumentError(f'Can not inverse argument id. angles id: {argument.index}!', ModuleErrorCode,
                                          FileErrorCode, 2)

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
                                      FileErrorCode, 3)

        if self.name == 'MS':
            if nAngles == 0:
                return CompositeGateOP('MS', 2, MSOpAllowArgumentCounts, [-math.pi / 2])
            else:
                return CompositeGateOP('MS', 2, MSOpAllowArgumentCounts, [-theta])
        elif self.name == 'CK':
            return CompositeGateOP('CK', 2, CKOpAllowArgumentCounts, [-theta])
        else:
            raise Error.ArgumentError(f'Unsupported inverse {self.name}!', ModuleErrorCode,
                                      FileErrorCode, 4)


MSOpAllowArgumentCounts = [0, 1]
CKOpAllowArgumentCounts = [1]

_MSNotified = False


def MS(theta: Optional['RotationArgument'] = None) -> 'OperationFunc':
    r"""
    Native two-qubit gate in trapped ion quantum computing, and can create maximum entangle state
    by one step. For example:

    :math: `MS|00>\rangle = \frac{1}{\sqrt 2}(|00\rangle - i|11\rangle)`

    Matrix form:

    :math:`MS = \frac{1}{\sqrt 2} \begin{bmatrix} 1 & 0  &0 & -i \\ 0 & 1  & -i& 0 \\ 0 & -i  & 1& 0 \\ -i & 0  & 0& 1 \end{bmatrix}`

    Sometimes, the MS gate can hold a rotating parameter which depend on laser devices property
    
    :param theta: the two qubits rotating angle

    Matrix form:
    
    :math:`MS = \begin{bmatrix} \cos\theta & 0  &0 & -i\sin\theta \\ 0 & \cos\theta  & -i\sin\theta & 0 \\ 0 & -i\sin\theta  & \cos\theta& 0 \\ -i\sin\theta & 0  & 0& \cos\theta \end{bmatrix}`

    MS(theta)(Q0, Q1)
    """
    global _MSNotified
    if not _MSNotified:
        _MSNotified = True
        print('**Attention** MS gate is a two qubit gate used in trapped ion quantum computing.\n'
              'We support MS as a native gate only when the target device type is an ion trap.\n'
              'In other cases, MS is a composite gate on platform.\n')

    angleList = [value for value in [theta] if value is not None]
    gate = CompositeGateOP('MS', 2, MSOpAllowArgumentCounts, angleList)
    return gate


MS.type = 'CompositeGateOP'
MS.allowArgumentCounts = MSOpAllowArgumentCounts

_CKNotified = False


def CK(kappa: 'RotationArgument') -> 'OperationFunc':
    r"""
    The cross-Kerr (CK) is a nonlinear crystal that can act on two incident photons.
    If two channels $i$ and $j$ have incident photons, then the emergent photons will change a phase

    :param kappa: the nonlinear crystal strength

    Matrix form:

    :math:`CK|n_in_j\rangle = \exp(i\kappa n_i\times n_j)|n_in_j\rangle`

    CK(kappa)(Q0, Q1)
    """
    global _CKNotified
    if not _CKNotified:
        _CKNotified = True
        print('**Attention** CK gate is a two qubit gate used in optical quantum computing.\n'
              'We support CK as a native gate only when the target device type is optical.\n'
              'In other cases, CK is a composite gate on platform.\n')

    gate = CompositeGateOP('CK', 2, CKOpAllowArgumentCounts, [kappa])
    return gate


CK.type = 'CompositeGateOP'
CK.allowArgumentCounts = CKOpAllowArgumentCounts


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
