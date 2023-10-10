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
from QCompute.QPlatform.ProcedureParameterExpression import ProcedureParameterExpression

FileErrorCode = 34

import importlib
import math
from typing import List, Optional, TYPE_CHECKING

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterStorage
from QCompute.QPlatform.QOperation import QOperation

if TYPE_CHECKING:
    from QCompute.QPlatform.QOperation import RotationArgument, OperationFunc
    from QCompute.QPlatform.QRegPool import QRegStorage


class CompositeGateOP(QOperation):
    """
    The composite gate. Used for users' convenience.

    The programmer can define more composite gate in this file.

    The real implementation of Composite Gate can be given in the file OpenModule/CompositeGateModule/__init__.py
    """

    argumentList: List['RotationArgument'] = None

    def __init__(self, gate: str, bits: int, allowArgumentCounts: List[int],
                 angleList: List['RotationArgument']) -> None:
        super().__init__(gate, bits)
        if len(angleList) not in allowArgumentCounts:
            raise Error.ArgumentError(f'allowArgumentCounts is not len(angleList)!', ModuleErrorCode, FileErrorCode, 1)

        self.allowArgumentCounts = allowArgumentCounts
        self.argumentList = angleList

    def __call__(self, *qRegList: 'QRegStorage') -> None:
        self._op(list(qRegList))

    def getInversed(self) -> 'CompositeGateOP':
        nAngles = len(self.argumentList)
        if nAngles == 0:
            pass
        elif nAngles == 1:
            [theta] = self.argumentList
        # elif nAngles == 2:
        #     [theta, phi] = self.argumentList
        # elif nAngles == 3:
        #     [theta, phi, lamda] = self.argumentList
        else:
            raise Error.ArgumentError(
                f'Wrong angles count. angles value: {self.argumentList}!', ModuleErrorCode, FileErrorCode, 2)

        if self.name == 'MS':
            if nAngles == 0:
                return MS(-math.pi / 2)
            else:
                return MS(-theta)
        elif self.name == 'CK':
            # kappa = theta
            return CK(-theta)
        else:
            raise Error.ArgumentError(f'Unsupported inverse {self.name}!', ModuleErrorCode, FileErrorCode, 3)

MSOpAllowArgumentCounts = [0, 1]
CKOpAllowArgumentCounts = [1]

_MSNotified = False


def MS(theta: Optional['RotationArgument'] = None) -> 'OperationFunc':
    r"""
    Native two-qubit gate in trapped ion quantum computing, and can create maximum entangle state
    by one step. For example:

    :math:`MS|00\rangle = \frac{1}{\sqrt 2}(|00\rangle - i|11\rangle)`

    Matrix form:

    :math:`MS = \frac{1}{\sqrt 2} \begin{bmatrix} 1 & 0  &0 & -i \\ 0 & 1  & -i& 0 \\ 0 & -i  & 1& 0 \\ -i & 0  & 0& 1 \end{bmatrix}`

    Sometimes, the MS gate can hold a rotating parameter which depend on laser devices property
    
    :param theta: the two qubits rotating angle.

    :type theta: Optional[Union[int, float, ProcedureParameterStorage]]

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
    If two channels :math:`i` and :math:`j` have incident photons, then the emergent photons will change a phase

    :param kappa: The nonlinear crystal strength.

    :type kappa: Union[int, float, ProcedureParameterStorage]

    Matrix form:

    :math:`CK|n_in_j\rangle = e^{i \kappa n_i \times n_j} |n_i n_j\rangle`

    Code example:

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
    Create a new gate according to name and angles.

    :param name: Rotation gate name.

    :type name: str

    :param angles: Angle parameter list.

    :type angles: List[Union[int, float, ProcedureParameterStorage]]
    
    :return: new gate.
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