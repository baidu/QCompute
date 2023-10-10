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
Rotation Gate Operation
"""
from QCompute.QPlatform.ProcedureParameterExpression import ProcedureParameterExpression

FileErrorCode = 39

import importlib
from typing import List, Optional, TYPE_CHECKING

import numpy

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterStorage
from QCompute.QPlatform.QOperation import QOperation
from QCompute.QPlatform.QRegPool import QRegStorage

if TYPE_CHECKING:
    from QCompute.QPlatform.QOperation import RotationArgument, OperationFunc


class RotationGateOP(QOperation):
    """
    Rotation gate.

    Use rotation parameters to create the quantum operators.
    """

    def __init__(self, gate: str, bits: int, allowArgumentCounts: List[int],
                 angleList: List['RotationArgument'],
                 uGateArgumentList: List['RotationArgument']) -> None:
        super().__init__(gate, bits)
        if len(angleList) not in allowArgumentCounts:
            raise Error.ArgumentError(f'allowArgumentCounts is not len(angleList)!', ModuleErrorCode, FileErrorCode, 1)

        self.allowArgumentCounts = allowArgumentCounts
        self.argumentList: List['RotationArgument'] = angleList
        self.uGateArgumentList: List['RotationArgument'] = uGateArgumentList

    def __call__(self, *qRegList: QRegStorage) -> None:
        self._op(list(qRegList))

    def generateMatrix(self) -> numpy.ndarray:
        pass

    def _u3Matrix(self, theta: float, phi: float, lamda: float) -> numpy.ndarray:
        """
        Generate a single-qubit rotation gate with 3 angles

        :param theta: angle

        :param phi: angle

        :param lamda: angle

        :return: U3 matrix
        """

        self.matrix = numpy.array([[numpy.cos(theta / 2.0), -numpy.exp(1j * lamda) * numpy.sin(theta / 2.0)],
                                   [numpy.exp(1j * phi) * numpy.sin(theta / 2.0),
                                    numpy.exp(1j * lamda + 1j * phi) * numpy.cos(theta / 2.0)]])
        return self.matrix

    def _u2Matrix(self, phi: float, lamda: float) -> numpy.ndarray:
        """
        Generate a single-qubit rotation gate with 2 angles

        :param phi: angle

        :param lamda: angle

        :return: U2 matrix
        """

        self.matrix = (1 / numpy.sqrt(2)) * numpy.array([[1, -numpy.exp(1j * lamda)],
                                                         [numpy.exp(1j * phi), numpy.exp(1j * (phi + lamda))]])
        return self.matrix

    def _u1Matrix(self, lamda: float) -> numpy.ndarray:
        """
        Generate a single-qubit rotation gate along the Z-axis

        :param lamda: angle

        :return: U1 matrix
        """

        self.matrix = numpy.array([[1, 0],
                                   [0, numpy.exp(1j * lamda)]])
        return self.matrix

    def _cu3Matrix(self, theta: float, phi: float, lamda: float) -> numpy.ndarray:
        self.matrix = numpy.kron(numpy.eye(2),
                                 numpy.array([[1, 0],
                                              [0, 0]])
                                 ) + \
                      numpy.kron(self._u3Matrix(theta, phi, lamda),
                                 numpy.array([[0, 0],
                                              [0, 1]])
                                 )
        return self.matrix

    def _generateUMatrix(self) -> numpy.ndarray:
        uGateArgumentCount = len(
            [value for value in self.uGateArgumentList if isinstance(value, (float, int))])
        if uGateArgumentCount != len(self.uGateArgumentList):
            pass  # has parameter
        elif uGateArgumentCount == 3:
            return self._u3Matrix(*self.uGateArgumentList)
        elif uGateArgumentCount == 2:
            return self._u2Matrix(*self.uGateArgumentList)
        elif uGateArgumentCount == 1:
            return self._u1Matrix(*self.uGateArgumentList)

    def _generateCUMatrix(self) -> numpy.ndarray:
        uGateArgumentCount = len([value for value in self.uGateArgumentList if isinstance(value, (float, int))])
        if uGateArgumentCount != len(self.uGateArgumentList):
            pass  # has parameter
        elif uGateArgumentCount == 3:
            return self._cu3Matrix(*self.uGateArgumentList)
        # elif uGateArgumentCount == 2:
        #     return self._cu2Matrix(*self.uGateArgumentList)
        # elif uGateArgumentCount == 1:
        #     return self._cu1Matrix(*self.uGateArgumentList)

    def getInversed(self) -> 'RotationGateOP':
        nAngles = len(self.argumentList)
        if nAngles == 1:
            [theta] = self.argumentList
        elif nAngles == 2:
            [theta, phi] = self.argumentList
        elif nAngles == 3:
            [theta, phi, lamda] = self.argumentList
        else:
            raise Error.ArgumentError(
                f'Wrong angles count! angles value: {self.argumentList}.',
                ModuleErrorCode, FileErrorCode, 2)

        if self.name == 'RX':
            return RX(-theta)
        elif self.name == 'RY':
            return RY(-theta)
        elif self.name == 'RZ':
            return RZ(-theta)
        elif self.name == 'CRX':
            return CRX(-theta)
        elif self.name == 'CRY':
            return CRY(-theta)
        elif self.name == 'CRZ':
            return CRZ(-theta)
        elif self.name == 'U':
            if nAngles == 1:
                angles = [-theta]
            elif nAngles == 2:
                angles = [numpy.pi - phi, numpy.pi - theta]
            elif nAngles == 3:
                angles = [theta, numpy.pi - lamda, numpy.pi - phi]
            else:
                raise Error.ArgumentError(
                    f'Wrong angles count! angles: {self.argumentList}.',
                    ModuleErrorCode, FileErrorCode, 3)

            return U(*angles)
        elif self.name == 'CU':
            return CU(theta, numpy.pi - lamda, numpy.pi - phi)


UOpAllowArgumentCounts = [1, 2, 3]
RXOpAllowArgumentCounts = [1]
RYOpAllowArgumentCounts = [1]
RZOpAllowArgumentCounts = [1]
CUOpAllowArgumentCounts = [3]
CRXOpAllowArgumentCounts = [1]
CRYOpAllowArgumentCounts = [1]
CRZOpAllowArgumentCounts = [1]


def U(theta: 'RotationArgument',
      phi: Optional['RotationArgument'] = None,
      lamda: Optional['RotationArgument'] = None) -> 'OperationFunc':
    """
    U Gate.

    Generate a single-qubit U1 (or U2 or U3) gate according to the number of angles.

    The reason is any single-qubit operator can be fully identified with three angles.
    """
    uGateArgumentList = angleList = [value for value in [theta, phi, lamda] if value is not None]
    gate = RotationGateOP('U', 1, UOpAllowArgumentCounts, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateUMatrix
    return gate


U.type = 'RotationGateOP'
U.allowArgumentCounts = UOpAllowArgumentCounts


def RX(theta: 'RotationArgument') -> 'OperationFunc':
    """
    RX Gate.

    Single-qubit rotation about the X-axis.

    According to the relation: U3(theta, -pi/2, pi/2) = RX(theta)
    """
    angleList = [theta]
    uGateArgumentList = [theta, -numpy.math.pi / 2, numpy.math.pi / 2]
    gate = RotationGateOP('RX', 1, RXOpAllowArgumentCounts, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateUMatrix
    return gate


RX.type = 'RotationGateOP'
RX.allowArgumentCounts = RXOpAllowArgumentCounts


def RY(theta: 'RotationArgument') -> 'OperationFunc':
    """
    RY Gate.

    Single-qubit rotation about the Y-axis.

    According to the relation: U3(theta, 0, 0) = RY(theta)
    """
    angleList = [theta]
    uGateArgumentList = [theta, 0, 0]
    gate = RotationGateOP('RY', 1, RYOpAllowArgumentCounts, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateUMatrix
    return gate


RY.type = 'RotationGateOP'
RY.allowArgumentCounts = RYOpAllowArgumentCounts


def RZ(lamda: 'RotationArgument') -> 'OperationFunc':
    """
    RZ Gate.

    Single-qubit rotation about the Z-axis.

    According to the relation: U3(0, 0, lamda) = RZ(lamda)
    """
    angleList = [lamda]
    uGateArgumentList = [0, 0, lamda]
    gate = RotationGateOP('RZ', 1, RZOpAllowArgumentCounts, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateUMatrix
    return gate


RZ.type = 'RotationGateOP'
RZ.allowArgumentCounts = RZOpAllowArgumentCounts


def CU(theta: 'RotationArgument',
       phi: 'RotationArgument',
       lamda: 'RotationArgument') -> 'OperationFunc':
    """
    Controlled-rotation gate.

    It contains two qubits: the control qubit and the target qubit.

    The rotation gate is performed on the target qubit only when the control qubit is taking effect.
    """
    uGateArgumentList = angleList = [theta, phi, lamda]
    gate = RotationGateOP('CU', 2, CUOpAllowArgumentCounts, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateCUMatrix
    return gate


CU.type = 'RotationGateOP'
CU.allowArgumentCounts = CUOpAllowArgumentCounts


def CRX(theta: 'RotationArgument') -> 'OperationFunc':
    """
    Controlled-RX gate.
    """
    angleList = [theta]
    uGateArgumentList = [theta, -numpy.math.pi / 2, numpy.math.pi / 2]
    gate = RotationGateOP('CRX', 2, CRXOpAllowArgumentCounts, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateCUMatrix
    return gate


CRX.type = 'RotationGateOP'
CRX.allowArgumentCounts = CRXOpAllowArgumentCounts


def CRY(theta: 'RotationArgument') -> 'OperationFunc':
    """
    Controlled-RY Gate.
    """
    angleList = [theta]
    uGateArgumentList = [theta, 0, 0]
    gate = RotationGateOP('CRY', 2, CRYOpAllowArgumentCounts, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateCUMatrix
    return gate


CRY.type = 'RotationGateOP'
CRY.allowArgumentCounts = CRYOpAllowArgumentCounts


def CRZ(lamda: 'RotationArgument') -> 'OperationFunc':
    """
    Controlled-RZ Gate.
    """
    angleList = [lamda]
    uGateArgumentList = [0, 0, lamda]
    gate = RotationGateOP('CRZ', 2, CRZOpAllowArgumentCounts, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateCUMatrix
    return gate


CRZ.type = 'RotationGateOP'
CRZ.allowArgumentCounts = CRZOpAllowArgumentCounts


def createRotationGateInstance(name: str, *angles: 'RotationArgument') -> 'OperationFunc':
    """
    Create a new gate according to name and angles.

    :param name: rotation gate name

    :param angles: angle param list

    :return: new gate
    """

    currentModule = importlib.import_module(__name__)
    gateClass = getattr(currentModule, name)
    gate = gateClass(*angles)
    return gate