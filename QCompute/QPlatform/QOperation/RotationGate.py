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
Rotation Gate Operation
"""
import importlib
from typing import List, Optional, TYPE_CHECKING

import numpy

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterStorage
from QCompute.QPlatform.QOperation import QOperation
from QCompute.QPlatform.QRegPool import QRegStorage

if TYPE_CHECKING:
    from QCompute.QPlatform.QOperation import RotationArgument, OperationFunc

FileErrorCode = 10


class RotationGateOP(QOperation):
    """
    Rotation gate

    Use rotation parameters to create the quantum operators
    """
    argumentList = None  # type: List['RotationArgument']
    uGateArgumentList = None  # type: List['RotationArgument']

    def __init__(self, gate: str, bits: int,
                 angleList: List['RotationArgument'],
                 uGateArgumentList: List['RotationArgument']) -> None:
        super().__init__(gate, bits)
        self.argumentList = angleList
        self.uGateArgumentList = uGateArgumentList

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

        self._matrix = numpy.array([[numpy.cos(theta / 2.0), -numpy.exp(1j * lamda) * numpy.sin(theta / 2.0)],
                                    [numpy.exp(1j * phi) * numpy.sin(theta / 2.0),
                                     numpy.exp(1j * lamda + 1j * phi) * numpy.cos(theta / 2.0)]])
        return self._matrix

    def _u2Matrix(self, phi: float, lamda: float) -> numpy.ndarray:
        """
        Generate a single-qubit rotation gate with 2 angles

        :param phi: angle
        :param lamda: angle
        :return: U2 matrix
        """

        self._matrix = (1 / numpy.sqrt(2)) * numpy.array([[1, -numpy.exp(1j * lamda)],
                                                          [numpy.exp(1j * phi), numpy.exp(1j * (phi + lamda))]])
        return self._matrix

    def _u1Matrix(self, lamda: float) -> numpy.ndarray:
        """
        Generate a single-qubit rotation gate along the Z-axis

        :param lamda: angle
        :return: U1 matrix
        """

        self._matrix = numpy.array([[1, 0],
                                    [0, numpy.exp(1j * lamda)]])
        return self._matrix

    def _cu3Matrix(self, theta: float, phi: float, lamda: float) -> numpy.ndarray:
        self._matrix = numpy.kron(numpy.eye(2),
                                  numpy.array([[1, 0],
                                               [0, 0]])
                                  ) + \
                       numpy.kron(self._u3Matrix(theta, phi, lamda),
                                  numpy.array([[0, 0],
                                               [0, 1]])
                                  )
        return self._matrix

    def _generateUMatrix(self) -> None:
        uGateArgumentCount = len(
            [value for value in self.uGateArgumentList if isinstance(value, (float, int))])
        if uGateArgumentCount != len(self.uGateArgumentList):
            pass  # has parameter
        elif uGateArgumentCount == 3:
            self._u3Matrix(*self.uGateArgumentList)
        elif uGateArgumentCount == 2:
            self._u2Matrix(*self.uGateArgumentList)
        elif uGateArgumentCount == 1:
            self._u1Matrix(*self.uGateArgumentList)

    def _generateCUMatrix(self) -> None:
        uGateArgumentCount = len([value for value in self.uGateArgumentList if isinstance(value, (float, int))])
        if uGateArgumentCount != len(self.uGateArgumentList):
            pass  # has parameter
        elif uGateArgumentCount == 3:
            self._cu3Matrix(*self.uGateArgumentList)
        # elif uGateArgumentCount == 2:
        #     self._cu2Matrix(*self.uGateArgumentList)
        # elif uGateArgumentCount == 1:
        #     self._cu1Matrix(*self.uGateArgumentList)

    def getInverse(self) -> 'RotationGateOP':
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
                angles = [-self.argumentList[0]]
            elif nAngles == 2:
                angles = [numpy.pi - self.argumentList[1], numpy.pi - self.argumentList[0]]
            elif nAngles == 3:
                angles = [self.argumentList[0], numpy.pi - self.argumentList[2], numpy.pi - self.argumentList[1]]
            else:
                raise Error.ArgumentError(f'Wrong angles count. angles: {self.argumentList}!', ModuleErrorCode,
                                          FileErrorCode, 3)
            return U(*angles)
        elif self.name == 'CU':
            return CU(theta, numpy.pi - lamda, numpy.pi - phi)


def U(theta: 'RotationArgument',
      phi: Optional['RotationArgument'] = None,
      lamda: Optional['RotationArgument'] = None) -> 'OperationFunc':
    """
    U Gate

    Generate a single-qubit U1 (or U2 or U3) gate according to the number of angles.

    The reason is any single-qubit operator can be fully identified with three angles.
    """
    uGateArgumentList = angleList = [value for value in [theta, phi, lamda] if value is not None]
    gate = RotationGateOP('U', 1, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateUMatrix()
    return gate


def RX(theta: 'RotationArgument') -> 'OperationFunc':
    """
    RX Gate

    Single-qubit rotation about the X-axis.

    According to the relation: U3(theta, -pi/2, pi/2) = RX(theta)
    """
    angleList = [theta]
    uGateArgumentList = [theta, -numpy.math.pi / 2, numpy.math.pi / 2]
    gate = RotationGateOP('RX', 1, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateUMatrix()
    return gate


def RY(theta: 'RotationArgument') -> 'OperationFunc':
    """
    RY Gate

    Single-qubit rotation about the Y-axis.

    According to the relation: U3(theta, 0, 0) = RY(theta)
    """
    angleList = [theta]
    uGateArgumentList = [theta, 0, 0]
    gate = RotationGateOP('RY', 1, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateUMatrix()
    return gate


def RZ(lamda: 'RotationArgument') -> 'OperationFunc':
    """
    RZ Gate

    Single-qubit rotation about the Z-axis.

    According to the relation: U3(0, 0, lamda) = RZ(lamda)
    """
    angleList = [lamda]
    uGateArgumentList = [0, 0, lamda]
    gate = RotationGateOP('RZ', 1, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateUMatrix()
    return gate


def CU(theta: 'RotationArgument',
       phi: 'RotationArgument',
       lamda: 'RotationArgument') -> 'OperationFunc':
    """
    Controlled-rotation gate

    It contains two qubits: the control qubit and the target qubit.

    The rotation gate is performed on the target qubit only when the control qubit is taking effect.
    """
    uGateArgumentList = angleList = [theta, phi, lamda]
    gate = RotationGateOP('CU', 2, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateCUMatrix()
    return gate


def CRX(theta: 'RotationArgument') -> 'OperationFunc':
    """
    Controlled-RX gate
    """
    angleList = [theta]
    uGateArgumentList = [theta, -numpy.math.pi / 2, numpy.math.pi / 2]
    gate = RotationGateOP('CRX', 2, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateCUMatrix()
    return gate


def CRY(theta: 'RotationArgument') -> 'OperationFunc':
    """
    Controlled-RY Gate
    """
    angleList = [theta]
    uGateArgumentList = [theta, 0, 0]
    gate = RotationGateOP('CRY', 2, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateCUMatrix()
    return gate


def CRZ(lamda: 'RotationArgument') -> 'OperationFunc':
    """
    Controlled-RZ Gate
    """
    angleList = [lamda]
    uGateArgumentList = [0, 0, lamda]
    gate = RotationGateOP('CRZ', 2, angleList, uGateArgumentList)
    gate.generateMatrix = gate._generateCUMatrix()
    return gate


def createRotationGateInstance(name: str, *angles: 'RotationArgument') -> 'OperationFunc':
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
