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

import numpy

from QCompute.QuantumPlatform.QuantumOperation import QuantumOperation
from QCompute.QuantumPlatform.Utilities import _mergePBList
from QCompute.QuantumProtobuf.Library.PlatformStruct_pb2 import CircuitLine
from QCompute.QuantumProtobuf.Library.QuantumOperation_pb2 import RotationGate as RotationGateEnum


class RotationGate(QuantumOperation):
    """
    Rotation gate

    Use rotation parameters to create the quantum operators
    """

    def __init__(self):
        self.angles = []  # rotation parameter values
        self.procedureParams = []  # placeholder parameter No.
        self.uGateParams = None  # compute universal rotation of U gate, provided that rotation parameters are complete
        self.matrix = None  # compute underlying matrix, provided that rotation parameters are complete

    def _setAngles(self, *angles):
        """
        Set angles and procedureParams
        """

        for angle in angles:
            if isinstance(angle, (int, float)):
                self.angles.append(angle)
                self.procedureParams.append(-1)
            else:
                self.angles.append(0)
                self.procedureParams.append(angle.index)  # ProcedureParamStorage
        if all([param != -1 for param in self.procedureParams]):
            self.angles = None
        if all([param == -1 for param in self.procedureParams]):
            self.procedureParams = None

    def _u3Matrix(self, theta, phi, lamda):
        """
        Generate a single-qubit rotation gate with 3 angles

        :param theta: angle
        :param phi: angle
        :param lamda: angle
        :return: U3 matrix
        """

        self.uGateParams = (theta, phi, lamda)  # record the rotation angle
        return numpy.array([[numpy.cos(theta / 2.0), -numpy.exp(1j * lamda) * numpy.sin(theta / 2.0)],
                            [numpy.exp(1j * phi) * numpy.sin(theta / 2.0),
                             numpy.exp(1j * lamda + 1j * phi) * numpy.cos(theta / 2.0)]])

    def _u2Matrix(self, phi, lamda):
        """
        Generate a single-qubit rotation gate with 2 angles

        :param phi: angle
        :param lamda: angle
        :return: U2 matrix
        """

        self.uGateParams = (phi, lamda)  # record the rotation angle
        return (1 / numpy.sqrt(2)) * numpy.array([[1, -numpy.exp(1j * lamda)],
                                                  [numpy.exp(1j * phi), numpy.exp(1j * (phi + lamda))]])

    def _u1Matrix(self, lamda):
        """
        Generate a single-qubit rotation gate along the Z-axis

        :param lamda: angle
        :return: U1 matrix
        """

        self.uGateParams = (lamda,)  # record the rotation angle
        return numpy.array([[1, 0],
                            [0, numpy.exp(1j * lamda)]])

    def _toPB(self, *qRegsIndex):
        """
        Convert to Protobuf object

        :param qRegsIndex: the quantum register list used in creating single circuit object
        :return: Protobuf object
        """

        ret = CircuitLine()

        if len(qRegsIndex) == 0:  # fill in the register list
            # The circuit object is already in the Python env.
            # Directly generate the circuit in Protobuf format according to member variables.
            for reg in self.qRegs:
                ret.qRegs.append(reg.index)
        else:
            # Insert the new circuit object in the module process.
            # Generate a Protobuf circuit according to parameters.
            _mergePBList(ret.qRegs, qRegsIndex)

        assert len(ret.qRegs) == self.bits  # The number of quantum registers of operators in circuit must conform to the setting.

        qRegSet = set(qReg for qReg in ret.qRegs)
        assert len(ret.qRegs) == len(qRegSet)  # Quantum registers should not be duplicated.

        ret.rotationGate = RotationGateEnum.Value(self.Gate)  # fill in the name of the rotation gate
        _mergePBList(ret.paramValues, self.angles)  # fill in rotation angles
        _mergePBList(ret.paramIds, self.procedureParams)  # fill in procedure parameters
        return ret


class U(RotationGate):
    """
    U Gate

    Generate a single-qubit U1 (or U2 or U3) gate according to the number of angles.

    The reason is any single-qubit operator can be fully identified with three angles.
    """

    Gate = 'U'

    def __init__(self, *angles):
        super().__init__()
        self.bits = 1

        self.theta = angles[0]
        if len(angles) >= 2:
            self.phi = angles[1]
        if len(angles) == 3:
            self.lamda = angles[2]

        self._setAngles(*angles)  # fill in rotation gate parameters
        self._generateMatrix(*angles)

    def _generateMatrix(self, *angles):
        """
        Generate a matrix according to angles
        :param angles: angle list
        """

        if self.procedureParams is not None:
            return

        if len(angles) == 1:
            self.matrix = self._u1Matrix(*angles)
        elif len(angles) == 2:
            self.matrix = self._u2Matrix(*angles)
        elif len(angles) == 3:
            self.matrix = self._u3Matrix(*angles)


class _R(RotationGate):
    """
    RX RY RZ base class
    """

    Gate = None  # User can't use.

    def __init__(self):
        super().__init__()
        self.bits = 1

    def _generateMatrix(self, theta, phi, lamda):
        """
        Generate the matrix according to three angles
        """

        if self.procedureParams is not None:
            return

        self.matrix = self._u3Matrix(theta, phi, lamda)  # fill in the generated matrix directly


class RX(_R):
    """
    RX Gate

    Single-qubit rotation about the X-axis.

    According to the relation: U3(theta, -pi/2, pi/2) = RX(theta)
    """

    Gate = 'RX'
    phi = -numpy.math.pi / 2
    lamda = numpy.math.pi / 2

    def __init__(self, theta):
        self.theta = theta
        super().__init__()
        self._setAngles(theta)  # fill in the rotation gate parameters
        self._generateMatrix(self.theta, self.phi, self.lamda)


class RY(_R):
    """
    RY Gate

    Single-qubit rotation about the Y-axis.

    According to the relation: U3(theta, 0, 0) = RY(theta)
    """

    Gate = 'RY'
    phi = 0
    lamda = 0

    def __init__(self, theta):
        self.theta = theta
        super().__init__()
        self._setAngles(theta)  # fill in the rotation gate parameters
        self._generateMatrix(self.theta, self.phi, self.lamda)


class RZ(_R):
    """
    RZ Gate

    Single-qubit rotation about the Z-axis.

    According to the relation: U3(0, 0, lamda) = RZ(lamda)
    """

    Gate = 'RZ'
    theta = 0
    phi = 0

    def __init__(self, lamda):
        self.lamda = lamda
        super().__init__()
        self._setAngles(lamda)  # fill in the rotation gate parameters
        self._generateMatrix(self.theta, self.phi, self.lamda)


class CU(RotationGate):
    """
    Controlled-rotation gate

    It contains two qubits: the control qubit and the target qubit.

    The rotation gate is performed on the target qubit only when the control qubit is taking effect.
    """

    Gate = 'CU'

    def __init__(self, theta, phi, lamda):
        super().__init__()
        self.bits = 2
        if type(self) == CU:
            self.theta = theta
            self.phi = phi
            self.lamda = lamda

            self._setAngles(theta, phi, lamda)  # fill in rotation gate parameters
            self._generateMatrix(theta, phi, lamda)

    def _generateMatrix(self, theta, phi, lamda):
        """
        Generate matrix
        """

        if self.procedureParams is not None:
            return

        self.matrix = numpy.kron(numpy.eye(2),
                                 numpy.array([[1, 0],
                                              [0, 0]])
                                 ) + \
                      numpy.kron(self._u3Matrix(theta, phi, lamda),
                                 numpy.array([[0, 0],
                                              [0, 1]])
                                 )


class CRX(CU):
    """
    Controlled-RX gate
    """

    Gate = 'CRX'
    phi = -numpy.math.pi / 2
    lamda = numpy.math.pi / 2

    def __init__(self, theta):
        self.theta = theta
        super().__init__(None, None, None)
        self._setAngles(theta)  # fill in the rotation gate parameters
        self._generateMatrix(self.theta, self.phi, self.lamda)


class CRY(CU):
    """
    Controlled-RY Gate
    """

    Gate = 'CRY'
    phi = 0
    lamda = 0

    def __init__(self, theta):
        self.theta = theta
        super().__init__(None, None, None)
        self._setAngles(theta)  # fill in the rotation gate parameters
        self._generateMatrix(self.theta, self.phi, self.lamda)


class CRZ(CU):
    """
    Controlled-RZ Gate
    """

    Gate = 'CRZ'
    theta = 0
    phi = 0

    def __init__(self, lamda):
        self.lamda = lamda
        super().__init__(None, None, None)
        self._setAngles(lamda)  # fill in the rotation gate parameters
        self._generateMatrix(self.theta, self.phi, self.lamda)


def createGate(name, *angles):
    """
    Create a new gate according to name and angles

    :param name : rotation gate name
    :param angles: angle param list
    :return: new gate
    """

    # import sys
    # currentModule = sys.modules[__name__]
    currentModule = __import__(__name__)
    gateClass = getattr(currentModule, name)
    gate = gateClass(*angles)
    return gate
