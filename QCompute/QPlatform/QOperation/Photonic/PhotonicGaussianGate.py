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
Photonic gate operation for simulating quantum circuits based on gaussian state
"""
FileErrorCode = 43

import numpy
from typing import List, TYPE_CHECKING

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.QOperation import QOperation
from QCompute.QPlatform.QRegPool import QRegStorage

if TYPE_CHECKING:
    from QCompute.QPlatform.QOperation import OperationFunc

class PhotonicGaussianGateOP(QOperation):
    """
    Photonic gate.
    """

    def __init__(self, gate: str, bits: int, allowArgumentCounts: int,
                 argumentList: List[float]) -> None:
        super().__init__(gate, bits)
        if len(argumentList) != allowArgumentCounts:
            raise Error.ArgumentError(f'allowArgumentCounts is not len(aargumentList)!', ModuleErrorCode, FileErrorCode, 1)

        self.allowArgumentCounts = allowArgumentCounts
        self.argumentList: List[float] = argumentList
        self.displace_vector: numpy.ndarray = None

    def __call__(self, *qRegList: QRegStorage) -> None:
        self._op(list(qRegList))

    def generateMatrixAndVector(self):
        pass

    def _generateDXMatrixAndVector(self):
        self.matrix = numpy.eye(2)
        self.displace_vector = numpy.array([self.argumentList,
                                            [0]])
        return self.matrix, self.displace_vector

    def _generateDPMatrixAndVector(self):
        self.matrix = numpy.eye(2)
        self.displace_vector = numpy.array([[0],
                                            self.argumentList])
        return self.matrix, self.displace_vector

    def _generatePHAMatrixAndVector(self):
        phi = self.argumentList[0]
        self.matrix = numpy.array([[numpy.cos(phi), numpy.sin(phi)],
                                   [-numpy.sin(phi), numpy.cos(phi)]])
        self.displace_vector = numpy.zeros((2, 1))
        return self.matrix, self.displace_vector

    def _generateBSMatrixAndVector(self):
        t = self.argumentList[0]
        self.matrix = numpy.block([[numpy.sqrt(t) * numpy.eye(2), numpy.sqrt(1 - t) * numpy.eye(2)],
                                   [-numpy.sqrt(1 - t) * numpy.eye(2), numpy.sqrt(t) * numpy.eye(2)]])
        self.displace_vector = numpy.zeros((4, 1))
        return self.matrix, self.displace_vector

    def _generateCZMatrixAndVector(self):
        phi = self.argumentList[0]
        self.matrix = numpy.eye(4)
        self.matrix[1, 2] = self.matrix[3, 0] = phi
        self.displace_vector = numpy.zeros((4, 1))
        return self.matrix, self.displace_vector

    def _generateCXMatrixAndVector(self):
        g = self.argumentList[0]
        self.matrix = numpy.eye(4)
        self.matrix[1, 3] = -g
        self.matrix[2, 0] = g
        self.displace_vector = numpy.zeros((4, 1))
        return self.matrix, self.displace_vector

    def _generateDISMatrixAndVector(self):
        [r, phi] = self.argumentList
        self.matrix = numpy.eye(2)
        self.displace_vector = 2 * numpy.array([[r * numpy.cos(phi)],
                                                [r * numpy.sin(phi)]])
        return self.matrix, self.displace_vector

    def _generateSQUMatrixAndVector(self):
        [r, phi] = self.argumentList
        self.matrix = numpy.cosh(r) * numpy.eye(2) - numpy.sinh(r) * numpy.array([[numpy.cos(phi), numpy.sin(phi)],
                                                                                  [numpy.sin(phi), -numpy.cos(phi)]])
        self.displace_vector = numpy.zeros((2, 1))
        return self.matrix, self.displace_vector

    def _generateTSQUMatrixAndVector(self):
        [r, phi] = self.argumentList
        matrix_phi = numpy.array([[numpy.cos(phi), numpy.sin(phi)],
                                  [numpy.sin(phi), -numpy.cos(phi)]])
        self.matrix = numpy.block([[numpy.cosh(r) * numpy.eye(2), numpy.sinh(r) * matrix_phi],
                                   [numpy.sinh(r) * matrix_phi, numpy.cosh(r) * numpy.eye(2)]])
        self.displace_vector = numpy.zeros((4, 1))
        return self.matrix, self.displace_vector

    def _generateMZMatrixAndVector(self):
        [phi_in, phi_ex] = self.argumentList
        cc = numpy.cos(phi_in) * numpy.cos(phi_ex)
        cs = numpy.cos(phi_in) * numpy.sin(phi_ex)
        sc = numpy.sin(phi_in) * numpy.cos(phi_ex)
        ss = numpy.sin(phi_in) * numpy.sin(phi_ex)
        matrix_lt = numpy.array([[cc - ss - numpy.cos(phi_ex), cs + sc - numpy.sin(phi_ex)],
                                 [-sc - cs + numpy.sin(phi_ex), -ss + cc - numpy.cos(phi_ex)]])
        matrix_rt = numpy.array([[numpy.sin(phi_in), -numpy.cos(phi_in) - 1],
                                 [numpy.cos(phi_in) + 1, numpy.sin(phi_in)]])
        matrix_ld = numpy.array([[sc + cs + numpy.sin(phi_ex), ss - cc - numpy.cos(phi_ex)],
                                 [cc - ss + numpy.cos(phi_ex), cs + sc + numpy.sin(phi_ex)]])
        matrix_rd = numpy.array([[-numpy.cos(phi_in) + 1, -numpy.sin(phi_in)],
                                 [numpy.sin(phi_in), -numpy.cos(phi_in) + 1]])
        self.matrix = 0.5 * numpy.block([[matrix_lt, matrix_rt],
                                   [matrix_ld, matrix_rd]])
        self.displace_vector = numpy.zeros((4, 1))
        return self.matrix, self.displace_vector


def PhotonicDX(d_x: float) -> 'OperationFunc':
    r"""
    Displacement gate

    .. math::
        \mathbf{S} = \mathbf{I} \qquad
        \mathbf{d} = \begin{bmatrix} d_x \\ 0 \end{bmatrix}

    :param d_x: displacement along the direction of position
    """

    return PhotonicGaussianGateOP('PhotonicGaussianDX', 1, 1, [d_x])


PhotonicDX.type = 'PhotonicGaussianGateOP'
PhotonicDX.allowArgumentCounts = 1


def PhotonicDP(d_p: float) -> 'OperationFunc':
    r"""
    Displacement gate

    .. math::
        \mathbf{S} = \mathbf{I} \qquad
        \mathbf{d} = \begin{bmatrix} 0 \\ d_p \end{bmatrix}

    :param d_p: displacement along the direction of momentum
    """

    return PhotonicGaussianGateOP('PhotonicGaussianDP', 1, 1, [d_p])


PhotonicDP.type = 'PhotonicGaussianGateOP'
PhotonicDP.allowArgumentCounts = 1


def PhotonicPHA(phi: float) -> 'OperationFunc':
    r"""
    Phase gate

    .. math::
        \mathbf{S} =
        \begin{bmatrix}
        \cos(\phi) & \sin(\phi) \\
        -\sin(\phi) & \cos(\phi) \\
        \end{bmatrix} \qquad
        \mathbf{d} = \mathbf{0}

    :param phi: phase shift
    """

    return PhotonicGaussianGateOP('PhotonicGaussianPHA', 1, 1, [phi])


PhotonicPHA.type = 'PhotonicGaussianGateOP'
PhotonicPHA.allowArgumentCounts = 1


def PhotonicBS(t: float) -> 'OperationFunc':
    r"""
    Beam Splitter (BS)

    .. math::
        \mathbf{S} =
        \begin{bmatrix}
        \sqrt{t} & 0 & \sqrt{1 - t} & 0 \\
        0 & \sqrt{t} & 0 & \sqrt{1 - t} \\
        -\sqrt{1 - t} & 0 & \sqrt{t} & 0 \\
        0 & -\sqrt{1 - t} & 0 & \sqrt{t}
        \end{bmatrix} \qquad
        \mathbf{d} = \mathbf{0}

    :param t: transmissivity rate of BS
    """

    assert 0 <= t <= 1
    return PhotonicGaussianGateOP('PhotonicGaussianBS', 2, 1, [t])


PhotonicBS.type = 'PhotonicGaussianGateOP'
PhotonicBS.allowArgumentCounts = 1


def PhotonicCZ(phi: float) -> 'OperationFunc':
    r"""
    Controlled phase gate

    .. math::
        \mathbf{S} =
        \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & \phi & 0 \\
        0 & 0 & 1 & 0 \\
        \phi & 0 & 0 & 1
        \end{bmatrix} \qquad
        \mathbf{d} = \mathbf{0}

    :param phi: phase shift
    """

    return PhotonicGaussianGateOP('PhotonicGaussianCZ', 2, 1, [phi])


PhotonicCZ.type = 'PhotonicGaussianGateOP'
PhotonicCZ.allowArgumentCounts = 1


def PhotonicCX(g: float) -> 'OperationFunc':
    r"""
    Quantum nondemolition (QND) sum gate

    .. math::
        \mathbf{S} =
        \begin{bmatrix}
        1 & 0 & 0 & 0 \\
        0 & 1 & 0 & -g \\
        g & 0 & 1 & 0 \\
        0 & 0 & 0 & 1
        \end{bmatrix} \qquad
        \mathbf{d} = \mathbf{0}

    :param g: interaction gain
    """

    return PhotonicGaussianGateOP('PhotonicGaussianCX', 2, 1, [g])


PhotonicCX.type = 'PhotonicGaussianGateOP'
PhotonicCX.allowArgumentCounts = 1


def PhotonicDIS(r: float, phi: float) -> 'OperationFunc':
    r"""
    Displacement gate

    .. math::
        \mathbf{S} = \mathbf{I} \qquad
        \mathbf{d} = \begin{bmatrix} 2r \cos(\phi) \\ 2r \sin(\phi) \end{bmatrix}

    :param r: amplitude
    :param phi: phase
    """

    assert r >= 0
    return PhotonicGaussianGateOP('PhotonicGaussianDIS', 1, 2, [r, phi])


PhotonicDIS.type = 'PhotonicGaussianGateOP'
PhotonicDIS.allowArgumentCounts = 2


def PhotonicSQU(r: float, phi: float) -> 'OperationFunc':
    r"""
    Single-qumode squeezing gate

    .. math::
        \mathbf{S} =
        \begin{bmatrix}
        \cosh(r) - \sinh(r) \cos (\phi) & - \sinh(r) \sin (\phi) \\
        -\sinh(r) \sin (\phi) & \cosh(r) + \sinh(r) \cos (\phi)
        \end{bmatrix}

    .. math::
        \mathbf{d} = \mathbf{0}

    :param r: amplitude
    :param phi: phase
    """

    assert r >= 0
    return PhotonicGaussianGateOP('PhotonicGaussianSQU', 1, 2, [r, phi])


PhotonicSQU.type = 'PhotonicGaussianGateOP'
PhotonicSQU.allowArgumentCounts = 2


def PhotonicTSQU(r: float, phi: float) -> 'OperationFunc':
    r"""
    Two-qumode squeezing gate

    .. math::
        \mathbf{S} =
        \begin{bmatrix}
        \cosh(r) & 0 & \sinh(r) \cos(\phi) & \sinh(r) \sin(\phi) \\
        0 & \cosh(r) & \sinh(r) \sin\phi) & -\sinh(r) \cos(\phi) \\
        \sinh(r) \cos(\phi) & \sinh(r) \sin(\phi) & \cosh(r) & 0 \\
        \sinh(r) \sin(\phi) & -\sinh(r) \cos(\phi) & 0 & \cosh(r)
        \end{bmatrix}

    .. math::
        \mathbf{d} = \mathbf{0}

    :param r: amplitude
    :param phi: phase
    """

    assert r >= 0
    return PhotonicGaussianGateOP('PhotonicGaussianTSQU', 2, 2, [r, phi])


PhotonicTSQU.type = 'PhotonicGaussianGateOP'
PhotonicTSQU.allowArgumentCounts = 2


def PhotonicMZ(phi_in: float, phi_ex: float) -> 'OperationFunc':  # CompositeGate
    r"""
    Mach-Zehnder interferometer (MZ)

    .. math::
        \mathbf{S} = \frac{1}{2}
        \begin{bmatrix}
        {\rm cc}-{\rm ss} - \cos(\phi_{\rm ex}) & {\rm cs}+{\rm sc} - \sin(\phi_{\rm ex}) & \sin(\phi_{\rm in}) & -\cos(\phi_{\rm in}) - 1 \\
        -{\rm sc}-{\rm cs} + \sin(\phi_{\rm ex}) & -{\rm ss}+{\rm cc} - \cos(\phi_{\rm ex}) & \cos(\phi_{\rm in}) + 1 & \sin(\phi_{\rm in}) \\
        {\rm sc}+{\rm cs} + \sin(\phi_{\rm ex}) & {\rm ss}-{\rm cc} -\cos(\phi_{\rm ex}) & -\cos(\phi_{\rm in}) + 1 & -\sin(\phi_{\rm in}) \\
        {\rm cs}-{\rm ss} + \cos(\phi_{\rm ex}) & {\rm cs}+{\rm sc} + \sin(\phi_{\rm ex}) & \sin(\phi_{\rm in}) & -\cos(\phi_{\rm in}) + 1
        \end{bmatrix}

    .. math::
        \mathbf{d} = \mathbf{0}

    where

    .. math::

        {\rm cc} = \cos(\phi_{\rm in}) \cos(\phi_{\rm ex}) \\
        {\rm cs} = \cos(\phi_{\rm in}) \sin(\phi_{\rm ex}) \\
        {\rm sc} = \sin(\phi_{\rm in}) \cos(\phi_{\rm ex}) \\
        {\rm ss} = \sin(\phi_{\rm in}) \sin(\phi_{\rm ex})

    :param phi_in: internal phase
    :param phi_ex: external phase
    """

    return PhotonicGaussianGateOP('PhotonicGaussianMZ', 2, 2, [phi_in, phi_ex])


PhotonicMZ.type = 'PhotonicGaussianGateOP'
PhotonicMZ.allowArgumentCounts = 2