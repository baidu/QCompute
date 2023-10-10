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
Photonic gate operations for simulating quantum circuits based on fock state
"""
FileErrorCode = 41

import numpy
from typing import List, TYPE_CHECKING, Union

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.QOperation import QOperation
from QCompute.QPlatform.QRegPool import QRegStorage

if TYPE_CHECKING:
    from QCompute.QPlatform.QOperation import OperationFunc

class PhotonicFockGateOP(QOperation):
    """
    Photonic gate.
    """

    def __init__(self, gate: str, bits: int, allowArgumentCounts: int, argumentList: List[Union[float, int]]) -> None:
        super().__init__(gate, bits)
        if len(argumentList) != allowArgumentCounts:
            raise Error.ArgumentError(f'allowArgumentCounts is not len(argumentList)!', ModuleErrorCode, FileErrorCode, 1)

        self.allowArgumentCounts = allowArgumentCounts
        self.argumentList: List[Union[float, int]] = argumentList
        self.matrix: numpy.ndarray = None

    def __call__(self, *qRegList: QRegStorage) -> None:
        self._op(list(qRegList))

    def generateMatrix(self):
        pass

    def _generateAPMatrix(self):
        self.matrix = numpy.array([[self.argumentList[0]]])
        return self.matrix

    def _generatePHAMatrix(self):
        self.matrix = numpy.array([[self.argumentList[0]]])
        return self.matrix

    def _generateBSMatrix(self):
        t = self.argumentList[0]
        self.matrix = numpy.array([[numpy.sqrt(t), 1j * numpy.sqrt(1 - t)],
                                   [1j * numpy.sqrt(1 - t), numpy.sqrt(t)]])
        return self.matrix

    def _generateMZMatrix(self):
        [phi_in, phi_ex] = self.argumentList
        expin = numpy.exp(1j * phi_in)
        expex = numpy.exp(1j * phi_ex)
        self.matrix = numpy.array([[-(1 - expin) * expex, 1j * (1 + expin)],
                                   [(1j * (1 + expin)) * expex, 1 - expin]])
        return self.matrix


def PhotonicAP(num_photons: int) -> 'OperationFunc':
    """
    Add photon(s) to a single-qumode.

    :param num_photons: the number of photon(s)
    """

    assert num_photons >= 0
    return PhotonicFockGateOP('PhotonicFockAP', 1, 1, [num_photons])


PhotonicAP.type = 'PhotonicFockGateOP'
PhotonicAP.allowArgumentCounts = 1


def PhotonicPHA(phi: float) -> 'OperationFunc':
    """
    Phase gate

    :param phi: phase shift
    """

    return PhotonicFockGateOP('PhotonicFockPHA', 1, 1, [phi])


PhotonicPHA.type = 'PhotonicFockGateOP'
PhotonicPHA.allowArgumentCounts = 1


def PhotonicBS(t: float) -> 'OperationFunc':
    r"""
    Beam Splitter (BS)

    :math:`BS =
    \begin{bmatrix}
    \sqrt{t} & i \sqrt{1-t} \\
    i \sqrt{1-t} & \sqrt{t}
    \end{bmatrix}`

    :param t: transmissivity rate of BS
    """

    assert 0 <= t <= 1
    return PhotonicFockGateOP('PhotonicFockBS', 2, 1, [t])


PhotonicBS.type = 'PhotonicFockGateOP'
PhotonicBS.allowArgumentCounts = 1


def PhotonicMZ(phi_in: float, phi_ex: float) -> 'OperationFunc':  # CompositeGate
    r"""
    Mach-Zehnder interferometer (MZ)

    :math:`MZ =
    \begin{bmatrix}
    (e^{i \phi_{\rm in}} - 1)e^{i \phi_{\rm ex}} & i(e^{i \phi_{\rm in}} + 1) \\
    i(e^{i \phi_{\rm in}} + 1)e^{i \phi_{\rm ex}} & -(e^{i \phi_{\rm in}} - 1)
    \end{bmatrix}`

    :param phi_in: internal phase
    :param phi_ex: external phase
    """

    return PhotonicFockGateOP('PhotonicFockMZ', 2, 2, [phi_in, phi_ex])


PhotonicMZ.type = 'PhotonicFockGateOP'
PhotonicMZ.allowArgumentCounts = 2