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
Quantum Operation
"""
from typing import List, Union, Optional, Callable, TYPE_CHECKING

import numpy

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterStorage

if TYPE_CHECKING:
    from QCompute.QPlatform.QOperation.FixedGate import FixedGateOP
    from QCompute.QPlatform.QOperation.RotationGate import RotationGateOP
    from QCompute.QPlatform.QOperation.CompositeGate import CompositeGateOP
    from QCompute.QPlatform.QOperation.CustomizedGate import CustomizedGateOP
    from QCompute.QPlatform.QOperation.QProcedure import QProcedureOP
    from QCompute.QPlatform.QOperation.Barrier import BarrierOP
    from QCompute.QPlatform.QOperation.Measure import MeasureOP
    from QCompute.QPlatform.QRegPool import QRegStorage

FileErrorCode = 9

OperationFunc = Callable[[*'QRegStorage'], None]
RotationArgument = Union[int, float, ProcedureParameterStorage]


class QOperation:
    """
    Basic classes for quantum operation
    """

    def __init__(self, name: Optional[str] = None, bits: Optional[int] = None,
                 matrix: Optional[numpy.ndarray] = None) -> None:
        self.name = name
        self.bits = bits
        self._matrix = matrix

    def getMatrix(self) -> numpy.ndarray:
        if self.__class__.__name__ == 'FixedGateOP':
            return self._matrix
        elif self.__class__.__name__ == 'RotationGateOP':
            if self._matrix is None:
                self._matrix = self.generateMatrix()
            return self._matrix
        elif self.__class__.__name__ == 'CustomizedGateOP':
            return self._matrix
        else:
            raise Error.ArgumentError(f'{self.__class__.__name__} do not have matrix!', ModuleErrorCode, FileErrorCode,
                                      1)

    def getInverse(self) -> 'QOperation':
        return self

    def _op(self, qRegList: List['QRegStorage']) -> None:
        """
        Quantum operation base

        :param qRegList: quantum register list
        """
        env = qRegList[0].env
        for qReg in qRegList:
            if qReg.env != env:
                raise Error.ArgumentError('QReg must belong to the same env!', ModuleErrorCode, FileErrorCode, 2)

        if env.__class__.__name__ == 'QProcedure':
            raise Error.ArgumentError('QProcedure should not be operated!', ModuleErrorCode, FileErrorCode, 3)

        if self.bits is not None and self.bits != len(
                qRegList):  # Barrier and QProcedure does not match bits configuration
            raise Error.ArgumentError('The number of QReg must match the setting!', ModuleErrorCode, FileErrorCode, 4)

        if len(qRegList) <= 0:
            raise Error.ArgumentError('Must have QReg in operation!', ModuleErrorCode, FileErrorCode, 5)

        if len(qRegList) != len(set(qReg for qReg in qRegList)):
            raise Error.ArgumentError('QReg of operators in circuit are not repeatable!', ModuleErrorCode,
                                      FileErrorCode, 6)

        circuitLine = CircuitLine()
        circuitLine.data = self
        circuitLine.qRegList = [qReg.index for qReg in qRegList]
        env.circuit.append(circuitLine)

    def _opMeasure(self, qRegList: List['QRegStorage'], cRegList: List[int]) -> None:
        """
        Measure operation base

        :param qRegList: quantum register list
        :param cRegList: classic register list
        """
        env = qRegList[0].env
        for qReg in qRegList:
            if qReg.env != env:
                raise Error.ArgumentError('QReg must belong to the same env!', ModuleErrorCode, FileErrorCode, 7)

        if env.__class__.__name__ == 'QProcedure':
            raise Error.ArgumentError('QProcedure must not be measured!', ModuleErrorCode, FileErrorCode, 8)

        if self.bits is not None and self.bits != len(
                qRegList):  # Barrier and QProcedure does not match bits configuration
            raise Error.ArgumentError('The number of QReg must match the setting!', ModuleErrorCode, FileErrorCode, 9)

        if len(qRegList) <= 0:
            raise Error.ArgumentError('Must have QReg in measure!', ModuleErrorCode, FileErrorCode, 10)

        if len(qRegList) != len(set(qReg for qReg in qRegList)):
            raise Error.ArgumentError('QReg of operators in circuit are not repeatable!', ModuleErrorCode,
                                      FileErrorCode, 11)

        for qReg in qRegList:  # Only in QEnv
            if qReg.index in env.measuredQRegSet:
                raise Error.ArgumentError('Measure must be once on a QReg!', ModuleErrorCode, FileErrorCode, 12)
            env.measuredQRegSet.add(qReg.index)

        if len(cRegList) <= 0:
            raise Error.ArgumentError('Must have CReg in measure!', ModuleErrorCode, FileErrorCode, 13)

        if len(cRegList) != len(set(cReg for cReg in cRegList)):
            raise Error.ArgumentError('CReg of operators in measure are not repeatable!', ModuleErrorCode,
                                      FileErrorCode, 14)

        for cReg in cRegList:  # Only in QEnv
            if cReg in env.measuredCRegSet:
                raise Error.ArgumentError('Measure must be once on a CReg!', ModuleErrorCode, FileErrorCode, 15)
            env.measuredCRegSet.add(cReg)

        if len(qRegList) != len(cRegList):
            raise Error.ArgumentError('QReg and CReg in measure must have same count!', ModuleErrorCode, FileErrorCode,
                                      16)

        circuitLine = CircuitLine()
        circuitLine.data = self
        circuitLine.qRegList = [qReg.index for qReg in qRegList]
        circuitLine.cRegList = cRegList
        env.circuit.append(circuitLine)


Operation = Union[
    'FixedGateOP', 'RotationGateOP', 'CompositeGateOP', 'CustomizedGateOP', 'QProcedureOP', 'BarrierOP', 'MeasureOP']


class CircuitLine:
    """
    Circuit Line
    """
    data: None  # type: Operation
    qRegList: None  # type: List[int]
    cRegList: None  # type: List[int]
