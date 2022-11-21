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
    Basic classes for quantum operation.
    """

    def __init__(self, name: Optional[str] = None, bits: Optional[int] = None,
                 matrix: Optional[numpy.ndarray] = None) -> None:
        self.name = name
        self.bits = bits
        self.matrix = matrix

    def getMatrix(self) -> numpy.ndarray:
        if self.__class__.__name__ == 'FixedGateOP':
            return self.matrix
        elif self.__class__.__name__ == 'RotationGateOP':
            if self.matrix is None:
                self.generateMatrix()
            return self.matrix
        elif self.__class__.__name__ == 'CustomizedGateOP':
            return self.matrix
        else:
            raise Error.ArgumentError(f'{self.__class__.__name__} do not have matrix!', ModuleErrorCode, FileErrorCode,
                                      1)

    def getInversed(self) -> 'QOperation':
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

        if len(qRegList) != len(set(qReg for qReg in qRegList)):
            raise Error.ArgumentError('QReg of operators in circuit are not repeatable!', ModuleErrorCode,
                                      FileErrorCode, 5)

        circuitLine = CircuitLine()
        circuitLine.data = self
        circuitLine.qRegList = [qReg.index for qReg in qRegList]
        env.circuit.append(circuitLine)

    def _opMeasure(self, qRegList: List['QRegStorage'], cRegList: List[int]) -> None:
        """
        Measure operation base.

        :param qRegList: quantum register list
        :param cRegList: classic register list
        """
        if len(qRegList) != len(cRegList):
            raise Error.ArgumentError('QReg and CReg in measure must have same count!', ModuleErrorCode, FileErrorCode,
                                      6)

        env = qRegList[0].env
        for qReg in qRegList:
            if qReg.env != env:
                raise Error.ArgumentError('QReg must belong to the same env!', ModuleErrorCode, FileErrorCode, 7)

        if env.__class__.__name__ == 'QProcedure':
            raise Error.ArgumentError('QProcedure must not be measured!', ModuleErrorCode, FileErrorCode, 8)

        if len(qRegList) <= 0:
            raise Error.ArgumentError('Must have QReg in measure!', ModuleErrorCode, FileErrorCode, 9)

        if len(qRegList) != len(set(qReg for qReg in qRegList)):
            raise Error.ArgumentError('QReg of measure in circuit are not repeatable!', ModuleErrorCode,
                                      FileErrorCode, 10)

        for qReg in qRegList:  # Only in QEnv
            if qReg.index in env.measuredQRegSet:
                raise Error.ArgumentError('Measure must be once on a QReg!', ModuleErrorCode, FileErrorCode, 11)
            env.measuredQRegSet.add(qReg.index)

        if len(cRegList) <= 0:
            raise Error.ArgumentError('Must have CReg in measure!', ModuleErrorCode, FileErrorCode, 12)

        if len(cRegList) != len(set(cReg for cReg in cRegList)):
            raise Error.ArgumentError('CReg of measure in circuit are not repeatable!', ModuleErrorCode,
                                      FileErrorCode, 13)

        for cReg in cRegList:  # Only in QEnv
            if cReg in env.measuredCRegSet:
                raise Error.ArgumentError('Measure must be once on a CReg!', ModuleErrorCode, FileErrorCode, 14)
            env.measuredCRegSet.add(cReg)

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

    def __init__(self, data: Operation = None, qRegList: List[int] = None, cRegList: List[int] = None):
        """
        Initialize a quantum gate instance.

        :param data: a Quanlse.QOperation.Operation instance,
                    the quantum gate to be applied
        :param qRegList: a list of qubit indices.
                    If `gate` is a single-qubit
                    gate, then `qubits` still be a List of the form `[i]`
        :param cRegList: a list of classical bit indices
        """
        self.data: Operation = data
        self.qRegList: List[int] = qRegList
        self.cRegList: List[int] = cRegList

    def inverse(self) -> 'CircuitLine':
        """
        Return a `CircuitLine` instance whose `QOperation` data is the inverse of the origin one.

        :return: a `CircuitLine` instance whose `QOperation` data is the inverse of the origin one
        """
        self.data = self.data.getInversed()
        return self


def getGateInstance(name: str) -> Union['FixedGateOP', 'OperationFunc']:
    """
    Get a gate according to name

    :param name : gate name
    :return: gate
    """
    from QCompute.QPlatform.QOperation.FixedGate import getFixedGateInstance
    from QCompute.QPlatform.QOperation.RotationGate import createRotationGateInstance

    gate = getFixedGateInstance(name)
    if not gate:
        gate = createRotationGateInstance(name, 0, 0, 0)
    return gate
