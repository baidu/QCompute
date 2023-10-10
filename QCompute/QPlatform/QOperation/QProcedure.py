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
Quantum Procedure
"""
FileErrorCode = 38

from functools import reduce
from typing import List, TYPE_CHECKING

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.ProcedureParameterExpression import ProcedureParameterExpression
from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterPool, ProcedureParameterStorage
from QCompute.QPlatform.QOperation import QOperation

if TYPE_CHECKING:
    from QCompute.QPlatform.QOperation import CircuitLine, OperationFunc, RotationArgument
    from QCompute.QPlatform.QRegPool import QRegPool, QRegStorage


class QProcedure:
    """
    Quantum procedure (sub-procedure).

    :param name: Name of quantum procedure operation.

    :type name: str

    :param Q: Quantum register dict.

    :type Q: QRegPool

    :param Parameter: Procedure parameter dict.

    :type Parameter: ProcedureParameterPool

    :param circuit: List of Circuit Lines. List[CircuitLine]
    """

    def __init__(self, name: str, Q: 'QRegPool', Parameter: ProcedureParameterPool,
                 circuit: List['CircuitLine']):
        """
        Initialize the sub-procedure of object.
        """

        def reduceFunc(previousValue: int, currentValue: int) -> int:
            return previousValue if previousValue > currentValue else currentValue

        self.parameterCount = 0 if len(Parameter.parameterMap) == 0 \
            else reduce(reduceFunc,
                        Parameter.parameterMap.keys()) + 1

        self.name = name
        self.Q = Q
        self.Parameter = Parameter
        self.circuit = circuit

        Q.changeEnv(self)

    def __call__(self, *argumentList: 'RotationArgument') -> 'OperationFunc':
        if len(argumentList) < self.parameterCount:
            raise Error.ArgumentError('Not enough QProcedure argument!', ModuleErrorCode, FileErrorCode, 1)
        for argument in argumentList:
            if not isinstance(argument, (int, float, ProcedureParameterStorage, ProcedureParameterExpression)):
                raise Error.ArgumentError('Wrong QProcedure argument!', ModuleErrorCode, FileErrorCode, 2)
        return QProcedureOP(self.name, self, list(argumentList))


class QProcedureOP(QOperation):
    """
    Quantum procedure operation.

    :param name: Name of quantum procedure operation.

    :type name: str

    :param procedureData: Quantum procedure.

    :type procedureData:  procedureData: QProcedure

    :param argumentList: List of rotation parameters.

    :type argumentList: List[Union[int, float, ProcedureParameterStorage]]
    """

    def __init__(self, name: str, procedureData: QProcedure, argumentList: List['RotationArgument']) -> None:
        super().__init__(name)
        self.procedureData = procedureData
        self.argumentList = argumentList

    def __call__(self, *qRegList: 'QRegStorage') -> None:
        self._op(list(qRegList))

    def getInversed(self) -> None:
        raise Error.ArgumentError("QProcedureOP can't getInversed! Please use QEnv.inverseProcedure.", ModuleErrorCode,
                                  FileErrorCode, 3)
