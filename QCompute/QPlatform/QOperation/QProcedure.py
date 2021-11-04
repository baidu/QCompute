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
Quantum Procedure
"""
from functools import reduce
from typing import List, TYPE_CHECKING

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.ProcedureParameterPool import ProcedureParameterPool, ProcedureParameterStorage
from QCompute.QPlatform.QOperation import QOperation

if TYPE_CHECKING:
    from QCompute.QPlatform.QOperation import CircuitLine, OperationFunc, RotationArgument
    from QCompute.QPlatform.QRegPool import QRegPool, QRegStorage

FileErrorCode = 13


class QProcedure:
    """
    Quantum procedure (sub-procedure)
    """

    def __init__(self, name: str, Q: 'QRegPool', Parameter: ProcedureParameterPool,
                 circuit: List['CircuitLine']):
        """
        Initialize the sub-procedure of object
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
            if not isinstance(argument, (int, float, ProcedureParameterStorage)):
                raise Error.ArgumentError('Wrong QProcedure argument!', ModuleErrorCode, FileErrorCode, 2)
        return QProcedureOP(self.name, self, list(argumentList))


class QProcedureOP(QOperation):
    """
    Quantum procedure operation
    """

    def __init__(self, name: str, procedureData: QProcedure, argumentList: List['RotationArgument']) -> None:
        super().__init__(name)
        self.procedureData = procedureData
        self.argumentList = argumentList

    def __call__(self, *qRegList: 'QRegStorage') -> None:
        self._op(list(qRegList))

    def getInverse(self) -> None:
        raise Error.ArgumentError("QProcedureOP can't getInverse! Please use QEnv.inverseProcedure.", ModuleErrorCode,
                                  FileErrorCode, 3)
