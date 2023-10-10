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
Quantum Register Poll
"""
FileErrorCode = 10

from typing import Dict, Union, TYPE_CHECKING, List, Tuple

from QCompute.QPlatform import Error, ModuleErrorCode

if TYPE_CHECKING:
    from QCompute.QPlatform.QEnv import QEnv
    from QCompute.QPlatform.QOperation.QProcedure import QProcedure


class QRegStorage:
    """
    The storage for quantum register.
    """

    def __init__(self, index: int, env: Union['QEnv', 'QProcedure']) -> None:
        """
        The quantum register object needs to know its index and related quantum environment.

        :param index: the quantum register index.

        :param env: the related quantum environment or procedure.
        """

        self.index = index
        self.env = env


class QRegPool:
    """
    The quantum register dict.
    """

    def __init__(self, env: Union['QEnv', 'QProcedure']) -> None:
        """
        The constructor of the QRegPool class.

        :param env: the related quantum environment or procedure.
        """

        # the quantum environment related with the quantum register dict
        self.env = env
        # the inner data for quantum register dict
        self.registerMap: Dict[int, QRegStorage] = {}

    def __getitem__(self, index: int) -> QRegStorage:
        return self._get(index)

    def __call__(self, index: int) -> QRegStorage:
        return self._get(index)

    def _get(self, index: int) -> QRegStorage:
        """
        Get the quantum register according to the index.

        Create the register when it does not exists.

        :param index: the quantum register index.

        :return: QuantumRegisterStorage
        """

        value = self.registerMap.get(index)
        if value is None:
            value = QRegStorage(index, self.env)
            self.registerMap[index] = value
        return value

    def changeEnv(self, env: Union['QEnv', 'QProcedure']) -> None:
        """
        For QProcedure
        """
        self.env = env
        for qReg in self.registerMap.values():
            qReg.env = env

    def createList(self, num: int) -> List[QRegStorage]:
        """
        Create QRegStorage list.

        Only can called when self.registerMap is empty.
        """
        if len(self.registerMap) > 0:
            raise Error.ArgumentError('createList is called when self.registerMap is not empty!', ModuleErrorCode, FileErrorCode, 1)

        for i in range(num):
            self.registerMap[i] = QRegStorage(i, self.env)
        return list(self.registerMap.values())

    def toListPair(self) -> Tuple[List[QRegStorage], List[int]]:
        """
        Get QReg list and index list
        """

        qRegList: List[QRegStorage] = []
        indexList: List[int] = []
        for index, qReg in self.registerMap.items():
            qRegList.append(qReg)
            indexList.append(index)
        return qRegList, indexList