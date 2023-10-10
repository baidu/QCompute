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
Measure Operation
"""
FileErrorCode = 37

import importlib
from typing import List, TYPE_CHECKING, Union
from QCompute.QPlatform.QOperation import QOperation

if TYPE_CHECKING:
    from QCompute.QPlatform.QRegPool import QRegStorage


class MeasureOP(QOperation):
    """
    The measure instruction.

    This class is to be implemented on the next version.

    Currently, only computational basis measure, MeasureZ, can be used in the program.
    """

    def __init__(self, gate: str) -> None:
        super().__init__(gate)

    def __call__(self, qRegList: List['QRegStorage'], cRegList: Union[List[int], range]) -> None:
        """
        Hack initialize by calling parent classes

        :param qRegList: The quantum register list. List[QRegStorage]

        :param cRegList: The classical register list. Union[List[int], range]
        """
        if isinstance(cRegList, range):  # compatible 'range'
            cRegList = list(cRegList)

        self._opMeasure(qRegList, cRegList)


MeasureZ = MeasureOP('Z')
"""
Z measure: measurement along computational basis.
"""


def getMeasureInstance(name: str) -> 'MeasureOP':
    """
    Get a measure according to name.

    :param name: measure name.

    :type name: str

    :return: gate.
    """

    currentModule = importlib.import_module(__name__)
    gate = getattr(currentModule, 'Measure' + name)
    return gate