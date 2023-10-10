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
Photonic Measure Operations
"""
FileErrorCode = 44

import numpy

from QCompute.QPlatform import Error, ModuleErrorCode, FileErrorCode
from typing import List, TYPE_CHECKING, Union, Tuple, Optional
from QCompute.QPlatform.QOperation import QOperation

if TYPE_CHECKING:
    from QCompute.QPlatform.QRegPool import QRegStorage


class PhotonicGaussianMeasureOP(QOperation):
    """
    The photonic measure instruction.
    """

    def __init__(self, gate: str, allowArgumentCounts: int,
                 photonicArgumentList: Optional[Union[List[Tuple[float, float]], int]]) -> None:
        super().__init__(gate)
        if allowArgumentCounts == 0 and photonicArgumentList is not None:
            raise Error.ArgumentError('allowArgumentCounts is invalid!', ModuleErrorCode, FileErrorCode, 1)
        elif isinstance(photonicArgumentList, int):
            if allowArgumentCounts != 1:
                raise Error.ArgumentError('allowArgumentCounts is invalid!', ModuleErrorCode, FileErrorCode, 2)
        elif isinstance(photonicArgumentList, (numpy.ndarray, list)) \
                and allowArgumentCounts != len(photonicArgumentList):
            raise Error.ArgumentError('allowArgumentCounts is invalid!', ModuleErrorCode, FileErrorCode, 3)
        self.allowArgumentCounts = allowArgumentCounts
        if gate == 'PhotonCount':
            self.cutoff: int = photonicArgumentList
        elif gate == 'Heterodyne':
            self.heterodyneArgument: List[Tuple[float, float]] = photonicArgumentList

    def __call__(self, qRegList: List['QRegStorage'], cRegList: Union[List[int], range]) -> None:
        """
        Hack initialize by calling parent classes
        :param qRegList: the quantum register list
        :param cRegList: the classical register list. [2, 4, 6] or range(0, 3)
        """

        if isinstance(cRegList, range):  # compatible 'range'
            cRegList = list(cRegList)

        self._opMeasure(qRegList, cRegList)


MeasureHomodyne = PhotonicGaussianMeasureOP('Homodyne', 0, None)
"""
Homodyne measurement
"""


def MeasureHeterodyne(arguments: List[Tuple[float, float]]):
    """
    Heterodyne measurement
    """

    for index in range(len(arguments)):
        r_mode_index = arguments[index][0]
        assert (r_mode_index >= 0)
    return PhotonicGaussianMeasureOP('Heterodyne', len(arguments), arguments)


def MeasurePhotonCount(cutoff: int):
    """
    Photon-count measurement

    :param cutoff: resolution of photon-count detector
    """

    assert (cutoff >= 1 and type(cutoff) == int)
    return PhotonicGaussianMeasureOP('PhotonCount', 1, cutoff)