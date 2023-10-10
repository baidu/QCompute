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
Photonic measurement operations for simulating quantum circuits based on fock state
"""
FileErrorCode = 42


from QCompute.QPlatform import Error, ModuleErrorCode, FileErrorCode
from typing import List, TYPE_CHECKING, Union
from QCompute.QPlatform.QOperation import QOperation

if TYPE_CHECKING:
    from QCompute.QPlatform.QRegPool import QRegStorage


class PhotonicFockMeasureOP(QOperation):
    """
    The instruction for photon-count measurement
    """

    def __init__(self, gate: str, allowArgumentCounts: int, photonicArgument: int) -> None:
        super().__init__(gate)

        if isinstance(photonicArgument, int):
            if allowArgumentCounts != 1:
                raise Error.ArgumentError(f'photonicArgument is invalid!', ModuleErrorCode, FileErrorCode, 1)
        else:
            raise Error.ArgumentError(f'photonicArgument is invalid!', ModuleErrorCode, FileErrorCode, 2)

        self.allowArgumentCounts = allowArgumentCounts
        if gate == 'PhotonCount':
            self.cutoff: int = photonicArgument

    def __call__(self, qRegList: List['QRegStorage'], cRegList: Union[List[int], range]) -> None:
        """
        Hack initialize by calling parent classes
        :param qRegList: the quantum register list
        :param cRegList: the classical register list. [2, 4, 6] or range(0, 3)
        """

        if isinstance(cRegList, range):  # compatible 'range'
            cRegList = list(cRegList)

        self._opMeasure(qRegList, cRegList)


def MeasurePhotonCount(cutoff: int):
    """
    Photon-count measurement

    :param cutoff: the number of photons in single shot
    """

    assert (cutoff >= 1 and type(cutoff) == int)
    return PhotonicFockMeasureOP('PhotonCount', 1, cutoff)