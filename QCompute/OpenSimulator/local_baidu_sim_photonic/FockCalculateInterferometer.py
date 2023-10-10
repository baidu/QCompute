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

r"""
Calculate overall unitary :math:`U` of interferometer
"""
import copy

FileErrorCode = 15

import math
import numpy


class FockStateTransferProcessor:
    r"""
    Calculate the overall unitary :math:`U` of interferometer.
    """

    def __init__(self) -> None:
        self.proc = self.WhichPhotonicGate

    def __call__(self, gate_name: str, unitary_trans_total: numpy.ndarray, argument_list: list, modes: list) \
            -> numpy.ndarray:
        """
        To enable the object callable
        """

        return self.proc(gate_name, unitary_trans_total, argument_list, modes)

    def WhichPhotonicGate(self, gate_name: str, unitary_trans_total: numpy.ndarray, argument_list: list, modes: list) \
            -> numpy.ndarray:
        """
        To judge which quantum gate.
        """
        if gate_name == 'PhotonicFockPHA':
            return self.CRTransferByPHA(unitary_trans_total, argument_list, modes)
        elif gate_name == 'PhotonicFockBS':
            return self.CRTransferByBS(unitary_trans_total, argument_list, modes)
        elif gate_name == 'PhotonicFockMZ':
            return self.CRTransferByMZ(unitary_trans_total, argument_list, modes)
        else:
            assert False

    def CRTransferByPHA(self, unitary_trans_total: numpy.ndarray, argument_list: list, modes: list) -> numpy.ndarray:
        """
        Update the unitary :math:`U` by phase gate.
        """

        phi = argument_list[0]
        coor = modes[0]

        row = copy.copy(unitary_trans_total[coor, :])
        unitary_trans_total[coor, :] = numpy.exp(1j * phi) * row

        return unitary_trans_total

    def CRTransferByBS(self, unitary_trans_total: numpy.ndarray, argument_list: list, modes: list) -> numpy.ndarray:
        """
        Update the unitary :math:`U` by beam splitter
        """

        t = argument_list[0]
        [coor1, coor2] = modes
        sqrtt, sqrtr = math.sqrt(t), math.sqrt(1 - t) * 1j

        row1 = copy.copy(unitary_trans_total[coor1, :])
        row2 = copy.copy(unitary_trans_total[coor2, :])

        unitary_trans_total[coor1, :] = sqrtt * row1 + sqrtr * row2
        unitary_trans_total[coor2, :] = sqrtr * row1 + sqrtt * row2

        return unitary_trans_total

    def CRTransferByMZ(self, unitary_trans_total: numpy.ndarray, argument_list: list, modes: list) -> numpy.ndarray:
        """
        Update the unitary :math:`U` by Mach-Zehnder interferometer
        """

        [phi_in, phi_ex] = argument_list
        expin = numpy.exp(1j * phi_in)
        expex = numpy.exp(1j * phi_ex)
        [coor1, coor2] = modes

        row1 = copy.copy(unitary_trans_total[coor1, :])
        row2 = copy.copy(unitary_trans_total[coor2, :])

        unitary_trans_total[coor1, :] = ((-(1 - expin) * expex) * row1 + (1j * (1 + expin)) * row2) / 2
        unitary_trans_total[coor2, :] = (((1j * (1 + expin)) * expex) * row1 + (1 - expin) * row2) / 2

        return unitary_trans_total