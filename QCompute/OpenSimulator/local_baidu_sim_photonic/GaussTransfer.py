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
This module transfers the gaussian state from one into another according to the sequence of gates.
"""
FileErrorCode = 22

from typing import List, Union
import numpy
import math
import copy



from QCompute.OpenSimulator.local_baidu_sim_photonic.InitGaussState import MatrixType


class GaussStateTransferProcessor:
    """
    Simulate gaussian state evolution.
    """

    def __init__(self, matrixType: MatrixType) -> None:

        if matrixType == MatrixType.Dense:
            self.proc = self.WhichPhotonicGate

    def __call__(self, gate_name: str, state_list: Union[List[numpy.ndarray], List['COO']],
                 argument_list: list, modes: list) \
            -> Union[List[numpy.ndarray], List['COO']]:
        """
        To enable the object callable
        """

        return self.proc(gate_name, state_list, argument_list, modes)

    def WhichPhotonicGate(self, gate_name: str, state_list: List[numpy.ndarray], argument_list: list,
                          modes: list) -> List[numpy.ndarray]:
        """
        To judge which quantum gate.
        """

        if gate_name == 'PhotonicGaussianDX':
            return self.CRTransferByDX(state_list, argument_list, modes)
        elif gate_name == 'PhotonicGaussianDP':
            return self.CRTransferByDP(state_list, argument_list, modes)
        elif gate_name == 'PhotonicGaussianPHA':
            return self.CRTransferByPHA(state_list, argument_list, modes)
        elif gate_name == 'PhotonicGaussianBS':
            return self.CRTransferByBS(state_list, argument_list, modes)
        elif gate_name == 'PhotonicGaussianCZ':
            return self.CRTransferByCZ(state_list, argument_list, modes)
        elif gate_name == 'PhotonicGaussianCX':
            return self.CRTransferByCX(state_list, argument_list, modes)
        elif gate_name == 'PhotonicGaussianDIS':
            return self.CRTransferByDIS(state_list, argument_list, modes)
        elif gate_name == 'PhotonicGaussianSQU':
            return self.CRTransferBySQU(state_list, argument_list, modes)
        elif gate_name == 'PhotonicGaussianTSQU':
            return self.CRTransferByTSQU(state_list, argument_list, modes)
        elif gate_name == 'PhotonicGaussianMZ':
            return self.CRTransferByMZ(state_list, argument_list, modes)
        else:
            assert False

    def CRTransferByDX(self, state_list: List[numpy.ndarray], argument_list: list, modes: list) \
            -> List[numpy.ndarray]:
        """
        Update the unitary :math:`U` by the displacement along the direction of position
        """

        state_list[0][2 * modes[0], 0] += argument_list[0]

        return state_list

    def CRTransferByDP(self, state_list: List[numpy.ndarray], argument_list: list, modes: list) \
            -> List[numpy.ndarray]:
        """
        Update the unitary :math:`U` by the displacement along the direction of momentum
        """

        state_list[0][2 * modes[0] + 1, 0] += argument_list[0]

        return state_list

    def CRTransferByDIS(self, state_list: List[numpy.ndarray], argument_list: list, modes: list) \
            -> List[numpy.ndarray]:
        """
        Update the unitary :math:`U` by displacement gate
        """

        [r, phi] = argument_list

        coor = 2 * modes[0]
        fir_mom = state_list[0]
        fir_mom[coor, 0] += 2 * r * math.cos(phi)
        fir_mom[coor + 1, 0] += 2 * r * math.sin(phi)

        return state_list

    def CRTransferByPHA(self, state_list: List[numpy.ndarray], argument_list: list, modes: list) \
            -> List[numpy.ndarray]:
        """
        Update the unitary :math:`U` by phase gate.
        """

        fir_mom, sec_mom = state_list
        phi = argument_list[0]

        # Define the coordinate of position of target qumode
        coor = 2 * modes[0]
        cos = math.cos(phi)
        sin = math.sin(phi)

        # Update rows of the second moment
        xp = sec_mom[coor: coor + 2, :]
        cosbyxp = cos * xp
        sinbyxp = sin * xp
        sec_mom[coor, :] = numpy.add(cosbyxp[0, :], sinbyxp[1, :])
        sec_mom[coor + 1, :] = numpy.subtract(cosbyxp[1, :], sinbyxp[0, :])

        # Update columns of the second moment
        xp = sec_mom[:, coor: coor + 2]
        cosbyxp = cos * xp
        sinbyxp = sin * xp
        sec_mom[:, coor] = numpy.add(cosbyxp[:, 0], sinbyxp[:, 1])
        sec_mom[:, coor + 1] = numpy.subtract(cosbyxp[:, 1], sinbyxp[:, 0])

        # Update the first moment
        x = fir_mom[coor, 0]
        p = fir_mom[coor + 1, 0]
        fir_mom[coor, 0] = cos * x + sin * p
        fir_mom[coor + 1, 0] = -sin * x + cos * p

        return state_list

    def CRTransferBySQU(self, state_list: List[numpy.ndarray], argument_list: list, modes: list) \
            -> List[numpy.ndarray]:
        """
        Update the unitary :math:`U` by single-qumode squeezing gate
        """

        fir_mom, sec_mom = state_list
        [r, phi] = argument_list

        # Define the coordinate of position of target qumode
        coor = 2 * modes[0]

        # Calculate matrix elements of gate matrix
        squ11 = math.cosh(r) - math.sinh(r) * math.cos(phi)
        squdiag = -math.sinh(r) * math.sin(phi)
        squ22 = math.cosh(r) + math.sinh(r) * math.cos(phi)

        # Update rows of the second moment
        squ11byx = squ11 * sec_mom[coor, :]
        squdiagbyxp = squdiag * sec_mom[coor: coor + 2, :]
        squ22byp = squ22 * sec_mom[coor + 1, :]
        sec_mom[coor, :] = numpy.add(squ11byx, squdiagbyxp[1, :])
        sec_mom[coor + 1, :] = numpy.add(squ22byp, squdiagbyxp[0, :])

        # Update columns of the second moment
        squ11byx = squ11 * sec_mom[:, coor]
        squdiagbyxp = squdiag * sec_mom[:, coor: coor + 2]
        squ22byp = squ22 * sec_mom[:, coor + 1]
        sec_mom[:, coor] = numpy.add(squ11byx, squdiagbyxp[:, 1])
        sec_mom[:, coor + 1] = numpy.add(squ22byp, squdiagbyxp[:, 0])

        # Update the first moment
        x = fir_mom[coor, 0]
        p = fir_mom[coor + 1, 0]
        fir_mom[coor, 0] = squ11 * x + squdiag * p
        fir_mom[coor + 1, 0] = squdiag * x + squ22 * p

        return state_list

    def CRTransferByCZ(self, state_list: List[numpy.ndarray], argument_list: list, modes: list) \
            -> List[numpy.ndarray]:
        """
        Update the unitary :math:`U` by controlled phase gate
        """

        fir_mom, sec_mom = state_list
        phi = argument_list[0]

        # Define the coordinates of position of target qumodes
        coor1 = 2 * modes[0]
        coor2 = 2 * modes[1]

        # Update rows of the second moment
        sec_mom[coor1 + 1, :] += phi * sec_mom[coor2, :]
        sec_mom[coor2 + 1, :] += phi * sec_mom[coor1, :]

        # Update columns of the second moment
        sec_mom[:, coor1 + 1] += phi * sec_mom[:, coor2]
        sec_mom[:, coor2 + 1] += phi * sec_mom[:, coor1]

        # Update the first moment
        fir_mom[coor1 + 1, 0] += phi * fir_mom[coor2, 0]
        fir_mom[coor2 + 1, 0] += phi * fir_mom[coor1, 0]

        return state_list

    def CRTransferByCX(self, state_list: List[numpy.ndarray], argument_list: list, modes: list) \
            -> List[numpy.ndarray]:
        """
        Update the unitary :math:`U` by quantum nondemolition (QND) sum gate
        """

        fir_mom, sec_mom = state_list
        g = argument_list[0]

        # Define the coordinates of position of target qumodes
        coor1 = 2 * modes[0]
        coor2 = 2 * modes[1]

        # Update rows of the second moment
        sec_mom[coor1 + 1, :] -= g * sec_mom[coor2 + 1, :]
        sec_mom[coor2, :] += g * sec_mom[coor1, :]

        # Update columns of the second moment
        sec_mom[:, coor1 + 1] -= g * sec_mom[:, coor2 + 1]
        sec_mom[:, coor2] += g * sec_mom[:, coor1]

        # Update the first moment
        fir_mom[coor1 + 1, 0] -= g * fir_mom[coor2 + 1, 0]
        fir_mom[coor2, 0] += g * fir_mom[coor1, 0]

        return state_list

    def CRTransferByTSQU(self, state_list: List[numpy.ndarray], argument_list: list, modes: list) \
            -> List[numpy.ndarray]:
        """
        Update the unitary :math:`U` by two-qumode squeezing gate
        """

        fir_mom, sec_mom = state_list
        [r, phi] = argument_list

        # Define the coordinates of position of target qumodes
        coor1 = 2 * modes[0]
        coor2 = 2 * modes[1]

        # Pre calculation
        ch = math.cosh(r)
        shc = math.sinh(r) * math.cos(phi)
        shs = math.sinh(r) * math.sin(phi)

        mode1xp = copy.copy(sec_mom[:, coor1: coor1 + 2])
        mode2xp = copy.copy(sec_mom[:, coor2: coor2 + 2])

        sec_mom[:, coor1] = ch * mode1xp[:, 0] + shc * mode2xp[:, 0] + shs * mode2xp[:, 1]
        sec_mom[:, coor1 + 1] = ch * mode1xp[:, 1] + shs * mode2xp[:, 0] - shc * mode2xp[:, 1]
        sec_mom[:, coor2] = ch * mode2xp[:, 0] + shc * mode1xp[:, 0] + shs * mode1xp[:, 1]
        sec_mom[:, coor2 + 1] = ch * mode2xp[:, 1] + shs * mode1xp[:, 0] - shc * mode1xp[:, 1]

        mode1xp = copy.copy(sec_mom[coor1: coor1 + 2, :])
        mode2xp = copy.copy(sec_mom[coor2: coor2 + 2, :])

        sec_mom[coor1, :] = ch * mode1xp[0, :] + shc * mode2xp[0, :] + shs * mode2xp[1, :]
        sec_mom[coor1 + 1, :] = ch * mode1xp[1, :] + shs * mode2xp[0, :] - shc * mode2xp[1, :]
        sec_mom[coor2, :] = ch * mode2xp[0, :] + shc * mode1xp[0, :] + shs * mode1xp[1, :]
        sec_mom[coor2 + 1, :] = ch * mode2xp[1, :] + shs * mode1xp[0, :] - shc * mode1xp[1, :]

        xp12 = copy.copy(fir_mom)
        fir_mom[coor1, 0] = ch * xp12[coor1, 0] + shc * xp12[coor2, 0] + shs * xp12[coor2 + 1, 0]
        fir_mom[coor1 + 1, 0] = ch * xp12[coor1 + 1, 0] + shs * xp12[coor2, 0] - shc * xp12[coor2 + 1, 0]
        fir_mom[coor2, 0] = ch * xp12[coor2, 0] + shc * xp12[coor1, 0] + shs * xp12[coor1 + 1, 0]
        fir_mom[coor2 + 1, 0] = ch * xp12[coor2 + 1, 0] + shs * xp12[coor1, 0] - shc * xp12[coor1 + 1, 0]

        return state_list

    def CRTransferByBS(self, state_list: List[numpy.ndarray], argument_list: list, modes: list) \
            -> List[numpy.ndarray]:
        """
        Update the unitary :math:`U` by beam splitter
        """

        fir_mom, sec_mom = state_list
        t = argument_list[0]

        # Define the coordinates of position of target qumodes
        coor1 = 2 * modes[0]
        coor2 = 2 * modes[1]
        coor_list = [coor1, coor1 + 1, coor2, coor2 + 1]

        # Update rows of the second moment
        by_transmission = math.sqrt(t) * sec_mom[coor_list, :]
        by_reflection = math.sqrt(1 - t) * sec_mom[coor_list, :]
        sec_mom[coor1: coor1 + 2, :] = numpy.add(by_transmission[0: 2, :], by_reflection[2: 4, :])
        sec_mom[coor2: coor2 + 2, :] = numpy.subtract(by_transmission[2: 4, :], by_reflection[0: 2, :])

        # Update columns of the second moment
        by_transmission = math.sqrt(t) * sec_mom[:, coor_list]
        by_reflection = math.sqrt(1 - t) * sec_mom[:, coor_list]
        sec_mom[:, coor1: coor1 + 2] = numpy.add(by_transmission[:, 0: 2], by_reflection[:, 2: 4])
        sec_mom[:, coor2: coor2 + 2] = numpy.subtract(by_transmission[:, 2: 4], by_reflection[:, 0: 2])

        # Update the first moment
        for index in range(2):
            x1orp1 = fir_mom[coor1 + index, 0]
            x2orp2 = fir_mom[coor2 + index, 0]
            fir_mom[coor1 + index, 0] = math.sqrt(t) * x1orp1 + math.sqrt(1 - t) * x2orp2
            fir_mom[coor2 + index, 0] = math.sqrt(t) * x2orp2 - math.sqrt(1 - t) * x1orp1

        return state_list

    def CRTransferByMZ(self, state_list: List[numpy.ndarray], argument_list: list, modes: list) \
            -> List[numpy.ndarray]:
        """
        Update the unitary :math:`U` by Mach-Zehnder interferometer
        """

        fir_mom, sec_mom = state_list
        [phi_in, phi_ex] = argument_list

        # Define the coordinates of position of target qumodes
        coor1 = 2 * modes[0]
        coor2 = 2 * modes[1]

        # Pre calculation
        cos_in = math.cos(phi_in)
        sin_in = math.sin(phi_in)
        cos_ex = math.cos(phi_ex)
        sin_ex = math.sin(phi_ex)

        mz11 = cos_in * cos_ex - sin_in * sin_ex - cos_ex
        mz12 = cos_in * sin_ex + sin_in * cos_ex - sin_ex
        mz31 = sin_in * cos_ex + cos_in * sin_ex + sin_ex
        mz32 = sin_in * sin_ex - cos_in * cos_ex - cos_ex

        mode1x = copy.copy(sec_mom[:, coor1])
        mode1p = copy.copy(sec_mom[:, coor1 + 1])
        mode2x = copy.copy(sec_mom[:, coor2])
        mode2p = copy.copy(sec_mom[:, coor2 + 1])
        sec_mom[:, coor1] = (mz11 * mode1x + mz12 * mode1p + sin_in * mode2x - (cos_in + 1) * mode2p) / 2
        sec_mom[:, coor1 + 1] = (-mz12 * mode1x + mz11 * mode1p + (cos_in + 1) * mode2x + sin_in * mode2p) / 2
        sec_mom[:, coor2] = (mz31 * mode1x + mz32 * mode1p + (1 - cos_in) * mode2x - sin_in * mode2p) / 2
        sec_mom[:, coor2 + 1] = (-mz32 * mode1x + mz31 * mode1p + sin_in * mode2x + (1 - cos_in) * mode2p) / 2

        mode1x = copy.copy(sec_mom[coor1, :])
        mode1p = copy.copy(sec_mom[coor1 + 1, :])
        mode2x = copy.copy(sec_mom[coor2, :])
        mode2p = copy.copy(sec_mom[coor2 + 1, :])
        sec_mom[coor1, :] = (mz11 * mode1x + mz12 * mode1p + sin_in * mode2x - (cos_in + 1) * mode2p) / 2
        sec_mom[coor1 + 1, :] = (-mz12 * mode1x + mz11 * mode1p + (cos_in + 1) * mode2x + sin_in * mode2p) / 2
        sec_mom[coor2, :] = (mz31 * mode1x + mz32 * mode1p + (1 - cos_in) * mode2x - sin_in * mode2p) / 2
        sec_mom[coor2 + 1, :] = (-mz32 * mode1x + mz31 * mode1p + sin_in * mode2x + (1 - cos_in) * mode2p) / 2

        x1 = copy.copy(fir_mom[coor1, 0])
        p1 = copy.copy(fir_mom[coor1 + 1, 0])
        x2 = copy.copy(fir_mom[coor2, 0])
        p2 = copy.copy(fir_mom[coor2 + 1, 0])
        fir_mom[coor1, 0] = (mz11 * x1 + mz12 * p1 + sin_in * x2 - (cos_in + 1) * p2) / 2
        fir_mom[coor1 + 1, 0] = (-mz12 * x1 + mz11 * p1 + (cos_in + 1) * x2 + sin_in * p2) / 2
        fir_mom[coor2, 0] = (mz31 * x1 + mz32 * p1 + (1 - cos_in) * x2 - sin_in * p2) / 2
        fir_mom[coor2 + 1, 0] = (-mz32 * x1 + mz31 * p1 + sin_in * x2 + (1 - cos_in) * p2) / 2

        return state_list