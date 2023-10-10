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
Add photon(s) to a single-qumode
"""
FileErrorCode = 13

from typing import Union
import numpy



from QCompute.OpenSimulator.local_baidu_sim_photonic.InitFockState import MatrixType


class AddPhotonsToInitFockState:
    r"""
    This class is used to generate the vector of :math:`N`-qumode fock state after adding photon(s).
    """

    def __init__(self, matrixType: MatrixType) -> None:
        if matrixType == MatrixType.Dense:
            self.proc = self.AddPhotonsDirectly

    def __call__(self, fock_state_vector: numpy.ndarray, gate_matrix: numpy.ndarray, modes: list) -> numpy.ndarray:
        """
        To enable the object callable
        """

        return self.proc(fock_state_vector, gate_matrix, modes)

    def AddPhotonsDirectly(self, fock_state_vector: numpy.ndarray, argument_list: list, modes: list) \
            -> numpy.ndarray:
        r"""
        Get the vector of :math:`N`-qumode fock state after adding photon(s)

        :param fock_state_vector: a :math:`N`-dimensional vector
        :param gate_matrix: the number of photons added to target qumode
        :param modes: target qumode
        :return fock_state_vector: numpy.ndarray, a vector of :math:`N`-qumode fock state after adding photon(s)
        """

        fock_state_vector[modes[0]] = fock_state_vector[modes[0]] + int(argument_list[0])
        return fock_state_vector