# -*- coding: UTF-8 -*-
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
QAOA Ansatz Circuit
"""

from typing import List
import numpy as np
from QCompute.QPlatform.QOperation.RotationGate import RX, RZ
from QCompute.QPlatform.QOperation.FixedGate import H, CX
from QCompute.QPlatform.QRegPool import QRegPool
from .parameterized_circuit import ParameterizedCircuit


class QAOAAnsatz(ParameterizedCircuit):
    """QAOA Ansatz class
    """
    def __init__(self, num: int, parameters: np.ndarray, hamiltonian: List, layer: int):
        """The constructor of the QAOAAnsatz class

        :param num: Number of qubits in this ansatz
        :param parameters: Parameters of parameterized gates in this ansatz
        :param hamiltonian: Hamiltonian used to construct the QAOA ansatz
        :param layer: Number of layers for this Ansatz
        """
        super().__init__(num, parameters)
        self._hamiltonian = hamiltonian
        self._layer = layer

    def add_circuit(self, q: QRegPool):
        """Adds circuit to the register according to the given hamiltonian

        :param q: Quantum register to which this circuit is added
        """
        for j in range(self._num):
            H(q[j])

        for i in range(self._layer):
            for pauli in self._hamiltonian:
                if pauli[1].count('i') == self._num:
                    continue
                pauli_list = []
                ind_list = []

                for j, k in enumerate(pauli[1]):
                    if k == 'i':
                        continue
                    elif k == 'x':
                        H(q[j])
                    elif k == 'y':
                        RX(np.pi / 2)(q[j])

                    pauli_list.append(k)
                    ind_list.append(j)

                for j in range(len(pauli_list) - 1):
                    CX(q[ind_list[j]], q[ind_list[j + 1]])
                RZ(self._parameters[2 * i])(q[ind_list[-1]])

                for j in range(len(pauli_list) - 1, 0, -1):
                    CX(q[ind_list[j - 1]], q[ind_list[j]])

                for j, k in enumerate(pauli_list):
                    if k == 'x':
                        H(q[ind_list[j]])
                    elif k == 'y':
                        RX(- np.pi / 2)(q[ind_list[j]])                  

            for j in range(self._num):
                RX(self._parameters[2 * i + 1])(q[j])  
