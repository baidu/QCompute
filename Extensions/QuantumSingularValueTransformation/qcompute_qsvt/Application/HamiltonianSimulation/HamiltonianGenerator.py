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
This script supplies a function to generate random Hamiltonian, given the numbers of qubits and Pauli terms.
"""

from typing import List, Tuple
import random

import numpy as np


def func_Hamiltonian_gen(num_qubits: int, num_terms: int) -> List[Tuple[float, str]]:
    r"""Generate a Hamiltonian randomly in Pauli string representation.

    :param num_qubits: :math:`n`, `int`, the number of qubits in such quantum system
    :param num_terms: :math:`s`, `int`, the number of Pauli terms in such Hamiltonian
    :return: **list_str_Pauli_rep** – :math:`\check H`, `List[Tuple[float, str]]` of length :math:`s`,
        a list of form :math:`(a_j, P_j)` describing a random Hamiltonian, such as
        `[(-0.09706, \'I\'), (-0.045302, \'X0X1Y2Y3\'), (0.17441, \'Z2Z3\'), (-0.2234, \'Z3\')]`,
        where :math:`a_j` is a float indicating the coefficient of :math:`P_j` in :math:`\check H`,
        and :math:`P_j` is a string indicating which multi-Pauli gate

    **Example**

        We could run the following program to generate and print
        a :math:`6` qubit Hamiltonian with :math:`2000` Pauli terms:

        >>> from qcompute_qsvt.Application.HamiltonianSimulation.HamiltonianGenerator import func_Hamiltonian_gen
        >>> print(func_Hamiltonian_gen(num_qubits=6, num_terms=2000))
    """
    int_dim = 4 ** num_qubits  # n qubits Hamiltonian should be regarded as a int_dim dimension vector in Pauli basis

    list_coor_Pauli = []  # a list to record which component is none-zero
    if num_terms < int_dim * 0.01:  # the sparse case
        list_coor_Pauli = random.sample(range(int_dim), k=num_terms)
    else:  # the dense case
        while len(list_coor_Pauli) < num_terms:
            int_coor_Pauli = random.randint(0, int_dim - 1)
            if int_coor_Pauli not in list_coor_Pauli:
                list_coor_Pauli.append(int_coor_Pauli)

    # generate the coefficients and normalize it
    list_coe = np.array([random.uniform(-1, 1) for _ in range(num_terms)])
    list_coe = list(list_coe / np.sqrt(np.dot(list_coe, list_coe)))

    # translate Pauli coordinate into Pauli string
    list_str = []
    for idx in range(num_terms):
        str_Pauli = ''
        if list_coor_Pauli[idx] == 0:
            str_Pauli = 'I'
        else:
            idx_qubit = 0
            int_coor_Pauli = list_coor_Pauli[idx]
            while int_coor_Pauli > 0:
                int_Pauli = int_coor_Pauli % 4
                if int_Pauli == 1:
                    str_Pauli += 'X{0}'.format(idx_qubit)
                elif int_Pauli == 2:
                    str_Pauli += 'Y{0}'.format(idx_qubit)
                elif int_Pauli == 3:
                    str_Pauli += 'Z{0}'.format(idx_qubit)
                int_coor_Pauli //= 4
                idx_qubit += 1
        list_str.append((list_coe[idx], str_Pauli))
    # return such Pauli string
    return list_str
