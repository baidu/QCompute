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
Molecular Ground State Energy
"""

from copy import deepcopy
from typing import List
import numpy as np
from qapp.utils import pauli_terms_to_matrix


class MolecularGroundStateEnergy:
    """Molecular Ground State Energy class
    """
    def __init__(self, num_qubits: int = 0, hamiltonian: List = None):
        """The constructor of the MolecularGroundStateEnergy class

        :param num_qubits: Number of qubits, defaults to 0
        :param hamiltonian: Hamiltonian of the molecular system, defaults to None
        """
        self._num_qubits = num_qubits
        self._hamiltonian = hamiltonian

    @property
    def num_qubits(self) -> int:
        """The number of qubits used to encoding this molecular system

        :return: Number of qubits
        """
        return self._num_qubits

    @property
    def hamiltonian(self) -> List:
        """The Hamiltonian of this molecular system

        :return: Hamiltonian of this molecular system
        """
        return deepcopy(self._hamiltonian)

    def compute_ground_state_energy(self) -> float:
        """Analytically computes the ground state energy
        """
        matrix = pauli_terms_to_matrix(self._hamiltonian)
        eigval, _ = np.linalg.eig(matrix)

        return min(eigval.real)

    def load_hamiltonian_from_file(self, filename: str, separator: str = ', '):
        """Loads Hamiltonian from a file

        :param filename: Path to the file storing the Hamiltonian in Pauli terms
        :param separator: Delimiter between coefficient and Pauli string, defaults to ', '
        """
        if not isinstance(filename, str):
            raise TypeError('The input filename should be a string.')
        self._hamiltonian = []
        with open(filename) as f:
            for line in f:
                coeff, pauli_str = line.strip().split(separator)
                coeff = float(coeff)
                self._hamiltonian.append([coeff, pauli_str])
            self._num_qubits = len(pauli_str)
        print(self._hamiltonian)
        print(self._num_qubits)
