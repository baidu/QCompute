# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2021 Baidu, Inc. All Rights Reserved.
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
import numpy as np
from ...utils import pauli_terms_to_matrix


class MolecularGroundStateEnergy:
    r"""Molecular Ground State Energy class"""

    def __init__(self, num_qubits: int = 0, hamiltonian: list = None):
        r"""The constructor of the MolecularGroundStateEnergy class

        Args:
            num_qubits (int): Number of qubits, defaults to 0
            hamiltonian (list): Hamiltonian of the molecular system, defaults to None

        """
        self._num_qubits = num_qubits
        self._hamiltonian = hamiltonian

    @property
    def num_qubits(self) -> int:
        r"""The number of qubits used to encoding this molecular system

        Returns:
            int: Number of qubits

        """
        return self._num_qubits

    @property
    def hamiltonian(self) -> list:
        r"""The Hamiltonian of this molecular system

        Returns:
            list: Hamiltonian of this molecular system

        """
        return deepcopy(self._hamiltonian)

    def compute_ground_state_energy(self) -> float:
        r"""Analytically computes the ground state energy

        Returns:
            float: minimum real part of eigenvalues

        """
        matrix = pauli_terms_to_matrix(self._hamiltonian)
        eigval, _ = np.linalg.eig(matrix)

        return min(eigval.real)

    def load_hamiltonian_from_file(self, filename: str, separator: str = ", ") -> None:
        r"""Loads Hamiltonian from a file

        Args:
            filename (str): Path to the file storing the Hamiltonian in Pauli terms
            separator (str): Delimiter between coefficient and Pauli string, defaults to ', '

        """
        if not isinstance(filename, str):
            raise TypeError("Error EA02001(QAPP): The input filename should be a string.")
        self._hamiltonian = []
        with open(filename) as f:
            for line in f:
                coeff, pauli_str = line.strip().split(separator)
                coeff = float(coeff)
                self._hamiltonian.append([coeff, pauli_str])
            self._num_qubits = len(pauli_str)
        print(self._hamiltonian)
        print(self._num_qubits)
