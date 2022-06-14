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
Max Cut Problem
"""

from typing import List
from copy import deepcopy
import networkx as nx


class MaxCut():
    """Max Cut Problem class
    """
    def __init__(self, num_qubits: int = 0, hamiltonian: List = None):
        """The constructor of the MaxCut class

        :param num_qubits: Number of qubits, defaults to 0
        :param hamiltonian: Hamiltonian of the target graph of the Max Cut problem, defaults to None
        """
        self._num_qubits = num_qubits
        self._hamiltonian = hamiltonian

    @property
    def num_qubits(self) -> int:
        """The number of qubits used to encoding this target graph

        :return: Number of qubits used to encoding this target graph
        """
        return self._num_qubits

    @property
    def hamiltonian(self) -> List:
        """The Hamiltonian of this target graph

        :return: Hamiltonian of this target graph
        """
        return deepcopy(self._hamiltonian)

    def graph_to_hamiltonian(self, graph: nx.Graph):
        """Constructs Hamiltonian from the target graph of the Max Cut problem

        :param graph: Undirected graph without weights
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError('Type of the input graph should be networkx.Graph.')

        edges = graph.edges()

        def convert_to_str(edge):
            new_pauli = ['i'] * self._num_qubits

            new_pauli[edge[0]] = 'z'
            new_pauli[edge[1]] = 'z'

            return ''.join(new_pauli)

        self._hamiltonian = [[-1.0, convert_to_str(edge)] for edge in edges]

    def decode_bitstring(self, bitstring: str) -> dict:
        """Decodes the measurement result into problem solution, i.e., set partition

        :param bitstring: Measurement result with the largest probability
        :return: Solution to the Max Cut problem
        """
        bitstring = bitstring[::-1]

        return {i: bitstring[i] for i in range(len(bitstring))}
