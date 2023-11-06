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
Max Cut Problem
"""

from copy import deepcopy
import networkx as nx


class MaxCut:
    r"""Max Cut Problem class"""

    def __init__(self, num_qubits: int = 0, hamiltonian: list = None):
        r"""The constructor of the MaxCut class

        Args:
            num_qubits (int): Number of qubits, defaults to 0
            hamiltonian (list): Hamiltonian of the target graph of the Max Cut problem, defaults to None

        """
        self._num_qubits = num_qubits
        self._hamiltonian = hamiltonian

    @property
    def num_qubits(self) -> int:
        r"""The number of qubits used to encoding this target graph

        Returns:
            int: Number of qubits used to encoding this target graph

        """
        return self._num_qubits

    @property
    def hamiltonian(self) -> list:
        r"""The Hamiltonian of this target graph

        Returns:
            list: Hamiltonian of this target graph

        """
        return deepcopy(self._hamiltonian)

    def graph_to_hamiltonian(self, graph: nx.Graph) -> None:
        r"""Constructs Hamiltonian from the target graph of the Max Cut problem

        Args:
            graph (nx.Graph): Undirected graph without weights

        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("Error EA02002(QAPP): Type of the input graph should be networkx.Graph.")

        edges = graph.edges()

        def convert_to_str(edge):
            new_pauli = ["i"] * self._num_qubits

            new_pauli[edge[0]] = "z"
            new_pauli[edge[1]] = "z"

            return "".join(new_pauli)

        self._hamiltonian = [[-1.0, convert_to_str(edge)] for edge in edges]

    def decode_bitstring(self, bitstring: str) -> dict:
        r"""Decodes the measurement result into problem solution, i.e., set partition

        Args:
            bitstring (str): Measurement result with the largest probability

        Returns:
            dict: Solution to the Max Cut problem

        """
        bitstring = bitstring[::-1]

        return {i: bitstring[i] for i in range(len(bitstring))}
