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
Minimum Clique Cover of the Hamiltonian
"""

from typing import List
import networkx as nx


def _check_qwc(pauli1: str, pauli2: str) -> bool:
    r"""Checks whether two pauli terms are qubitwise commute to each other

    Args:
        pauli1 (str): Pauli term in the form of a string
        pauli2 (str): Pauli term in the form of a string

    Returns:
        bool: whether two pauli terms are qubitwise commute to each other

    """
    for index in range(len(pauli1)):
        sigma1 = pauli1[index]
        sigma2 = pauli2[index]
        if "i" in [sigma1, sigma2]:
            continue
        elif sigma1 == sigma2:
            continue
        else:
            return False

    return True


def _generate_graph(h: list) -> nx.Graph:
    r"""Generates a Hamiltonian graph with edges representing the connection of qubitwise terms

    Args:
        h (list): Target Hamiltonian in the form of list of lists of int and string, i.e. [[1.0, 'xx'],[-1.0,  'yy']]

    Returns:
        nx.graph: Hamiltonian graph with edges representing the connection of qubitwise terms

    """
    # Combine same terms
    h_dict = {}
    for coefficient, pauli in h:
        if pauli in h_dict:
            h_dict[pauli] += coefficient
        else:
            h_dict[pauli] = coefficient
    h_ls = list(h_dict)

    # Generate graph
    g = nx.Graph()
    g.add_nodes_from(h_ls)
    # Add edges according to qwc property
    edges = []
    for i in range(len(h_ls)):
        for j in range(i + 1, len(h_ls)):
            if _check_qwc(h_ls[i], h_ls[j]):
                edges.append((h_ls[i], h_ls[j]))
    g.add_edges_from(edges)

    return g


def grouping_hamiltonian(hamiltonian: list, coloring_strategy: str = "largest_first") -> List[List[str]]:
    r"""Finds the minimum clique cover of the Hamiltonian graph, which is used for simultaneous Pauli measurement

    Args:
        hamiltonian (list): Hamiltonian of the target system
        coloring_strategy (str): Graph coloring strategy chosen from the following: 'largest_first', 'random_sequential',
            'smallest_last', 'independent_set', 'connected_sequential_bfs', 'connected_sequential_dfs', 'connected_sequential',
            'saturation_largest_first', and 'DSATUR'; defaults to 'largest_first'

    Returns:
        List[List[str]]: List of cliques consisting of Pauli strings to be measured together
    """

    g = _generate_graph(hamiltonian)
    g_bar = nx.complement(g)

    coloring = nx.coloring.greedy_color(g_bar, strategy=coloring_strategy)
    cliques_dict = {}
    for pauli_term in hamiltonian:
        coloring_index = coloring[pauli_term[1]]

        if coloring_index in cliques_dict:
            cliques_dict[coloring_index].append(pauli_term)
        else:
            cliques_dict[coloring_index] = [pauli_term]
    cliques = list(cliques_dict.values())

    return cliques
