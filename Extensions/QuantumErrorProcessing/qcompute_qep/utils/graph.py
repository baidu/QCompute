#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
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

"""This script aims to collect functions related to the graph data structure
manipulation.

We use the ``networkx`` package to represent the graph data structure.
"""

from typing import List

import networkx as nx


def connected_subgraphs(G: nx.Graph, k: int) -> List[nx.Graph]:
    r"""Find all subgraphs of size :math:`k` of a graph.

    We assume that the input graph is undirected. Each subgraph must satisfy the following two conditions:

    1. The subgraph is connected; and
    2. The number of nodes of the subgraph is k.

    References: https://www.py4u.net/discuss/199398

    We used ESU(Exact Subgraph Enumeration) algorithm to improve the performance.

    :param G: nx.Graph, an undirected graph whose connected subgraphs to be extracted
    :param k: int, the number of nodes in each connected subgraph
    :return: List[nx.Graph], a list of connected subgraphs of type ``networkx.Graph``
    """
    subgraphs = []

    def dfs(selected_nodes: List, extension, removed):
        if len(selected_nodes) == k - 1:
            for final_node in extension:
                subgraphs.append(G.subgraph(selected_nodes + [final_node]))
            return

        for i in range(len(extension)):
            current_node = extension[i]
            dfs(selected_nodes + [current_node],
                extension[i + 1:] + [nxt_node
                                     for nxt_node in G.neighbors(current_node)
                                     if nxt_node > start and
                                     nxt_node not in selected_nodes and
                                     nxt_node not in removed
                                     and nxt_node not in extension],
                removed + [current_node], )
            removed.append(current_node)

    for node in G.nodes:
        start = node
        dfs([node], [x for x in G.neighbors(node) if x > node], [])

    return subgraphs
