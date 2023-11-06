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

r"""
Module for utility functions used in quantum key distribution simulation.
"""

from typing import List
import matplotlib.pyplot as plt
import networkx
from Extensions.QuantumNetwork.qcompute_qnet.topology.network import Network
from Extensions.QuantumNetwork.qcompute_qnet.models.qkd.node import EndNode, BackboneNode, RMPEndNode, RMPRepeaterNode

__all__ = ["summary", "print_traffic"]


def summary(network: "Network") -> None:
    r"""Summarize the delivered requests and keys."""
    sum_reqs = 0
    sum_keys = 0

    for node in network.nodes:
        if isinstance(node, RMPEndNode):
            sum_reqs += node.reqs_delivered
            sum_keys += node.keys_delivered

    sum_reqs = sum_reqs // 2
    sum_keys = sum_keys // 2

    print(f"{sum_reqs} requests are processed.")
    print(f"{sum_keys} keys are delivered.")


def print_traffic(network: "Network", num_color: int) -> None:
    r"""Print the traffic of the network. The color of a repeater node is determined by the number of
    keys it delivered.

    Args:
        network (Network): network to print traffic
        num_color (int): number of gradient colors
    """

    # Record the delivered keys of all repeater nodes
    delivered_keys_set = {}
    for node in network.nodes:
        if isinstance(node, RMPRepeaterNode):
            delivered_keys_set[node] = node.keys_delivered

    # Get the gradient color list
    def gen_colors(num_color: int) -> List:
        """Get the gradient color list.

        Args:
            num_color (int): number of the colors to generate

        Returns:
            List: generated color list
        """
        values = [int(i * 250 / num_color) for i in range(num_color)]
        color_list = ["#%02x%02x%02x" % (200, int(g), 40) for g in values]
        color_list.reverse()
        return color_list

    color_list = gen_colors(num_color)

    # Set color for each repeater node
    step = (max(delivered_keys_set.values()) - min(delivered_keys_set.values())) // num_color
    assert step != 0, f"Illegal step value: 0"
    color_dict = {}
    for node in network.nodes:
        if isinstance(node, RMPRepeaterNode):
            idx = delivered_keys_set[node] // step
            if idx >= num_color:
                idx = num_color - 1
            color_dict[node] = color_list[idx]

    # Get the quantum topology
    network.get_quantum_topology()
    topology = network.quantum_topology

    edge_options = {"width": 1, "edge_color": "tab:orange", "connectionstyle": "arc3, rad=0.05"}
    plt.rcParams["figure.figsize"] = (16, 12)  # set figure size
    nodes_pos = {node: node.location for node in network.nodes}

    # End nodes
    end_nodes = [node for node in network.nodes if isinstance(node, EndNode)]
    ends_pos = {node: node.location for node in end_nodes}
    ends_options = {"nodelist": end_nodes, "node_size": 200, "node_color": "black"}

    # Backbone nodes
    backbone_nodes = [node for node in network.nodes if isinstance(node, BackboneNode)]
    backbones_pos = {node: node.location for node in backbone_nodes}
    backbones_options = {"nodelist": backbone_nodes, "node_size": 200, "node_color": "blue", "edgecolors": "black"}
    # Labels
    labels = {node: node.name for node in network.nodes}
    labels_pos = {node: (node.location[0], node.location[1] - 0.01) for node in network.nodes}

    networkx.draw_networkx_nodes(topology, ends_pos, **ends_options)  # draw end nodes

    # Draw repeater nodes
    for node in network.nodes:
        if isinstance(node, RMPRepeaterNode):
            repeaters_options = {
                "nodelist": [node],
                "node_size": 200,
                "node_color": color_dict[node],
                "edgecolors": "black",
            }
            networkx.draw_networkx_nodes(topology, {node: node.location}, **repeaters_options)

    networkx.draw_networkx_nodes(topology, backbones_pos, **backbones_options)  # draw backbone nodes
    networkx.draw_networkx_edges(topology, nodes_pos, **edge_options)
    networkx.draw_networkx_labels(topology, labels_pos, labels, font_size=10, font_color="black")

    plt.show()
