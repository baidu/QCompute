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
Module for building a quantum network.
"""

from typing import List
import networkx
import matplotlib.pyplot as plt
from qcompute_qnet.core.des import DESEnv, Entity
from qcompute_qnet.quantum.circuit import Circuit
from qcompute_qnet.topology.node import Node
from qcompute_qnet.topology.link import Link
from qcompute_qnet.devices.channel import Channel, ClassicalChannel, QuantumChannel
from qcompute_qnet.devices.channel import DuplexChannel, DuplexClassicalFiberChannel, DuplexQuantumFiberChannel

__all__ = [
    "Network"
]


class Network(Entity):
    r"""Class for building a quantum network.

    Attributes:
        classical_topology (networkx.MultiDiGraph): classical topology of the network
        quantum_topology (networkx.MultiDiGraph): quantum topology of the network
        default_circuit (Circuit): the default global quantum circuit that records the evolution of network status
        circuits (List[Circuit]): a list of circuits that record the evolutions of network status

    Note:
        ``Network`` is the class for constructing a quantum network and saving its topology. In a network, there are
        communication nodes and links connecting these nodes generally. So we should install nodes and links
        to a network before running the simulation.

    Examples:
        Here we introduce the procedures of building a quantum network. Firstly, we need to create a network.

        >>> network = Network(name="Network")

        Secondly, we should create nodes and links in a network and connect the nodes.

        >>> alice = Node(name="Alice")
        >>> bob = Node(name="Bob")
        >>> link_ab = Link(name="Link_Alice_Bob", ends=(alice, bob))

        Finally, we should install the nodes and links to the network.

        >>> network.install([alice, bob, link_ab])
    """

    def __init__(self, name: str, env=None):
        """Constructor for the Network class.

        Args:
            name (str): name of the network
            env (DESEnv): discrete-event simulation environment of the network
        """
        super().__init__(name, env)
        self.classical_topology = networkx.MultiDiGraph()
        self.quantum_topology = networkx.MultiDiGraph()
        self.default_circuit = Circuit()
        self.circuits = []

    def init(self) -> None:
        r"""Network initialization.

        The ``init`` method will be triggered by the environment initialization.
        """
        self.get_topology()
        self.assign_routing_table()

    @property
    def nodes(self) -> List[Node]:
        r"""Filter the nodes from the network components.

        Returns:
            List[Node]: a list of nodes
        """
        return [component for component in self.components if isinstance(component, Node)]

    @property
    def links(self) -> List[Link]:
        r"""Filter the links from the network components.

        Returns:
            List[Link]: a list of links
        """
        return [component for component in self.components if isinstance(component, Link)]

    def load_topology_from(self, filename: str) -> None:
        r"""Load a network topology from the json file.

        Args:
            filename (str): name of the file saving a network topology

        Note:
            - The ``filename`` should be given as an absolute path to avoid ``FileNotFoundError``.
            - The ``load_topology_from`` method will create duplex fiber channels for a link which connects two nodes
              in this version.
        """
        import json
        topo = json.load(open(filename))
        nodes = []
        links = []

        from qcompute_qnet.models.qkd.node import EndNode, TrustedRepeaterNode, BackboneNode
        for node in topo['nodes']:
            if node['type'] == "EndNode":
                nodes.append(EndNode(node['name'], location=(node['longitude'], node['latitude'])))
            elif node['type'] == "TrustedRepeaterNode":
                nodes.append(TrustedRepeaterNode(node['name'], location=(node['longitude'], node['latitude'])))
            elif node['type'] == "BackboneNode":
                nodes.append(BackboneNode(node['name'], location=(node['longitude'], node['latitude'])))
            else:
                nodes.append(Node(node['name']))

        for link in topo['links']:
            lk = Link(f"{link['node1']}_{link['node2']}")
            n1 = lk.env.get_node(link['node1'])
            n2 = lk.env.get_node(link['node2'])
            lk.connect(n1, n2)
            # Set a duplex classical channel
            c_distance = link['cchannel']['distance']
            c_loss = link['cchannel']['loss'] if "loss" in link['cchannel'].keys() else 0
            c_delay = link['cchannel']['delay'] if "delay" in link['cchannel'].keys() else None
            ch = DuplexClassicalFiberChannel(f"c_{link['node1']}_{link['node2']}",
                                             distance=c_distance, loss=c_loss, delay=c_delay)
            ch.connect(n1, n2)
            # Set a duplex quantum channel
            q_distance = link['qchannel']['distance']
            q_loss = link['qchannel']['loss'] if "loss" in link['qchannel'].keys() else None
            q_delay = link['qchannel']['delay'] if "delay" in link['qchannel'].keys() else None
            qh = DuplexQuantumFiberChannel(f"q_{link['node1']}_{link['node2']}",
                                           distance=q_distance, loss=q_loss, delay=q_delay)
            qh.connect(n1, n2)
            lk.install([ch, qh])
            links.append(lk)

        self.install(nodes)
        self.install(links)

    def get_topology(self) -> None:
        r"""Get the classical and quantum topology of the network.
        """
        self.get_classical_topology()
        self.get_quantum_topology()

    def get_classical_topology(self) -> None:
        r"""Get the classical topology of the network.
        """
        self.classical_topology.add_nodes_from(self.nodes)
        self.__add_channels(self.classical_topology, [ClassicalChannel, DuplexClassicalFiberChannel])

    def get_quantum_topology(self) -> None:
        r"""Get the quantum topology of the network.
        """
        self.quantum_topology.add_nodes_from(self.nodes)
        self.__add_channels(self.quantum_topology, [QuantumChannel, DuplexQuantumFiberChannel])

    def __add_channels(self, topology: networkx.MultiDiGraph, channel_types: List) -> None:
        r"""Add channels of given types to the topology.

        Args:
            topology (networkx.MultiDiGraph): topology to add channels
            channel_types (List): specified channel types
        """
        for link in self.links:
            for component in link.components:
                if type(component) in channel_types:
                    if isinstance(component, Channel):
                        topology.add_weighted_edges_from([(component.sender, component.receiver,
                                                           component.get_distance())])
                    elif isinstance(component, DuplexChannel):
                        topology.add_weighted_edges_from([(component.channel1_2.sender,
                                                           component.channel1_2.receiver,
                                                           component.channel1_2.get_distance())])
                        topology.add_weighted_edges_from([(component.channel2_1.sender,
                                                           component.channel2_1.receiver,
                                                           component.channel2_1.get_distance())])

    def assign_routing_table(self) -> None:
        r"""Assign routing tables for trusted repeater nodes in the network.

        Note:
            We use the static routing here and apply the Dijkstra's method to compute the shortest weighted path.
            The classical and quantum routing tables are computed separately.
            Both routing costs depend on the length of the communication channel.
        """
        from qcompute_qnet.models.qkd.node import TrustedRepeaterNode
        for node in self.nodes:
            if isinstance(node, TrustedRepeaterNode):
                for dst in self.nodes:
                    if node != dst:
                        classical_path = networkx.dijkstra_path(self.classical_topology,
                                                                source=node, target=dst, weight='weight')
                        node.classical_routing_table[dst] = classical_path[1]
                        quantum_path = networkx.dijkstra_path(self.quantum_topology,
                                                              source=node, target=dst, weight='weight')
                        node.quantum_routing_table[dst] = quantum_path[1]

    def print_classical_topology(self, geo=False) -> None:
        r"""Print the classical topology of the network.

        Args:
            geo (bool): whether to organize the layout by locations of the nodes
        """
        self.get_classical_topology()

        edge_options = {"width": 1, "style": "dashed", "edge_color": "tab:orange", "connectionstyle": "arc3, rad=0.05"}
        self.__print_topology(self.classical_topology, geo, **edge_options)

    def print_quantum_topology(self, geo=False) -> None:
        r"""Print the quantum topology of the network.

        Args:
            geo (bool): whether to organize the layout by locations of the nodes
        """
        self.get_quantum_topology()

        edge_options = {"width": 1, "edge_color": "tab:orange", "connectionstyle": "arc3, rad=0.05"}
        self.__print_topology(self.quantum_topology, geo, **edge_options)

    def __print_topology(self, topology: networkx.MultiDiGraph, geo: bool, **edge_options) -> None:
        r"""Print the topology of the network.

        Args:
            topology (networkx.MultiDiGraph): topology to print
            geo (bool): whether to organize the layout by locations of the nodes
            **edge_options: keyword arguments for edges
        """
        if geo:  # print the topology with locations
            assert all(node.location is not None for node in self.nodes), f"Not all nodes have a geographical location!"
            plt.rcParams['figure.figsize'] = (16, 12)  # set figure size
            nodes_pos = {node: node.location for node in self.nodes}

            from qcompute_qnet.models.qkd.node import EndNode, TrustedRepeaterNode, BackboneNode
            # End nodes
            end_nodes = [node for node in self.nodes if isinstance(node, EndNode)]
            ends_pos = {node: node.location for node in end_nodes}
            ends_options = {"nodelist": end_nodes, "node_size": 200, "node_color": "black"}
            # Repeater nodes
            repeater_nodes = [node for node in self.nodes if isinstance(node, TrustedRepeaterNode)]
            repeaters_pos = {node: node.location for node in repeater_nodes}
            repeaters_options = {"nodelist": repeater_nodes, "node_size": 200,
                                 "node_color": "blue", "edgecolors": "black"}
            # Backbone nodes
            backbone_nodes = [node for node in self.nodes if isinstance(node, BackboneNode)]
            backbones_pos = {node: node.location for node in backbone_nodes}
            backbones_options = {"nodelist": backbone_nodes, "node_size": 200,
                                 "node_color": "red", "edgecolors": "black"}
            # Labels
            labels = {node: node.name for node in self.nodes}
            labels_pos = {node: (node.location[0], node.location[1] - 0.01) for node in self.nodes}

            networkx.draw_networkx_nodes(topology, ends_pos, **ends_options)  # draw end nodes
            networkx.draw_networkx_nodes(topology, repeaters_pos, **repeaters_options)  # draw repeater nodes
            networkx.draw_networkx_nodes(topology, backbones_pos, **backbones_options)  # draw backbone nodes
            networkx.draw_networkx_edges(topology, nodes_pos, **edge_options)
            networkx.draw_networkx_labels(topology, labels_pos, labels, font_size=10)

        else:
            labels = {node: node.name for node in self.nodes}
            pos = networkx.spring_layout(topology)

            networkx.draw_networkx_nodes(topology, pos, node_size=200, node_color="tab:red")
            networkx.draw_networkx_edges(topology, pos, **edge_options)
            networkx.draw_networkx_labels(topology, pos, labels, font_size=10)

        plt.show()
