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
Module for mobility of nodes.
"""

from abc import ABC, abstractmethod

__all__ = [
    "Mobility",
    "Track"
]


class Mobility:
    r"""Class for the mobility of a node.

    A mobile node moves on its preset track.

    Attributes:
        track (Track): track of the mobile node
    """

    def __init__(self):
        r"""Constructor for Mobility class.
        """
        self.track = None

    def set_track(self, track: "Track") -> None:
        r"""Set a track for the mobile node.

        Args:
            track (Track): track to set
        """
        self.track = track


class Track(ABC):
    r"""Class for the track of a mobile node.

    Attributes:
        ref_node (Node): reference node
        ref_time (int): time elapsed from the start of the simulation to the start of the movement
    """

    def __init__(self, ref_node: "Node", ref_time=0):
        r"""Constructor for Track class.

        Args:
            ref_node (Node): reference node
            ref_time (int): time elapsed from the start of the simulation to the start of the movement
        """
        self.ref_node = ref_node
        self.ref_time = ref_time

    @abstractmethod
    def time2distance(self, current_time: int) -> float:
        r"""Mapping the time to the distance from the mobile node to the reference node.

        Args:
            current_time (int): current time of the simulation

        Returns:
            float: the distance from the mobile node to the reference node
        """
        pass

    @abstractmethod
    def distance2loss(self, distance: float) -> float:
        r"""Mapping the distance to the communication loss.

        Args:
            distance (float): the distance from the mobile node to the reference node

        Returns:
            float: communication loss from the mobile node to the reference node
        """
        pass
