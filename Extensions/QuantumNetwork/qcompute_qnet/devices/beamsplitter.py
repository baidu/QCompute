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
Module for beam splitters.
"""

from typing import Tuple
from qcompute_qnet.core.des import Entity

__all__ = [
    "PolarizationBeamSplitter"
]


class PolarizationBeamSplitter(Entity):
    r"""Class for creating a polarization beam splitter.

    Attributes:
        clock (int): scheduled time of photon reception in picoseconds
        frequency (float): frequency of receiving photon pulse in hertz
        bases (List[numpy.ndarray]): measurement bases
        receivers (Tuple[Entity]): entities to receive transmitted photons
    """

    def __init__(self, name: str, env=None):
        r"""Constructor for polarization beam splitters.

        Args:
            name (str): name of the polarization beam splitter
            env (DESEnv): discrete-event simulation environment
        """
        super().__init__(name, env)
        self.clock = 0
        self.frequency = 0
        self.bases = []
        self.receivers = ()

    def init(self) -> None:
        r"""Polarization beam splitter initialization.
        """
        assert self.owner != self, f"The polarization beam splitter {self.name} has no owner!"

    def set(self, **kwargs) -> None:
        r"""Set given parameters.

        Args:
            **kwargs: keyword arguments to set
        """
        for attr in kwargs:
            if attr == "clock":
                assert isinstance(kwargs[attr], int), "'clock' should be an int value."
                self.__setattr__(attr, kwargs[attr])
            elif attr == "frequency":
                assert isinstance(kwargs[attr], float) or isinstance(kwargs[attr], int), \
                    "'frequency' should be a float or int value."
                self.__setattr__(attr, kwargs[attr])
            elif attr == "bases":
                assert isinstance(kwargs[attr], list), "'bases' should be a list."
                self.__setattr__(attr, kwargs[attr])
            elif attr == "receivers":
                assert isinstance(kwargs[attr], tuple) and len(kwargs[attr]) == 2, \
                    f"'receivers' should be a tuple of length 2."
                self.__setattr__(attr, kwargs[attr])
            else:
                raise TypeError(f"Setting {attr} is not allowed in {self.name}")

    def receive(self, photon: "Photon") -> None:
        r"""Receive an incoming photon.

        Args:
            photon (Photon): received photon
        """
        assert self.frequency != 0, "Should set a frequency first."
        outcome = photon.state.measure(self.bases[photon.index])
        self.receivers[outcome].receive(photon.index)

    def print_parameters(self) -> None:
        r"""Print parameters of the polarization beam splitter.
        """
        print("-" * 50)
        print(f"Details of Polarization BeamSplitter: {self.name}")
        print(f"frequency: {self.frequency}\n"
              f"receivers: {self.receivers}")
