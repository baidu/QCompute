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
Module for photon detectors.
"""

from typing import List, Tuple
from numpy import random
from qcompute_qnet.core.des import Entity, EventHandler
from qcompute_qnet.devices.beamsplitter import PolarizationBeamSplitter

__all__ = [
    "SinglePhotonDetector",
    "PolarizationDetector"
]


class SinglePhotonDetector(Entity):
    r"""Class for creating a single photon detector.

    Attributes:
        _is_working (bool): whether the single photon detector is working
        efficiency (float): probability that successfully detect a photon when it arrives
        dark_count (float): frequency in hertz that the detector is triggered in the absence of any incident photons
        count_rate (float): maximum photon detection rate in hertz
        resolution (int): how accurately the time in picoseconds the detector can record
        _valid_time (int): next valid time for photon detection
    """

    def __init__(self, name: str, env=None, efficiency=0.4, dark_count=0.0, count_rate=10e6, resolution=1):
        r"""Constructor for SinglePhotonDetector class.

        Args:
            name (str): name of the single photon detector
            env (DESEnv): discrete-event simulation environment
            efficiency (float): probability that successfully detect a photon when it arrives
            dark_count (float): frequency in hertz that the detector is triggered in the absence of any incident photons
            count_rate (float): maximum photon detection rate in hertz
            resolution (int): how accurately the time in picoseconds the detector can record
        """
        super().__init__(name, env)
        self._is_working = False
        self.efficiency = efficiency
        self.dark_count = dark_count
        self.count_rate = count_rate
        self.resolution = resolution
        self._valid_time = 0
        self.__records = []

    def init(self) -> None:
        r"""Detector initialization.
        """
        assert self.owner != self, f"The single photon detector {self.name} has no owner!"

    def turn_on(self) -> None:
        r"""Turn on the single photon detector.
        """
        self._is_working = True
        if self.dark_count > 0:
            delay = round(random.exponential(1 / self.dark_count) * 1e12)
            self.scheduler.schedule_after(delay, EventHandler(self, "_receive_dark_count"))

    def turn_off(self) -> None:
        r"""Turn off the single photon detector.
        """
        self._is_working = False
        self.__records = []

    def is_ready(self) -> bool:
        r"""Check if the detector is ready for detection.

        Returns:
            bool: ready or not
        """
        return self.env.now >= self._valid_time

    def _receive_dark_count(self) -> None:
        r"""Simulate the dark count events.
        """
        pass

    def set(self, **kwargs) -> None:
        r"""Set given parameters.

        Args:
            **kwargs: keyword arguments to set
        """
        for attr in kwargs:
            if attr == "efficiency":
                assert isinstance(kwargs[attr], float), "'efficiency' should be a float value."
                self.__setattr__(attr, kwargs[attr])
            elif attr == "dark_count":
                assert isinstance(kwargs[attr], float), "'dark_count' should be a float value."
                self.__setattr__(attr, kwargs[attr])
            elif attr == "count_rate":
                assert isinstance(kwargs[attr], float), "'count_rate' should be a float value."
                self.__setattr__(attr, kwargs[attr])
            elif attr == "resolution":
                assert isinstance(kwargs[attr], int), f"'resolution' should be an int value."
                self.__setattr__(attr, kwargs[attr])
            else:
                raise TypeError(f"Setting {attr} is not allowed in {self.name}")

    def print_parameters(self) -> None:
        r"""Print parameters of the single photon detector.
        """
        print("-" * 50)
        print(f"Details of Single Photon Detector: {self.name}\n"
              f"efficiency: {self.efficiency}\n"
              f"dark count: {self.dark_count} hertz\n"
              f"count rate: {self.count_rate} hertz\n"
              f"resolution: {self.resolution} picoseconds")

    def receive(self, photon_index: int) -> None:
        r"""Receive an incoming photon and record its index and arrival time.

        Args:
            photon_index (int): index of the incoming photon in a pulse
        """
        if self._is_working:
            if random.random_sample() < self.efficiency and self.is_ready():
                self.__records.append((photon_index, round(self.env.now / self.resolution) * self.resolution))
                self._valid_time = self.env.now + round(1e12 / self.count_rate)

    def pop_records(self) -> List[Tuple]:
        r"""Pop the detection records.

        Returns:
            List[Tuple]: photon indices and time records of photon detection
        """
        records, self.__records = self.__records, []
        return records


class PolarizationDetector(Entity):
    r"""Class for polarization detector.

    Attributes:
        detectors (Tuple): single photon detector components of the polarization detector
        beamsplitter (BeamSplitter): beam splitter component of the polarization detector
    """

    def __init__(self, name: str, env=None):
        r"""Constructor for PolarizationDetector class.

        Args:
            name (str): name of the polarization detector
            env (DESEnv): discrete-event simulation environment
        """
        super().__init__(name, env)
        self.detectors = (SinglePhotonDetector(name + ".SPD0", env),
                          SinglePhotonDetector(name + ".SPD1", env))
        self.beamsplitter = PolarizationBeamSplitter(name + ".PBS", env)
        self.install(list(self.detectors))
        self.install(self.beamsplitter)
        self.beamsplitter.set(receivers=self.detectors)

    def init(self) -> None:
        r"""Detector initialization.
        """
        assert self.owner != self, f"The polarization detector {self.name} has no owner!"

    def set_beamsplitter(self, **kwargs) -> None:
        r"""Set given parameters for the beam splitter.

        Args:
            **kwargs: keyword arguments to set
        """
        self.beamsplitter.set(**kwargs)

    def set_detectors(self, **kwargs) -> None:
        r"""Set given parameters for the two single photon detectors.

        Args:
            **kwargs: keyword arguments to set
        """
        self.detectors[0].set(**kwargs)
        self.detectors[1].set(**kwargs)

    def receive(self, photon: "Photon") -> None:
        r"""Receive an incoming photon.

        Args:
            photon (Photon): photon to receive
        """
        self.beamsplitter.receive(photon)

    def print_parameters(self) -> None:
        r"""Print parameters of the polarization detector.
        """
        self.detectors[0].print_parameters()
        self.detectors[1].print_parameters()
        self.beamsplitter.print_parameters()
