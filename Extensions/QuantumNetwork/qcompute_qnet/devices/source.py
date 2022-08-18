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
Module for photon sources.
"""

from typing import List
from numpy import random
from qcompute_qnet.core.des import Entity, EventHandler
from qcompute_qnet.devices.photon import Photon
from qcompute_qnet.quantum.state import MixedState

__all__ = [
    "PhotonSource"
]


class PhotonSource(Entity):
    r"""Class for the simulation of a photon source.

    Attributes:
        frequency (float): frequency at which light pulses are emitted in hertz
        wavelength (float): wavelength of photons in nanometers
        bandwidth (float): bandwidth of photon wavelength in nanometers
        mean_photon_num (float): average number of photons per signal pulse
    """

    def __init__(self, name: str, env=None, frequency=80e6, wavelength=1550, bandwidth=0, mean_photon_num=0.5):
        r"""Constructor for the PhotonSource class.

        Args:
            name (str): name of the photon source
            env (DESEnv): discrete-event simulation environment
            frequency (float): frequency at which light pulses are emitted in hertz
            wavelength (float): wavelength of photons in nanometers
            bandwidth (float): bandwidth of photon wavelength in nanometers
            mean_photon_num (float): average number of photons per signal pulse
        """
        super().__init__(name, env)
        self.frequency = frequency
        self.wavelength = wavelength
        self.bandwidth = bandwidth
        self.mean_photon_num = mean_photon_num

    def init(self) -> None:
        r"""Photon source initialization.
        """
        assert self.owner != self, f"The photon source {self.name} has no owner!"

    def set(self, **kwargs) -> None:
        r"""Set given parameters for the photon source.

        Args:
            **kwargs: keyword arguments to set
        """
        for attr in kwargs:
            if attr == "frequency":
                assert isinstance(kwargs[attr], float), "'frequency' should be a float value."
                self.__setattr__(attr, kwargs[attr])
            elif attr == "wavelength":
                assert isinstance(kwargs[attr], float), "'wavelength' should be a float value."
                self.__setattr__(attr, kwargs[attr])
            elif attr == "bandwidth":
                assert isinstance(kwargs[attr], float), "'bandwidth' should be a float value."
                self.__setattr__(attr, kwargs[attr])
            elif attr == "mean_photon_num":
                assert isinstance(kwargs[attr], float), f"'mean_photon_num' should be a float value."
                self.__setattr__(attr, kwargs[attr])
            else:
                raise TypeError(f"Setting {attr} is not allowed in {self.name}")

    def emit(self, dst: "Node", states: List, mean_photon_num_list=None) -> None:
        r"""Emit photons in specific states to the destination.

        Args:
            dst (Node): the destination node of the emitted photons
            states (List[numpy.ndarray]): expected quantum states to emit
            mean_photon_num_list (list): source intensity for each emitted states

        Important:
            The ``mean_photon_num_list`` is a list of the same length as ``states``, indicating the intensity of
            each emitted state. This will be used in protocols like decoy-state BB84.
            If ``mean_photon_num_list == None``, all states will be emitted with the same ``mean_photon_num``.
        """
        interval = round(1e12 / self.frequency)

        for i, state in enumerate(states):
            # Number of photons in a pulse (Poisson distribution)
            mean_photon_num_list = [self.mean_photon_num] * len(states) if mean_photon_num_list is None \
                else mean_photon_num_list
            photons_num = random.poisson(mean_photon_num_list[i])
            for _ in range(photons_num):
                wavelength = self.bandwidth * random.randn() + self.wavelength
                photon = Photon(wavelength=wavelength, state=MixedState(state, [0]), index=i)
                handler = EventHandler(self.owner, "send_quantum_msg", [dst, photon])
                self.scheduler.schedule_after(i * interval, handler)

    def print_parameters(self) -> None:
        r"""Print parameters of the photon source.
        """
        print(f"Details of Photon Source: {self.name}\n"
              f"frequency: {self.frequency} hertz\n"
              f"wavelength: {self.wavelength} nanometers\n"
              f"bandwidth: {self.bandwidth} nanometers\n"
              f"mean photon number: {self.mean_photon_num}")
