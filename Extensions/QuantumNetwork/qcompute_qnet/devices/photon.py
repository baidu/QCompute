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
Module for photons.
"""

from qcompute_qnet.messages.message import QuantumMessage

__all__ = [
    "Photon"
]


class Photon(QuantumMessage):
    r"""Class for the simulation of photons.

    Note that ``Photon`` is a specific type of ``QuantumMessage``.

    Attributes:
        wavelength (float): wavelength of the photon in nanometers
        state (MixedState): quantum state of the photon
        index (int): index of the photon in a pulse sequence

    Note:
        The ``index`` is actually obtained from a time synchronization module, but we regard it as an attribute of
        the photon for simplicity.
    """

    def __init__(self, wavelength=None, state=None, index=None):
        r"""Constructor for Photon class.

        Args:
            wavelength (float): wavelength of the photon in nanometers
            state (MixedState): quantum state of the photon
            index (int): index of the photon in a pulse sequence
        """
        self.wavelength = wavelength
        self.state = state
        self.index = index
