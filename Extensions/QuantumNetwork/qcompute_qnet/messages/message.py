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
Module for messages.
"""

from abc import ABC
from typing import Any, Dict

__all__ = [
    "Message",
    "ClassicalMessage",
    "QuantumMessage",
]


class Message(ABC):
    r"""Abstract class for the classical message in a network.

    Attributes:
        data: message content
    """

    def __init__(self, data: Any):
        r"""Constructor for Message class.

        Args:
            data (Any): message content
        """
        self.data = data


class ClassicalMessage(Message):
    r"""Class for the classical message as the carrier of classical information.

    Attributes:
        src (Node): source of the classical message
        dst (Node): destination of the classical message
        protocol (type) : protocol related to the classical message
        data (Dict): message content
    """

    def __init__(self, src: "Node", dst: "Node", protocol: type, data: Dict):
        r"""Constructor for ClassicalMessage class.

        Args:
            src (Node): source of the classical message
            dst (Node): destination of the classical message
            protocol (type) : protocol related to the classical message
            data (Dict): message content
        """
        super().__init__(data)
        self.src = src
        self.dst = dst
        self.protocol = protocol


class QuantumMessage(Message):
    r"""Class for the quantum message as the carrier of quantum information.

    Attributes:
        data (QuantumState): quantum state of the quantum message
    """

    def __init__(self, data: "QuantumState"):
        r"""Constructor for QuantumMessage class.

        Args:
            data (QuantumState): quantum state of the quantum message
        """
        super().__init__(data)
