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
Module for messages in quantum key distribution.
"""

from enum import Enum, unique
from typing import Dict
from qcompute_qnet.messages.message import ClassicalMessage

__all__ = [
    "QKDMessage"
]


class QKDMessage(ClassicalMessage):
    r"""Class for the classical control messages in quantum key distribution.

    Attributes:
        src (Node): source of QKDMessage
        dst (Node): destination of QKDMessage
        protocol (type): protocol of QKDMessage
        data (Dict): message content
    """

    def __init__(self, src: "Node", dst: "Node", protocol: type, data: Dict):
        r"""Constructor for QKDMessage class.

        Args:
            src (Node): source of QKDMessage
            dst (Node): destination of QKDMessage
            protocol (type): protocol of QKDMessage
            data (Dict): message content
        """
        super().__init__(src, dst, protocol, data)

    @unique
    class Type(Enum):
        r"""Class for QKDMessage types.
        """

        REQUEST = "Request"
        ACCEPT = "Accept"
        READY = "Ready"
        CIPHERTEXT = "Ciphertext"
        ACKNOWLEDGE = "Acknowledge"
        DONE = "Done"
