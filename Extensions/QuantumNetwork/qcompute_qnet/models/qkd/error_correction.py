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
Module for error correction protocols.
"""

from abc import ABC
from qcompute_qnet.protocols.protocol import Protocol

__all__ = [
    "ErrorCorrection",
    "Cascade",
    "LDPC"
]


class ErrorCorrection(Protocol, ABC):
    r"""Class for error correction protocols.
    """

    def __init__(self, name: str):
        r"""Constructor for ErrorCorrection class.

        Args:
            name (str): name of the error correction protocol
        """
        super().__init__(name)


class Cascade(ErrorCorrection):
    r"""Class for the cascade protocol.
    """

    def __init__(self, name: str):
        r"""Constructor for Cascade class.

        Args:
            name (str): name of the cascade protocol
        """
        super().__init__(name)


class LDPC(ErrorCorrection):
    r"""Class for the low-density parity-check protocol.
    """

    def __init__(self, name: str):
        r"""Constructor for LDPC class.

        Args:
            name (str): name of the LDPC protocol
        """
        super().__init__(name)
