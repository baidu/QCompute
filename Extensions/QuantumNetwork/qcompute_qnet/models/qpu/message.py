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
Module for messages utilized in the qpu module.
"""

from Extensions.QuantumNetwork.qcompute_qnet.messages.message import QuantumMessage


class QuantumMsg(QuantumMessage):
    r"""Class for the quantum messages utilized in the qpu module.

    Attributes:
        index (int): index of the quantum message
    """

    def __init__(self, data: "QuantumState", index: int):
        r"""Constructor for QuantumMsg class.

        Args:
            data (QuantumState): quantum state of the quantum message
            index (int): index of the quantum message
        """
        super().__init__(data)
        self.index = index
        