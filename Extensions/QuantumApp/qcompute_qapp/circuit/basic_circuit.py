# !/usr/bin/env python3
# -*- coding: UTF-8 -*-
# Copyright (c) 2021 Baidu, Inc. All Rights Reserved.
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

"""
Basic Circuit
"""

from abc import ABC, abstractmethod
from QCompute.QPlatform.QRegPool import QRegPool


class BasicCircuit(ABC):
    r"""Basic Circuit class
    """
    def __init__(self, num: int):
        r"""The constructor of the BasicCircuit class

        Args:
            num (int): Number of qubits

        """
        self._num = num

    @abstractmethod
    def add_circuit(self, q: QRegPool) -> None:
        """Adds circuit to the register.

        Args:
            q (QRegPool): Quantum register to which this circuit is added

        """
        raise NotImplementedError
