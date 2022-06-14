# -*- coding: UTF-8 -*-
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

"""
Basic Circuit
"""

from abc import ABC, abstractmethod
from QCompute.QPlatform.QRegPool import QRegPool


class BasicCircuit(ABC):
    """Basic Circuit class
    """
    def __init__(self, num: int):
        """The constructor of the BasicCircuit class

        :param num: Number of qubits
        """
        self._num = num

    @abstractmethod
    def add_circuit(self, q: QRegPool):
        """Adds circuit to the register.

        :param q: Quantum register to which this circuit is added
        """
        raise NotImplementedError
