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
Encoding Circuit
"""

import numpy as np
from QCompute.QPlatform.QRegPool import QRegPool
from QCompute.QPlatform.QOperation.RotationGate import RZ
from QCompute.QPlatform.QOperation.FixedGate import X, H, CX
from .basic_circuit import BasicCircuit


class IQPEncodingCircuit(BasicCircuit):
    """IQP Encoding Circuit class
    """
    def __init__(self, num: int, inverse: bool = False):
        """The constructor of the IQPEncodingCircuit class

        :param num: Number of qubits
        :param inverse: Whether the encoding circuit will be inverted, i.e. U^dagger(x) if True, defaults to False
        """
        super().__init__(num)
        self._is_inverse = inverse

    def add_circuit(self, q: QRegPool, x: np.ndarray):
        """Adds the encoding circuit used to map a classical data vector into its quantum feature state

        :param q: Quantum register to which this circuit is added
        :param x: Classical data vector to be encoded
        """
        if not self._is_inverse:

            # Data encoding layer
            for i in range(self._num):
                H(q[i])
                RZ(x[i])(q[i])

            # Entanglement layer
            i = 0
            while i < self._num - 1:
                CX(q[i], q[i + 1])
                RZ((x[i] * x[i + 1]))(q[i + 1])
                CX(q[i], q[i + 1])
                i += 2

            i = 1
            while i < self._num - 1:
                CX(q[i], q[i + 1])
                RZ((x[i] * x[i + 1]))(q[i + 1])
                CX(q[i], q[i + 1])
                i += 2

        else:
            # Put everything in a reverse order
            # Entanglement layer
            i = 1
            while i < self._num - 1:
                CX(q[i], q[i + 1])
                RZ((-x[i] * x[i + 1]))(q[i + 1])
                CX(q[i], q[i + 1])
                i += 2

            i = 0
            while i < self._num - 1:
                CX(q[i], q[i + 1])
                RZ((-x[i] * x[i + 1]))(q[i + 1])
                CX(q[i], q[i + 1])
                i += 2

            # Encoding layer (angles are inverted)
            for i in range(self._num):
                RZ(-x[i])(q[i])
                H(q[i])


class BasisEncodingCircuit(BasicCircuit):
    """Basis Encoding Circuit class
    """

    def __init__(self, num: int, bit_string: str):
        """The constructor of the BasisEncodingCircuit class

        :param num: Number of qubits
        :param bit_string: Bit string to be encoded as a quantum state
        """
        super().__init__(num)
        self._bit_string = bit_string

    def add_circuit(self, q: QRegPool):
        """Adds the basis encoding circuit to the register

        :param q: Quantum register to which this circuit is added
        """
        for pos, bit in enumerate(self._bit_string):
            if bit == '1':
                X(q[pos])
