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
PQC templates
"""

import numpy as np
from QCompute.QPlatform.QRegPool import QRegPool
from QCompute.QPlatform.QOperation.FixedGate import CZ, CX
from QCompute.QPlatform.QOperation.RotationGate import RX, RY, RZ, U
from .parameterized_circuit import ParameterizedCircuit


class UniversalCircuit(ParameterizedCircuit):
    """Universal Circuit class
    """
    def __init__(self, num: int, parameters: np.ndarray):
        """The constructor of the UniversalCircuit class

        :param num: Number of qubits in this ansatz
        :param parameters: Parameters of parameterized gates in this circuit, whose shape should be ``(3,)`` for
            single-qubit cases and should be ``(15,)`` for 2-qubit cases
        """
        super().__init__(num, parameters)
        assert num == 1 or num == 2, "The number of qubits should be 1 or 2."

    def add_circuit(self, q: QRegPool):
        """Adds the universal circuit to the register. Only support single-qubit and 2-qubit cases

        :param q: Quantum register to which this circuit is added
        """
        if self._num == 1:
            assert self._param_shape == (3,), "The shape of parameters should be (3,)."
            U(self._parameters[0], self._parameters[1], self._parameters[2])(q[0])
        elif self._num == 2:
            assert self._param_shape == (15,), "The shape of parameters should be (15,)."
            U(self._parameters[0], self._parameters[1], self._parameters[2])(q[0])
            U(self._parameters[3], self._parameters[4], self._parameters[5])(q[1])
            CX(q[1], q[0])
            RZ(self._parameters[6])(q[0])
            RY(self._parameters[7])(q[1])
            CX(q[0], q[1])
            RY(self._parameters[8])(q[1])
            CX(q[1], q[0])
            U(self._parameters[9], self._parameters[10], self._parameters[11])(q[0])
            U(self._parameters[12], self._parameters[13], self._parameters[14])(q[1])


class RealEntangledCircuit(ParameterizedCircuit):
    """Real Entangled Circuit class
    """
    def __init__(self, num: int, layer: int, parameters: np.ndarray):
        """The constructor of the RealEntangledCircuit class

        :param num: Number of qubits in this ansatz
        :param layer: Number of layers for this ansatz
        :param parameters: Parameters of parameterized gates in this circuit, whose shape should be ``(num * layer,)``
        """
        super().__init__(num, parameters)
        self._layer = layer

    def add_circuit(self, q: QRegPool):
        """Adds the real entangled circuit to the register

        :param q: Quantum register to which this circuit is added
        """
        assert self._param_shape == (self._num * self._layer,), "The shape of parameters should be (num * layer,)."
        for i in range(self._layer):
            for j in range(self._num):
                RY(self._parameters[i * self._num + j])(q[j])
            for j in range(0, self._num - 1, 2):
                CZ(q[j], q[j + 1])
            for j in range(1, self._num - 1, 2):
                CZ(q[j], q[j + 1])


class ComplexEntangledCircuit(ParameterizedCircuit):
    """Complex Entangled Circuit class
    """
    def __init__(self, num: int, layer: int, parameters: np.ndarray):
        """The constructor of the ComplexEntangledCircuit class

        :param num: Number of qubits in this Ansatz
        :param layer: Number of layer for this Ansatz
        :param parameters: Parameters of parameterized gates in this circuit, whose shape should be ``(num * layer * 2,)``
        """
        super().__init__(num, parameters)
        self._layer = layer

    def add_circuit(self, q: QRegPool):
        """Adds the complex entangled circuit to the register

        :param q: Quantum register to which this circuit is added
        """
        assert self._param_shape == (2 * self._num * self._layer,),\
            "The shape of parameters should be (2 * num * layer,)."
        for j in range(self._layer):
            for k in range(self._num):
                RX(self._parameters[2 * self._num * j + 2 * k])(q[k])
                RY(self._parameters[2 * self._num * j + 2 * k + 1])(q[k])
            for k in range(self._num):
                if (k % 2 == 0) and (k < self._num - 1):
                    CZ(q[k], q[k + 1])
            for k in range(self._num):
                if (k % 2 == 1) and (k < self._num - 1):
                    CZ(q[k], q[k + 1])


class RealAlternatingLayeredCircuit(ParameterizedCircuit):
    """Real Alternating Layered Circuit class
    """
    def __init__(self, num: int, layer: int, parameters: np.ndarray):
        """The constructor of the RealAlternatingLayeredCircuit class

        :param num: Number of qubits in this Ansatz
        :param layer: Number of layer for this Ansatz
        :param parameters: Parameters of parameterized gates in this circuit, whose shape should be ``((2 * num - 2) * layer,)``
        """
        super().__init__(num, parameters)
        assert self._num > 1, "The number of qubits should be larger than 1."
        self._layer = layer

    def add_circuit(self, q: QRegPool):
        """Adds the real alternating layered circuit to the register

        :param q: Quantum register to which this circuit is added
        """
        assert self._param_shape == ((2 * self._num - 2) * self._layer,),\
            "The shape of parameters should be ((2 * num - 2) * layer,)."
        if self._num % 2 == 0:
            for j in range(self._layer):
                for k in range(self._num):
                    RY(self._parameters[(2 * self._num - 2) * j + k])(q[k])
                for k in range(self._num):
                    if (k % 2 == 0) and (k < self._num - 1):
                        CZ(q[k], q[k + 1])
                for k in range(1, self._num - 1):
                    RY(self._parameters[(2 * self._num - 2) * j + self._num + k - 1])(q[k])
                for k in range(1, self._num - 1):
                    if (k % 2 == 1) and (k < self._num - 1):
                        CZ(q[k], q[k + 1])
        else:
            for j in range(self._layer):
                for k in range(self._num - 1):
                    RY(self._parameters[(2 * self._num - 2) * j + k])(q[k])
                for k in range(self._num - 1):
                    if (k % 2 == 0) and (k < self._num - 1):
                        CZ(q[k], q[k + 1])
                for k in range(1, self._num):
                    RY(self._parameters[(2 * self._num - 2) * j + self._num - 1 + k - 1])(q[k])
                for k in range(1, self._num):
                    if (k % 2 == 1) and (k < self._num - 1):
                        CZ(q[k], q[k + 1])


class ComplexAlternatingLayeredCircuit(ParameterizedCircuit):
    """Complex Alternating Layered Circuit class
    """
    def __init__(self, num: int, layer: int, parameters: np.ndarray):
        """The constructor of the ComplexAlternatingLayeredCircuit class

        :param num: Number of qubits in this Ansatz
        :param layer: Number of layer for this Ansatz
        :param parameters: Parameters of parameterized gates in this circuit, whose shape should be ``((4 * num - 4) * layer,)``
        """
        super().__init__(num, parameters)
        assert self._num > 1, "The number of qubits should be larger than 1."
        self._layer = layer

    def add_circuit(self, q: QRegPool):
        """Adds the complex alternating layered circuit to the register

        :param q: Quantum register to which this circuit is added
        """
        assert self._param_shape == ((4 * self._num - 4) * self._layer,),\
            "The shape of parameters should be ((4 * num - 4) * layer,)."
        if self._num % 2 == 0:
            for j in range(self._layer):
                for k in range(self._num):
                    RX(self._parameters[2 * (2 * self._num - 2) * j + 2 * k])(q[k])
                    RY(self._parameters[2 * (2 * self._num - 2) * j + 2 * k + 1])(q[k])
                for k in range(self._num):
                    if (k % 2 == 0) and (k < self._num - 1):
                        CZ(q[k], q[k + 1])
                for k in range(1, self._num - 1):
                    RX(self._parameters[2 * (2 * self._num - 2) * j + 2 * self._num + 2 * (k - 1)])(q[k])
                    RY(self._parameters[2 * (2 * self._num - 2) * j + 2 * self._num + 2 * (k - 1) + 1])(q[k])
                for k in range(1, self._num - 1):
                    if (k % 2 == 1) and (k < self._num - 1):
                        CZ(q[k], q[k + 1])
        else:
            for j in range(self._layer):
                for k in range(self._num - 1):
                    RX(self._parameters[2 * (2 * self._num - 2) * j + 2 * k])(q[k])
                    RY(self._parameters[2 * (2 * self._num - 2) * j + 2 * k + 1])(q[k])
                for k in range(self._num - 1):
                    if (k % 2 == 0) and (k < self._num - 1):
                        CZ(q[k], q[k + 1])
                for k in range(1, self._num):
                    RX(self._parameters[2 * (2 * self._num - 2) * j + 2 * (self._num - 1) + 2 * (k - 1)])(q[k])
                    RY(self._parameters[2 * (2 * self._num - 2) * j + 2 * (self._num - 1) + 2 * (k - 1) + 1])(q[k])
                for k in range(1, self._num):
                    if (k % 2 == 1) and (k < self._num - 1):
                        CZ(q[k], q[k + 1])
