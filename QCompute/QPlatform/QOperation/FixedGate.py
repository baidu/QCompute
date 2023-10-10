#!/usr/bin/python3
# -*- coding: utf8 -*-

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
Fixed Gate Operation
"""
FileErrorCode = 36

import importlib
from typing import TYPE_CHECKING

import numpy

from QCompute.QPlatform.QOperation import QOperation

if TYPE_CHECKING:
    from QCompute.QPlatform.QRegPool import QRegStorage


class FixedGateOP(QOperation):
    """
    Fixed gates are set in built-in quantum tool chain.

    Only some solid gates with concrete definitions (without parameters) are set here,

    like Identity, Pauli X, Y, Z, Hadamard, Phase, T, CNOT (CX), etc.
    """

    def __init__(self, gate: str, bits: int, matrix: numpy.ndarray) -> None:
        super().__init__(gate, bits, matrix)

    def __call__(self, *qRegList: 'QRegStorage') -> None:
        self._op(list(qRegList))

    def getInversed(self) -> 'FixedGateOP':
        inversedName = InverseDict.get(self.name)
        if inversedName is not None:
            return getFixedGateInstance(inversedName)
        return self


ID = FixedGateOP('ID', 1,
                 numpy.array([[1. + 0.j, 0. + 0.j],
                              [0. + 0.j, 1. + 0.j]])
                 )
r"""
Identity operator, means doing nothing.

Matrix form:

:math:`I= \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
"""

X = FixedGateOP('X', 1,
                numpy.array([[0. + 0.j, 1. + 0.j],
                             [1. + 0.j, 0. + 0.j]])
                )
r"""
Pauli-X operator, also called NOT gate, means flipping the qubit.

For example:

:math:`X|0 \rangle = |1\rangle \quad \text{and} \quad  X|1 \rangle = |0\rangle` 

Matrix form:

:math:`X= \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}`
"""

Y = FixedGateOP('Y', 1,
                numpy.array([[0. + 0.j, 0. - 1.j],
                             [0. + 1.j, 0. + 0.j]])
                )
r"""
Pauli-Y operator, similar to Pauli-X operator.

Matrix form:

:math:`Y= \begin{bmatrix}     0 & -i \\     i & 0    \end{bmatrix}`
"""

Z = FixedGateOP('Z', 1,
                numpy.array([[1. + 0.j, 0. + 0.j],
                             [0. + 0.j, -1. + 0.j]])
                )  # Pauli-Z operator
r"""
Pauli-Z operator, means changing a local phase.

For example: 

:math:`Z(|0 \rangle +|1 \rangle )= |0 \rangle - |1 \rangle`  

Matrix form:

:math:`Z= \begin{bmatrix}     1 & 0 \\     0 & -1    \end{bmatrix}`
"""

H = FixedGateOP('H', 1,
                1 / numpy.sqrt(2) * numpy.array([[1, 1],
                                                 [1, -1]], dtype=complex)
                )
r"""
Hadamard gate: it's the most important single qubit gate. 

And it can prepare a superposed state via applied on zero state, i.e.,

:math:`H|0 \rangle =\frac{1}{\sqrt{2}}( |0 \rangle + |1 \rangle)`  

Matrix form:

:math:`H=\frac{1}{\sqrt{2}} \begin{bmatrix}     1 & 1 \\     1 & -1    \end{bmatrix}`
"""

S = FixedGateOP('S', 1,
                numpy.array([[1. + 0.j, 0. + 0.j],
                             [0. + 0.j, 0. + 1.j]])
                )
r"""
Phase gate or :math:`\frac{\pi}{4}`-gate, it equals :math:`S=e^{-i\frac{\pi}{4}}= Z^{\frac{1}{2}}`.

It changes a local phase, similar to Pauli-Z operator, i.e.,

:math:`S (|0 \rangle +|1 \rangle )= |0 \rangle +i  |1 \rangle`  

Matrix form:

:math:`S=  \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}`
"""

SDG = FixedGateOP('SDG', 1,
                  numpy.array([[1. + 0.j, 0. + 0.j],
                               [0. + 0.j, 0. - 1.j]])
                  )
r"""
S-adjoint gate or :math:`(-\frac{\pi}{4})`-gate, it equals :math:`SDG=e^{i\frac{\pi}{4}} =Z^{\frac{1}{2}}`.

It changes a local phase, similar to :math:`S` gate, i.e.,

:math:`SDG (|0 \rangle +|1 \rangle )= |0 \rangle - i  |1 \rangle`  

Matrix form:

:math:`SDG =  \begin{bmatrix} 1 & 0 \\ 0 & - i \end{bmatrix}`
"""

T = FixedGateOP('T', 1,
                numpy.array([[1. + 0.j, 0. + 0.j],
                             [0. + 0.j, numpy.exp(1j * numpy.pi / 4)]])
                )
r"""
T gate or :math:`(\frac{\pi}{8})`-gate, it equals :math:`T =e^{-i\frac{\pi}{8}}=Z^{\frac{1}{4}}`.

It changes a local phase, similar to :math:`Z` gate, i.e.,

:math:`T (|0 \rangle +|1 \rangle )= |0 \rangle + e^{i\frac{\pi}{4}} |1 \rangle`

Matrix form:

:math:`T = \begin{bmatrix} 1 & 0 \\ 0 & e^{i\frac{\pi}{4}} \end{bmatrix}`
"""

TDG = FixedGateOP('TDG', 1,
                  numpy.array([[1. + 0.j, 0. + 0.j],
                               [0. + 0.j, numpy.exp(-1j * numpy.pi / 4)]])
                  )
r"""
T-adjoint gate or :math:`(-\frac{\pi}{8})`-gate, it equals :math:`TDG =e^{i\frac{\pi}{8}}=Z^{\frac{1}{4}}`.

It changes a local phase, similar to :math:`T` gate, i.e.,

:math:`TDG (|0 \rangle +|1 \rangle )= |0 \rangle + e^{i\frac{\pi}{4}} |1 \rangle`  

Matrix form:

:math:`TDG = \begin{bmatrix} 1 & 0 \\ 0 & e^{-i\frac{\pi}{4}} \end{bmatrix}`
"""

CX = FixedGateOP('CX', 2,
                 numpy.array([[1, 0, 0, 0],
                              [0, 0, 0, 1],
                              [0, 0, 1, 0],
                              [0, 1, 0, 0]], dtype=complex)
                 )
r"""
CNOT gate, or control-X gate, is the most important two-qubit gate.

On the hybrid mode (use outside the simulator): If the first qubit is 1, then flip the second qubit, else do nothing. 

On the simulator mode (use inside the simulator): If the second qubit is 1, then flip the first qubit, else do nothing.

It can prepare an entangled state together with Hadamard gate, i.e.,

On the simulator mode:
:math:`CNOT \left(|0\rangle  \otimes H |0\rangle \right) =\frac{1}{\sqrt{2}}( |00\rangle +|11\rangle )` 

Matrix form:

:math:`CNOT = \begin{bmatrix} 1 & 0  &0 & 0 \\ 0 & 0  & 0& 1 \\ 0 & 0  & 1& 0 \\ 0 & 1  & 0& 0 \end{bmatrix}`
"""

CY = FixedGateOP('CY', 2,
                 numpy.kron(numpy.eye(2),
                            numpy.array([[1, 0],
                                         [0, 0]])
                            ) + \
                 numpy.kron(numpy.array([[0. + 0.j, 0. - 1.j],
                                         [0. + 1.j, 0. + 0.j]]),
                            numpy.array([[0, 0],
                                         [0, 1]]),
                            )
                 )
r"""
CY gate, or control-Y gate. It's similar to CNOT gate.

On the simulator mode (use inside the simulator): 

Matrix form:

:math:`CY = \begin{bmatrix} 1 & 0  & 0 & 0 \\ 0 & 0  & 0 & -i \\ 0 & 0  & 1 & 0 \\ 0 & i  & 0 & 0 \end{bmatrix}`
"""

CZ = FixedGateOP('CZ', 2,
                 numpy.kron(numpy.eye(2),
                            numpy.array([[1, 0],
                                         [0, 0]])
                            ) + \
                 numpy.kron(numpy.array([[1. + 0.j, 0. + 0.j],
                                         [0. + 0.j, -1. + 0.j]]),
                            numpy.array([[0, 0],
                                         [0, 1]])
                            )
                 )
r"""
CZ gate, or control-Z gate. It's similar to CNOT gate.

On the simulator mode (use inside the simulator):

Matrix form:

:math:`CZ = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1  & 0 & 0 \\ 0 & 0  & 1 & 0 \\ 0 & 0  & 0 & -1 \end{bmatrix}`
"""

CH = FixedGateOP('CH', 2,
                 numpy.kron(numpy.eye(2),
                            numpy.array([[1, 0],
                                         [0, 0]])
                            ) + \
                 numpy.kron(1 / numpy.sqrt(2) * numpy.array([[1, 1],
                                                             [1, -1]]),
                            numpy.array([[0, 0],
                                         [0, 1]])
                            )
                 )
r"""
CH gate, or control-Hadamard gate. It's similar to CNOT gate.

On the simulator mode (use inside the simulator):

Matrix form:

:math:`CH = \begin{bmatrix} 1 & 0  &0 & 0 \\ 0 & \frac{1}{\sqrt{2}}  & 0& \frac{1}{\sqrt{2}} \\ 0 & 0  & 1& 0 \\ 0 & \frac{1}{\sqrt{2}}  & 0& -\frac{1}{\sqrt{2}} \end{bmatrix}`
"""

SWAP = FixedGateOP('SWAP', 2,
                   numpy.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                                [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                                [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                                [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]])
                   )
r"""
SWAP gate is another important two-qubit gate.

It swaps the contents of the first qubit and the second qubit, i.e.,

:math:`SWAP |x,y\rangle =|y,x\rangle` 

On the simulator mode (use inside the simulator): 

Matrix form:

:math:`SWAP = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0  & 1 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}`
"""

CCX = FixedGateOP('CCX', 3,
                  numpy.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                               [0., 1., 0., 0., 0., 0., 0., 0.],
                               [0., 0., 1., 0., 0., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 0., 1.],
                               [0., 0., 0., 0., 1., 0., 0., 0.],
                               [0., 0., 0., 0., 0., 1., 0., 0.],
                               [0., 0., 0., 0., 0., 0., 1., 0.],
                               [0., 0., 0., 1., 0., 0., 0., 0.]], dtype=complex)
                  )
r"""
Toffoli gate, or control-control-X gate, is an important three-qubit gate.

It flips the first qubit only when the other two qubits are both 1, i.e.,

On the simulator mode (use inside the simulator):

:math:`CCX |011\rangle =|111\rangle  \quad \text{and }\quad  CCX |111\rangle =|011\rangle` 

:math:`CCX |xyz\rangle =|xyz\rangle,  \quad \text{where}\  yz \ \text{is not 11}`

Matrix form:

:math:`CCX=  \begin{bmatrix} 1 & 0  &0 & 0 & 0  &0 & 0 & 0  \\ 0 & 1  &0 & 0 & 0  &0 & 0 & 0  \\ 0 & 0  &1 & 0 & 0  &0 & 0 & 0 \\ 0 & 0  &0 & 0 & 0  &0 & 0 & 1 \\ 0 & 0  &0 & 0 & 1  &0 & 0 & 0 \\ 0 & 0  &0 & 0 & 0  &1 & 0 & 0 \\ 0 & 0  &0 & 0 & 0  &0 & 1 & 0 \\ 0 & 0  &0 & 1 & 0  &0 & 0 & 0 \\ \end{bmatrix}`
"""

CSWAP = FixedGateOP('CSWAP', 3,
                    numpy.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 1., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 1., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 1., 0., 0.],
                                 [0., 0., 0., 0., 1., 0., 0., 0.],
                                 [0., 0., 0., 1., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 1., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 1.]], dtype=complex)
                    )
r"""
Control swap gate

cx c,b; ccx a,b,c; cx c,b;

On the simulator mode (use inside the simulator):

Matrix form:

:math:`CSWAP=  \begin{bmatrix} 1 & 0  &0 & 0 & 0  &0 & 0 & 0  \\ 0 & 1  &0 & 0 & 0  &0 & 0 & 0  \\ 0 & 0  &1 & 0 & 0  &0 & 0 & 0 \\ 0 & 0  &0 & 0 & 0  &1 & 0 & 0 \\ 0 & 0  &0 & 0 & 1  &0 & 0 & 0 \\ 0 & 0  &0 & 1 & 0  &0 & 0 & 0 \\ 0 & 0  &0 & 0 & 0  &0 & 1 & 0 \\ 0 & 0  &0 & 0 & 0  &0 & 0 & 1 \end{bmatrix}`
"""


def getFixedGateInstance(name: str) -> 'FixedGateOP':
    """
    Get a gate according to name.

    :param name: fixed gate name

    :return: gate.
    """

    currentModule = importlib.import_module(__name__)
    gate = getattr(currentModule, name)
    return gate


InverseDict = {
    'S': 'SDG',
    'SDG': 'S',
    'T': 'TDG',
    'TDG': 'T',
}