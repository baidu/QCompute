#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2020 Baidu, Inc. All Rights Reserved.
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

import numpy

from QCompute.QuantumPlatform.QuantumOperation import QuantumOperation
from QCompute.QuantumPlatform.Utilities import _mergePBList
from QCompute.QuantumProtobuf.Library.PlatformStruct_pb2 import CircuitLine
from QCompute.QuantumProtobuf.Library.QuantumOperation_pb2 import FixedGate as FixedGateEnum


class FixedGate(QuantumOperation):
    """
    Fixed gates are set in built-in quantum tool chain.

    Only some solid gates with concrete definitions (without parameters) are set here,

    like Identity, Pauli X, Y, Z, Hadamard, Phase, T, CNOT (CX), etc.
    """

    def _toPB(self, *qRegsIndex):
        """
        Convert to Protobuf object
        :param qRegsIndex: Quantum registers list used in creating single circuit object
        :return: the circuit in Protobuf format. Filled with the name of fixed gates and parameters of quantum registers.
        """

        ret = CircuitLine()

        if len(qRegsIndex) == 0:  # fill in the register list
            # The circuit object is already in Python env.
            # Directly generate the circuit in Protobuf format according to member variables.
            for reg in self.qRegs:
                ret.qRegs.append(reg.index)
        else:
            # Insert the new circuit object in module process.
            # Generate the Protobuf circuit according to parameters.
            _mergePBList(ret.qRegs, qRegsIndex)  # fill in quantum registers

        assert len(ret.qRegs) == self.bits  # The number of quantum registers in circuit must match to the setting.

        qRegSet = set(qReg for qReg in ret.qRegs)
        assert len(ret.qRegs) == len(qRegSet)  # Quantum registers of operators in circuit should not be duplicated.

        ret.fixedGate = FixedGateEnum.Value(self.Gate)  # fill in the name of fixed gates
        return ret


ID = FixedGate()
r"""
Identity operator, means doing nothing.

Matrix form:

:math:`I= \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}`
"""

ID.Gate = 'ID'
ID.matrix = numpy.array([[1. + 0.j, 0. + 0.j],
                         [0. + 0.j, 1. + 0.j]])
ID.bits = 1

X = FixedGate()
r"""
Pauli-X operator, also called NOT gate, means flipping the qubit.

For example:

:math:`X|0 \rangle = |1\rangle \quad \text{and} \quad  X|1 \rangle = |0\rangle` 

Matrix form:

:math:`X= \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}`
"""
X.Gate = 'X'
X.matrix = numpy.array([[0. + 0.j, 1. + 0.j],
                        [1. + 0.j, 0. + 0.j]])
X.bits = 1

Y = FixedGate()
r"""
Pauli-Y operator, similar to Pauli-X operator

Matrix form:

:math:`Y= \begin{bmatrix}     0 & -i \\     i & 0    \end{bmatrix}`
"""
Y.Gate = 'Y'
Y.matrix = numpy.array([[0. + 0.j, 0. - 1.j],
                        [0. + 1.j, 0. + 0.j]])
Y.bits = 1

Z = FixedGate()  # Pauli-Z operator
r"""
Pauli-Z operator, means changing a local phase

For example: 

:math:`Z(|0 \rangle +|1 \rangle )= |0 \rangle - |1 \rangle`  

Matrix form:

:math:`Z= \begin{bmatrix}     1 & 0 \\     0 & -1    \end{bmatrix}`
"""
Z.Gate = 'Z'
Z.matrix = numpy.array([[1. + 0.j, 0. + 0.j],
                        [0. + 0.j, -1. + 0.j]])
Z.bits = 1

H = FixedGate()
r"""
Hadamard gate: it's the most important single qubit gate. 

And it can prepare a superposed state via applied on zero state, i.e.,

:math:`H|0 \rangle =\frac{1}{\sqrt{2}}( |0 \rangle + |1 \rangle)`  

Matrix form:

:math:`H=\frac{1}{\sqrt{2}} \begin{bmatrix}     1 & 1 \\     1 & -1    \end{bmatrix}`
"""
H.Gate = 'H'
H.matrix = 1 / numpy.sqrt(2) * numpy.array([[1, 1],
                                            [1, -1]], dtype=complex)
H.bits = 1

S = FixedGate()
r"""
Phase gate or :math:`\frac{\pi}{4}`-gate, it equals :math:`S=e^{-i\frac{\pi}{4}}= Z^{\frac{1}{2}}`.

It changes a local phase, similar to Pauli-Z operator, i.e.,

:math:`S (|0 \rangle +|1 \rangle )= |0 \rangle +i  |1 \rangle`  

Matrix form:

:math:`S=  \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}`
"""
S.Gate = 'S'
S.matrix = numpy.array([[1. + 0.j, 0. + 0.j],
                        [0. + 0.j, 0. + 1.j]])
S.bits = 1

SDG = FixedGate()
r"""
S-adjoint gate or :math:`(-\frac{\pi}{4})`-gate, it equals :math:`SDG=e^{i\frac{\pi}{4}} =Z^{\frac{1}{2}}`.

It changes a local phase, similar to :math:`S` gate, i.e.,

:math:`SDG (|0 \rangle +|1 \rangle )= |0 \rangle - i  |1 \rangle`  

Matrix form:

:math:`SDG =  \begin{bmatrix} 1 & 0 \\ 0 & - i \end{bmatrix}`
"""
SDG.Gate = 'SDG'
SDG.matrix = numpy.array([[1. + 0.j, 0. + 0.j],
                          [0. + 0.j, 0. - 1.j]])
SDG.bits = 1

T = FixedGate()
r"""
T gate or :math:`(\frac{\pi}{8})`-gate, it equals :math:`T =e^{-i\frac{\pi}{8}}=Z^{\frac{1}{4}}`.

It changes a local phase, similar to :math:`Z` gate, i.e.,

:math:`T (|0 \rangle +|1 \rangle )= |0 \rangle + e^{i\frac{\pi}{4}} |1 \rangle`

Matrix form:

:math:`T = \begin{bmatrix} 1 & 0 \\ 0 & e^{i\frac{\pi}{4}} \end{bmatrix}`
"""
T.Gate = 'T'
T.matrix = numpy.array([[1. + 0.j, 0. + 0.j],
                        [0. + 0.j, numpy.exp(1j * numpy.pi / 4)]])
T.bits = 1

TDG = FixedGate()
r"""
T-adjoint gate or :math:`(-\frac{\pi}{8})`-gate, it equals :math:`TDG =e^{i\frac{\pi}{8}}=Z^{\frac{1}{4}}`.

It changes a local phase, similar to :math:`T` gate, i.e.,

:math:`TDG (|0 \rangle +|1 \rangle )= |0 \rangle + e^{i\frac{\pi}{4}} |1 \rangle`  

Matrix form:

:math:`TDG = \begin{bmatrix} 1 & 0 \\ 0 & e^{-i\frac{\pi}{4}} \end{bmatrix}`
"""
TDG.Gate = 'TDG'
TDG.matrix = numpy.array([[1. + 0.j, 0. + 0.j],
                          [0. + 0.j, numpy.exp(-1j * numpy.pi / 4)]])
TDG.bits = 1

CX = FixedGate()
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
CX.Gate = 'CX'
CX.matrix = numpy.array([[1, 0, 0, 0],
                         [0, 0, 0, 1],
                         [0, 0, 1, 0],
                         [0, 1, 0, 0]], dtype=complex)
CX.bits = 2

CY = FixedGate()
r"""
CY gate, or control-Y gate. It's similar to CNOT gate.

On the simulator mode (use inside the simulator): Matrix form:

:math:`CY = \begin{bmatrix} 1 & 0  &0 & 0 \\ 0 & 0  & 0& -i \\ 0 & 0  & 1& 0 \\ 0 & i  & 0& 0 \end{bmatrix}`
"""
CY.Gate = 'CY'
CY.matrix = numpy.kron(numpy.eye(2),
                       numpy.array([[1, 0],
                                    [0, 0]])
                       ) + \
            numpy.kron(numpy.array([[0. + 0.j, 0. - 1.j],
                                    [0. + 1.j, 0. + 0.j]]),
                       numpy.array([[0, 0],
                                    [0, 1]]),
                       )
CY.bits = 2

CZ = FixedGate()
r"""
CZ gate, or control-Z gate. It's similar to CNOT gate.

On the simulator mode (use inside the simulator): Matrix form:

:math:`CZ = \begin{bmatrix} 1 & 0  &0 & 0 \\ 0 & 1  & 0& 0 \\ 0 & 0  & 1& 0 \\ 0 & 0  & 0& -1 \end{bmatrix}`
"""
CZ.Gate = 'CZ'
CZ.matrix = numpy.kron(numpy.eye(2),
                       numpy.array([[1, 0],
                                    [0, 0]])
                       ) + \
            numpy.kron(numpy.array([[1. + 0.j, 0. + 0.j],
                                    [0. + 0.j, -1. + 0.j]]),
                       numpy.array([[0, 0],
                                    [0, 1]])
                       )
CZ.bits = 2

CH = FixedGate()
r"""
CH gate, or control-Hadamard gate. It's similar to CNOT gate.

On the simulator mode (use inside the simulator): Matrix form:

:math:`CH = \begin{bmatrix} 1 & 0  &0 & 0 \\ 0 & \frac{1}{\sqrt{2}}  & 0& \frac{1}{\sqrt{2}} \\ 0 & 0  & 1& 0 \\ 0 & \frac{1}{\sqrt{2}}  & 0& -\frac{1}{\sqrt{2}} \end{bmatrix}`
"""
CH.Gate = 'CH'
CH.matrix = numpy.kron(numpy.eye(2),
                       numpy.array([[1, 0],
                                    [0, 0]])
                       ) + \
            numpy.kron(1 / numpy.sqrt(2) * numpy.array([[1, 1],
                                                        [1, -1]]),
                       numpy.array([[0, 0],
                                    [0, 1]])
                       )
CH.bits = 2

SWAP = FixedGate()
r"""
SWAP gate is another important two-qubit gate.

It swaps the contents of the first qubit and the second qubit, i.e.,

:math:`SWAP |x,y\rangle =|y,x\rangle` 

On the simulator mode (use inside the simulator): Matrix form:

:math:`SWAP = \begin{bmatrix} 1 & 0  &0 & 0 \\ 0 & 0  & 1& 0 \\ 0 & 1 & 0& 0 \\ 0 & 0 & 0& 1 \end{bmatrix}`
"""
SWAP.Gate = 'SWAP'
SWAP.matrix = numpy.array([[1. + 0.j, 0. + 0.j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.j, 1. + 0.j, 0. + 0.j],
                           [0. + 0.j, 1. + 0.j, 0. + 0.j, 0. + 0.j],
                           [0. + 0.j, 0. + 0.j, 0. + 0.j, 1. + 0.j]])
SWAP.bits = 2

CCX = FixedGate()
r"""
Toffoli gate, or control-control-X gate, is an important three-qubit gate.

It flips the first qubit only when the other two qubits are both 1, i.e.,

On the simulator mode (use inside the simulator):

:math:`CCX |011\rangle =|111\rangle  \quad \text{and }\quad  CCX |111\rangle =|011\rangle` 

:math:`CCX |xyz\rangle =|xyz\rangle,  \quad \text{where}\  yz \ \text{is not 11}`

Matrix form:

:math:`CCX=  \begin{bmatrix} 1 & 0  &0 & 0 & 0  &0 & 0 & 0  \\ 0 & 1  &0 & 0 & 0  &0 & 0 & 0  \\ 0 & 0  &1 & 0 & 0  &0 & 0 & 0 \\ 0 & 0  &0 & 0 & 0  &0 & 0 & 1 \\ 0 & 0  &0 & 0 & 1  &0 & 0 & 0 \\ 0 & 0  &0 & 0 & 0  &1 & 0 & 0 \\ 0 & 0  &0 & 0 & 0  &0 & 1 & 0 \\ 0 & 0  &0 & 1 & 0  &0 & 0 & 0 \\ \end{bmatrix}`
"""
CCX.Gate = 'CCX'
CCX.matrix = numpy.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                          [0., 1., 0., 0., 0., 0., 0., 0.],
                          [0., 0., 1., 0., 0., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 0., 1.],
                          [0., 0., 0., 0., 1., 0., 0., 0.],
                          [0., 0., 0., 0., 0., 1., 0., 0.],
                          [0., 0., 0., 0., 0., 0., 1., 0.],
                          [0., 0., 0., 1., 0., 0., 0., 0.]], dtype=complex)

CCX.bits = 3

CSWAP = FixedGate()
r"""
Control swap gate

cx c,b; ccx a,b,c; cx c,b;

On the simulator mode (use inside the simulator):

Matrix form:

:math:`CSWAP=  \begin{bmatrix} 1 & 0  &0 & 0 & 0  &0 & 0 & 0  \\ 0 & 1  &0 & 0 & 0  &0 & 0 & 0  \\ 0 & 0  &1 & 0 & 0  &0 & 0 & 0 \\ 0 & 0  &0 & 0 & 0  &1 & 0 & 0 \\ 0 & 0  &0 & 0 & 1  &0 & 0 & 0 \\ 0 & 0  &0 & 1 & 0  &0 & 0 & 0 \\ 0 & 0  &0 & 0 & 0  &0 & 1 & 0 \\ 0 & 0  &0 & 0 & 0  &0 & 0 & 1 \end{bmatrix}`
"""
CSWAP.Gate = 'CSWAP'
CSWAP.matrix = numpy.array([[1., 0., 0., 0., 0., 0., 0., 0.],
                            [0., 1., 0., 0., 0., 0., 0., 0.],
                            [0., 0., 1., 0., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 1., 0., 0.],
                            [0., 0., 0., 0., 1., 0., 0., 0.],
                            [0., 0., 0., 1., 0., 0., 0., 0.],
                            [0., 0., 0., 0., 0., 0., 1., 0.],
                            [0., 0., 0., 0., 0., 0., 0., 1.]], dtype=complex)
CSWAP.bits = 3
