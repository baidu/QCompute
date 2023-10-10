# Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.
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

# !/usr/bin/env python3

"""
qobject
"""
FileErrorCode = 5


from typing import List
from numpy import ndarray
from numpy import matmul
from numpy import linalg
from functools import reduce

from QCompute.OpenService import ModuleErrorCode
from QCompute.QPlatform import Error

from QCompute.OpenService.service_ubqc.client.utils import h_gate, s_gate, t_gate, u_gate
from QCompute.OpenService.service_ubqc.client.utils import pauli_gate, rotation_gate
from QCompute.OpenService.service_ubqc.client.utils import eps, decompose

__all__ = [
    "Pattern",
    "Circuit"
]


class Pattern:
    r"""Define the ``Pattern`` class.

    This class represents the measurement pattern in MBQC model.
    Please see the reference [The measurement calculus, arXiv: 0704.1263] for more details.

    Attributes:
        name (str): pattern name
        space (list): space vertices
        input_ (list): input vertices
        output_ (list): output vertices
        commands (list): command list
    """

    def __init__(self, name, space, input_, output_, commands):
        r"""``Pattern`` constructor, used to instantiate a ``Pattern`` object.

        This class represents the measurement pattern in MBQC model.
        Please see the reference [The measurement calculus, arXiv: 0704.1263] for more details.

        Args:
            name (str): pattern name
            space (list): space vertices
            input_ (list): input vertices
            output_ (list): output vertices
            commands (list): command list
        """
        self.name = name
        self.space = space
        self.input_ = input_
        self.output_ = output_
        self.commands = commands

    class CommandE:
        r"""Define the ``CommandE`` class.

        This class represents the entanglement command in MBQC model.
        It entangles two adjacent vertices by operating a Control Z (CZ) gate on them.

        Attributes:
            which_qubits (list): a list of two vertices to be entangled
        """

        def __init__(self, which_qubits: List[int]):
            r"""``CommandE`` constructor, used to instantiate a ``CommandE`` object.

            This class represents the entanglement command in MBQC model.
            It entangles two adjacent vertices by operating a Control Z (CZ) gate on them.

            Args:
                which_qubits (list): a list of two vertices to be entangled
            """
            self.name = 'E'
            self.which_qubits = which_qubits

    class CommandM:
        r"""Define the ``CommandM`` class.

        This class represents the measurement command in MBQC model.
        It has five attributes including vertex label, primitive angle, measurement plane, 'domain_s', and 'domain_t'.
        Let :math:`\alpha` be the primitive measurement angle. After considering the dependencies to the
        measurement outcomes of some other vertices, the adaptive angle is calculated using the formula:

        .. math::

            \theta = (-1)^s \times \alpha + t \times \pi

        Note:
            'domain_s' and 'domain_t' are crucial concepts in the MBQC model.
            For detailed definitions, please refer to the reference [The measurement calculus, arXiv: 0704.1263].
            These two domains respectively record the effect of Pauli X and Pauli Z operators to the measurement angles.
            In other words, in order to calculate a measurement angle, the dependencies to the outcomes
            of some other vertices have to be taken into consideration in most cases.

        Warning:
            This command only supports the measurement in XY plane or in YZ plane.

        Attributes:
            which_qubit (any): vertex label
            angle (float / int): primitive angle
            plane (str): measurement plane
            domain_s (list): a list of vertices in 'domain_s' that have dependencies to this command
            domain_t (list): a list of vertices in 'domain_t' that have dependencies to this command
        """

        def __init__(self, which_qubit: int, angle, plane, domain_s, domain_t):
            r"""``CommandM`` constructor, used to instantiate a ``CommandM`` object.

            This class represents the measurement command in MBQC model.
            It has five attributes including vertex label, primitive angle, measurement plane, 'domain_s', and 'domain_t'.
            Let :math:`\alpha` be the primitive measurement angle. After considering the dependencies to the
            measurement outcomes of some other vertices, the adaptive angle is calculated using the formula:

            .. math::

                \theta = (-1)^s \times \alpha + t \times \pi

            Note:
                'domain_s' and 'domain_t' are crucial concepts in the MBQC model.
                For detailed definitions, please refer to the reference [The measurement calculus, arXiv: 0704.1263].
                These two domains respectively record the effect of Pauli X and Pauli Z operators to the measurement angles.
                In other words, in order to calculate a measurement angle, the dependencies to the outcomes
                of some other vertices have to be taken into consideration in most cases.

            Warning:
                This command only supports the measurement in XY plane or in YZ plane.

            Args:
                which_qubit (any): vertex label
                angle (float / int): primitive angle
                plane (str): measurement plane
                domain_s (list): a list of vertices in 'domain_s' that have dependencies to this command
                domain_t (list): a list of vertices in 'domain_t' that have dependencies to this command
            """
            if not isinstance(angle, float) and not isinstance(angle, int):
                raise Error.ArgumentError(f'Invalid measurement angle ({angle}) with the type: `{type(angle)}`!\nOnly `float` and `int` are supported as the type of measurement angle.', ModuleErrorCode, FileErrorCode, 1)

            self.name = 'M'
            self.which_qubit = which_qubit
            self.angle = angle
            self.plane = plane
            self.domain_s = domain_s
            self.domain_t = domain_t

    class CommandX:
        r"""Define the ``CommandX`` class.

        This class represents the Pauli X byproduct correction command in MBQC model.

        Attributes:
            which_qubit (any): vertex label
            domain (list): a list of vertices that have dependencies to this command
        """

        def __init__(self, which_qubit: int, domain):
            r"""``CommandX`` constructor, used to instantiate a ``CommandX`` object.

            This class represents the Pauli X byproduct correction command in MBQC model.

            Args:
                which_qubit (any): vertex label
                domain (list): a list of vertices that have dependencies to this command
            """
            self.name = 'X'
            self.which_qubit = which_qubit
            self.domain = domain

    class CommandZ:
        r"""Define the ``CommandZ`` class.

        This class represents the Pauli Z byproduct correction command in MBQC model.

        Attributes:
            which_qubit (any): vertex label
            domain (list): a list of vertices that have dependencies to this command
        """

        def __init__(self, which_qubit: int, domain):
            r"""``CommandZ`` constructor, used to instantiate a ``CommandZ`` object.

            This class represents the Pauli Z byproduct correction command in MBQC model.

            Args:
                which_qubit (any): vertex label
                domain (list): a list of vertices that have dependencies to this command
            """
            self.name = 'Z'
            self.which_qubit = which_qubit
            self.domain = domain

    class CommandS:
        r"""Define the ``CommandS`` class.

        This class represents the signal shifting command in MBQC model.

        Note:
            Signal shifting is an unique operation in MBQC.
            In most cases, it can simplify the measurement commands by excluding the dependencies in 'domain_t'.

        Attributes:
            which_qubit (any): vertex label
            domain (list): a list of vertices that have dependencies to this command
        """

        def __init__(self, which_qubit: int, domain):
            r"""``CommandS`` constructor, used to instantiate a ``CommandS`` object.

            This class represents the signal shifting command in MBQC model.

            Note:
                Signal shifting is an unique operation in MBQC.
                In most cases, it can simplify the measurement commands by excluding the dependencies in 'domain_t'.

            Args:
                which_qubit (any): vertex label
                domain (list): a list of vertices that have dependencies to this command
            """
            self.name = 'S'
            self.which_qubit = which_qubit
            self.domain = domain

    def print_command_list(self):
        r"""Print all commands in the command list of a ``Pattern`` class.

        This is a method to view the information of commands in a concise way.

        Code Example:

        .. code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            from QCompute.OpenService.service_ubqc.client.mcalculus import MCalculus

            width = 2
            cir = Circuit(width)
            cir.s(0)
            cir.t(1)
            cir.measure()
            cir.simplify_by_merging()
            cir.to_brickwork()
            mc = MCalculus()
            mc.set_circuit(cir)
            mc.to_brickwork_pattern()
            mc.standardize()
            brickwork_pattern = mc.get_pattern()
            brickwork_pattern.print_command_list()

        ::

            -----------------------------------------------------------
                                Current Command List
            -----------------------------------------------------------
            Command:        E
            which_qubits:   [(0, 0), (0, 1)]
            -----------------------------------------------------------
            Command:        E
            which_qubits:   [(1, 0), (1, 1)]
            -----------------------------------------------------------
            Command:        E
            which_qubits:   [(0, 1), (0, 2)]
            -----------------------------------------------------------
            Command:        E
            which_qubits:   [(1, 1), (1, 2)]
            -----------------------------------------------------------
            Command:        E
            which_qubits:   [(0, 2), (0, 3)]
            -----------------------------------------------------------
            Command:        E
            which_qubits:   [(1, 2), (1, 3)]
            -----------------------------------------------------------
            Command:        E
            which_qubits:   [(0, 3), (0, 4)]
            -----------------------------------------------------------
            Command:        E
            which_qubits:   [(1, 3), (1, 4)]
            -----------------------------------------------------------
            Command:        E
            which_qubits:   [(0, 2), (1, 2)]
            -----------------------------------------------------------
            Command:        E
            which_qubits:   [(0, 4), (1, 4)]
            -----------------------------------------------------------
            Command:        M
            which_qubit:    (0, 0)
            plane:          XY
            angle:          -0.0
            domain_s:       []
            domain_t:       []
            -----------------------------------------------------------
            Command:        M
            which_qubit:    (1, 0)
            plane:          XY
            angle:          -2.356194490192345
            domain_s:       []
            domain_t:       []
            -----------------------------------------------------------
            Command:        M
            which_qubit:    (0, 1)
            plane:          XY
            angle:          1.5707963267948966
            domain_s:       []
            domain_t:       [(0, 0)]
            -----------------------------------------------------------
            Command:        M
            which_qubit:    (1, 1)
            plane:          XY
            angle:          -1.5707963267948968
            domain_s:       []
            domain_t:       [(1, 0)]
            -----------------------------------------------------------
            Command:        M
            which_qubit:    (0, 2)
            plane:          XY
            angle:          1.5707963267948966
            domain_s:       []
            domain_t:       [(0, 0), (1, 1), (0, 1)]
            -----------------------------------------------------------
            Command:        M
            which_qubit:    (1, 2)
            plane:          XY
            angle:          -1.5707963267948966
            domain_s:       []
            domain_t:       [(1, 0), (0, 1), (1, 1)]
            -----------------------------------------------------------
            Command:        M
            which_qubit:    (0, 3)
            plane:          XY
            angle:          0
            domain_s:       []
            domain_t:       [(0, 1)]
            -----------------------------------------------------------
            Command:        M
            which_qubit:    (1, 3)
            plane:          XY
            angle:          0
            domain_s:       []
            domain_t:       [(1, 1)]
            -----------------------------------------------------------
            Command:        M
            which_qubit:    (0, 4)
            plane:          XY
            angle:          0
            domain_s:       []
            domain_t:       [(1, 3), (0, 2)]
            -----------------------------------------------------------
            Command:        M
            which_qubit:    (1, 4)
            plane:          XY
            angle:          0
            domain_s:       []
            domain_t:       [(1, 2), (0, 3)]
            -----------------------------------------------------------
        """
        print("-----------------------------------------------------------")
        print("                    Current Command List                   ")
        print("-----------------------------------------------------------")
        # Print commands list
        for cmd in self.commands:
            print('\033[91m' + 'Command:'.ljust(16) + cmd.name + '\033[0m')
            if cmd.name == 'E':
                print('which_qubits:'.ljust(15), cmd.which_qubits)
            elif cmd.name == 'M':
                print('which_qubit:'.ljust(15), cmd.which_qubit)
                print('plane:'.ljust(15), cmd.plane)
                print('angle:'.ljust(15), cmd.angle)
                print('domain_s:'.ljust(15), cmd.domain_s)
                print('domain_t:'.ljust(15), cmd.domain_t)
            else:
                print('which_qubit:'.ljust(15), cmd.which_qubit)
                print('domain:'.ljust(15), cmd.domain)
            print("-----------------------------------------------------------")


class Circuit:
    r"""Define the ``Circuit`` class.

    This circuit represents the quantum circuit which supports the translation to its equivalent MBQC model.

    Note:
        This class is similar to the class ``UAnsatz``.
        Users can instantiate a class to build a quantum circuit.

    Warning:
        The current version only supports gates in [H, X, Y, Z, S, T, Rx, Ry, Rz, Rz_5, U, CNOT, CNOT_15, CZ].
        And it only supports the Z measurement.

    Attributes:
        width (int): circuit width (qubit number)
    """

    def __init__(self, width: int):
        r"""``Circuit`` constructor, used to instantiate a ``Circuit`` object.

        This circuit represents the quantum circuit which supports the translation to MBQC model.

        Note:
            This class is similar to the class ``UAnsatz``.
            Users can instantiate a class to build a quantum circuit.

        Warning:
            The current version only supports gates in [H, X, Y, Z, S, T, Rx, Ry, Rz, Rz_5, U, CNOT, CNOT_15, CZ].
            And it only supports the Z measurement.

        Args:
            width (int): circuit width (qubit number)
        """
        if not isinstance(width, int):
            raise Error.ArgumentError(f"Invalid circuit width {width} with the type: `{type(width)}`!\nOnly 'int' is supported as the type of circuit width.", ModuleErrorCode, FileErrorCode, 2)

        self.__history = []  # a list to record the circuit information
        self.__measured_qubits = []  # a list to record the measurement indices in the circuit
        self.__width = width  # the circuit width

        # Here are all attributes used for brickwork mould generation
        self.__bw_depth = None  # brickwork mould depth
        self.__bw_history = None  # a list to record the brickwork circuit information
        self.__bw_mould = None  # brickwork mould
        self.__to_xy_measurement = None  # whether to transform the measurements to XY-plane

        # Record valid columns
        self.__sgl_col = None  # record the valid columns to map a single qubit gate
        self.__dbl_col = None  # record the valid columns to map a double qubit gate

    def __add_a_single_qubit_gate(self, name: str, which_qubit: int, params=None):
        r"""Add a single qubit gate to the circuit list.

        Note:
            This is an intrinsic method. No need to call it externally.
            Check the validity before adding the gate.

        Args:
            name (str): gate name
            which_qubit (int): qubit index
            params (any): gate parameters. Set as 'None' if there are no parameters
        """
        if not isinstance(which_qubit, int):
            raise Error.ArgumentError(f"Invalid qubit index {which_qubit} with the type: `{type(which_qubit)}`!\nOnly 'int' is supported as the type of qubit index.", ModuleErrorCode, FileErrorCode, 3)

        if not (0 <= which_qubit < self.__width):
            raise Error.ArgumentError(f'Invalid qubit index: {which_qubit}!\nQubit index must be smaller than the circuit width.', ModuleErrorCode, FileErrorCode, 4)

        if which_qubit in self.__measured_qubits:
            raise Error.ArgumentError(f'Invalid qubit index: {which_qubit}!\nThis qubit has already been measured.', ModuleErrorCode, FileErrorCode, 5)

        self.__history.append([name, [which_qubit], params])

    def __add_a_double_qubit_gate(self, name: str, which_qubits: List[int], params=None):
        r"""Add a double qubit gate to the circuit list.

        Note:
            This is an intrinsic method. No need to call it externally.
            Check the validity before adding the gate.

        Args:
            name (str): gate name
            which_qubits (list): qubit indexes.
                                 The first element is the index of control qubit.
                                 The second element is the index of target qubit.
            params (any): gate parameters. Set as 'None' if there are no parameters
        """
        ctrl = which_qubits[0]
        tar = which_qubits[1]

        if not isinstance(ctrl, int):
            raise Error.ArgumentError(f"Invalid qubit index {ctrl} with the type: `{type(ctrl)}`!\nOnly 'int' is supported as the type of qubit index.", ModuleErrorCode, FileErrorCode, 6)

        if not (0 <= ctrl < self.__width):
            raise Error.ArgumentError(f'Invalid qubit index: {ctrl}!\nQubit index must be smaller than the circuit width.', ModuleErrorCode, FileErrorCode, 7)

        if ctrl in self.__measured_qubits:
            raise Error.ArgumentError(f'Invalid qubit index: {ctrl}!\nThis qubit has already been measured.', ModuleErrorCode, FileErrorCode, 8)

        if not isinstance(tar, int):
            raise Error.ArgumentError(f"Invalid qubit index {tar} with the type: `{type(tar)}`!\nOnly 'int' is supported as the type of qubit index.", ModuleErrorCode, FileErrorCode, 9)

        if not (0 <= tar < self.__width):
            raise Error.ArgumentError(f'Invalid qubit index: {tar}!\nQubit index must be smaller than the circuit width.', ModuleErrorCode, FileErrorCode, 10)

        if tar in self.__measured_qubits:
            raise Error.ArgumentError(f'Invalid qubit index: {tar}!\nThis qubit has already been measured.', ModuleErrorCode, FileErrorCode, 11)

        if ctrl == tar:
            raise Error.ArgumentError(f'Invalid qubit indexes: {ctrl} and {tar}!\nControl qubit must not be the same as target qubit.', ModuleErrorCode, FileErrorCode, 12)

        self.__history.append([name, which_qubits, params])

    def h(self, which_qubit: int):
        r"""Add a ``Hadamard`` gate.

        The matrix form is:

        .. math::

            H = \frac{1}{\sqrt{2}} \begin{bmatrix} 1 & 1 \\ 1 & -1 \end{bmatrix}

        Args:
            which_qubit (int): qubit index

        Code Example:

        .. code-block:: python
            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.h(which_qubit)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
             [['h', [0], None]]
        """
        self.__add_a_single_qubit_gate('h', which_qubit)

    def x(self, which_qubit: int):
        r"""Add a Pauli `` X`` gate.

        The matrix form is:

        .. math::

            \begin{bmatrix} 0 & 1 \\ 1 & 0 \end{bmatrix}

        Args:
            which_qubit (int): qubit index

        Code Example:

        .. code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.x(which_qubit)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
            [['x', [0], None]]
        """
        self.__add_a_single_qubit_gate('x', which_qubit)

    def y(self, which_qubit: int):
        r"""Add a Pauli `` Y`` gate.

        The matrix form is:

        .. math::

            \begin{bmatrix} 0 & -i \\ i & 0 \end{bmatrix}

        Args:
            which_qubit (int): qubit index

        Code Example:

        .. code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.y(which_qubit)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
            [['y', [0], None]]
        """
        self.__add_a_single_qubit_gate('y', which_qubit)

    def z(self, which_qubit: int):
        r"""Add a Pauli `` Z`` gate.

        The matrix form is:

        .. math::

            \begin{bmatrix} 1 & 0 \\ 0 & -1 \end{bmatrix}

        Args:
            which_qubit (int): qubit index

        Code Example:

        .. code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.z(which_qubit)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
            [['z', [0], None]]
        """
        self.__add_a_single_qubit_gate('z', which_qubit)

    def s(self, which_qubit: int):
        r"""Add a ``S`` gate.

        The matrix form is:

        .. math::

            T = \begin{bmatrix} 1 & 0 \\ 0 & i \end{bmatrix}

        Args:
            which_qubit (int): qubit index

        Code Example:

        .. code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.s(which_qubit)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
            [['s', [0], None]]
        """
        self.__add_a_single_qubit_gate('s', which_qubit)

    def t(self, which_qubit: int):
        r"""Add a ``T`` gate.

        The matrix form is:

        .. math::

            T = \begin{bmatrix} 1 & 0 \\ 0 & e^{i \pi / 4} \end{bmatrix}

        Args:
            which_qubit (int): qubit index

        Code Example:

        .. code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            cir.t(which_qubit)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
            [['t', [0], None]]
        """
        self.__add_a_single_qubit_gate('t', which_qubit)

    def rx(self, theta: float, which_qubit: int):
        r"""Add a rotation gate around x axis.

        The matrix form is:

        .. math::

            \begin{bmatrix}
            \cos\frac{\theta}{2} & -i\sin\frac{\theta}{2} \\
            -i\sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

        Args:
            theta (float / int): rotation angle
            which_qubit (int): qubit index

        Code Example:

        ..  code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            angle = 1
            cir.rx(angle, which_qubit)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
            [['rx', [0], 1.0]]
        """
        if not isinstance(theta, float) and not isinstance(theta, int):
            raise Error.ArgumentError(f'Invalid rotation angle ({theta}) with the type: `{type(theta)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 13)

        self.__add_a_single_qubit_gate('rx', which_qubit, theta)

    def ry(self, theta: float, which_qubit: int):
        r"""Add a rotation gate around y axis.

        The matrix form is:

        .. math::

            \begin{bmatrix}
            \cos\frac{\theta}{2} & -\sin\frac{\theta}{2} \\
            \sin\frac{\theta}{2} & \cos\frac{\theta}{2}
            \end{bmatrix}

        Args:
            theta (float / int): rotation angle
            which_qubit (int): qubit index

        Code Example:

        ..  code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            angle = 1
            cir.ry(angle, which_qubit)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
            [['ry', [0], 1.0]]
        """
        if not isinstance(theta, float) and not isinstance(theta, int):
            raise Error.ArgumentError(f'Invalid rotation angle ({theta}) with the type: `{type(theta)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 14)

        self.__add_a_single_qubit_gate('ry', which_qubit, theta)

    def rz(self, theta: float, which_qubit: int):
        r"""Add a rotation gate around z axis.

        The matrix form is:

        .. math::

            \begin{bmatrix}
            1 & 0 \\
            0 & e^{i\theta}
            \end{bmatrix}

        Args:
            theta (float / int): rotation angle
            which_qubit (int): qubit index

        Code Example:

        ..  code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            angle = 1
            cir.rz(angle, which_qubit)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
            [['rz', [0], 1.0]]
        """
        if not isinstance(theta, float) and not isinstance(theta, int):
            raise Error.ArgumentError(f'Invalid rotation angle ({theta}) with the type: `{type(theta)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 15)

        self.__add_a_single_qubit_gate('rz', which_qubit, theta)

    def u(self, theta: float, phi: float, lamda: float, which_qubit: int):
        r"""Add a single qubit unitary gate.

        Warning：
            The unitary gate generated by this method is a unique gate in MBQC.
            Unlike the commonly used ``U3`` gate, it has a decomposition form of 'Rz, Rx, Rz' .

        It has a decomposition form:

        .. math::

            U(\theta, \phi, \lambda) = Rz(\phi) Rx(\theta) Rz (\lambda)
            = \begin{pmatrix}
            e^{-i(\lambda / 2 + \phi / 2)} cos(\theta / 2)      &   -e^{i(\lambda / 2 - \phi / 2)} sin(\theta / 2) i \\
            -e^{-i(\lambda / 2 - \phi / 2)} sin(\theta / 2) i   &    e^{i(\lambda / 2 + \phi / 2)} cos(\theta / 2)
            \end{pmatrix}

        Args:
            theta (float / int): the rotation angle of the Rx gate
            phi (float / int): the rotation angle of the left Rz gate
            lamda (float / int): the rotation angle of the right Rz gate
            which_qubit (int): qubit index

        Code Example:

        ..  code-block:: python

            from numpy import pi
            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 1
            cir = Circuit(width)
            which_qubit = 0
            theta = pi / 2
            phi = pi
            lamda = - pi / 2
            cir.u([theta, phi, lamda], which_qubit)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
            [['u', [0], [1.57079633, 3.14159265, -1.57079633]]
       """
        if not isinstance(theta, float) and not isinstance(theta, int):
            raise Error.ArgumentError(f'Invalid rotation angle ({theta}) with the type: `{type(theta)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 16)

        if not isinstance(phi, float) and not isinstance(phi, int):
            raise Error.ArgumentError(f'Invalid rotation angle ({phi}) with the type: `{type(phi)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 17)

        if not isinstance(lamda, float) and not isinstance(lamda, int):
            raise Error.ArgumentError(f'Invalid rotation angle ({lamda}) with the type: `{type(lamda)}`!\nOnly `float` and `int` are supported as the type of rotation angle.', ModuleErrorCode, FileErrorCode, 18)

        params = [theta, phi, lamda]
        self.__add_a_single_qubit_gate('u', which_qubit, params)

    def cnot(self, which_qubits: List[int]):
        r"""Add a Control NOT gate.

        Let ``which_qubits`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
            CNOT &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes X\\
            &=\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}
            \end{align}

        Args:
            which_qubits (list): a list of qubit indexes.
                                 The first element is the index of control qubit.
                                 The second element is the index of target qubit

        Code Example:

        ..  code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 2
            cir = Circuit(width)
            which_qubits = [0, 1]
            cir.cnot(which_qubits)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
            [['cnot', [0, 1], None]]
        """
        self.__add_a_double_qubit_gate('cnot', which_qubits)

    def cz(self, which_qubits: List[int]):
        r"""Add a Control Z gate.

        Let ``which_qubits`` be ``[0, 1]``, the matrix form is:

        .. math::

            \begin{align}
            CZ &=|0\rangle \langle 0|\otimes I + |1 \rangle \langle 1|\otimes Z\\
            &=\begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & -1 \end{bmatrix}
            \end{align}

        Args:
            which_qubits (list): a list of qubit indexes
                                 The first element is the index of control qubit.
                                 The second element is the index of target qubit.

        Code Example:

        ..  code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            width = 2
            cir = Circuit(width)
            which_qubits = [0, 1]
            cir.cz(which_qubits)
            print('The quantum circuit is:\n', cir.get_circuit())

        ::

            The quantum circuit is:
            [['cz', [0, 1], None]]
        """
        self.__add_a_double_qubit_gate('cz', which_qubits)

    def measure(self, which_qubit: int = None, basis_list=None):
        r"""Measure the output state of a quantum circuit.

        Note:
            Unlike the measurements in Paddle Quantum ``UAnsatz`` class,
            measurements generated with this method can be customized.
            Please remember that the manipulation must be consistent with the measurement qubits.

        Warning:
            This method only supports three types of inputs:
            1. Call this method without any argument. It sets Z-measurements to all qubits in the circuit model;
            2. Input a qubit index with no measurement basis. It sets Z-measurement to the tagged qubit;
            3. Input a qubit index together with a measurement basis.
               It sets a customized measurement to the tagged qubit with the given basis.
            If the input is customized, please be aware of the format: [angle, plane, domain_s, domain_t].
            Additionally, only ``XY`` and ``YZ`` are supported as the measurement ``plane`` in this version.

        Args:
            which_qubit (int, optional): qubit index
            basis_list (list, optional): measurement basis
        """
        # Set Z-measurements to all qubits
        if which_qubit is None and basis_list is None:
            basis_list = [0, 'YZ', [], []]  # Z measurement as default
            for which_qubit in range(self.__width):
                self.__add_a_single_qubit_gate('m', which_qubit, basis_list)
                self.__measured_qubits.append(which_qubit)

        # Set a Z-measurement to the 'which_qubit'
        elif which_qubit is not None and basis_list is None:
            basis_list = [0, 'YZ', [], []]  # Z measurement as default
            self.__add_a_single_qubit_gate('m', which_qubit, basis_list)
            self.__measured_qubits.append(which_qubit)

        # Set a customized measurement to the 'which_qubit'
        elif which_qubit is not None and basis_list is not None:
            if not isinstance(basis_list, list):
                raise Error.ArgumentError(f'Invalid measurement basis list: ({basis_list}) with the type: `{type(basis_list)}`!\nOnly `List` is supported as the type of measurement basis list.', ModuleErrorCode, FileErrorCode, 19)

            else:
                if not len(basis_list) == 4:
                    raise Error.ArgumentError(f'Invalid measurement basis list: ({basis_list})!\nMeasurement basis list must have four elements in total.', ModuleErrorCode, FileErrorCode, 20)

                measurement_angle = basis_list[0]
                if not isinstance(measurement_angle, float) and not isinstance(measurement_angle, int):
                    raise Error.ArgumentError(f"Invalid measurement basis list: ({basis_list})!\nOnly 'float' and 'int' are supported as the type of measurement angle.", ModuleErrorCode, FileErrorCode, 21)

                measurement_plane = basis_list[1]
                if not {measurement_plane}.issubset(['XY', 'YZ']):
                    raise Error.ArgumentError(f"Invalid measurement basis list: ({basis_list})!\nOnly 'XY' and 'YZ' are supported as the measurement plane in UBQC in this version.", ModuleErrorCode, FileErrorCode, 22)

            self.__add_a_single_qubit_gate('m', which_qubit, basis_list)
            self.__measured_qubits.append(which_qubit)

        else:
            raise Error.ArgumentError(f'Invalid input: ({which_qubit}) and ({basis_list})!\nSuch input is not supported in this version. Please choose another way to call the method.', ModuleErrorCode, FileErrorCode, 23)

    def is_valid(self):
        r"""Check the validity of the quantum circuit.

        It is required that each qubit of the circuit must be operated by at least one quantum gate.

        Returns:
            bool: a boolean value. It represents the validity of the quantum circuit.
        """
        all_qubits = []
        for gate in self.__history:
            if gate[0] != 'm':
                all_qubits += gate[1]
        effective_qubits = list(set(all_qubits))

        return self.__width == len(effective_qubits)

    def get_width(self):
        r"""Return the quantum circuit width.

        Returns:
           int: the circuit width
        """
        return self.__width

    def get_circuit(self):
        r"""Return a quantum circuit list.

        Returns:
            list: a quantum circuit list.
        """
        return self.__history

    def get_measured_qubits(self):
        r"""Return indexes of measurement qubits in the circuit.

        Returns:
            list: a list of indexes of measurement qubits in the circuit
        """
        return self.__measured_qubits

    def print_circuit_list(self):
        r"""Print the quantum circuit list.

        Returns:
            string: a string including the information of the quantum circuit

        Code Example:

        .. code-block:: python

            from QCompute.OpenService.service_ubqc.client.qobject import Circuit
            from numpy import pi

            width = 2
            theta = pi
            cir = Circuit(width)
            cir.h(0)
            cir.cnot([0, 1])
            cir.rx(theta, 1)
            cir.measure()
            cir.print_circuit_list()

        ::

            --------------------------------------------------
                             Current circuit
            --------------------------------------------------
            Gate Name       Qubit Index     Parameter
            --------------------------------------------------
            h               [0]             None
            cnot            [0, 1]          None
            rx              [1]             3.141592653589793
            m               [0]             [0.0, 'YZ', [], []]
            m               [1]             [0.0, 'YZ', [], []]
            --------------------------------------------------
        """
        print("--------------------------------------------------")
        print("                 Current circuit                  ")
        print("--------------------------------------------------")
        print("Gate Name".ljust(16) + "Qubit Index".ljust(16) + "Parameter".ljust(16))
        print("--------------------------------------------------")

        for gate in self.__history:
            name = gate[0]
            which_qubits = gate[1]
            parameters = gate[2]
            if isinstance(parameters, float) or isinstance(parameters, int):
                par_show = parameters
            elif name == 'm':
                par_show = [parameters[0]] + parameters[1:]
            else:
                par_show = parameters
            print(str(name).ljust(16) + str(which_qubits).ljust(16) + str(par_show).ljust(16))
        print("--------------------------------------------------")

    @staticmethod
    def __swap_two_gates(counter: int, two_gates):
        r"""Swap two sequential gates in the circuit list.

        In this method, the quantum gates in the circuit list are reordered.
        The qubit indexes are arranged in an order from small to large.
        So the corresponding gates operated to these qubits are arranged from left to right in the list.
        But those quantum gates with the same row remain unchanged.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            counter (int): a counter to check whether the gate order is standard
            two_gates (list): a list of two gates to be swapped

        Returns:
            int: a counter to check whether the gate order is standard
            list: a list of two new gates after swapping
        """
        gate_1 = two_gates[0]
        gate_2 = two_gates[1]
        vertex_list_1 = gate_1[1]
        vertex_list_2 = gate_2[1]

        # If there is at least one qubit in 'vertex_list_2' and 'vertex_list_1', do not swap them
        if [bit2 for bit2 in vertex_list_2 if bit2 in vertex_list_1]:
            swap_gates = two_gates

        else:
            # We want a order from small to large
            bit1_min = min(vertex_list_1)
            bit2_min = min(vertex_list_2)
            if bit2_min < bit1_min:
                counter += 1
                swap_gates = [gate_2, gate_1]
            else:
                swap_gates = two_gates
        return counter, swap_gates

    def __align_to_left(self):
        r"""Align all gates in the circuit by swapping.

        Note:
            This is an intrinsic method. No need to call it externally.
        """
        # The idea of this method is the same as the idea of swapping commands in ``MCalculus``
        counter = 1
        while counter > 0:
            counter = 0
            for i in range(len(self.__history) - 1):
                before = self.__history[:i]
                after = self.__history[i + 2:]
                __swap_gates = self.__history[i:i + 2]

                counter, new_gates = self.__swap_two_gates(counter, __swap_gates)
                self.__history = before + new_gates + after

    def __to_matrix(self, gate):
        r"""Convert a single qubit gate in the circuit list to its matrix form.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            gate (list): a single qubit gate

        Returns:
            ndarray: the matrix form of the gate
        """
        name = gate[0]
        param = gate[2]
        if name == 'h':
            return h_gate()
        elif name == 'x':
            return pauli_gate('X')
        elif name == 'y':
            return pauli_gate('Y')
        elif name == 'z':
            return pauli_gate('Z')
        elif name == 's':
            return s_gate()
        elif name == 't':
            return t_gate()
        elif name == 'rx':
            return rotation_gate('x', param)
        elif name == 'ry':
            return rotation_gate('y', param)
        elif name == 'rz':
            return rotation_gate('z', param)
        elif name == 'u':
            theta, phi, lamda = param
            return u_gate(theta, phi, lamda)
        elif name == 'm':
            plane = param[1]
            if self.__to_xy_measurement:
                if plane == 'YZ':
                    return h_gate()
                else:
                    return pauli_gate('I')
            else:
                return pauli_gate('I')
        else:
            raise Error.ArgumentError(f'Invalid gate: ({name})!\nThis gate is not supported in this version.', ModuleErrorCode, FileErrorCode, 24)

    def __merge_single_qubit_gates(self, gates):
        r"""Merge those sequential single qubit gates to form a new unitary gate.

        In this method, those sequential single qubit gates which operate on the same qubit are merged together.
        So that a new single qubit unitary gate is formed. In this way, the quantum circuit is simplified.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            gates (list): a list of single qubit gates to be merged

        Returns:
            list: a new single qubit unitary gate
        """
        u = []
        if gates:
            merged_gates = [[] for _ in range(self.__width)]
            for gate in gates:
                which_qubit = gate[1][0]
                matrix = self.__to_matrix(gate)
                merged_gates[which_qubit].append(matrix)
            for i in range(len(merged_gates)):
                row = merged_gates[i]
                if row:
                    # Gate list should be inverse in consistence with the operation order
                    row_inv = row[::-1]
                    u_mat = reduce(lambda mat1, mat2: mat1 if len(row_inv) == 1 else matmul(mat1, mat2), row_inv)

                    # Drop the gates if the 'u_mat' is an identity matrix
                    if linalg.norm(u_mat - pauli_gate('I')) < eps:
                        continue
                    else:
                        theta, phi, lamda = decompose(u_mat)
                        u.append(['u', [i], [theta, phi, lamda]])
        else:
            pass
        return u

    @staticmethod
    def __merge_double_qubit_gates(gates):
        r"""Merge those sequential double qubit gates to form a new one.

        In this method, those sequential double qubit gates which operate on the same qubit are merged together.
        So that a new double gate is formed. In this way, the quantum circuit is simplified.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            gates (list): a list of double qubit gates to be merged

        Returns:
            list: a new double qubit gate
        """
        merged_gates = []
        if gates:
            if len(gates) == 1:
                merged_gates = gates
            else:
                for i in range(len(gates)):
                    if not merged_gates:
                        merged_gates.append(gates[i])
                    else:
                        gate_1 = merged_gates[-1]
                        name_1 = gate_1[0]
                        which_qubits_1 = gate_1[1]
                        gate_2 = gates[i]
                        name_2 = gate_2[0]
                        which_qubits_2 = gate_2[1]

                        # Merge CNOT gates if they operate on the same control and target qubits
                        # Merge CZ gates if they operate on the same control and target qubits
                        # Indeed for the CZ gate, it is no matter which qubit is control and which qubit is target
                        if name_1 == name_2:
                            if which_qubits_1 == which_qubits_2:
                                merged_gates.remove(gate_1)
                            else:
                                if name_1 == name_2 == 'cz' and set(which_qubits_1) == set(which_qubits_2):
                                    merged_gates.remove(gate_1)
                                else:
                                    merged_gates.append(gate_2)

                        else:
                            merged_gates.append(gate_2)

        return merged_gates

    def __merge_sequential_gates(self):
        r"""Merge the sequential gates to simplify the quantum circuit.

        All sequential single qubit gates are merged by ``__merge_single_qubit_gates``.
        All sequential double qubit gates are merged by ``__merge_double_qubit_gates``.
        In this way, the quantum circuit is simplified.

        Note:
            This is an intrinsic method. No need to call it externally.
        """
        simple_circuit = []

        sgl_idx = 0  # index of single qubit gate
        dbl_idx = 0  # index of double qubit gate

        for i in range(len(self.__history)):
            gate = self.__history[i]
            name = gate[0]

            # Find double qubit gates and merge those single qubit gates before them
            if name == 'cnot' or name == 'cz':
                # Merge single qubit gates before them
                # The distance between two 'count_single' is the number of single qubit gates
                gates = self.__history[sgl_idx:i]
                simple_circuit += self.__merge_single_qubit_gates(gates)
                sgl_idx = i + 1

            # Find single qubit gates and merge those double qubit gates before them
            else:
                # Merge double qubit gates before them
                # The distance between two 'count_double' is the number of double qubit gates
                gates = self.__history[dbl_idx:i]
                simple_circuit += self.__merge_double_qubit_gates(gates)
                dbl_idx = i + 1

        # There are also some sequential single qubit gates in the end, so we need to merge them
        sgl_end = self.__history[sgl_idx:]
        simple_circuit += self.__merge_single_qubit_gates(sgl_end)

        # There are also some sequential double qubit gates in the end, so we need to merge them
        dbl_end = self.__history[dbl_idx:]
        simple_circuit += self.__merge_double_qubit_gates(dbl_end)

        # Add measurements to the circuit list
        mea_gates = [gate for gate in self.__history if gate[0] == 'm']
        if self.__to_xy_measurement:
            mea_gates = [['m', gate[1], [gate[2][0], 'XY', gate[2][2], gate[2][3]]] for gate in mea_gates]

        self.__history = simple_circuit + mea_gates

    def simplify_by_merging(self, to_xy_measurement=True):
        r"""The quantum circuit is simplified by merging all sequential single or double qubit gates.

        In this method, the quantum gates are arranged in a standard order.
        All sequential single or double qubit gates are merged together.
        So that the quantum circuit is simplified.

        Note:
            This method applies to any quantum circuit.
            The circuit depth is greatly reduced through this method.

        Args:
            to_xy_measurement (bool): whether or not to convert all measurements to the measurements in XY plane
        """
        self.__to_xy_measurement = to_xy_measurement
        self.__align_to_left()
        self.__merge_sequential_gates()

    def __update_single_column(self, idx: int, col: int):
        r"""Update the list of the valid columns to fill in a single qubit gate.

        Args:
            idx (int): the index of a single qubit gate
            col (int): the column to fill in a single qubit gate
        """
        self.__sgl_col[idx] = col + 4

        if idx == 0 or idx == self.__width - 1:
            self.__sgl_col[idx] += 4
        else:
            self.__sgl_col[idx] += 0

    def __update_double_column(self, row_1: int, row_2: int, idx: int):
        r"""Update the list of the valid columns to fill in a double qubit gate.

        Args:
            row_1 (int): the smaller row to fill in a double qubit gate
            row_2 (int): the larger row to fill in a double qubit gate
            idx (int): the index of a double qubit gate
        """
        col_1 = self.__sgl_col[row_1]
        col_2 = self.__sgl_col[row_2]
        sgl_col = max(col_1, col_2)
        dbl_col = self.__dbl_col[idx]
        if sgl_col > dbl_col:
            self.__dbl_col[idx] += 8
        else:
            self.__dbl_col[idx] += 0

    def __fill_a_single_qubit_gate(self, idx: int):
        r"""Fill in a single qubit gate to the brickwork mould and update the valid columns.

        Args:
            idx (int): the index of a single qubit gate
        """
        col = self.__sgl_col[idx]

        self.__update_single_column(idx, col)

        # If the index of a single qubit gate is also the index of a double qubit gate
        if (col % 8) / 4 == idx % 2:
            if idx != self.__width - 1:
                cor_row = idx + 1
                self.__update_double_column(idx, cor_row, idx)
            else:
                raise Error.ArgumentError('Invalid brickwork mould!\nThis brickwork mould is not supported in UBQC.', ModuleErrorCode, FileErrorCode, 25)

        # If the index of a single qubit gate is not the index of a double qubit gate
        else:
            if idx != 0:
                cor_row = idx - 1
                self.__update_double_column(idx, cor_row, cor_row)
            else:
                raise Error.ArgumentError('Invalid brickwork mould!\nThis brickwork mould is not supported in UBQC.', ModuleErrorCode, FileErrorCode, 26)

    def __fill_a_double_qubit_gate(self, idx: int):
        r"""Fill in a double qubit gate to the brickwork mould and update the valid columns.

        Args:
            idx (int): the index of a double qubit gate
        """
        col = self.__dbl_col[idx]

        self.__dbl_col[idx] += 8

        self.__update_single_column(idx, col)
        self.__update_single_column(idx + 1, col)

        if idx + 1 != self.__width - 1:
            if self.__dbl_col[idx + 1] < col:
                self.__dbl_col[idx + 1] = col + 4

        if idx - 1 != -1:
            if self.__dbl_col[idx - 1] < col:
                self.__dbl_col[idx - 1] = col + 4

    def __gain_position(self, row_list: List[int], gate_type: str):
        r"""Fill in a quantum gate to the brickwork mould to obtain its position.

        Args:
            row_list (list): the rows operated by quantum gates
            gate_type (str): 'single' represents the single qubit gate; 'double' represents the double qubit gate

        Returns:
            list: a list of positions
        """
        if gate_type == 'single':
            row = row_list[0]
            col = self.__sgl_col[row]
            self.__fill_a_single_qubit_gate(row)
            pos = [(row, col)]

        elif gate_type == 'double':
            row = min(row_list)
            col = self.__dbl_col[row]
            self.__fill_a_double_qubit_gate(row)
            pos = [(row_list[0], col), (row_list[1], col)]

        else:
            raise Error.ArgumentError(f'Invalid gate type: ({gate_type})!\nOnly single qubit gates and double qubit gates are supported in UBQC in this version.', ModuleErrorCode, FileErrorCode, 27)

        return pos

    def __fill_gates(self):
        r"""Fill in all quantum gates to the brickwork mould one after another.

        After each filling, the lists of valid columns and the depth of brickwork mould are updated.
        And the gate positions are obtained.
        """
        # Initialize parameters
        self.__bw_depth = 0
        self.__bw_history = []
        if self.__width % 2 == 0:
            self.__sgl_col = [0 for _ in range(self.__width)]
        else:
            self.__sgl_col = [0 for _ in range(self.__width - 1)] + [4]
        self.__dbl_col = [0 if i % 2 == 0 else 4 for i in range(self.__width - 1)]
        mea_gates = [gate for gate in self.__history if gate[0] == 'm']
        qu_gates = [gate for gate in self.__history if gate[0] != 'm']

        # Fill in the gates in 'history' to the brickwork mould
        for gate in qu_gates:
            name = gate[0]
            which_qubit = gate[1]
            param = gate[2]

            # Single qubit gates
            if name != 'cnot' and name != 'cz':
                position = self.__gain_position(which_qubit, 'single')

            # Double qubit gates
            else:
                position = self.__gain_position(which_qubit, 'double')

            # Update brickwork mould depth
            self.__bw_depth = max(self.__bw_depth, int(position[0][1] / 4))

            input_ = position
            output_ = [(pos[0], pos[1] + 4) for pos in input_]
            gate_with_pos = [name, input_, output_, param]

            # Record gates information in a list 'bw_history'
            # Note: the gates in 'bw_history' are not exactly the same gates in 'history'
            # Because these gates also have position parameters
            self.__bw_history.append(gate_with_pos)

        # Add one for counting the circuit depth
        self.__bw_depth += 1

        # Add measurements
        for gate in mea_gates:
            name = gate[0]
            which_qubit = gate[1][0]
            param = gate[2]

            input_ = [(which_qubit, 4 * self.__bw_depth)]
            output_ = None
            gate_with_pos = [name, input_, output_, param]

            self.__bw_history.append(gate_with_pos)

    def __fill_identity(self):
        r"""Fill the blanks of the brickwork mould with identity gates.

        Note:
            To simplify the codes, a trick is implemented here in this method.
            The trick is that we initialize a brickwork mould and fill all blanks with identity gates at the beginning.
            In this way, instead of spending much effort to seek for all blanks of the brickwork mould,
            we just need to replace certain identity gates with those gates to be filled in the brickwork mould.
            And so a complete brickwork circuit is obtained automatically.
        """
        zero_params = [0, 0, 0]

        # Initialize a brickwork mould and fill all blanks with identity gates
        self.__bw_mould = {}
        for row in range(self.__width):
            if row in self.__measured_qubits:
                for col in range(self.__bw_depth + 1):
                    self.__bw_mould[(row, 4 * col)] = ['u', [(row, 4 * col)], [(row, 4 * col + 4)], zero_params]
            else:
                for col in range(self.__bw_depth):
                    self.__bw_mould[(row, 4 * col)] = ['u', [(row, 4 * col)], [(row, 4 * col + 4)], zero_params]

        # Replace identity gates with those in 'bw_history'
        for gate in self.__bw_history:
            pos = gate[1]
            for v in pos:
                self.__bw_mould[v] = gate

    def to_brickwork(self):
        r"""Convert the circuit list to a brickwork circuit.
        """
        self.__fill_gates()
        self.__fill_identity()

    def get_brickwork_circuit(self):
        r"""Return a dictionary of the brickwork circuit.

        Warning:
            This method must be called after ``to_brickwork``.

        Returns:
           dict: a dictionary of the brickwork circuit
        """
        return self.__bw_mould

    def get_brickwork_depth(self):
        r"""Return the depth of brickwork mould.

        Warning:
            This method must be called after ``to_brickwork``.

        Returns:
           int: the depth of brickwork mould
        """
        return self.__bw_depth