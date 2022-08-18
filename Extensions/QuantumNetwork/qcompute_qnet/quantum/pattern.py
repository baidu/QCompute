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
Module for measurement patterns used in the measurement-based quantum computation.
"""

from typing import Union, Any

__all__ = [
    "Pattern"
]


class Pattern:
    r"""Class for creating a measurement pattern.

    This class represents the measurement pattern in the MBQC model.
    Please refer to [The measurement calculus, arXiv: 0704.1263] for technical details.

    Attributes:
        name (str): pattern name
        space (list): space vertices
        input_ (list): input vertices
        output_ (list): output vertices
        commands (list): command list
    """

    def __init__(self, name: str, space: list, input_: list, output_: list, commands: list):
        r"""Constructor for Pattern class.

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
        r"""Class for creating an entanglement command.

        It entangles two adjacent vertices by operating a Control Z (CZ) gate on them.

        Attributes:
            which_qubits (list): a list of two vertices to be entangled
        """

        def __init__(self, which_qubits: list):
            r"""Constructor for CommandE class.

            Args:
                which_qubits (list): a pair of vertex labels
            """
            self.name = 'E'
            self.which_qubits = which_qubits

    class CommandM:
        r"""Class for creating a measurement command.

        It has five attributes including vertex label, primitive angle, measurement plane, 'domain_s', and 'domain_t'.
        Let :math:`\alpha` be the primitive measurement angle. After considering the dependencies to the
        measurement outcomes of some other vertices, the adaptive angle is calculated using the formula:

        .. math::

            \theta = (-1)^s \times \alpha + t \times \pi

        Note:
            'domain_s' and 'domain_t' are crucial concepts in the MBQC model.
            For detailed definitions, please refer to the reference [The measurement calculus, arXiv: 0704.1263].
            These two domains respectively record the effect of Pauli-X and Pauli-Z operators to the measurement angles.
            In other words, in order to calculate a measurement angle, the dependencies to the outcomes
            of some other vertices have to be taken into consideration in most cases.

        Warning:
            This command only supports measurement in the XY or YZ plane in the current version.

        Attributes:
            which_qubit (any): vertex label
            angle (Union[float, int]): primitive angle
            plane (str): measurement plane, can be 'XY' or 'YZ'
            domain_s (list): a list of vertices in 'domain_s' that have dependencies to this command
            domain_t (list): a list of vertices in 'domain_t' that have dependencies to this command
        """

        def __init__(self, which_qubit: Any, angle: Union[float, int], plane: str, domain_s: list, domain_t: list):
            r"""Constructor for CommandM class.

            Args:
                which_qubit (any): vertex label
                angle (Union[float, int]): primitive angle
                plane (str): measurement plane
                domain_s (list): a list of vertices in 'domain_s' that have dependencies to this command
                domain_t (list): a list of vertices in 'domain_t' that have dependencies to this command
            """
            assert isinstance(angle, float) or isinstance(angle, int), \
                f"Invalid measurement angle ({angle}) with the type: `{type(angle)}`!\n"\
                "Only `float` and `int` are supported as the type of measurement angle."

            self.name = 'M'
            self.which_qubit = which_qubit
            self.angle = angle
            self.plane = plane
            self.domain_s = domain_s
            self.domain_t = domain_t

    class CommandX:
        r"""Class for creating a Pauli-X correction command.

        Attributes:
            which_qubit (any): vertex label
            domain (list): a list of vertices that have dependencies to this command
        """

        def __init__(self, which_qubit: Any, domain: list):
            r"""Constructor for CommandX class.

            Args:
                which_qubit (any): vertex label
                domain (list): a list of vertices that have dependencies to this command
            """
            self.name = 'X'
            self.which_qubit = which_qubit
            self.domain = domain

    class CommandZ:
        r"""Class for creating a Pauli-Z correction command.

        Attributes:
            which_qubit (any): vertex label
            domain (list): a list of vertices that have dependencies to this command
        """

        def __init__(self, which_qubit: Any, domain: list):
            r"""Constructor for CommandZ class.

            Args:
                which_qubit (any): vertex label
                domain (list): a list of vertices that have dependencies to this command
            """
            self.name = 'Z'
            self.which_qubit = which_qubit
            self.domain = domain

    class CommandS:
        r"""Class for creating a signal shifting command.

        Note:
            Signal shifting is a unique operation in MBQC.
            It can be used to simplify the measurement commands by excluding the dependencies in 'domain_t'.

        Attributes:
            which_qubit (any): vertex label
            domain (list): a list of vertices that have dependencies to this command
        """

        def __init__(self, which_qubit: Any, domain: list):
            r"""Constructor for CommandS class.

            Args:
                which_qubit (any): vertex label
                domain (list): a list of vertices that have dependencies to this command
            """
            self.name = 'S'
            self.which_qubit = which_qubit
            self.domain = domain

    def print(self) -> None:
        r"""Print all commands in the command list.
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
