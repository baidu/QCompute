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
Module for the measurement calculus.

This will be used to manipulate the measurement patterns in the measurement-based quantum computation.
"""

from argparse import ArgumentTypeError
from typing import List, Tuple
from numpy import pi
from Extensions.QuantumNetwork.qcompute_qnet.quantum.pattern import Pattern
from Extensions.QuantumNetwork.qcompute_qnet.quantum.utils import print_progress

__all__ = ["MCalculus"]


class MCalculus:
    r"""Class for manipulating measurement patterns.

    This class provides various basic operations to compute measurement patterns.
    Please see the reference [The measurement calculus, arXiv: 0704.1263] for technical details.
    """

    def __init__(self):
        r"""Constructor for MCalculus class."""
        self._circuit = None  # circuit
        self._circuit_width = None  # circuit width
        self._circuit_slice = []  # sliced circuit
        self._wild_pattern = []  # wild pattern
        self._pattern = None  # standard pattern
        self._measured_qubits = []  # measured qubits
        self._track = False  # progress tracking

    def set_circuit(self, circuit: "Circuit") -> None:
        r"""Set a quantum circuit.

        Args:
            circuit (Circuit): quantum circuit to set
        """
        self._circuit = circuit
        self._circuit_width = circuit.width

    def __slice_circuit(self, gate_history: list) -> None:
        r"""Slice a quantum circuit and label the input and output qubits for each gate.

        Args:
            gate_history (list): a list of gates in the circuit
        """
        # Slice the circuit to mark the input_/output_ labels for measurement patterns
        counter = []
        for gate in gate_history:
            name = gate["name"]
            which_qubits = gate["which_qubit"]
            qubit_number = len(which_qubits)
            if qubit_number not in [1, 2]:
                raise ArgumentTypeError(
                    f"Invalid quantum gate: ({gate})!\n"
                    f"({qubit_number})-qubit gate is not supported in this version.\n"
                )

            if qubit_number == 1:  # single qubit gate
                which_qubit = which_qubits[0]  # take the qubit index
                if which_qubit in self._measured_qubits:
                    raise ArgumentTypeError(
                        f"Invalid qubit index: ({which_qubit})!\n"
                        f"Please check your qubit index as this qubit has already been measured."
                    )

                input_ = [(which_qubit, int(counter.count(which_qubit)))]

                if name == "m":
                    # Measurements have no output labels
                    output_ = []
                else:
                    output_ = [(which_qubit, int(counter.count(which_qubit) + 1))]
                    counter += [which_qubit]  # count the qubit index

            else:  # double qubit gate
                control = which_qubits[0]  # take the control qubit
                target = which_qubits[1]  # take the target qubit
                if control in self._measured_qubits:
                    raise ArgumentTypeError(
                        f"Invalid qubit index: ({control})!\n"
                        f"Please check your qubit index as this qubit has already been measured."
                    )
                if target in self._measured_qubits:
                    raise ArgumentTypeError(
                        f"Invalid qubit index: ({target})!\n"
                        f"Please check your qubit index as this qubit has already been measured."
                    )

                input_ = [(control, int(counter.count(control))), (target, int(counter.count(target)))]

                if name == "cz":  # input and output labels are the same
                    output_ = input_[:]

                elif name == "cx":
                    output_ = [(control, int(counter.count(control))), (target, int(counter.count(target) + 1))]
                    counter += [target]  # count the qubit index

                else:
                    output_ = [(control, int(counter.count(control) + 1)), (target, int(counter.count(target) + 1))]
                    counter += which_qubits  # count the qubit index

            # The gate after slicing has a form: [original_gate, input_, output_]
            # E.g. = [[cx, [0, 1], None], input_, output_]
            self._circuit_slice.append([gate, input_, output_])

    @staticmethod
    def __set_ancilla(input_: list, output_: list, ancilla_number_list=None) -> list:
        r"""Insert ancilla qubits.

        Insert ancilla qubits between the input and output qubits.
        The coordinates of the ancilla qubits are evenly distributed.

        Args:
            input_ (list): input vertices
            output_ (list): output vertices
            ancilla_number_list (list, optional): number of ancilla qubits

        Returns:
            list: ancilla qubit labels
        """
        ancilla_num_in = len(input_)
        ancilla_num_out = len(output_)
        if ancilla_num_in != ancilla_num_out:
            raise ArgumentTypeError(
                f"Invalid input and output labels: ({input_}) and ({output_})!\n"
                f"Input labels and output labels must have the same length."
            )
        if ancilla_num_in not in [1, 2]:
            raise ArgumentTypeError(
                f"Invalid ancilla number: ({ancilla_num_in})!\n"
                f"({ancilla_num_in})-qubit gate is not supported in this version.\n"
            )

        ancilla_nums = [] if ancilla_number_list is None else ancilla_number_list
        ancilla_idxes = []

        for i in range(len(ancilla_nums)):
            input_qubit = input_[i]  # obtain input qubit index
            row_in = input_qubit[0]  # input qubit row
            col_in = input_qubit[1]  # input qubit column

            output_qubit = output_[i]  # obtain output qubit
            row_out = output_qubit[0]  # output qubit row
            col_out = output_qubit[1]  # output qubit column

            ancilla = ancilla_nums[i]

            if row_in != row_out:
                raise ArgumentTypeError(
                    f"Invalid input and output labels: ({input_}) and ({output_})!\n"
                    f"Each input label and output label must have the same qubit index."
                )

            # Calculate ancillary qubits' positions
            col_len = col_out - col_in
            pos = [(int(col_in * (ancilla + 1) + col * col_len) / (ancilla + 1)) for col in range(1, ancilla + 1)]

            # Set ancillary qubits' labels
            for idx in range(ancilla):
                ancilla_idxes.append((row_in, pos[idx]))

        return ancilla_idxes

    def __to_pattern(self, gate: list) -> None:
        r"""Translate a quantum gate or measurement to its equivalent pattern.

        Args:
            gate (list): gate or measurement to translate
        """
        original_gate, input_, output_ = gate
        name = original_gate["name"]
        which_qubit = original_gate["which_qubit"]

        ancilla = []

        if name == "h":  # Hadamard gate
            E = Pattern.CommandE([input_[0], output_[0]])
            M = Pattern.CommandM(input_[0], 0, "XY", [], [])
            X = Pattern.CommandX(output_[0], [input_[0]])
            commands = [E, M, X]

        elif name == "x":  # Pauli X gate
            ancilla = self.__set_ancilla(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], 0, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], -pi, "XY", [], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == "y":  # Pauli Y gate
            ancilla = self.__set_ancilla(input_, output_, [3])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], ancilla[1]])
            E34 = Pattern.CommandE([ancilla[1], ancilla[2]])
            E45 = Pattern.CommandE([ancilla[2], output_[0]])
            M1 = Pattern.CommandM(input_[0], pi / 2, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], pi / 2, "XY", [], [])
            M3 = Pattern.CommandM(ancilla[1], -pi / 2, "XY", [], [input_[0], ancilla[0]])
            M4 = Pattern.CommandM(ancilla[2], 0, "XY", [], [ancilla[0]])
            X5 = Pattern.CommandX(output_[0], [ancilla[2]])
            Z5 = Pattern.CommandZ(output_[0], [ancilla[1]])
            commands = [E12, E23, E34, E45, M1, M2, M3, M4, X5, Z5]

        elif name == "z":  # Pauli Z gate
            ancilla = self.__set_ancilla(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], -pi, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], 0, "XY", [], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == "s":  # phase gate
            ancilla = self.__set_ancilla(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], -pi / 2, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], 0, "XY", [], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == "t":  # T gate
            ancilla = self.__set_ancilla(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], -pi / 4, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], 0, "XY", [], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == "rx":  # rotation gate around x-axis
            ancilla = self.__set_ancilla(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], 0, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], -original_gate["angle"], "XY", [input_[0]], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == "ry":  # rotation gate around y-axis
            ancilla = self.__set_ancilla(input_, output_, [3])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], ancilla[1]])
            E34 = Pattern.CommandE([ancilla[1], ancilla[2]])
            E45 = Pattern.CommandE([ancilla[2], output_[0]])
            M1 = Pattern.CommandM(input_[0], pi / 2, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], -original_gate["angle"], "XY", [input_[0]], [])
            M3 = Pattern.CommandM(ancilla[1], -pi / 2, "XY", [], [input_[0], ancilla[0]])
            M4 = Pattern.CommandM(ancilla[2], 0, "XY", [], [ancilla[0]])
            X5 = Pattern.CommandX(output_[0], [ancilla[2]])
            Z5 = Pattern.CommandZ(output_[0], [ancilla[1]])
            commands = [E12, E23, E34, E45, M1, M2, M3, M4, X5, Z5]

        elif name == "rz":  # rotation gate around z axis
            ancilla = self.__set_ancilla(input_, output_, [1])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], output_[0]])
            M1 = Pattern.CommandM(input_[0], -original_gate["angle"], "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], 0, "XY", [], [])
            X3 = Pattern.CommandX(output_[0], [ancilla[0]])
            Z3 = Pattern.CommandZ(output_[0], [input_[0]])
            commands = [E12, E23, M1, M2, X3, Z3]

        elif name == "rz5":  # rotation gate around z axis
            ancilla = self.__set_ancilla(input_, output_, [3])
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], ancilla[1]])
            E34 = Pattern.CommandE([ancilla[1], ancilla[2]])
            E45 = Pattern.CommandE([ancilla[2], output_[0]])
            M1 = Pattern.CommandM(input_[0], 0, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], 0, "XY", [], [])
            M3 = Pattern.CommandM(ancilla[1], -original_gate["angle"], "XY", [ancilla[0]], [input_[0]])
            M4 = Pattern.CommandM(ancilla[2], 0, "XY", [], [ancilla[0]])
            X5 = Pattern.CommandX(output_[0], [ancilla[2]])
            Z5 = Pattern.CommandZ(output_[0], [ancilla[1]])
            commands = [E12, E23, E34, E45, M1, M2, M3, M4, X5, Z5]

        elif name == "u":  # single-qubit unitary gate
            # Note: U = Rz(\phi) Rx(\theta) Rz(\gamma)
            ancilla = self.__set_ancilla(input_, output_, [3])
            theta, phi, gamma = original_gate["angles"]
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], ancilla[1]])
            E34 = Pattern.CommandE([ancilla[1], ancilla[2]])
            E45 = Pattern.CommandE([ancilla[2], output_[0]])
            M1 = Pattern.CommandM(input_[0], -gamma, "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], -theta, "XY", [input_[0]], [])
            M3 = Pattern.CommandM(ancilla[1], -phi, "XY", [ancilla[0]], [input_[0]])
            M4 = Pattern.CommandM(ancilla[2], 0, "XY", [], [ancilla[0]])
            X5 = Pattern.CommandX(output_[0], [ancilla[2]])
            Z5 = Pattern.CommandZ(output_[0], [ancilla[1]])
            commands = [E12, E23, E34, E45, M1, M2, M3, M4, X5, Z5]

        elif name == "u3":  # single-qubit unitary gate
            # Note: U3 = Rz(\phi) Ry(\theta) Rz(\gamma) usually used in MBQC
            ancilla = self.__set_ancilla(input_, output_, [3])
            theta, phi, gamma = original_gate["angles"]
            E12 = Pattern.CommandE([input_[0], ancilla[0]])
            E23 = Pattern.CommandE([ancilla[0], ancilla[1]])
            E34 = Pattern.CommandE([ancilla[1], ancilla[2]])
            E45 = Pattern.CommandE([ancilla[2], output_[0]])
            M1 = Pattern.CommandM(input_[0], -(gamma + -pi / 2), "XY", [], [])
            M2 = Pattern.CommandM(ancilla[0], -theta, "XY", [input_[0]], [])
            M3 = Pattern.CommandM(ancilla[1], -(phi + pi / 2), "XY", [ancilla[0]], [input_[0]])
            M4 = Pattern.CommandM(ancilla[2], 0, "XY", [], [ancilla[0]])
            X5 = Pattern.CommandX(output_[0], [ancilla[2]])
            Z5 = Pattern.CommandZ(output_[0], [ancilla[1]])
            commands = [E12, E23, E34, E45, M1, M2, M3, M4, X5, Z5]

        elif name == "cx":  # control NOT gate
            ancilla = self.__set_ancilla(input_, output_, [0, 1])
            E23 = Pattern.CommandE([input_[1], ancilla[0]])
            E13 = Pattern.CommandE([input_[0], ancilla[0]])
            E34 = Pattern.CommandE([ancilla[0], output_[1]])
            M2 = Pattern.CommandM(input_[1], 0, "XY", [], [])
            M3 = Pattern.CommandM(ancilla[0], 0, "XY", [], [])
            X4 = Pattern.CommandX(output_[1], [ancilla[0]])
            Z1 = Pattern.CommandZ(output_[0], [input_[1]])
            Z4 = Pattern.CommandZ(output_[1], [input_[1]])
            commands = [E23, E13, E34, M2, M3, X4, Z1, Z4]

        elif name == "cz":  # control Z gate
            commands = [Pattern.CommandE(input_)]

        elif name == "m":  # single qubit measurement
            self._measured_qubits.append(which_qubit[0])
            commands = [
                Pattern.CommandM(
                    input_[0],
                    original_gate["basis"]["angle"],
                    original_gate["basis"]["plane"],
                    original_gate["basis"]["domain_s"],
                    original_gate["basis"]["domain_t"],
                )
            ]

        else:
            raise ArgumentTypeError(
                f"Invalid gate: ({gate})!\n" f"Translation of such a gate is not supported in this version."
            )

        self._wild_pattern.append(
            Pattern(
                str([name] + [str(qubit) for qubit in which_qubit]),
                list(set(input_ + output_ + ancilla)),
                input_,
                output_,
                commands,
            )
        )

    def _list_to_net(self, vertices: list) -> List[List]:
        r"""Map a vertex list to a vertex net.

        This method maps a one-dimensional vertex list to a two-dimensional vertex net.
        The net contains the labels of all vertices on the graph.

        Args:
            vertices (list): one-dimensional list containing the labels of input or output vertices

        Returns:
            list: a two-dimensional vertex net. This net has a special network structure of List[List],
                  Each row of the net records the column number of input or output vertices.
        """
        # Get width
        rows = {qubit[0] for qubit in vertices}
        self._circuit_width = len(rows) if self._circuit_width is None else self._circuit_width
        # Map to net
        bit_net = [[] for _ in range(self._circuit_width)]
        for qubit in vertices:
            bit_net[qubit[0]].append(qubit[1])
        return bit_net

    def _get_input(self) -> list:
        r"""Get the labels of the input vertices in the list.

        Returns:
            list: a list containing the labels of input vertices
        """
        input_list = []
        for pat in self._wild_pattern:
            input_list += pat.input_
        # Map to net
        bit_net = self._list_to_net(input_list)
        input_ = [(i, min(bit_net[i])) for i in range(self._circuit_width)]
        return input_

    def _get_output(self) -> list:
        r"""Get the labels of the output vertices in the list.

        Returns:
            list: a list containing the labels of output vertices
        """
        output_list = []
        for pat in self._wild_pattern:
            output_list += pat.output_
        bit_net = self._list_to_net(output_list)
        # Divide the output into classical and quantum parts
        c_out = [
            (i, max(bit_net[i]) if bit_net[i] != [] else 0)
            for i in range(self._circuit_width)
            if i in self._measured_qubits
        ]
        q_out = [
            (i, max(bit_net[i]) if bit_net[i] != [] else 0)
            for i in range(self._circuit_width)
            if i not in self._measured_qubits
        ]
        output_ = [c_out, q_out]
        return output_

    def _splice_patterns(self, input_: list, output_: list) -> None:
        r"""Splice all measurement patterns together to form a complete pattern.

        The spliced pattern is equivalent to the complete quantum circuit.

        Args:
            input_ (list): pattern input vertices
            output_ (list): pattern output vertices
        """
        names = ""
        space = []
        cmds = []

        for pat in self._wild_pattern:
            names += pat.name
            space += [qubit for qubit in pat.space if qubit not in space]
            cmds += pat.commands

        self._pattern = Pattern(names, space, input_, output_, cmds)

    def to_pattern(self) -> None:
        r"""Translate a quantum circuit to its equivalent measurement pattern."""
        self.__slice_circuit(self._circuit.gate_history)
        for gate in self._circuit_slice:
            self.__to_pattern(gate)

        input_ = self._get_input()
        output_ = self._get_output()

        self._splice_patterns(input_, output_)

    @staticmethod
    def __swap(two_cmds: list) -> list:
        r"""Swap two commands.

        Hint:
            The commands in the list are executed from the left to the right as default.
            For example, the command 'E' represents the entanglement command.
            According to the MBQC procedure, entanglements are executed in the first step.
            Therefore, the command 'E' should be on the left side of the list.
            After the entanglements, the measurement commands are executed, and so on.

        Warning:
            This method only swaps two commands in ['E', 'M', 'X', 'Z', 'S'].
            Swapping two commands is non-trivial. Please refer to [arXiv:0704.1263] for technical details.

        Args:
            two_cmds (list): a list of two commands to swap

        Returns:
            list: a list of two new commands after swapping
        """
        cmd1 = two_cmds[0]
        cmd2 = two_cmds[1]
        name1 = cmd1.name
        name2 = cmd2.name

        if not {name1, name2}.issubset(["E", "M", "X", "Z", "S"]):
            raise ArgumentTypeError(
                f"Invalid command names: ({name1}) and ({name2})!"
                "Only 'E', 'M', 'X', 'Z' and 'S' are supported as the command names."
            )

        # [X, E] --> [E, X]
        if name1 == "X" and name2 == "E":
            X_qubit = cmd1.which_qubit
            E_qubits = cmd2.which_qubit[:]
            if X_qubit not in E_qubits:
                return [cmd2, cmd1]  # swap the independent commands
            else:
                op_qubit = list(set(E_qubits).difference([X_qubit]))
                new_cmd = Pattern.CommandZ(op_qubit[0], cmd1.domain)
                return [cmd2, new_cmd, cmd1]  # create a new command

        # [Z, E] --> [E, Z]
        elif name1 == "Z" and name2 == "E":
            return [cmd2, cmd1]  # swap the independent commands

        # [M, E] --> [E, M]
        elif name1 == "M" and name2 == "E":
            if cmd1.which_qubit not in cmd2.which_qubit:
                return [cmd2, cmd1]  # swap the independent commands
            else:
                raise ArgumentTypeError(
                    f"Invalid command names: ({name1}) and ({name2})!"
                    f"The measurement command must be executed after the entanglement command."
                )

        # [X, M] --> [M, X]
        elif name1 == "X" and name2 == "M":
            X_qubit = cmd1.which_qubit
            M_qubit = cmd2.which_qubit

            if X_qubit != M_qubit:
                return [cmd2, cmd1]  # swap the independent commands
            else:
                measurement_plane = cmd2.plane
                if measurement_plane == "XY":
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, "XY", cmd2.domain_s + cmd1.domain, cmd2.domain_t)
                elif measurement_plane == "YZ":
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, "YZ", cmd2.domain_s, cmd2.domain_t + cmd1.domain)
                else:
                    raise ArgumentTypeError(
                        f"Invalid measurement plane: ({measurement_plane})! Only 'XY' and 'YZ' are"
                        f" supported as the measurement plane in this version."
                    )

                return [M_new]  # create a new command

        # [Z, M] --> [M, Z]
        elif name1 == "Z" and name2 == "M":
            Z_qubit = cmd1.which_qubit
            M_qubit = cmd2.which_qubit

            if Z_qubit != M_qubit:
                return [cmd2, cmd1]  # swap the independent commands
            else:
                measurement_plane = cmd2.plane
                if measurement_plane == "YZ":
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, "YZ", cmd2.domain_s + cmd1.domain, cmd2.domain_t)
                elif measurement_plane == "XY":
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, "XY", cmd2.domain_s, cmd2.domain_t + cmd1.domain)
                else:
                    raise ArgumentTypeError(
                        f"Invalid measurement plane: ({measurement_plane})! Only 'XY' and 'YZ' are"
                        f" supported as the measurement plane in this version."
                    )
                return [M_new]  # create a new command

        # [Z, X] --> [X, Z]
        elif name1 == "Z" and name2 == "X":
            return [cmd2, cmd1]  # swap the independent commands

        # [X, X] --> [X]
        elif name1 == "X" and name2 == "X":
            X1_qubit = cmd1.which_qubit
            X2_qubit = cmd2.which_qubit
            if X1_qubit == X2_qubit:
                return [Pattern.CommandX(X1_qubit, cmd1.domain + cmd2.domain)]  # create a new command
            else:
                return two_cmds  # keep both commands

        # [Z, Z] --> [Z]
        elif name1 == "Z" and name2 == "Z":
            Z1_qubit = cmd1.which_qubit
            Z2_qubit = cmd2.which_qubit
            if Z1_qubit == Z2_qubit:
                return [Pattern.CommandZ(Z1_qubit, cmd1.domain + cmd2.domain)]  # create a new command
            else:
                return two_cmds  # keep both commands

        # [S, M / X / Z / S] --> [M / X / Z / S, S]
        elif name1 == "S":
            # Swap command S with command M
            # Measurements in 'XY' plane and in 'YZ' plane have the same rule
            # [S, M] --> [M, S]
            if name2 == "M":
                S_qubit = cmd1.which_qubit
                # According to the reference [arXiv:0704.1263],
                # if this 'S_qubit' is in the 'domain_s' or 'domain_t' of the command M,
                # the 'domain_s' or 'domain_t' will be summed with the domain of command S
                S_domains = cmd1.domain
                if S_qubit in cmd2.domain_s:
                    cmd2.domain_s += S_domains
                if S_qubit in cmd2.domain_t:
                    cmd2.domain_t += S_domains
                # According to the reference [arXiv:0704.1263],
                # if this 'S_qubit' is not in the 'domain_s' or 'domain_t' of the command M,
                # command S and command M can swap independently without domain modification
                return [cmd2, cmd1]  # swap the commands with domain modification
            # [S, X] --> [X, S], [S, Z] --> [Z, S], [S, S] --> [S, S]
            elif name2 in ["X", "Z", "S"]:
                S_qubit = cmd1.which_qubit
                S_domains = cmd1.domain
                if S_qubit in cmd2.domain:
                    cmd2.domain += S_domains
                return [cmd2, cmd1]  # swap the commands with domain modification
            else:
                return two_cmds  # keep both commands

        else:
            return two_cmds  # keep both commands

    def __propagate(self, cmd_type: str, cmds: list) -> list:
        r"""Propagate a given type of commands in the list to the front.

        Warning:
            Propagation to the front does not mean to swap this type of commands
            with brute force to the leftmost of the whole list.
            Instead, just propagate this type of commands to the available leftmost positions.
            For example, the measurement commands must be executed after the entanglement commands.
            So when calling this method, all measurement commands are propagated to the
            leftmost positions after the entanglement commands.

        Args:
            cmd_type (str): a command type to be propagated. It must be in ["E", "M", "X", "Z", "S"]
            cmds (list): a list of commands

        Returns:
            list: a list of new commands after propagation
        """
        if not isinstance(cmds, list):
            raise ArgumentTypeError(
                f"Invalid commands list ({cmds}) with the type: ({type(cmds)})!\n"
                f"Only `List` is supported as the type of commands list."
            )

        # Back to front, propagate command E, command M, command X and command Z sequentially
        if {cmd_type}.issubset(["E", "M", "X", "Z"]):
            for i in range(len(cmds) - 1, 0, -1):
                if cmds[i].name == cmd_type:
                    cmds = cmds[: i - 1] + self.__swap([cmds[i - 1], cmds[i]]) + cmds[i + 1 :]

        # Front to back, propagate command S
        else:
            for i in range(0, len(cmds) - 1):
                if cmds[i].name == cmd_type:
                    cmds = cmds[:i] + self.__swap([cmds[i], cmds[i + 1]]) + cmds[i + 2 :]

        return cmds

    @staticmethod
    def __cmds_to_nums(cmds: list) -> Tuple[list, list, list]:
        r"""Map the commands to the numbers.

        Args:
            cmds (list): a list of commands

        Returns:
            Tuple[list, list, list]: a tuple containing a list of counts on each type of the commands,
            a list of numbers, and a list of numbers arranged from small to large
        """
        cmd_map = {
            "E": 1,  # order of commands E
            "M": 2,  # order of commands M
            "X": 3,  # order of commands X
            "Z": 4,  # order of commands Z
            "S": 5,
        }  # order of commands S
        cmd_num_wild = [cmd_map[cmd.name] for cmd in cmds]
        cmd_num_std = cmd_num_wild[:]
        cmd_num_std.sort(reverse=False)
        cmds_count = [cmd_num_std.count(i) for i in [1, 2, 3, 4, 5]]  # count each type of commands

        return cmds_count, cmd_num_wild, cmd_num_std

    def __distance_to_standard(self, cmds: list) -> float:
        r"""Hamming distance of the current commands to the standard order.

        Args:
            cmds (list): current command list

        Returns:
            float: the hamming distance divided by the total length
        """
        cmds_count, cmd_wild, cmd_std = self.__cmds_to_nums(cmds[:])

        return sum([cmd_wild[i] == cmd_std[i] for i in range(len(cmd_wild))]) / len(cmd_wild)

    def __is_standard(self, cmd_type: str, cmds: list) -> bool:
        r"""Check if a certain type of commands is in a standard order.

        Args:
            cmd_type (str): a command type in ["E", "M", "X", "Z", "S"]
            cmds (list): a list of commands

        Returns:
            bool: whether the commands are in a standard order
        """
        if not {cmd_type}.issubset(["E", "M", "X", "Z", "S"]):
            raise ArgumentTypeError(f"Input {cmd_type} should be 'E', 'M', 'X', 'Z' or 'S'.")

        # Map the commands to numbers
        cmds_count, cmd_num_wild, cmd_num_std = self.__cmds_to_nums(cmds)
        pointer_map = {
            "E": sum(cmds_count[:1]),  # number of commands E
            "M": sum(cmds_count[:2]),  # number of commands E + M
            "X": sum(cmds_count[:3]),  # number of commands E + M + X
            "Z": sum(cmds_count[:4]),  # number of commands E + M + X + Z
            "S": sum(cmds_count[:5]),
        }  # number of commands E + M + X + Z + S

        return cmd_num_wild[: pointer_map[cmd_type]] == cmd_num_std[: pointer_map[cmd_type]]

    def __simplify_pauli_measurements(self) -> None:
        r"""Simplify the dependencies of Pauli measurements.

        In certain cases, the dependencies to other vertices can be excluded with Pauli simplification method.

         .. math::

            \text{Let } \alpha \text{ be a measurement angle without adaptation. }
            \text{Indeed, the adaptive angle of } \alpha \text{ has only these four values:}

            \theta_{\text{ad}} = \alpha

            \theta_{\text{ad}} = \alpha + \pi

            \theta_{\text{ad}} = - \alpha

            \theta_{\text{ad}} = - \alpha + \pi

            \text{When the } \alpha \text{ is in } [0, \pi / 2, \pi, 3 \times \pi / 2]
            \text{, the dependencies can be simplified. For example, when } \alpha = \pi
            \text{, the measurement angle of } \pi + t \times \pi \text{ and } - \pi + t \times \pi
            \text{ are equal. Therefore, it has no dependencies to the vertices in } domain\_s
            \text{. So the } domain\_s \text{ can be excluded, and so on.}
        """
        for cmd in self._pattern.commands:
            if cmd.name == "M":
                remainder = cmd.angle % (2 * pi)
                if remainder in [0, pi]:
                    cmd.domain_s = []
                elif remainder in [pi / 2, (3 * pi) / 2]:
                    cmd.domain_t += cmd.domain_s[:]
                    cmd.domain_s = []

    def standardize(self) -> None:
        r"""Standardize the measurement pattern.

        This method transforms a wild pattern to a standard 'EMC' pattern.
        Entanglement commands are propagated to the leftmost of the command list.
        Then come to the measurement commands and last come to the byproduct correction commands.
        To simplify the pattern, Pauli simplification method is automatically implemented to
        exclude some dependencies.
        """
        cmds = self._pattern.commands

        for cmd_type in ["E", "M", "X", "Z"]:
            while not self.__is_standard(cmd_type, cmds):
                cmds = self.__propagate(cmd_type, cmds)
                print_progress(self.__distance_to_standard(cmds), "Standardization Progress", self._track)

        self._pattern.commands = cmds
        self.__simplify_pauli_measurements()

    @staticmethod
    def __pull_out_domain_t(cmds: list) -> list:
        r"""Pull out the signal shifting from the measurement command.

        Args:
            cmds (list): command list

        Returns:
            list: processed command list
        """
        cmds_len = len(cmds)
        for i in range(cmds_len - 1, -1, -1):
            cmd = cmds[i]
            if cmd.name == "M":
                signal_cmd = Pattern.CommandS(cmd.which_qubit, cmd.domain_t)
                cmd.domain_t = []
                cmds = cmds[:i] + [cmd, signal_cmd] + cmds[i + 1 :]
        return cmds

    def shift_signals(self) -> None:
        r"""Shift signal commands."""
        cmds = self._pattern.commands
        cmds = self.__pull_out_domain_t(cmds)

        # Propagate CommandS
        while not self.__is_standard("S", cmds):
            cmds = self.__propagate("S", cmds)
            print_progress(self.__distance_to_standard(cmds), "Signal Shifting Progress", self._track)

        self._pattern.commands = cmds

    def get_pattern(self) -> "Pattern":
        r"""Get the measurement pattern.

        Returns:
            Pattern: a measurement pattern
        """
        return self._pattern

    def track_progress(self, track=True) -> None:
        r"""Track the progress of measurement calculus.

        Args:
            track (bool, optional): whether to turn on the progress tracking
        """
        self._track = track

    def optimize_by_row(self, order=None) -> None:
        r"""Optimize the measurement order by the row-first principle.

        Args:
            order (list, optional): the measurement order

        Warning:
            This is a heuristic algorithm that aims to measure each qubits row by row.
        """

        def _default_order_by_row(labels: list, order: list) -> list:
            r"""The default order of labels by row.

            Ordering rule: smaller row/column number has a higher priority.

            Args:
                labels (list): a list of labels
                order (list): the measurement order

            Returns:
                list: a list of ordered labels
            """
            container = {i: [] for i in order}
            for label in labels:
                container[label[0]].append(label)

            return sum([container[item] for item in order], [])

        order = list(range(self._circuit_width)) if order is None else order

        cmds = self._pattern.commands

        # Split the commands by type
        cmdE_list = [cmd for cmd in cmds if cmd.name == "E"]
        cmdM_list = [cmd for cmd in cmds if cmd.name == "M"]
        cmdC_list = [cmd for cmd in cmds if cmd.name in ["X", "Z"]]
        cmdS_list = [cmd for cmd in cmds if cmd.name == "S"]

        # Construct a dict from qubit labels and their measurement commands
        cmdM_map = {cmd.which_qubit: cmd for cmd in cmdM_list}

        # Sort all the qubit labels by row
        cmdM_qubit_list = _default_order_by_row([cmdM.which_qubit for cmdM in cmdM_list], order=order)
        mea_length = len(cmdM_qubit_list)

        for i in range(mea_length):
            optimal = False
            while not optimal:  # if the qubits list is not standard
                # Slice measurement qubit list into three parts
                measured = cmdM_qubit_list[:i]
                measuring = cmdM_qubit_list[i]
                to_measure = cmdM_qubit_list[i + 1 :]

                domains = set(cmdM_map[measuring].domain_s + cmdM_map[measuring].domain_t)
                # Find the qubits in domain but not in front of the current measurement
                push = _default_order_by_row(list(domains.difference(measured)), order=order)
                if push:  # Remove qubits from the to_measure list and push it to the front
                    to_measure = [qubit for qubit in to_measure if qubit not in push]
                    cmdM_qubit_list = cmdM_qubit_list[:i] + push + [measuring] + to_measure
                else:  # if no push qubits then jump out of the while loop
                    optimal = True

            print_progress((i + 1) / mea_length, "Optimization Progress", self._track)

        # Sort the measurement commands by the sorted qubit labels
        cmdM_opt = [cmdM_map[which_qubit] for which_qubit in cmdM_qubit_list]

        # Update pattern
        cmds = cmdE_list + cmdM_opt + cmdC_list + cmdS_list
        self._pattern.commands = cmds

    def optimize_by_column(self) -> None:
        r"""Optimize the measurement orders by the column-first principle.

        Warning:
            This is a heuristic algorithm that aims to measure each qubits column by column.
        """

        def _default_order_by_column(labels: list) -> list:
            r"""The default order of labels by column.

            Ordering rule: smaller column/row number has a higher priority.

            Args:
                labels (list): a list of labels

            Returns:
                list: a list of ordered labels
            """
            # Construct a dict by string labels and their float values
            labels_dict = {label: (label[1], label[0]) for label in labels}
            # Sort the dict by values (sort column first and then row)
            sorted_dict = dict(sorted(labels_dict.items(), key=lambda item: item[1]))
            # Extract the keys in the dict
            labels_sorted = list(sorted_dict.keys())

            return labels_sorted

        cmds = self._pattern.commands

        # Split the commands by type
        cmdE_list = [cmd for cmd in cmds if cmd.name == "E"]
        cmdM_list = [cmd for cmd in cmds if cmd.name == "M"]
        cmdC_list = [cmd for cmd in cmds if cmd.name in ["X", "Z"]]
        cmdS_list = [cmd for cmd in cmds if cmd.name == "S"]

        # Construct a dict from qubit labels and their measurement commands
        cmdM_map = {cmd.which_qubit: cmd for cmd in cmdM_list}

        # Sort all the qubit labels by column
        cmdM_qubit_list = _default_order_by_column([cmdM.which_qubit for cmdM in cmdM_list])
        mea_length = len(cmdM_qubit_list)

        for i in range(mea_length):
            optimal = False
            while not optimal:  # if the qubits list is not standard
                # Slice measurement qubit list into three parts
                measured = cmdM_qubit_list[:i]
                measuring = cmdM_qubit_list[i]
                to_measure = cmdM_qubit_list[i + 1 :]

                domains = set(cmdM_map[measuring].domain_s + cmdM_map[measuring].domain_t)
                # Find the qubits in domain but not in front of the current measurement
                push = _default_order_by_column(list(domains.difference(measured)))
                if push:  # Remove qubits from the to_measure list and push it to the front
                    to_measure = [qubit for qubit in to_measure if qubit not in push]
                    cmdM_qubit_list = cmdM_qubit_list[:i] + push + [measuring] + to_measure
                else:  # if no push qubits then jump out of the while loop
                    optimal = True
            print_progress((i + 1) / mea_length, "Optimization Progress", self._track)

        # Sort the measurement commands by the sorted qubit labels
        cmdM_opt = [cmdM_map[which_qubit] for which_qubit in cmdM_qubit_list]

        # Update pattern
        cmds = cmdE_list + cmdM_opt + cmdC_list + cmdS_list
        self._pattern.commands = cmds
