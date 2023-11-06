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

from typing import Tuple, Union, Any
from Extensions.QuantumNetwork.qcompute_qnet.quantum.state import PureState, Zero, Plus
from Extensions.QuantumNetwork.qcompute_qnet.quantum.utils import kron, find_keys_by_value

__all__ = ["Pattern"]


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

    class CommandN:
        r"""Class for creating a state preparation command.

        Attributes:
            name (str): name of the command
            which_qubit (Any): vertex label
            matrix (numpy.ndarray): matrix representation of the quantum state
        """

        def __init__(self, which_qubit: Any, matrix=None):
            r"""Constructor for CommandN class.

            Args:
                which_qubit (Any): vertex label
                matrix (numpy.ndarray, optional): matrix representation of the quantum state
            """
            self.name = "N"
            self.which_qubit = which_qubit
            self.matrix = Plus.SV if matrix is None else matrix

    class CommandE:
        r"""Class for creating an entanglement command.

        It entangles two adjacent vertices by operating a Control Z (CZ) gate on them.

        Attributes:
            name (str): name of the command
            which_qubit (list): a list of two vertices to be entangled
        """

        def __init__(self, which_qubit: list):
            r"""Constructor for CommandE class.

            Args:
                which_qubit (list): a pair of vertex labels
            """
            self.name = "E"
            self.which_qubit = which_qubit

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
            name (str): name of the command
            which_qubit (Any): vertex label
            angle (Union[float, int]): primitive angle
            plane (str): measurement plane, can be 'XY' or 'YZ'
            domain_s (list): a list of vertices in 'domain_s' that have dependencies to this command
            domain_t (list): a list of vertices in 'domain_t' that have dependencies to this command
        """

        def __init__(self, which_qubit: Any, angle: Union[float, int], plane: str, domain_s: list, domain_t: list):
            r"""Constructor for CommandM class.

            Args:
                which_qubit (Any): vertex label
                angle (Union[float, int]): primitive angle
                plane (str): measurement plane
                domain_s (list): a list of vertices in 'domain_s' that have dependencies to this command
                domain_t (list): a list of vertices in 'domain_t' that have dependencies to this command
            """
            assert isinstance(angle, float) or isinstance(angle, int), (
                f"Invalid measurement angle ({angle}) with the type: `{type(angle)}`!\n"
                "Only `float` and `int` are supported as the type of measurement angle."
            )

            self.name = "M"
            self.which_qubit = which_qubit
            self.angle = angle
            self.plane = plane
            self.domain_s = domain_s
            self.domain_t = domain_t

    class CommandX:
        r"""Class for creating a Pauli-X correction command.

        Attributes:
            name (str): name of the command
            which_qubit (Any): vertex label
            domain (list): a list of vertices that have dependencies to this command
        """

        def __init__(self, which_qubit: Any, domain: list):
            r"""Constructor for CommandX class.

            Args:
                which_qubit (Any): vertex label
                domain (list): a list of vertices that have dependencies to this command
            """
            self.name = "X"
            self.which_qubit = which_qubit
            self.domain = domain

    class CommandZ:
        r"""Class for creating a Pauli-Z correction command.

        Attributes:
            name (str): name of the command
            which_qubit (Any): vertex label
            domain (list): a list of vertices that have dependencies to this command
        """

        def __init__(self, which_qubit: Any, domain: list):
            r"""Constructor for CommandZ class.

            Args:
                which_qubit (Any): vertex label
                domain (list): a list of vertices that have dependencies to this command
            """
            self.name = "Z"
            self.which_qubit = which_qubit
            self.domain = domain

    class CommandS:
        r"""Class for creating a signal shifting command.

        Note:
            Signal shifting is a unique operation in MBQC.
            It can be used to simplify the measurement commands by excluding the dependencies in 'domain_t'.

        Attributes:
            name (str): name of the command
            which_qubit (Any): vertex label
            domain (list): a list of vertices that have dependencies to this command
        """

        def __init__(self, which_qubit: Any, domain: list):
            r"""Constructor for CommandS class.

            Args:
                which_qubit (Any): vertex label
                domain (list): a list of vertices that have dependencies to this command
            """
            self.name = "S"
            self.which_qubit = which_qubit
            self.domain = domain

    def has_preparation_cmds(self) -> bool:
        r"""Check if the current pattern has state preparation commands.

        Returns:
            bool: whether the pattern has state preparation commands
        """
        return any(cmd.name == "N" for cmd in self.commands)

    def fill_preparation_cmds(self, input_states=None) -> None:
        r"""Complete state preparation commands for the pattern.

        Args:
            input_states (str, optional): whether to replace the input state as zero states
        """
        if input_states is None:
            cmds_N = [self.CommandN(which_qubit) for which_qubit in self.space]
            self.commands = cmds_N + self.commands
        elif input_states == "zero_states":
            cmds_N_input = [self.CommandN(which_qubit, Zero.SV) for which_qubit in self.input_]
            other_vertex = set(self.space).difference(self.input_)
            cmds_N_others = [self.CommandN(which_qubit) for which_qubit in other_vertex]
            self.commands = cmds_N_input + cmds_N_others + self.commands

    def is_standard(self) -> bool:
        r"""Check if the current pattern is a standard pattern.

        A standard pattern should have commands ordered in [N, E, M, X, Z, S].

        Returns:
            bool: whether the pattern is standard
        """
        cmds = self.commands[:]

        # Check if the pattern is of a standard EMC form
        cmd_map = {"N": 0, "E": 1, "M": 2, "X": 3, "Z": 4, "S": 5}
        cmd_num_wild = [cmd_map[cmd.name] for cmd in cmds]
        cmd_num_standard = cmd_num_wild[:]
        cmd_num_standard.sort(reverse=False)

        return cmd_num_wild == cmd_num_standard

    def postpone_preparation_and_entanglement(self) -> None:
        r"""Postpone the state preparation and entanglement commands.

        Note:
            This is equivalent to the vertex dynamic classification in the ``MBQC`` simulator.
        """
        cmd_lst = {"N": [], "E": [], "M": [], "X": [], "Z": [], "S": []}
        for cmd in self.commands:
            cmd_lst[cmd.name].append(cmd)

        new_commands = cmd_lst["M"][:]

        # Postpone entanglement
        for cmdE in cmd_lst["E"]:
            # find the index to insert
            for i, cmd in enumerate(new_commands):
                if cmd.name == "M" and cmd.which_qubit in cmdE.which_qubit:
                    new_commands = new_commands[:i] + [cmdE] + new_commands[i:]
                    break
        # Postpone preparation
        for cmdN in cmd_lst["N"]:
            # find the index to insert
            for i, cmd in enumerate(new_commands):
                if cmd.name == "E" and cmdN.which_qubit in cmd.which_qubit:
                    new_commands = new_commands[:i] + [cmdN] + new_commands[i:]
                    break

        self.commands = new_commands

    def to_dynamic_circuit(self, input_states=None) -> "Circuit":
        r"""Transpile the current measurement pattern into an equivalent dynamic quantum circuit.

        Args:
            input_states (str, optional): input state of the transpiled circuit

        Returns:
            Circuit: dynamic quantum circuit
        """

        def _allocate_reg_unit(qreg: dict, vertex: Any) -> Tuple[int, dict]:
            r"""Allocate a register unit for the vertex.

            Args:
                qreg (dict): a dict that records the register information
                vertex (Any): a vertex in the pattern space

            Returns:
                tuple: a register unit for the vertex and the updated register
            """
            # If the vertex has been loaded
            if vertex in qreg.values():
                idx = find_keys_by_value(qreg, vertex)[0]
                return idx, qreg

            # If the vertex has not been loaded, then allocate a register unit
            available_regs = find_keys_by_value(qreg, None)
            idx = min(available_regs) if available_regs else len(qreg)
            qreg[idx] = vertex
            return idx, qreg

        self.fill_preparation_cmds(input_states=input_states)
        self.postpone_preparation_and_entanglement()

        qreg = {}  # a dict to record the connection between vertex and its execution register
        cir_history = []  # dynamic circuit to construct
        for cmd in self.commands:
            if cmd.name == "N":
                vertex = cmd.which_qubit
                idx, qreg = _allocate_reg_unit(qreg, vertex)
                gate = {"name": "r", "which_qubit": [idx], "signature": None, "matrix": cmd.matrix}
            elif cmd.name == "E":
                vertex0, vertex1 = cmd.which_qubit[0], cmd.which_qubit[1]
                idx0, qreg = _allocate_reg_unit(qreg, vertex0)
                idx1, qreg = _allocate_reg_unit(qreg, vertex1)
                gate = {"name": "cz", "which_qubit": [idx0, idx1], "signature": None}
            elif cmd.name == "M":
                vertex = cmd.which_qubit
                idx, qreg = _allocate_reg_unit(qreg, vertex)
                gate = {
                    "name": "m",
                    "which_qubit": [idx],
                    "signature": None,
                    "basis": {
                        "angle": cmd.angle,
                        "plane": cmd.plane,
                        "domain_s": cmd.domain_s,
                        "domain_t": cmd.domain_t,
                    },
                    "mid": vertex,
                }  # used to identify the measurement outcome
                qreg[idx] = None  # recycle the register unit
            else:
                raise NotImplementedError

            cir_history.append(gate)

        from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

        circuit = Circuit()
        circuit._history = cir_history
        circuit.output_ids = self.output_[0]  # indicate which measurement outcomes are the final result
        return circuit

    def run(self, shots: int, input_state=None) -> dict:
        r"""Run a measurement pattern multiple times and obtain the sampling results.

        Args:
            shots (int): number of sampling
            input_state (PureState, optional): input quantum state

        Warning:
            The input states are taken as plus states by default.

        Returns:
            dict: classical results
        """

        n = len(self.input_)
        if input_state == "zero_states":
            input_state = PureState(kron([Zero.SV for _ in range(n)]), list(range(n)))

        samples = []

        for shot in range(shots):
            from Extensions.QuantumNetwork.qcompute_qnet.quantum.backends.mbqc import MBQC

            mbqc = MBQC()
            mbqc.set_pattern(self)
            # mbqc.draw_process(pos=True)
            mbqc.set_input_state(input_state)
            mbqc.run_pattern()
            c_output = mbqc.get_classical_output()
            samples.append(c_output)

        sample_dict = {}
        for string in list(set(samples)):
            sample_dict[string] = samples.count(string)

        from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit

        counts = Circuit.sort_results(sample_dict)
        results = {"pattern_name": self.name, "shots": shots, "counts": counts}

        return results

    def print(self) -> None:
        r"""Print all commands in the command list."""
        print("------------------------------------------------------------")
        print("                    Current Command List                    ")
        print("------------------------------------------------------------")
        # Print commands list
        for cmd in self.commands:
            print("\033[91m" + "Command:".ljust(16) + cmd.name + "\033[0m")
            if cmd.name == "N":
                print("which_qubit:".ljust(15), cmd.which_qubit)
                print("matrix:".ljust(15), cmd.matrix)
            elif cmd.name == "E":
                print("which_qubit:".ljust(15), cmd.which_qubit)
            elif cmd.name == "M":
                print("which_qubit:".ljust(15), cmd.which_qubit)
                print("plane:".ljust(15), cmd.plane)
                print("angle:".ljust(15), cmd.angle)
                print("domain_s:".ljust(15), cmd.domain_s)
                print("domain_t:".ljust(15), cmd.domain_t)
            else:
                print("which_qubit:".ljust(15), cmd.which_qubit)
                print("domain:".ljust(15), cmd.domain)
            print("-----------------------------------------------------------")
