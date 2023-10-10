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
mcalculus
"""
FileErrorCode = 4


from typing import Tuple
from numpy import pi

from QCompute.OpenService import ModuleErrorCode
from QCompute.QPlatform import Error

from QCompute.OpenService.service_ubqc.client.qobject import Pattern, Circuit

__all__ = [
    "MCalculus"
]


class MCalculus:
    r"""Define the ``MCalculus`` class.

    This class provides various basic operations for calculating measurement patterns.
    Please see the reference [The measurement calculus, arXiv: 0704.1263] for more details.
    """

    def __init__(self):
        r"""``MCalculus`` constructor, used to instantiate a ``MCalculus`` object.

        This class provides various basic operations for calculating measurement patterns.
        Please see the reference [The measurement calculus, arXiv: 0704.1263] for more details.
        """
        self.__circuit = None  # circuit
        self.__circuit_slice = []  # sliced circuit
        self.__wild_pattern = []  # wild pattern
        self.__pattern = None  # standard pattern
        self.__measured_qubits = []  # measured qubits
        self.__width: int = None  # circuit width
        self.__bw_depth: int = None  # circuit depth

    def set_circuit(self, circuit):
        r"""Set a quantum circuit to ``MCalculus`` class.

        Args:
            circuit (Circuit): a quantum circuit
        """
        if not isinstance(circuit, Circuit):
            raise Error.ArgumentError(f'Invalid circuit ({circuit}) with the type: ({type(circuit)})!\nOnly `Circuit` is supported as the type of quantum circuit.', ModuleErrorCode, FileErrorCode, 1)

        if not circuit.is_valid():
            raise Error.ArgumentError(f'Invalid circuit: ({circuit})!\nThis quantum circuit is not valid after the simplification! There is at least one qubit not operated by any effective quantum gates.', ModuleErrorCode, FileErrorCode, 2)

        self.__circuit = circuit
        self.__width = circuit.get_width()

    def __list_to_net(self, vertices):
        r"""Map a vertex list to a vertex net.

        This method maps a one-dimensional vertex list to a two-dimensional vertex net.
        The net contains the labels of all vertices on the graph.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            vertices (list): a one-dimensional list containing the labels of input or output vertices

        Returns:
            list: a two-dimensional vertex net. This net has a special network structure of List[List]，
                  Each row of the net records the column number of input or output vertices
        """
        # Get width
        rows = {qubit[0] for qubit in vertices}
        self.__width = len(rows) if self.__width is None else self.__width
        # Map to net
        bit_net = [[] for _ in range(self.__width)]
        for qubit in vertices:
            bit_net[qubit[0]].append(qubit[1])
        return bit_net

    def __get_input(self):
        r"""Get the labels of the input vertices in the list.

        Note:
            This is an intrinsic method. No need to call it externally.

        Returns:
            list: a list containing the labels of input vertices
        """
        input_list = []
        for pat in self.__wild_pattern:
            input_list += pat.input_
        # Map to net
        bit_net = self.__list_to_net(input_list)
        input_ = [(i, min(bit_net[i])) for i in range(self.__width)]
        return input_

    def __get_output(self):
        r"""Get the labels of the output vertices in the list.

        Note:
            This is an intrinsic method. No need to call it externally.

        Returns:
            list: a list containing the labels of output vertices
        """
        output_list = []
        for pat in self.__wild_pattern:
            output_list += pat.output_
        bit_net = self.__list_to_net(output_list)
        # Divide the output into classical channels and quantum channels
        c_out = [(i, max(bit_net[i])) for i in range(self.__width) if i in self.__measured_qubits]
        q_out = [(i, max(bit_net[i])) for i in range(self.__width) if i not in self.__measured_qubits]
        output_ = [c_out, q_out]
        return output_

    def __splice_patterns(self, input_, output_):
        r"""Splice all measurement patterns together to form a complete pattern.

        The spliced pattern is equivalent to the whole quantum circuit.

        Note:
            This is an intrinsic method. No need to call it externally.
        """
        names = ''
        space = []
        cmds = []

        for pat in self.__wild_pattern:
            names += pat.name
            space += [qubit for qubit in pat.space if qubit not in space]
            cmds += pat.commands

        self.__pattern = Pattern(names, space, input_, output_, cmds)

    def __build_odd_width_bw(self, bw_cir, pos: Tuple[int, int]):
        r"""Build a brickwork mould with odd width.

        Note:
            This is an intrinsic method. No need to call it externally.
            According to the reference [arXiv:0807.4154], there are two types of brickwork moulds.
            These two moulds have different widths.
            One brickwork mould has an odd width. The other one has an even width.

        Args:
            bw_cir (dict): a brickwork circuit
            pos (tuple): a quantum gate position

        Returns:
            list: a brickwork mould with odd width
        """
        row, col = pos
        if col % 8 == 0:
            if row == self.__width - 1:
                gate_list = [bw_cir[pos]]
            else:
                if row % 2 == 0:
                    gate_list = [bw_cir[pos], bw_cir[(row + 1, col)]]
                else:
                    gate_list = []
        else:
            if row == 0:
                gate_list = [bw_cir[pos]]
            else:
                if row % 2 == 1:
                    gate_list = [bw_cir[pos], bw_cir[(row + 1, col)]]
                else:
                    gate_list = []

        return gate_list

    def __build_even_width_bw(self, bw_cir, pos: Tuple[int, int]):
        r"""Build a brickwork mould with even width.

        Note:
            This is an intrinsic method. No need to call it externally.
            According to the reference [arXiv:0807.4154], there are two types of brickwork moulds.
            These two moulds have different widths.
            One brickwork mould has an odd width. The other one has an even width.

        Args:
            bw_cir (dict): a brickwork circuit
            pos (tuple): a quantum gate position

        Returns:
            list: a brickwork mould with even width
        """
        row, col = pos
        if col % 8 == 0:
            if row % 2 == 0:
                gate_list = [bw_cir[pos], bw_cir[(row + 1, col)]]
            else:
                gate_list = []
        else:
            if row == 0 or row == self.__width - 1:
                gate_list = [bw_cir[pos]]
            else:
                if row % 2 == 1:
                    gate_list = [bw_cir[pos], bw_cir[(row + 1, col)]]
                else:
                    gate_list = []

        return gate_list

    def __build_brickwork(self, bw_cir):
        r"""Build a brickwork mould.

        Note:
            This is an intrinsic method. No need to call it externally.
            According to the reference [arXiv:0807.4154], there are two types of brickwork moulds.
            These two moulds have different width.
            One brickwork mould has an odd width. The other one has an even width.

        Args:
            bw_cir (dict): a brickwork circuit
        """
        for i in range(self.__bw_depth + 1):
            col = i * 4
            for row in range(self.__width):
                pos = (row, col)

                if pos in bw_cir.keys():
                    gate = bw_cir[pos]
                    name = gate[0]

                    # Build logic gate pattern
                    if name != 'm':
                        # Brickwork type 1, with even width
                        if self.__width % 2 == 0:
                            gate_list = self.__build_even_width_bw(bw_cir, pos)
                        # Brickwork type 2, with odd width
                        else:
                            gate_list = self.__build_odd_width_bw(bw_cir, pos)

                    # Build measurement pattern
                    else:
                        gate_list = [gate]

                    # As this is the traversal algorithm, there exists empty gate list
                    if gate_list:
                        pattern = self.__to_bw_pat(gate_list)
                        self.__wild_pattern.append(pattern)
                    else:
                        continue

                else:
                    continue

    @staticmethod
    def __set_cmds(name: str, space, params, domains):
        r"""Set the commands of the brickwork patterns.

        Please see the reference [arXiv:0807.4154] for more details.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            name (str): gate name
            space (list): space vertices
            params (list): command parameters
            domains (list): command domains

        Returns:
            list: a list of commands
        """
        if not {name}.issubset(['u', 'm', 'd']):
            raise Error.ArgumentError(f"Invalid gate name ({name})!\nOnly 'u', 'm' and 'd' are supported as the gate names in UBQC in this version, where 'u' is the 'unitary gate', 'm' is the 'measurement' and 'd' is the 'double qubit gate'.", ModuleErrorCode, FileErrorCode, 3)

        # Single qubit unitary gate commands
        if name == 'u':
            cmdE = [Pattern.CommandE([space[i], space[i + 1]]) for i in range(4)]

            cmdM = []
            for i in range(4):
                pos = space[i]
                param = -params[i]
                cmd = Pattern.CommandM(pos, param, 'XY', domains[i][0], domains[i][1])
                cmdM.append(cmd)

            cmdX = [Pattern.CommandX(space[4], [space[3]])]
            cmdZ = [Pattern.CommandZ(space[4], [space[2]])]

        # Single qubit measurement commands
        elif name == 'm':
            cmdE = []
            cmdM = [Pattern.CommandM(space[0], *params)]
            cmdX = []
            cmdZ = []

        # Double qubit gate commands
        else:
            cmdE = [Pattern.CommandE([space[5 * j + i], space[5 * j + i + 1]]) for i in range(4) for j in range(2)] + \
                   [Pattern.CommandE([space[2], space[7]]), Pattern.CommandE([space[4], space[9]])]

            cmdM = []
            for i in range(4):
                for j in range(2):
                    pos = space[5 * j + i]
                    param = - params[4 * j + i]
                    cmd = Pattern.CommandM(pos, param, 'XY', domains[4 * j + i][0], domains[4 * j + i][1])
                    cmdM.append(cmd)

            cmdX = [Pattern.CommandX(space[4], [space[3]]), Pattern.CommandX(space[9], [space[8]])]
            cmdZ = [Pattern.CommandZ(space[4], [space[2]]), Pattern.CommandZ(space[9], [space[3]]),
                    Pattern.CommandZ(space[9], [space[7]]), Pattern.CommandZ(space[4], [space[8]])]

        cmds = cmdE + cmdM + cmdX + cmdZ
        return cmds

    def __to_bw_pat(self, gate_list):
        r"""Translate the quantum gates and measurements to their equivalent brickwork patterns in MBQC.

        Note:
            This is an intrinsic method. No need to call it externally.

        Warning:
            Only one single qubit gate, two single qubit gates and one CNOT gate are supported
            as the valid input to this method in UBQC in this version.
            Please also be aware that the measurement patterns are unique
            because they all have specific brickwork structures.

        Args:
            gate_list (list): a list of gates to be translated
        """
        # Single qubit gate
        if len(gate_list) == 1:
            # Initialize parameters
            gate = gate_list[0]
            name = gate[0]
            input_ = gate[1][0]
            params = gate[3]
            row = input_[0]
            col_in = input_[1]

            if not {name}.issubset(['u', 'm']):
                raise Error.ArgumentError(f"Invalid gate list: ({gate_list})!\nOnly 'u' and 'm' are supported as the single qubit gate names in UBQC in this version, where 'u' is the 'unitary gate' and 'm' is the 'measurement'.", ModuleErrorCode, FileErrorCode, 4)

            # Unitary gate
            if name == 'u':
                output_ = gate[2][0]
                col_out = output_[1]
                # Set parameters
                space = [(row, col_in + i) for i in range(col_out - col_in + 1)]
                # Add a zero to the end of the list 'params'
                theta, phi, lamda = params
                params = [lamda, theta, phi, 0]
                domains = [[[], []], [[space[0]], []], [[space[1]], [space[0]]], [[space[2]], [space[1]]]]
                input_ = [(row, col_in)]
                output_ = [(row, col_out)]

            # Measurement
            else:
                self.__measured_qubits.append(row)
                # Set parameters
                space = [(row, col_in)]
                domains = None
                input_ = space
                output_ = []

        # Double qubit gate
        else:
            # Initialize parameters
            name = 'd'
            gate1 = gate_list[0]
            name1 = gate1[0]
            gate2 = gate_list[1]
            name2 = gate1[0]

            if not {name1, name2}.issubset(['u', 'cnot']):
                raise Error.ArgumentError(f"Invalid gate list: ({gate_list})!\nOnly 'u' and 'cnot' are supported as the double qubit gate names in UBQC in this version, where 'u' is the 'unitary gate' and 'cnot' is the 'CNOT gate'.", ModuleErrorCode, FileErrorCode, 5)

            # Two unitary gates
            if name1 == name2 == 'u':
                input1 = gate1[1][0]
                input2 = gate2[1][0]
            # CNOT gate
            else:
                input1 = gate1[1][0]
                input2 = gate1[1][1]

            # Set input and output qubits
            output1 = gate1[2][0]
            row1 = input1[0]
            row2 = input2[0]
            if not abs(row1 - row2) == 1:
                raise Error.ArgumentError(f'Invalid row indexes: {row1} and {row2}!\nOnly the adjacent qubits are supported as indexes of the quantum gates in UBQC in this version.', ModuleErrorCode, FileErrorCode, 6)

            col_in = input1[1]
            col_out = output1[1]
            # Set parameters
            space = [(row1, col_in + i) for i in range(col_out - col_in + 1)] + \
                    [(row2, col_in + i) for i in range(col_out - col_in + 1)]

            # Two unitary gates
            if name1 == name2 == 'u':
                params = []
                for i in range(2):
                    param = gate_list[i][3]
                    theta, phi, lamda = param
                    params += [lamda, theta, phi, 0]  # add a zero to the end of the list 'params'
            # CNOT gate
            else:
                params = [0, 0, pi / 2, 0, 0, pi / 2, 0, -pi / 2]

            domains = [[[], []], [[space[0]], []], [[space[1]], [space[0], space[6]]], [[space[2]], [space[1]]],
                       [[], []], [[space[5]], []], [[space[6]], [space[5], space[1]]], [[space[7]], [space[6]]]]
            input_ = [(row1, col_in), (row2, col_in)]
            output_ = [(row1, col_out), (row2, col_out)]

        cmds = self.__set_cmds(name, space, params, domains)
        return Pattern(name, space, input_, output_, cmds)

    def to_brickwork_pattern(self):
        r"""Translate a quantum circuit to its equivalent brickwork wild pattern in MBQC.

        Through this method, the equivalent wild pattern of the original quantum circuit can be obtained.
        """
        bw_cir = self.__circuit.get_brickwork_circuit()
        self.__bw_depth = self.__circuit.get_brickwork_depth()

        self.__build_brickwork(bw_cir)
        if self.__measured_qubits != list(range(self.__width)):
            raise Error.ArgumentError('Invalid brickwork pattern! Brickwork pattern does not support quantum output in UBQC in this version.', ModuleErrorCode, FileErrorCode, 7)

        input_ = self.__get_input()
        output_ = self.__get_output()
        self.__splice_patterns(input_, output_)

    @staticmethod
    def __swap(two_cmds):
        r"""Swap two commands.

        The rule to swap two commands are illustrated in the reference [arXiv:0704.1263].
        Please see the reference [arXiv:0704.1263] for more details.

        Note:
            This is an intrinsic method. No need to call it externally.

        Hint:
            The commands in the list are executed from the left to the right as default.
            For example, the command 'E' represents the entanglement command.
            According to the MBQC procedure, entanglements are executed in the first step.
            Therefore the command 'E' should be in the left side of the list.
            After the entanglements, the measurement commands are executed, and so on.

        Warning:
            This method only swaps two commands in ['E', 'M', 'X', 'Z', 'S'].

        Args:
            two_cmds (list): a list of two commands to be swapped

        Returns:
            list: a list of two new commands after swapping
        """
        cmd1 = two_cmds[0]
        cmd2 = two_cmds[1]
        name1 = cmd1.name
        name2 = cmd2.name

        if not {name1, name2}.issubset(['E', 'M', 'X', 'Z', 'S']):
            raise Error.ArgumentError(f"Invalid command names: ({name1}) and ({name2})! Only 'E', 'M', 'X', 'Z' and 'S' are supported as the command names in UBQC in this version, where 'E' is the entanglement command, 'M' is the measurement command, 'X' is the X byproduct correction command, 'Z' is the Z byproduct correction command and 'S' is the signal shifting command.", ModuleErrorCode, FileErrorCode, 8)

        # [X, E] --> [E, X]
        if name1 == 'X' and name2 == 'E':
            X_qubit = cmd1.which_qubit
            E_qubits = cmd2.which_qubits[:]
            if X_qubit not in E_qubits:
                return [cmd2, cmd1]  # swap the independent commands
            else:
                op_qubit = list(set(E_qubits).difference([X_qubit]))
                new_cmd = Pattern.CommandZ(op_qubit[0], cmd1.domain)
                return [cmd2, new_cmd, cmd1]  # create a new command

        # [Z, E] --> [E, Z]
        elif name1 == 'Z' and name2 == 'E':
            return [cmd2, cmd1]  # swap the independent commands

        # [M, E] --> [E, M]
        elif name1 == 'M' and name2 == 'E':
            if cmd1.which_qubit not in cmd2.which_qubits:
                return [cmd2, cmd1]  # swap the independent commands
            else:
                raise Error.ArgumentError(f'Invalid command names: ({name1}) and ({name2})! The measurement command must be executed after the entanglement command.', ModuleErrorCode, FileErrorCode, 9)

        # [X, M] --> [M, X]
        elif name1 == 'X' and name2 == 'M':
            X_qubit = cmd1.which_qubit
            M_qubit = cmd2.which_qubit
            if X_qubit != M_qubit:
                return [cmd2, cmd1]  # swap the independent commands
            else:
                measurement_plane = cmd2.plane
                if measurement_plane == 'XY':
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, 'XY', cmd2.domain_s + cmd1.domain, cmd2.domain_t)
                elif measurement_plane == 'YZ':
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, 'YZ', cmd2.domain_s, cmd2.domain_t + cmd1.domain)
                else:
                    raise Error.ArgumentError(f"Invalid measurement plane: ({measurement_plane})! Only 'XY' and 'YZ' are supported as the measurement plane in UBQC in this version.", ModuleErrorCode, FileErrorCode, 10)

                return [M_new]  # create a new command

        # [Z, M] --> [M, Z]
        elif name1 == 'Z' and name2 == 'M':
            Z_qubit = cmd1.which_qubit
            M_qubit = cmd2.which_qubit
            if Z_qubit != M_qubit:
                return [cmd2, cmd1]  # swap the independent commands
            else:
                measurement_plane = cmd2.plane
                if measurement_plane == 'YZ':
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, 'YZ', cmd2.domain_s + cmd1.domain, cmd2.domain_t)
                elif measurement_plane == 'XY':
                    M_new = Pattern.CommandM(M_qubit, cmd2.angle, 'XY', cmd2.domain_s, cmd2.domain_t + cmd1.domain)
                else:
                    raise Error.ArgumentError(f"Invalid measurement plane: ({measurement_plane})! Only 'XY' and 'YZ' are supported as the measurement plane in UBQC in this version.", ModuleErrorCode, FileErrorCode, 11)

                return [M_new]  # create a new command

        # [Z, X] --> [X, Z]
        elif name1 == 'Z' and name2 == 'X':
            return [cmd2, cmd1]  # swap the independent commands

        # [X, X] --> [X]
        elif name1 == 'X' and name2 == 'X':
            X1_qubit = cmd1.which_qubit
            X2_qubit = cmd2.which_qubit
            if X1_qubit == X2_qubit:
                return [Pattern.CommandX(X1_qubit, cmd1.domain + cmd2.domain)]  # create a new command
            else:
                return two_cmds  # keep both commands

        # [Z, Z] --> [Z]
        elif name1 == 'Z' and name2 == 'Z':
            Z1_qubit = cmd1.which_qubit
            Z2_qubit = cmd2.which_qubit
            if Z1_qubit == Z2_qubit:
                return [Pattern.CommandZ(Z1_qubit, cmd1.domain + cmd2.domain)]  # create a new command
            else:
                return two_cmds  # keep both commands

        # [S, M / X / Z / S] --> [M / X / Z / S, S]
        elif name1 == 'S':
            # Swap command S with command M
            # Measurements in 'XY' plane and in 'YZ' plane have the same rule
            # [S, M] --> [M, S]
            if name2 == 'M':
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
            elif name2 == 'X' or name2 == 'Z' or name2 == 'S':
                S_qubit = cmd1.which_qubit
                S_domains = cmd1.domain
                if S_qubit in cmd2.domain:
                    cmd2.domain += S_domains
                return [cmd2, cmd1]  # swap the commands with domain modification
            else:
                return two_cmds  # keep both commands

        else:
            return two_cmds  # keep both commands

    def __propagate(self, cmd_type: str, cmds):
        r"""Propagate a certain type of commands in the list to the front.

        Note:
            This is an intrinsic method. No need to call it externally.

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
            raise Error.ArgumentError(f'Invalid commands list ({cmds}) with the type: ({type(cmds)})!\nOnly `List` is supported as the type of commands list.', ModuleErrorCode, FileErrorCode, 12)

        # Back to front, propagate command E, command M, command X and command Z sequentially
        if {cmd_type}.issubset(['E', 'M', 'X', 'Z']):
            for i in range(len(cmds) - 1, 0, -1):
                if cmds[i].name == cmd_type:
                    cmds = cmds[:i - 1] + self.__swap([cmds[i - 1], cmds[i]]) + cmds[i + 1:]

        # Front to back, propagate command S
        else:
            for i in range(0, len(cmds) - 1):
                if cmds[i].name == cmd_type:
                    cmds = cmds[:i] + self.__swap([cmds[i], cmds[i + 1]]) + cmds[i + 2:]

        return cmds

    @staticmethod
    def __cmds_to_nums(cmds):
        r"""Map the commands to the numbers.

        The mapping rules are:
            CommandE --> 1
            CommandM --> 2
            CommandX --> 3
            CommandZ --> 4
            CommandS --> 5

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            cmds (list): a list of commands

        Returns:
            list: a list of counts on each type of the commands
            list: a list of numbers
            list: a list of numbers arranged from small to large
        """
        cmd_map = {'E': 1,  # order of commands E
                   'M': 2,  # order of commands M
                   'X': 3,  # order of commands X
                   'Z': 4,  # order of commands Z
                   'S': 5}  # order of commands S
        cmd_num_wild = [cmd_map[cmd.name] for cmd in cmds]
        cmd_num_std = cmd_num_wild[:]
        cmd_num_std.sort(reverse=False)
        cmds_count = [cmd_num_std.count(i) for i in [1, 2, 3, 4, 5]]  # count each type of commands

        return cmds_count, cmd_num_wild, cmd_num_std

    def __is_standard(self, cmd_type: str, cmds):
        r"""Check if a certain type of commands is in a standard order from small to large.

        Note:
            This is an intrinsic method. No need to call it externally.

        Args:
            cmd_type (str): a command type in ["E", "M", "X", "Z", "S"]
            cmds (list): a list of commands

        Returns:
            bool: True if the type of commands is in a standard order;
                  False if the type of commands is not in a standard order
        """
        if not {cmd_type}.issubset(['E', 'M', 'X', 'Z', 'S']):
            raise Error.ArgumentError(f"Invalid command name: ({cmd_type}! Only 'E', 'M', 'X', 'Z' and 'S' are supported as the command names in UBQC in this version, where 'E' is the entanglement command, 'M' is the measurement command, 'X' is the X byproduct correction command, 'Z' is the Z byproduct correction command and 'S' is the signal shifting command.", ModuleErrorCode, FileErrorCode, 13)

        # Map the commands to numbers
        cmds_count, cmd_num_wild, cmd_num_std = self.__cmds_to_nums(cmds)
        pointer_map = {'E': sum(cmds_count[:1]),  # number of commands E
                       'M': sum(cmds_count[:2]),  # number of commands E + M
                       'X': sum(cmds_count[:3]),  # number of commands E + M + X
                       'Z': sum(cmds_count[:4]),  # number of commands E + M + X + Z
                       'S': sum(cmds_count[:5])}  # number of commands E + M + X + Z + S

        return cmd_num_wild[:pointer_map[cmd_type]] == cmd_num_std[:pointer_map[cmd_type]]

    def __simplify_pauli_measurements(self):
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

        Note:
            This is an intrinsic method. No need to call it externally.
        """
        for cmd in self.__pattern.commands:
            if cmd.name == 'M':
                remainder = cmd.angle % (2 * pi)
                if remainder in [0, pi]:
                    cmd.domain_s = []
                elif remainder in [pi / 2, (3 * pi) / 2]:
                    cmd.domain_t += cmd.domain_s[:]
                    cmd.domain_s = []

    def standardize(self):
        r"""Standardize the measurement pattern.

        This method swaps the commands in the wild pattern to form a standard 'EMC' pattern.
        Entanglement commands are propagated to the leftmost of the command list.
        Then come to the measurement commands and last come to the byproduct correction commands.
        To simplify the pattern, Pauli simplification method is automatically implemented to
        exclude some of the dependencies.
        """
        cmds = self.__pattern.commands

        for cmd_type in ['E', 'M', 'X', 'Z']:
            while not self.__is_standard(cmd_type, cmds):
                cmds = self.__propagate(cmd_type, cmds)

        self.__pattern.commands = cmds
        self.__simplify_pauli_measurements()

    def get_pattern(self):
        r"""Return the standard measurement pattern.

        Warning:
            This method must be called after ``to_brickwork_pattern``.

        Returns:
            Pattern: a standard pattern
        """
        return self.__pattern