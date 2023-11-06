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
Module for brickwork pattern in blind quantum computation.
"""

from argparse import ArgumentTypeError
from math import pi
from typing import List, Tuple

from Extensions.QuantumNetwork.qcompute_qnet.quantum.circuit import Circuit
from Extensions.QuantumNetwork.qcompute_qnet.quantum.mcalculus import MCalculus
from Extensions.QuantumNetwork.qcompute_qnet.quantum.pattern import Pattern

__all__ = ["BrickworkCircuit", "BrickworkCalculus"]


class BrickworkCircuit(Circuit):
    r"""Define the ``BrickworkCircuit`` class.

    This class is responsible for the manipulation of brickwork circuit.
    """

    def __init__(self, circuit: "Circuit"):
        r"""Initialize a brickwork circuit from a static quantum circuit.

        Args:
            circuit (Circuit): static quantum circuit to transpile
        """
        super().__init__(circuit.name)
        self._history = circuit.gate_history
        # Here are all attributes used for brickwork mould generation
        self._bw_depth = None  # brickwork mould depth
        self.__bw_history = None  # a list to record the brickwork circuit information
        self.__bw_mould = None  # brickwork mould
        # self.__to_xy_measurement = None  # whether to transform the measurements to XY-plane

        # Record valid columns
        self.__sgl_col = None  # record the valid columns to map a single qubit gate
        self.__dbl_col = None  # record the valid columns to map a double qubit gate

    def __update_single_column(self, idx: int, col: int) -> None:
        r"""Update the list of the valid columns to fill in a single qubit gate.

        Args:
            idx (int): the index of a single qubit gate
            col (int): the column to fill in a single qubit gate
        """
        self.__sgl_col[idx] = col + 4
        self.__sgl_col[idx] += 4 if idx == 0 or idx == self.width - 1 else 0

    def __update_double_column(self, row_1: int, row_2: int, idx: int) -> None:
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

        self.__dbl_col[idx] += 8 if sgl_col > dbl_col else 0

    def __fill_a_single_qubit_gate(self, idx: int) -> None:
        r"""Fill in a single qubit gate to the brickwork mould and update the valid columns.

        Args:
            idx (int): the index of a single qubit gate
        """
        col = self.__sgl_col[idx]

        self.__update_single_column(idx, col)

        # If the index of a single qubit gate is also the index of a double qubit gate
        if (col % 8) / 4 == idx % 2:
            if idx != self.width - 1:
                cor_row = idx + 1
                self.__update_double_column(idx, cor_row, idx)
            else:
                raise ArgumentTypeError("Invalid brickwork mould!\n" "This brickwork mould is not supported in UBQC.")

        # If the index of a single qubit gate is not the index of a double qubit gate
        else:
            if idx != 0:
                cor_row = idx - 1
                self.__update_double_column(idx, cor_row, cor_row)
            else:
                raise ArgumentTypeError("Invalid brickwork mould!\n" "This brickwork mould is not supported in UBQC.")

    def __fill_a_double_qubit_gate(self, idx: int) -> None:
        r"""Fill in a double qubit gate to the brickwork mould and update the valid columns.

        Args:
            idx (int): the index of a double qubit gate
        """
        col = self.__dbl_col[idx]

        self.__dbl_col[idx] += 8

        self.__update_single_column(idx, col)
        self.__update_single_column(idx + 1, col)

        if idx + 1 != self.width - 1:
            if self.__dbl_col[idx + 1] < col:
                self.__dbl_col[idx + 1] = col + 4

        if idx - 1 != -1:
            if self.__dbl_col[idx - 1] < col:
                self.__dbl_col[idx - 1] = col + 4

    def __gain_position(self, row_list: List[int], gate_type: str) -> list:
        r"""Fill in a quantum gate to the brickwork mould to obtain its position.

        Args:
            row_list (list): the rows operated by quantum gates
            gate_type (str): 'single' represents the single qubit gate; 'double' represents the double qubit gate

        Returns:
            list: a list of positions
        """
        if gate_type == "single":
            row = row_list[0]
            col = self.__sgl_col[row]
            self.__fill_a_single_qubit_gate(row)
            pos = [(row, col)]

        elif gate_type == "double":
            row = min(row_list)
            col = self.__dbl_col[row]
            self.__fill_a_double_qubit_gate(row)
            pos = [(row_list[0], col), (row_list[1], col)]

        else:
            raise ArgumentTypeError(
                f"Invalid gate type: ({gate_type})!\n"
                "Only single qubit gates and double qubit gates are supported "
                "in UBQC in this version."
            )

        return pos

    def __fill_gates(self) -> None:
        r"""Fill in all quantum gates to the brickwork mould one after another.

        After each filling, the lists of valid columns and the depth of brickwork mould are updated.
        And the gate positions are obtained.
        """
        # Initialize parameters
        self._bw_depth = 0
        self.__bw_history = []
        self.__sgl_col = [0 for _ in range(self.width - 1)] + [4] if self.width % 2 else [0 for _ in range(self.width)]
        self.__dbl_col = [0 if i % 2 == 0 else 4 for i in range(self.width - 1)]
        meas_gates = [gate for gate in self._history if gate["name"] == "m"]
        qu_gates = [gate for gate in self._history if gate["name"] != "m"]

        # Fill in the gates in 'history' to the brickwork mould
        for gate in qu_gates:
            which_qubit = gate["which_qubit"]

            # Single qubit gates
            if len(which_qubit) == 1:
                position = self.__gain_position(which_qubit, "single")
            # Double qubit gates
            elif len(which_qubit) == 2:
                position = self.__gain_position(which_qubit, "double")
            else:
                raise ArgumentTypeError("The gate is not supported in this version!")

            # Update brickwork mould depth
            self._bw_depth = max(self._bw_depth, int(position[0][1] / 4))

            input_ = position
            output_ = [(pos[0], pos[1] + 4) for pos in input_]
            gate["input_"] = input_
            gate["output_"] = output_
            # Record gates information in a list 'bw_history'
            # Note: the gates in 'bw_history' are not exactly the same gates in 'history'
            # Because these gates also have position parameters
            self.__bw_history.append(gate)

        # Add one for counting the circuit depth
        self._bw_depth += 1

        # Add measurements
        for gate in meas_gates:
            which_qubit = gate["which_qubit"][0]

            input_ = [(which_qubit, 4 * self._bw_depth)]
            output_ = None
            gate["input_"] = input_
            gate["output_"] = output_
            self.__bw_history.append(gate)

    def __fill_identity(self) -> None:
        r"""Fill the blanks of the brickwork mould with identity gates.

        Note:
            To simplify the codes, a trick is implemented here in this method.
            The trick is that we initialize a brickwork mould and fill all blanks with identity gates at the beginning.
            In this way, instead of spending much effort to seek for all blanks of the brickwork mould,
            we just need to replace certain identity gates with those gates to be filled in the brickwork mould.
            In this way, a complete brickwork circuit is obtained automatically.
        """
        zero_params = [0, 0, 0]

        # Initialize a brickwork mould and fill all blanks with identity gates
        self.__bw_mould = {}
        for row in range(self.width):
            if row in self.measured_qubits:
                for col in range(self._bw_depth + 1):
                    gate = {
                        "name": "u",
                        "input_": [(row, 4 * col)],
                        "output_": [(row, 4 * col + 4)],
                        "angles": zero_params,
                    }
                    self.__bw_mould[(row, 4 * col)] = gate
            else:
                for col in range(self._bw_depth):
                    gate = {
                        "name": "u",
                        "input_": [(row, 4 * col)],
                        "output_": [(row, 4 * col + 4)],
                        "angles": zero_params,
                    }
                    self.__bw_mould[(row, 4 * col)] = gate

        # Replace identity gates with those in 'bw_history'
        for gate in self.__bw_history:
            pos = gate["input_"]
            for v in pos:
                self.__bw_mould[v] = gate

    def to_brickwork_circuit(self) -> None:
        r"""Transpile a static quantum circuit into a brickwork circuit."""
        if not self.is_static():
            print("\nIn 'to_brickwork_circuit': The transpiled circuit should be a static quantum circuit.")
            return

        # Replace CZ gate by one CNOT gate and two H gates
        new_history = []
        for i, gate in enumerate(self._history):
            if gate["name"] == "cz":
                new_history.append(
                    {"name": "h", "which_qubit": [gate["which_qubit"][1]], "signature": gate["signature"]}
                )
                new_history.append({"name": "cx", "which_qubit": gate["which_qubit"], "signature": gate["signature"]})
                new_history.append(
                    {"name": "h", "which_qubit": [gate["which_qubit"][1]], "signature": gate["signature"]}
                )
            else:
                new_history.append(gate)
        self._history = new_history
        self.simplify()  # convert the circuit to u gates and cnot gates
        self.__fill_gates()
        self.__fill_identity()

    def get_brickwork_circuit(self) -> dict:
        r"""Return a dictionary of the brickwork circuit.

        Warning:
            This method must be called after ``to_brickwork``.

        Returns:
           dict: a dictionary of the brickwork circuit
        """
        return self.__bw_mould

    def get_brickwork_depth(self) -> int:
        r"""Return the depth of brickwork mould.

        Warning:
            This method must be called after ``to_brickwork``.

        Returns:
           int: the depth of brickwork mould
        """
        return self._bw_depth


class BrickworkCalculus(MCalculus):
    r"""Define the ``BrickworkCalculus`` class.

    This class provides various basic operations for
    transpiling a static quantum circuit into its equivalent brickwork pattern.

    Warning:
        A brickwork pattern can only be transpiled from a static quantum circuit.
    """

    def __init__(self):
        r"""``BrickworkCalculus`` constructor, used to instantiate a ``BrickworkCalculus`` object.

        This class provides various basic operations
        for converting a static quantum circuit to its equivalent Brickwork pattern.
        """
        super().__init__()
        self._circuit_width = None  # type: int  # circuit width
        self._bw_depth = None  # type: int # circuit depth

    def __build_odd_width_bw(self, bw_cir: dict, pos: Tuple[int, int]) -> list:
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
            if row == self._circuit_width - 1:
                gate_list = [bw_cir[pos]]
            else:
                gate_list = [bw_cir[pos], bw_cir[(row + 1, col)]] if row % 2 == 0 else []
        else:
            if row == 0:
                gate_list = [bw_cir[pos]]
            else:
                gate_list = [bw_cir[pos], bw_cir[(row + 1, col)]] if row % 2 == 1 else []

        return gate_list

    def __build_even_width_bw(self, bw_cir: dict, pos: Tuple[int, int]) -> list:
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
            gate_list = [bw_cir[pos], bw_cir[(row + 1, col)]] if row % 2 == 0 else []
        else:
            if row == 0 or row == self._circuit_width - 1:
                gate_list = [bw_cir[pos]]
            else:
                gate_list = [bw_cir[pos], bw_cir[(row + 1, col)]] if row % 2 == 1 else []

        return gate_list

    def __build_brickwork(self, bw_cir: dict) -> None:
        r"""Build a brickwork mould.

        Note:
            This is an intrinsic method. No need to call it externally.
            According to the reference [arXiv:0807.4154], there are two types of brickwork moulds.
            These two moulds have different width.
            One brickwork mould has an odd width. The other one has an even width.

        Args:
            bw_cir (dict): a brickwork circuit
        """
        for i in range(self._bw_depth + 1):
            col = i * 4
            for row in range(self._circuit_width):
                pos = (row, col)

                if bw_cir.get(pos) is None:
                    continue

                gate = bw_cir[pos]
                name = gate["name"]

                # Build logic gate pattern
                if name != "m":
                    # Brickwork type 1, with even width
                    if self._circuit_width % 2 == 0:
                        gate_list = self.__build_even_width_bw(bw_cir, pos)
                    # Brickwork type 2, with odd width
                    else:
                        gate_list = self.__build_odd_width_bw(bw_cir, pos)

                # Build measurement pattern
                else:
                    gate_list = [gate]

                # As this is the traversal algorithm, there exists empty gate list
                if not gate_list:
                    continue

                pattern = self.__to_bw_pat(gate_list)
                self._wild_pattern.append(pattern)

    @staticmethod
    def __set_cmds(name: str, space: list, params: list, domains: list) -> list:
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
        if not {name}.issubset(["u", "m", "d"]):
            raise ArgumentTypeError(
                f"Invalid gate name ({name})!\n"
                "Only 'u', 'm' and 'd' are supported as the gate names "
                "in UBQC in this version, where "
                "'u' is the 'unitary gate', "
                "'m' is the 'measurement' and "
                "'d' is the 'double qubit gate'."
            )

        # Single qubit unitary gate commands
        if name == "u":
            cmdE = [Pattern.CommandE([space[i], space[i + 1]]) for i in range(4)]

            cmdM = []
            for i in range(4):
                pos = space[i]
                param = -params[i]
                cmd = Pattern.CommandM(pos, param, "XY", domains[i][0], domains[i][1])
                cmdM.append(cmd)

            cmdX = [Pattern.CommandX(space[4], [space[3]])]
            cmdZ = [Pattern.CommandZ(space[4], [space[2]])]

        # Single qubit measurement commands
        elif name == "m":
            cmdE = []
            cmdM = [
                Pattern.CommandM(space[0], params["angle"], params["plane"], params["domain_s"], params["domain_t"])
            ]
            cmdX = []
            cmdZ = []

        # Double qubit gate commands
        else:
            cmdE = [Pattern.CommandE([space[5 * j + i], space[5 * j + i + 1]]) for i in range(4) for j in range(2)] + [
                Pattern.CommandE([space[2], space[7]]),
                Pattern.CommandE([space[4], space[9]]),
            ]

            cmdM = []
            for i in range(4):
                for j in range(2):
                    pos = space[5 * j + i]
                    param = -params[4 * j + i]
                    cmd = Pattern.CommandM(pos, param, "XY", domains[4 * j + i][0], domains[4 * j + i][1])
                    cmdM.append(cmd)

            cmdX = [Pattern.CommandX(space[4], [space[3]]), Pattern.CommandX(space[9], [space[8]])]
            cmdZ = [
                Pattern.CommandZ(space[4], [space[2]]),
                Pattern.CommandZ(space[9], [space[3]]),
                Pattern.CommandZ(space[9], [space[7]]),
                Pattern.CommandZ(space[4], [space[8]]),
            ]

        cmds = cmdE + cmdM + cmdX + cmdZ
        return cmds

    def __to_bw_pat(self, gate_list: list) -> Pattern:
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

        Returns:
            Pattern: equivalent brickwork pattern in MBQC
        """
        # Single qubit gate
        if len(gate_list) == 1:
            # Initialize parameters
            gate = gate_list[0]
            name = gate["name"]
            input_ = gate["input_"][0]
            params = gate["angles"] if gate.get("angles") else gate["basis"]
            row = input_[0]
            col_in = input_[1]

            if not {name}.issubset(["u", "m"]):
                raise ArgumentTypeError(
                    f"Invalid gate list: ({gate_list})!\n"
                    "Only 'u' and 'm' are supported as the single qubit gate names "
                    "in UBQC in this version, where "
                    "'u' is the 'unitary gate' and "
                    "'m' is the 'measurement'."
                )

            # Unitary gate
            if name == "u":
                output_ = gate["output_"][0]
                col_out = output_[1]
                # Set parameters
                space = [(row, col_in + i) for i in range(col_out - col_in + 1)]
                # Add a zero to the end of the list 'params'
                theta, phi, gamma = params
                params = [gamma, theta, phi, 0]
                domains = [[[], []], [[space[0]], []], [[space[1]], [space[0]]], [[space[2]], [space[1]]]]
                input_ = [(row, col_in)]
                output_ = [(row, col_out)]

            # Measurement
            else:
                self._measured_qubits.append(row)
                # Set parameters
                space = [(row, col_in)]
                domains = None
                input_ = space
                output_ = []

        # Double qubit gate
        else:
            # Initialize parameters
            name = "d"
            gate1 = gate_list[0]
            name1 = gate1["name"]
            gate2 = gate_list[1]
            name2 = gate2["name"]

            if not {name1, name2}.issubset(["u", "cx"]):
                raise ArgumentTypeError(f"Invalid gate list!")

            # Two unitary gates
            if name1 == name2 == "u":
                input1 = gate1["input_"][0]
                input2 = gate2["input_"][0]
            # CNOT gate
            else:
                input1 = gate1["input_"][0]
                input2 = gate1["input_"][1]

            # Set input and output qubits
            output1 = gate1["output_"][0]
            row1 = input1[0]
            row2 = input2[0]
            if not abs(row1 - row2) == 1:
                raise ArgumentTypeError(
                    f"Invalid row indexes: {row1} and {row2}!\n"
                    "Only the adjacent qubits are supported as indexes of the quantum gates "
                    "in UBQC in this version."
                )

            col_in = input1[1]
            col_out = output1[1]
            # Set parameters
            space = [(row1, col_in + i) for i in range(col_out - col_in + 1)] + [
                (row2, col_in + i) for i in range(col_out - col_in + 1)
            ]

            # Two unitary gates
            if name1 == name2 == "u":
                params = []
                for i in range(2):
                    param = gate_list[i]["angles"]
                    theta, phi, gamma = param
                    params += [gamma, theta, phi, 0]  # add a zero to the end of the list 'params'
            # CNOT gate
            else:
                params = [0, 0, pi / 2, 0, 0, pi / 2, 0, -pi / 2]

            domains = [
                [[], []],
                [[space[0]], []],
                [[space[1]], [space[0], space[6]]],
                [[space[2]], [space[1]]],
                [[], []],
                [[space[5]], []],
                [[space[6]], [space[5], space[1]]],
                [[space[7]], [space[6]]],
            ]
            input_ = [(row1, col_in), (row2, col_in)]
            output_ = [(row1, col_out), (row2, col_out)]

        cmds = self.__set_cmds(name, space, params, domains)
        return Pattern(name, space, input_, output_, cmds)

    def to_brickwork_pattern(self) -> None:
        r"""Transpile a static quantum circuit into its equivalent brickwork wild pattern in MBQC."""
        if not self._circuit.is_static():
            print("\nIn 'to_brickwork_pattern': The transpiled circuit should be a static quantum circuit.")
            return

        bw_cir = self._circuit.get_brickwork_circuit()
        self._bw_depth = self._circuit.get_brickwork_depth()

        self.__build_brickwork(bw_cir)
        if self._measured_qubits != list(range(self._circuit_width)):
            raise ArgumentTypeError(
                "Invalid brickwork pattern!"
                "Brickwork pattern does not support quantum output in UBQC in this version."
            )

        input_ = self._get_input()
        output_ = self._get_output()
        self._splice_patterns(input_, output_)
