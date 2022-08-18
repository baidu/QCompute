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

"""
The backend for measurement-based quantum computation.
"""

from argparse import ArgumentTypeError
from typing import List, Dict, Tuple, Any, Union
from matplotlib import pyplot as plt
import networkx
from networkx import Graph, spring_layout, draw_networkx
import numpy
from numpy import reshape, pi, conj, real, random, sqrt
from qcompute_qnet import EPSILON
from qcompute_qnet.quantum.basis import Basis
from qcompute_qnet.quantum.gate import Gate
from qcompute_qnet.quantum.mcalculus import transpile_to_pattern
from qcompute_qnet.quantum.pattern import Pattern
from qcompute_qnet.quantum.state import PureState, Plus
from qcompute_qnet.quantum.utils import kron, print_progress

__all__ = [
    "MBQC",
    "run_circuit"
]


class MBQC:
    r"""Class for the measurement-based quantum computation.

    Attributes:
        vertex (Vertex): vertices in MBQC algorithm
        max_active (int): maximum number of active vertices
    """

    def __init__(self):
        r"""Constructor for MBQC class.
        """
        self.__graph = None
        self.__pattern = None
        self.vertex = None
        self.max_active = 0  # maximum number of active vertices

        self.__vertex_to_state = {}  # mapping of quantum states and the vertices on the graph
        self.__bg_state = PureState()  # initialize a trivial state
        self.__outcome = {}  # store the measurement outcomes

        self.__history = [self.__bg_state]  # background state history
        self.__status = self.__history[-1] if self.__history != [] else None

        self.__draw = False  # drawing the change of vertex classification
        self.__pause_time = None  # drawing pause time
        self.__track = False  # tracking the computational progress
        self.__pos = None  # vertex positions

    class Vertex:
        r"""Class for vertices in MBQC.

        Each vertex corresponds to a qubit.
        We classify vertices in MBQC to three categories and manipulate them dynamically.
        This helps to run MBQC algorithms in a large scale.

        Attributes:
            total (list): all vertices in the MBQC algorithm
            pending (list): pending vertices to activate
            active (list): active vertices in the current measurement step
            measured (list): measured vertices
        """

        def __init__(self, total=None, pending=None, active=None, measured=None):
            r"""Constructor for Vertex class.

            Args:
                total (list): all vertices in the MBQC algorithm
                pending (list): pending vertices to activate
                active (list): active vertices in the current measurement step
                measured (list): measured vertices
            """
            self.total = [] if total is None else total
            self.pending = [] if pending is None else pending
            self.active = [] if active is None else active
            self.measured = [] if measured is None else measured

    def set_graph(self, graph: List[List]) -> None:
        r"""Set the underlying graph of MBQC algorithm.

        Args:
            graph (List[List]): the underlying graph of MBQC algorithm

        Examples:
            The graph is given by a list as follows.

            >>> mbqc = MBQC()
            >>> V = ['1', '2', '3', '4', '5']  # a list of vertices
            >>> E = [('1', '2'), ('2', '3'), ('3', '4'), ('4', '5')]  # a list of edges
            >>> G = [V, E]
            >>> mbqc.set_graph(G)
        """
        if not isinstance(graph, List):
            raise ArgumentTypeError(f"Input {graph} should be a list.")

        vertices, edges = graph
        if not isinstance(vertices, List):
            raise ArgumentTypeError(f"The first element of {graph} should be a list of vertices.")
        if not isinstance(edges, List):
            raise ArgumentTypeError(f"The second element of {graph} should be a list of edges.")

        vertices_of_edges = set([vertex for edge in edges for vertex in list(edge)])
        if not vertices_of_edges.issubset(vertices):
            raise ArgumentTypeError(f"Invalid graph: ({graph})! The edges must link two vertices on the graph.")

        self.__graph = Graph()
        self.__graph.add_nodes_from(vertices)
        self.__graph.add_edges_from(edges)

        self.vertex = self.Vertex(total=vertices, pending=vertices)
        # Initialize each vertex on the graph with a plus state by default
        self.__vertex_to_state = {vertex: PureState(Plus.SV, [vertex]) for vertex in vertices}

    def get_graph(self) -> networkx.Graph:
        r"""Get the underlying graph of MBQC.

        Returns:
            networkx.Graph: the underlying graph of MBQC
        """
        return self.__graph

    def set_pattern(self, pattern: "Pattern") -> None:
        r"""Set a given measurement pattern.

        Args:
            pattern (Pattern): measurement pattern
        """
        if not isinstance(pattern, Pattern):
            raise ArgumentTypeError(f"Input {pattern} should be a 'Pattern'.")

        self.__pattern = pattern
        cmds = self.__pattern.commands[:]

        # Check if the pattern is of a standard EMC form
        cmd_map = {'E': 1, 'M': 2, 'X': 3, 'Z': 4, 'S': 5}
        cmd_num_wild = [cmd_map[cmd.name] for cmd in cmds]
        cmd_num_standard = cmd_num_wild[:]
        cmd_num_standard.sort(reverse=False)

        if not cmd_num_wild == cmd_num_standard:
            raise ArgumentTypeError(f"Input {pattern} is not a standard EMC pattern.")

        # Set graph according to entanglement commands
        edges = [tuple(cmd.which_qubits) for cmd in cmds if cmd.name == 'E']
        vertices = list(set([vertex for edge in edges for vertex in list(edge)]))
        graph = [vertices, edges]
        self.set_graph(graph)

    def get_pattern(self) -> "Pattern":
        r"""Get the measurement pattern.

        Returns:
            Pattern: a measurement pattern
        """
        return self.__pattern

    def __update_history(self) -> None:
        r"""Update the history and the status of the computation.
        """
        self.__history.append(self.__bg_state)
        self.__status = self.__history[-1]

    def get_history(self) -> list:
        r"""Get the computational history of MBQC.

        Returns:
            list: a list of computational history
        """
        return self.__history

    def set_input_state(self, state=None) -> None:
        r"""Set a given input quantum state.

        Warning:
            Different from the circuit model, MBQC uses plus states by default.

        Args:
            state (PureState): input state to set
        """
        if self.__graph is None:
            raise ArgumentTypeError(f"Please set a 'graph' or 'pattern' before calling 'set_input_state'.")

        if state is not None and not isinstance(state, PureState):
            raise ArgumentTypeError(f"Input {state} should be a 'PureState'.")

        vertices = list(self.__graph.nodes)

        if state is None:
            state_vector = numpy.array([[1]])
            systems = []
        else:
            state_vector = state.state
            # If a pattern is set, map the input state systems to the pattern's input
            if self.__pattern is not None:
                for label in state.systems:
                    if not isinstance(label, int):
                        raise ArgumentTypeError(f"System label {label} should be an int value.")
                    if label < 0:
                        raise ArgumentTypeError(f"System label {label} should be an non-negative integer.")

                systems = [label for label in self.__pattern.input_ if int(label[0]) in state.systems]
            else:
                systems = state.systems

        if not set(systems).issubset(vertices):
            raise ArgumentTypeError(f"Input systems {systems} must be a subset of all vertices on the graph.")

        self.__bg_state = PureState(state_vector, systems)
        self.__update_history()
        self.vertex = self.Vertex(total=vertices,
                                  pending=list(set(vertices).difference(systems)),
                                  active=systems,
                                  measured=[])
        self.max_active = len(self.vertex.active)

    def replace_state_on_vertex(self, vertex: Any, state: "PureState") -> None:
        r"""Replace the state on a given vertex.

        Warning:
            This method should be called after ``set_graph`` or ``set_pattern``.

        Args:
            vertex (Any): vertex to replace
            state (PureState): state to replace
        """
        if not isinstance(state, PureState):
            raise ArgumentTypeError(f"Input {state} should be a 'PureState'.")
        if vertex not in self.vertex.total:
            raise ArgumentTypeError(f"Invalid vertex: ({vertex})! This vertex is not on the graph.")

        self.__vertex_to_state[vertex] = state

    def __apply_cz(self, which_qubits_list: List[Tuple[int, int]]) -> None:
        r"""Apply Controlled-Z gate.

        Warning:
            The qubits to manipulate must be active vertices.

        Args:
            which_qubits_list (list): qubits to manipulate
        """
        for which_qubits in which_qubits_list:
            if not set(which_qubits).issubset(self.vertex.active):
                raise ArgumentTypeError(f"Invalid qubits: ({which_qubits})!\n"
                                        f"The qubits in 'which_qubits_list' must be activated first.")
            qubit1 = which_qubits[0]
            qubit2 = which_qubits[1]
            if qubit1 == qubit2:
                raise ArgumentTypeError(f"Invalid qubits: ({which_qubits})!\n"
                                        f"Control qubit must not be the same as target qubit.")

            # Find the control and target qubits and permute them to the front
            self.__bg_state.permute_to_front(qubit1)
            self.__bg_state.permute_to_front(qubit2)

            new_state = self.__bg_state
            new_state_len = new_state.length
            qua_length = int(new_state_len / 4)
            # Reshape the state, apply CZ and reshape it back
            new_state.state = reshape(Gate.CZ() @ reshape(new_state.state, [4, qua_length]), [new_state_len, 1])

            # Update the order of active vertices and the background state
            self.vertex.active = new_state.systems
            self.__bg_state = PureState(new_state.state, new_state.systems)

    def __apply_pauli_gate(self, gate: str, which_qubit: Any) -> None:
        r"""Apply Pauli gate.

        Args:
            gate (str): name of the Pauli gate
            which_qubit (Any): qubit to manipulate
        """
        self.__bg_state.permute_to_front(which_qubit)
        new_state = self.__bg_state
        new_state_len = new_state.length
        half_length = int(new_state_len / 2)

        if gate == 'X':
            gate_mat = Gate.X()
        elif gate == 'Z':
            gate_mat = Gate.Z()
        else:
            raise ArgumentTypeError(f"Input {gate} should be string 'X' or 'Z'.")

        # Reshape the state, apply X and reshape it back
        new_state.state = reshape(gate_mat @ reshape(new_state.state, [2, half_length]), [new_state_len, 1])
        # Update the order of active vertices and the background state
        self.vertex.active = new_state.systems
        self.__bg_state = PureState(new_state.state, new_state.systems)

    def __create_graph_state(self, which_qubit: Any) -> None:
        r"""Create a graph state based on the current qubit to measure.

        Args:
            which_qubit (any): qubit to measure
        """
        which_qubit_neighbors = set(self.__graph.neighbors(which_qubit))
        neighbors_not_measured = which_qubit_neighbors.difference(set(self.vertex.measured))
        # Create a list of systems and apply cz gates
        cz_list = [(which_qubit, qubit) for qubit in neighbors_not_measured]

        # Obtain the qubits to be activated
        append_qubits = {which_qubit}.union(neighbors_not_measured).difference(set(self.vertex.active))
        # Update active and pending lists
        self.vertex.active += list(append_qubits)
        self.vertex.pending = list(set(self.vertex.pending).difference(self.vertex.active))

        # Compute the new background state vector
        new_bg_state_vector = kron([self.__bg_state.state] +
                                   [self.__vertex_to_state[vertex].state for vertex in append_qubits])

        # Update the background state and apply cz
        self.__bg_state = PureState(new_bg_state_vector, self.vertex.active)
        self.__apply_cz(cz_list)
        self.__draw_process('active', which_qubit)

    def measure(self, which_qubit: Any, basis: numpy.ndarray) -> None:
        r"""Measure a given qubit.

        Args:
            which_qubit (Any): qubit to measure
            basis (numpy.ndarray): measurement basis
        """
        self.__draw_process('measuring', which_qubit)
        self.__create_graph_state(which_qubit)

        if which_qubit not in self.vertex.active:
            raise ArgumentTypeError(f"Invalid qubit: ({which_qubit})! The qubit must be activated before measurement.")

        self.__bg_state.permute_to_front(which_qubit)
        new_bg_state = self.__bg_state
        self.vertex.active = new_bg_state.systems
        half_length = int(new_bg_state.length / 2)

        prob = [0, 0]
        state_unnorm = [0, 0]

        # Calculate the probability and post-measurement states
        for result in [0, 1]:
            basis_dag = conj(basis[result]).T
            # Reshape the state, multiply the basis and reshape it back
            state_unnorm[result] = reshape(basis_dag @ reshape(new_bg_state.state, [2, half_length]), [half_length, 1])
            probability = conj(state_unnorm[result]).T @ state_unnorm[result]
            prob[result] = real(probability) if probability.dtype.name == 'COMPLEX128' else probability

        # Randomly choose a result and its corresponding post-measurement state
        prob_zero = real(prob[0].item())
        prob_one = real(prob[1].item())
        if prob_zero < EPSILON:
            result = 1
            post_state_vector = state_unnorm[1]
        elif prob_one < EPSILON:
            result = 0
            post_state_vector = state_unnorm[0]
        else:
            result = random.choice(2, 1, p=[prob_zero, prob_one]).item()
            # Normalize the state after measurement
            post_state_vector = state_unnorm[result] / sqrt(prob[result])

        # Write the measurement result into the dictionary
        self.__outcome.update({which_qubit: int(result)})
        # Update measured list and active list
        self.vertex.measured.append(which_qubit)
        self.max_active = max(len(self.vertex.active), self.max_active)
        self.vertex.active.remove(which_qubit)

        # Update the background state and the history list
        self.__bg_state = PureState(post_state_vector, self.vertex.active)
        self.__update_history()

        self.__draw_process('measured', which_qubit)

    def sum_outcomes(self, which_qubits: list, add_number=None) -> int:
        r"""Sum the measurement outcome of given qubits.

        Args:
            which_qubits (list): qubits to manipulate
            add_number (int): extra number to add to the summation

        Returns:
            int: summation result
        """
        if add_number is None:
            add_number = 0
        else:
            if not isinstance(add_number, int):
                raise ArgumentTypeError(f"Input {add_number} should be an int value.")

        return sum([self.__outcome[label] for label in which_qubits], add_number)

    def correct_byproduct(self, gate: str, which_qubit: Any, power: int) -> None:
        r"""Correct the byproduct operators.

        Args:
            gate (str): correction type to make
            which_qubit (Any): qubit to correct
            power (int): power of the correction operator
        """
        if gate not in ['X', 'Z']:
            raise ArgumentTypeError(f"Input {gate} should be a string 'X' or 'Z'.")
        if not isinstance(power, int):
            raise ArgumentTypeError(f"Input {power} should be an int value.")

        if power % 2 == 1:
            self.__apply_pauli_gate(gate, which_qubit)

        self.__update_history()

    def __run_cmd(self, cmd: Union[Pattern.CommandM, Pattern.CommandX, Pattern.CommandZ]) -> None:
        r"""Run the given command.

        Args:
            cmd (Union[Pattern.CommandM, Pattern.CommandX, Pattern.CommandZ]): command to run
        """
        if cmd.name not in ['M', 'X', 'Z']:
            raise ArgumentTypeError(f"Invalid command ({cmd}) with the name: ({cmd.name})!\n"
                                    f"Only 'M', 'X' and 'Z' are supported as the command name.")

        if cmd.name == 'M':  # execute measurement commands
            signal_s = self.sum_outcomes(cmd.domain_s)
            signal_t = self.sum_outcomes(cmd.domain_t)

            # The adaptive angle is (-1)^{signal_s} * angle + {signal_t} * pi
            adaptive_angle = (-1) ** signal_s * cmd.angle + signal_t * pi
            self.measure(cmd.which_qubit, Basis.Plane(cmd.plane, adaptive_angle))

        else:  # execute X and Z byproduct correction commands
            power = self.sum_outcomes(cmd.domain)
            self.correct_byproduct(cmd.name, cmd.which_qubit, power)

    def __run_cmd_lst(self, cmd_lst: list, bar_start: int, bar_end: int) -> None:
        r"""Run a list of commands

        Args:
            cmd_lst (list): a list of commands to run
            bar_start (int): start point of the progress bar
            bar_end (int): end point of the progress bar
        """
        for i in range(len(cmd_lst)):
            self.__run_cmd(cmd_lst[i])
            print_progress((bar_start + i + 1) / bar_end, "Pattern Running Progress", self.__track)

    def __flip_outcomes(self, cmd_s_lst: list, bar_start: int, bar_end: int) -> None:
        r"""Flip the measurement outcome from the signal shifting commands.

        Args:
            cmd_s_lst (list): a list of commands to run
            bar_start (int): start point of the progress bar
            bar_end (int): end point of the progress bar
        """
        if not isinstance(cmd_s_lst, List):
            raise ArgumentTypeError(f"Input {cmd_s_lst} should be a list.")

        flip = {}
        for i in range(len(cmd_s_lst)):
            cmd_s = cmd_s_lst[i]
            # Execute signal shifting commands
            if cmd_s.which_qubit not in self.vertex.measured:
                raise ArgumentTypeError(f"Invalid vertex index: ({cmd_s.which_qubit})!\n"
                                        f"This qubit is not measured.")

            power = self.sum_outcomes(cmd_s.domain)
            flip[cmd_s.which_qubit] = (self.__outcome[cmd_s.which_qubit] + power % 2) % 2
            print_progress((bar_start + i + 1) / bar_end, "Pattern Running Progress", self.__track)

        # Update outcome dictionary
        self.__outcome = {i: flip[i] if i in flip.keys() else self.__outcome[i] for i in self.__outcome.keys()}

    def kron_unmeasured_qubits(self) -> None:
        r"""Kronecker product to all unmeasured qubits.

        Warning:
            This method is called when user runs MBQC from a measurement pattern.
        """
        self.__draw = False  # turn off the plot switch
        # As the create_graph_state function would change the measured qubits list, we need to record it
        measured_qubits = self.vertex.measured[:]

        for qubit in list(self.__graph.nodes):
            if qubit not in self.vertex.measured:
                self.__create_graph_state(qubit)
                self.vertex.measured.append(qubit)
                self.max_active = max(len(self.vertex.active), self.max_active)
                self.__bg_state = PureState(self.__bg_state.state, self.vertex.active)

        self.vertex.measured = measured_qubits  # restore the measured qubits

    def run_pattern(self) -> None:
        r"""Run the measurement pattern.

        Warning:
            This method is called after ``set_pattern``.
        """
        if self.__pattern is None:
            raise ArgumentTypeError(f"Invalid pattern: ({self.__pattern})!\n"
                                    f"Please set 'pattern' before calling 'run_pattern'.")

        # Execute measurement commands and correction commands
        cmd_m_lst = [cmd for cmd in self.__pattern.commands if cmd.name == 'M']
        cmd_c_lst = [cmd for cmd in self.__pattern.commands if cmd.name in ['X', 'Z']]
        cmd_s_lst = [cmd for cmd in self.__pattern.commands if cmd.name == 'S']
        bar_end = len(cmd_m_lst + cmd_c_lst + cmd_s_lst)

        self.__run_cmd_lst(cmd_m_lst, 0, bar_end)
        # Activate unmeasured qubits before byproduct corrections
        self.kron_unmeasured_qubits()

        self.__run_cmd_lst(cmd_c_lst, len(cmd_m_lst), bar_end)

        # Flip vertices outcomes according to "commandS"
        self.__flip_outcomes(cmd_s_lst, len(cmd_m_lst + cmd_c_lst), bar_end)

        # The output state's label is messy (e.g. [(2, 0), (0, 1), (1, 3)...]),
        # so we permute the systems in order
        q_output = self.__pattern.output_[1]
        self.__status.permute_systems(q_output)
        self.__bg_state = self.__status

        self.__update_history()

    @staticmethod
    def __map_qubit_to_row(out_lst: list) -> Dict:
        r"""Map the output qubits to row index.

        Args:
            out_lst (list): a list of output qubits

        Returns:
            dict: the relation between output qubits and their row index
        """
        return {qubit[0]: qubit for qubit in out_lst}

    def get_classical_output(self) -> Union[str, dict]:
        r"""Get the measurement outcome of MBQC.

        Note:
            If user sets a measurement pattern, then this returns the measurement outcome of the output qubits.
            If user sets a graph of MBQC, then this returns measurement outcomes of all qubits.

        Returns:
            Union[str, dict]: return the measurement outcomes
        """
        # If the input is a pattern, return the string result equivalent to the circuit model
        if self.__pattern is not None:
            width = len(self.__pattern.input_)
            c_output = self.__pattern.output_[0]
            q_output = self.__pattern.output_[1]
            # Obtain the relationship between row number and the output qubit index
            output_lst = c_output + q_output
            row2qubit = self.__map_qubit_to_row(output_lst)

            # Mark the classical outputs with their measurement outcomes
            # Mark a quantum register with a place holder '_'
            # bit_lst = [str(self.__outcome[row2qubit[i]]) if row2qubit[i] in c_output else '_' for i in range(width)]
            bit_lst = [str(self.__outcome[row2qubit[i]]) if row2qubit[i] in c_output else '' for i in range(width)]
            bit_str = ''.join(bit_lst)
            return bit_str

        # If the input is a graph, return the outcome dictionary
        else:
            return self.__outcome

    def get_quantum_output(self) -> "PureState":
        r"""Get quantum output.

        Returns:
            PureState: quantum state after the MBQC algorithm
        """
        return self.__status

    def __set_position(self, pos: Union[dict, bool]) -> None:
        r"""Set the position of graph plotting.

        Args:
            pos (Union[dict, bool]): position to set
        """
        if isinstance(pos, dict):
            self.__pos = pos
        elif isinstance(pos, bool):
            if pos:
                self.__pos = {}
                for vertex in list(self.__graph.nodes):
                    self.__pos[vertex] = [vertex[1], - vertex[0]]
            else:
                self.__pos = spring_layout(self.__graph)  # use 'spring_layout' method
        else:
            raise ArgumentTypeError(f"Invalid position ({pos}) with the type: ({type(pos)})!\n"
                                    f"Only `Bool` and `Dict` are supported as the type of position.")

    def __draw_process(self, which_process: str, which_qubit: Any) -> None:
        r"""Draw the computational process of MBQC.

        Args:
            which_process (str): which process to plot, can be "measuring", "active" or "measured"
            which_qubit (Any): current vertex to focus on
        """
        if self.__draw:
            if which_process not in ['measuring', 'active', 'measured']:
                raise ArgumentTypeError(f"Invalid process name: ({which_process})!\n"
                                        f"Only `measuring`, 'active' and `measured` are supported as the process name.")

            # Find where the 'which_qubit' is
            if which_qubit in self.vertex.pending:
                pending = self.vertex.pending[:]
                pending.remove(which_qubit)
                vertex_sets = [pending, self.vertex.active, [which_qubit], self.vertex.measured]
            elif which_qubit in self.vertex.active:
                active = self.vertex.active[:]
                active.remove(which_qubit)
                vertex_sets = [self.vertex.pending, active, [which_qubit], self.vertex.measured]
            elif which_qubit in self.vertex.measured:
                vertex_sets = [self.vertex.pending, self.vertex.active, [], self.vertex.measured]
            else:
                raise ArgumentTypeError(f"Invalid vertex: ({which_qubit})! This vertex is not on the graph.")

            # Characterize ancilla vertices
            ancilla_qubits = []
            if self.__pattern is not None:
                for vertex in list(self.__graph.nodes):
                    row = vertex[0]
                    col = vertex[1]
                    # Ancilla vertices do not have integer rows and cols
                    if abs(col - int(col)) >= EPSILON or abs(row - int(row)) >= EPSILON:
                        ancilla_qubits.append(vertex)

            plt.cla()
            plt.title("MBQC Running Process", fontsize=15)
            plt.xlabel("Measuring (RED)  Active (GREEN)  Pending (BLUE)  Measured (GRAY)", fontsize=12)
            plt.grid()
            # mngr = plt.get_current_fig_manager()
            # mngr.window.setGeometry(500, 100, 800, 600)
            colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:gray']
            for j in range(4):
                for vertex in vertex_sets[j]:
                    options = {
                        'nodelist': [vertex],
                        'node_color': colors[j],
                        'node_shape': '8' if vertex in ancilla_qubits else 'o',
                        'with_labels': False,
                        'width': 3,
                    }
                    draw_networkx(self.__graph, self.__pos, **options)
                    ax = plt.gca()
                    ax.margins(0.20)
                    plt.axis('on')
                    ax.set_axisbelow(True)
            plt.pause(self.__pause_time)

    def draw_process(self, draw=True, pos=False, pause_time=0.5) -> None:
        r"""Draw the computational process of MBQC.

        Args:
            draw (bool, optional): whether to draw the process
            pos (Union[bool, Dict], optional): position of the graph
            pause_time (float, optional): refresh time of the plot
        """
        if self.__graph is None:
            raise ArgumentTypeError(f"Please set 'graph' or 'pattern' before calling 'draw_process'.")
        if not isinstance(draw, bool):
            raise ArgumentTypeError(f"Input {draw} should be a bool value.")
        if not isinstance(pos, bool):
            if not isinstance(pos, Dict):
                raise ArgumentTypeError(f"Invalid position ({pos}) with the type: ({type(pos)})!\n"
                                        f"Only `Bool` and `Dict` are supported as the type of position.")
        if pause_time <= 0:
            raise ArgumentTypeError(f"Invalid drawing pause time: ({pause_time})!\n"
                                    f"Drawing pause time must be a positive float value.")

        self.__draw = draw
        self.__pause_time = pause_time

        if self.__draw:
            plt.figure()
            plt.ion()
            self.__set_position(pos)

    def track_progress(self, track=True) -> None:
        r""" Track the progress of MBQC running.

        Args:
            track (bool, optional): whether to track the progress
        """
        if not isinstance(track, bool):
            raise ArgumentTypeError(f"Input {track} should be a bool value.")

        self.__track = track


def run_circuit(circuit: "Circuit", shots=1024, input_state=None, optimize=True) -> dict:
    r"""Run a quantum circuit by its equivalent MBQC model.

    Note:
        This method transpiles a quantum circuit to its equivalent MBQC model first
        and then runs the MBQC pattern to get equivalent sampling result.
        It can work well for large-scale quantum shallow circuits with the option ``optimize=True``.

    Warnings:
        We should check if the circuit has sequential registers first.
        If not, we need to perform remapping before running the circuit.

    Args:
        circuit (Circuit): quantum circuit to run
        shots (int, optional): number of sampling
        input_state (PureState, optional): input quantum state
        optimize (bool): whether to optimize the measurement order

    Returns:
        dict: classical results
    """
    # Check if the circuit has sequential registers
    if circuit.width != max(circuit.occupied_indices) + 1:
        remap_circuit = circuit.copy()
        remap_circuit.remap_indices()
        circuit = remap_circuit

    # Add a layer of H gates to start from zero states
    if input_state is None:
        h_layer = [{"name": 'h', "which_qubit": [i], "params": None} for i in range(circuit.width)]
        new_circuit = circuit.copy()
        new_circuit._history = h_layer + circuit.gate_history
        circuit = new_circuit

    pattern = transpile_to_pattern(circuit, shift_signal=True, optimize=optimize, track=False)

    samples = []

    for shot in range(shots):
        mbqc = MBQC()
        mbqc.set_pattern(pattern)
        mbqc.set_input_state(input_state)
        mbqc.run_pattern()
        c_output = mbqc.get_classical_output()
        samples.append(c_output)

    sample_dict = {}
    for string in list(set(samples)):
        sample_dict[string] = samples.count(string)

    return sample_dict
