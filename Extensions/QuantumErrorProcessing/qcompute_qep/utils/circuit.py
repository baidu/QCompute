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
Utils functions for manipulating quantum circuits.

In ``qcompute_qep``, a quantum circuit is represented by the
``QProgram`` abstract data structure and described by a list of
``CircuitLine`` instances. Each ``CircuitLine`` instance describes a
quantum gate operating on target qubits.
"""

import copy
import functools
import random
from typing import List, Optional, Union, Dict, Tuple, Callable
import networkx as nx
import numpy as np
import qiskit
from qiskit.providers.ibmq.ibmqbackend import IBMQBackend
from qiskit import transpile
from qiskit.providers.ibmq.managed import IBMQJobManager
from scipy.stats import unitary_group

import QCompute
from QCompute.QPlatform import QOperation
from QCompute.QPlatform.QOperation import CircuitLine, RotationGate, FixedGate
from qcompute_qep.interface import conversion
from qcompute_qep.utils.types import QProgram, QComputer, number_of_qubits
from qcompute_qep.utils.linalg import expand, permute_systems, dagger, basis
from qcompute_qep.utils.utils import limit_angle, decompose_yzy, COLOR_TABLE
from qcompute_qep.utils.gate import CPauliOP, decompose_CPauli
from qcompute_qep.exceptions.QEPError import ArgumentError


def inv_opr(ops: List[QOperation.QOperation]) -> RotationGate.RotationGateOP:
    r"""Computer the inverse of a list of single-qubit quantum gates.

    Let :math:`[U_1,\cdots, U_d]` is a list of single-qubit quantum gates and these gates act from left to right.
    Then the inverse gate of this list is defined as:

    .. math:: V = U_d^\dagger \cdots U_1^\dagger,

    where :math:`U_i^\dagger` is the inverse version of the gate :math:`U_i`.

    :param ops: List[QOperation], a list of `QOperation`-type quantum gates
    :return: a `QOperation`-type inverse quantum operation
    """
    inv_mat = functools.reduce(np.dot, [g.getInverse().getMatrix() for g in ops])
    # yzy-decomposition of a U3 gate
    _, theta, phi, lam = decompose_yzy(inv_mat)
    # construct a U3 gate
    return RotationGate.U(theta, phi, lam)


def _obtain_font_idx(dag: nx.MultiDiGraph) -> List[int]:
    r"""Obtain the font layer of a directed acyclic graph (DAG).

    :param dag: nx.MultiDiGraph, a directed acyclic graph
    :return: List[int], indices of nodes without any predecessor(s)
    """
    font_idx = []
    for gate in dag.nodes:
        if len(list(dag.predecessors(gate))) == 0:
            font_idx.append(gate)
    return font_idx


def circuit_to_dag(circuit: List[CircuitLine]) -> nx.MultiDiGraph:
    r"""Convert a circuit into a directed acyclic graph (DAG).

    From the quantum circuit, we construct a directed acyclic graph,
    such that the nodes represent the quantum gates, and the directed edges represent the execution orders.

    :param circuit: List[CircuitLine], a quantum circuit composed of ``CircuitLine`` instances
    :return: nx.MultiDiGraph, a directed acyclic graph

    **Examples**

        >>> qp = QEnv()
        >>> qp.Q.createList(2)
        >>> H(qp.Q[0])
        >>> CX(qp.Q[0], qp.Q[1])
        >>> MeasureZ(*qp.Q.toListPair())
        >>> dag = circuit_to_dag(qp.circuit)
        >>> nx.draw(dag)
    """
    circ_tmp = (copy.deepcopy(circuit))

    num_gates = len(circ_tmp)
    dag = nx.MultiDiGraph()
    # add nodes -- indices of gates)
    dag.add_nodes_from(range(num_gates))
    # construct edges (related to sequence of gates) -- from back to font
    while len(circ_tmp) > 0:
        cl_back = circ_tmp.pop()
        idx_back = len(circ_tmp)
        qreg_back = set(cl_back.qRegList)
        union_set = set()
        for idx_font in range(idx_back - 1, -1, -1):
            cl_font = circ_tmp[idx_font]
            # judge if there is dependent relation about qreg(s) acted
            qreg_font = set(cl_font.qRegList)
            if len(qreg_font & qreg_back) > 0 and len(qreg_font & union_set) == 0:
                dag.add_edge(idx_font, idx_back)
                union_set = union_set | qreg_font
    return dag


def circuit_to_layers(circuit: List[CircuitLine]) -> List[List[CircuitLine]]:
    r"""Separate a quantum circuit into a list of layers.

    A quantum circuit can be separated into a list of layers, where each layer is composed of several gates that
    can operate parallelly. On the other hand, different layers must operate sequentially determined by their order.
    In ``qcompute_qep``, each layer is represented by ``List[CircuitLine]``, a list of quantum gates.

    .. note::

        Consider the following Bell state preparation circuit

        ::

            0: ---H---@---
                      |
            1: -------X---

        It can be separated into two layers: the first layer contains the H gate and
        the second layer contains the CNOT gate.

    :param circuit: quantum circuit, in ``qcompute`` it is a list of ``CircuitLine`` elements
    :return: List[List[CircuitLine]], a list of quantum layers, each layer is of type ``List[CircuitLine]``
    """
    layers = []
    # building on parameter circuit and a corresponding DAG
    dag = circuit_to_dag(circuit)
    while len(dag.nodes) > 0:
        indices_font = _obtain_font_idx(dag)
        layers.append([circuit[idx] for idx in indices_font])
        dag.remove_nodes_from(indices_font)

    layers = list(map(sort_layer_on_qreg, layers))
    return layers


def inverse_layer(layer: List[CircuitLine]) -> List[CircuitLine]:
    r"""Compute the inverse operation of a quantum layer.

    Assume the quantum layer :math:`L = [g_1,\cdots, g_n]`. Mathematically, its inverse is defined as

    .. math:: L \mapsto L^\dagger =[g_n^\dagger, \cdots, g_1^\dagger],

    where :math:`g_i^\dagger` is the inverse gate of :math:`g_i`.

    :param layer: List[CircuitLine], :math:`L`, is a list of ``QOperation.CircuitLine`` objects
    :return: List[CircuitLine], :math:`L^\dagger`, short as the dagger of :math:`L`
    """
    layer = copy.deepcopy(layer)  # deepcopy is necessary
    return list(reversed([cl.inverse() for cl in layer]))


def inverse_layers(layers: List[List[CircuitLine]]) -> List[List[CircuitLine]]:
    r"""Compute the inverse operations of a list of quantum layers.

    Assume the list of quantum layers :math:`[L_1, L_2, \cdots, L_n]`. Its inverse is defined as

    .. math:: [L_1, L_2, \cdots, L_n] \mapsto [L_n^\dagger, \cdots, L_2^\dagger, L_1^\dagger],

    where :math:`L_i^\dagger` is the inverse of the layer :math:`L_i` and can be computed using ``inverse_layer``.

    :param layers: List[List[CircuitLine]], a list of quantum layers, each layer is of type ``List[CircuitLine]``
    :return: List[List[CircuitLine]], a list of quantum layers
    """
    return list(reversed(list(map(inverse_layer, layers))))


def layers_to_circuit(layers: List[List[CircuitLine]]) -> List[CircuitLine]:
    r"""Convert a list of quantum layers to a quantum circuit.

    In ``qcompute_qep``, a quantum circuit/quantum layer is represented by a list of ``CircuitLine`` objects.
    Thus, to convert a list of quantum layers to a quantum circuit,
    we need to regroup the ``CircuitLine`` objects in quantum layers.

    :param layers: List[List[CircuitLine]], a list of quantum layers
    :return: List[CircuitLine], a quantum circuit
    """
    return sum(layers, [])


def depth_of_circuit(circuit: List[CircuitLine], measure=True) -> int:
    r"""Compute the number of quantum layers of a quantum circuit.

    If ``measure=True``, the last measurement layer is excluded when computing the number of quantum layers.

    :param circuit: List[CircuitLine], a quantum circuit
    :param measure: bool, whether the last measurement layer should be taken into account
    :return: int, the number of quantum layers of a quantum circuit

    **Examples**

        >>> qp = QEnv()
        >>> qp.Q.createList(2)
        >>> H(qp.Q[0])
        >>> CX(qp.Q[0], qp.Q[1])
        >>> MeasureZ(*qp.Q.toListPair())
        >>> depth_of_circuit(qp.circuit, measure=False)
        2
        >>> depth_of_circuit(qp.circuit)
        3
    """
    # remove all barrier gates
    remove_barrier(circuit)
    # parse the layers
    layers = circuit_to_layers(circuit)
    has_meas = isinstance(circuit[-1].data, QOperation.Measure.MeasureOP)
    if measure is False and has_meas:
        return len(layers) - 1
    else:
        return len(layers)


def sort_layer_on_qreg(layer: List[CircuitLine], descend=False) -> List[CircuitLine]:
    r"""Sort the quantum gates of a quantum layer.

    Sort the quantum gates of a quantum layer, based on the indices of quantum registers that these quantum gates act.
    By default, sort these gates ascendingly.

    :param layer: a list of ``CircuitLine`` instances
    :param descend: sort whether in descending
    :return: a sorted layer
    """
    if descend:
        return sorted(layer, key=lambda cl: max(cl.qRegList))
    else:
        return sorted(layer, key=lambda cl: min(cl.qRegList))


def num_qubits_of_circuit(circuit: List[CircuitLine]) -> int:
    r"""Number of working qubits in the quantum circuit.

    The Number of working qubits in the quantum circuit is computed
    by collecting the qubit indices and count the number.

    :param circuit: quantum circuit, a list of `CircuitLine` instances
    :return: number of qubit registers used in this quantum circuit
    """
    qubit_indices = []
    for cl in circuit:
        qubit_indices.extend(cl.qRegList)
    qubit_indices = np.unique(qubit_indices)
    return len(qubit_indices)


def _gates_span_overlap(cl1: CircuitLine, cl2: CircuitLine) -> bool:
    r"""Check if two quantum gates have working qubit overlap.

    :param cl1: one ``CiruitLine`` instance
    :param cl2: another ``CiruitLine`` instance
    :return: bool type
    """
    idx1 = set(range(min(cl1.qRegList), max(cl1.qRegList) + 1))
    idx2 = set(range(min(cl2.qRegList), max(cl2.qRegList) + 1))
    return not set.isdisjoint(idx1, idx2)


def _width_of_gate(cl: CircuitLine) -> int:
    r"""Width of a quantum gate's name in alphabets.

    For example, the 'CZ' gates has width of :math:`2` while the 'TDG' (dagger of 'T' gate) has width of :math:`3`.

    :param cl: a `CircuitLine` instance
    :return: width, int type, char unit
    """
    gate_widths = {
        'ID': 1, 'X': 1, 'Y': 1, 'Z': 1, 'S': 1, 'H': 1, 'T': 1,
        'CX': 1, 'CY': 1, 'CZ': 1, 'CH': 1, 'SWAP': 1,
        'CSWAP': 1, 'CCX': 1, 'CPauli': 1,
        'TDG': 3, 'SDG': 3,
        'measure': 4, 'barrier': 1}
    if isinstance(cl.data, QOperation.FixedGate.FixedGateOP):
        return gate_widths[cl.data.name]
    elif isinstance(cl.data, QOperation.RotationGate.RotationGateOP):
        # if not :
        # RX, RY, RZ, U; CRX,CRY, CRZ, CU
        if len(cl.data.argumentList) == 1:
            # e.g. Rx(+1.23)
            return 9
        else:
            # U, CU
            # e.g. U3(+0.12,-2.12,+0.23)
            return 21
    elif isinstance(cl.data, QOperation.Barrier.BarrierOP):
        return gate_widths['barrier']
    elif isinstance(cl.data, QOperation.Measure.MeasureOP):
        return gate_widths['measure']
    elif isinstance(cl.data, CPauliOP):
        return gate_widths['CPauli']
    else:
        raise TypeError('print_circuit(): {} is not a supported gate type'.format(type(cl.data)))


def print_circuit(circuit: List[CircuitLine],
                  style='text',
                  show=True,
                  num_qubits=None,
                  colors=None) -> Optional[str]:
    r"""Print a quantum circuit in the simple text style.

    The @colors parameter is a dictionary that specifies the colors of the qubits. It has the form:

    > {'red': [0, 1, 2], 'blue': [3, 4, 5], 'green': [6, 7, 8], ...}

    which means that qubits `[0, 1, 2]` are printed in red color, the qubits `[3, 4, 5]` are printed in blue color,
    and so on. Currently, we support the following colors: 'red', 'yellow', 'blue', 'violet', 'beige', and 'white'.

    :param circuit: a list including a series of `CircuitLine` instances
    :param style: decide which style to be represented as, optional: 'text', 'mpl' or others
    :param show: bool type, decides whether show the circuit in form of text lines
    :param num_qubits: The number of qubits of quantum circuit
    :param colors: a dictionary specifying what the colors of the qubits
    :return: Optional[str], the quantum circuit in the simple text style
    """
    if style != 'text':
        raise ValueError('currently only support text display style')
    # If there is no quantum gate, simply print the circuit line
    # For example
    #
    # 0: ------------------
    # 1: ------------------

    if not circuit:
        if num_qubits is None:
            print('in print_circuit(): the given circuit is empty!')
            return None
        str_lines = ['{}: '.format(i) + '------------' for i in range(num_qubits)]
        str_longitude = [' ' * 3 + '   '] * (num_qubits - 1)
        str_print = [[str_lines[i], str_longitude[i]] for i in range(num_qubits - 1)]
        str_print.append([str_lines[-1]])  # each element is a list
        str_print = sum(str_print, [])
        str_print = '\n'.join(str_print)
        if show:
            print(str_print)
        return str_print

    # notice that the number of qubits should be the max qregs in qRegList plus one, otherwise it will cause problem.
    if num_qubits is None:
        num_qubits = max(x for cl in circuit for x in cl.qRegList) + 1
    else:
        num_qubits = max(max(x for cl in circuit for x in cl.qRegList) + 1, num_qubits)

    # Decompose the quantum circuit into layers
    layers = circuit_to_layers(circuit)
    layers = list(map(sort_layer_on_qreg, layers))
    # Parse each layer for print
    layers_print = []
    while len(layers) > 0:
        layer = layers.pop(0)
        layer_font_part = []
        while len(layer) > 0:
            cl = layer.pop(0)
            layer_font_part.append(cl)  # split the layer
            if any(map(functools.partial(_gates_span_overlap, cl), layer)):
                layers.insert(0, layer)
                break  # break while loop
        layers_print.append(layer_font_part)

    # Gate to string dictionary
    gate_strings = {'qw': '---', 'cw': '===', 'qwx': '|', 'cwx': '||',
                    'ctrl': '●', 'barrier': '≡', 'space': '   ', 'meas': 'MEAS',
                    'ID': 'I', 'X': 'X', 'Y': 'Y', 'Z': 'Z', 'H': 'H', 'S': 'S', 'T': 'T', 'TDG': 'TDG', 'SDG': 'SDG',
                    'CX': ('●', 'X'), 'CY': ('●', 'Y'), 'CZ': ('●', 'Z'), 'CH': ('●', 'H'),
                    'SWAP': ('x', 'x'), 'CSWAP': ('●', 'x', 'x'), 'CCX': ('●', '●', 'X'),
                    '0': '○', '1': '●', 'I': 'I'}

    def single_qubit_gate_str(cl: CircuitLine):
        gname = cl.data.name
        if isinstance(cl.data, QOperation.FixedGate.FixedGateOP):
            return gate_strings[gname]
        elif isinstance(cl.data, QOperation.RotationGate.RotationGateOP):
            if gname in {'RX', 'RY', 'RZ'}:
                angle = limit_angle(cl.data.argumentList[0])
                return '{}({:+.2f})'.format(gname.capitalize(), angle)
            else:  # U3 gate
                angles = list(map(limit_angle, cl.data.argumentList))
                return 'U3({:+.2f},{:+.2f},{:+.2f})'.format(*angles)
        else:
            raise TypeError(
                '{} is not a supported gate type to print'.format(type(cl.data)))

    def multi_qubit_gate_str(cl: CircuitLine):
        gname = cl.data.name
        if isinstance(cl.data, QOperation.FixedGate.FixedGateOP):
            return gate_strings[gname]
        elif isinstance(cl.data, QOperation.RotationGate.RotationGateOP):
            if gname in {'CRX', 'CRY', 'CRZ'}:
                angle = limit_angle(cl.data.argumentList[0])
                return gate_strings['ctrl'], '{}({:+.2f})'.format(gname[1:].capitalize(), angle)
            else:
                # CU3 gate
                angles = list(map(limit_angle, cl.data.argumentList))
                return gate_strings['ctrl'], 'U3({:+.2f},{:+.2f},{:+.2f})'.format(*angles)
        else:
            raise TypeError('{} is not a supported gate type to print'.format(type(cl.data)))

    def extend_gate_str(gstr, lw):
        return int((lw - len(gstr)) / 2) * '-' + gstr + int((lw - len(gstr)) / 2) * '-'

    def extend_space_str(spstr, lw):
        return int((lw - len(spstr)) / 2) * ' ' + spstr + int((lw - len(spstr)) / 2) * ' '

    # Initialize the quantum circuit strings.
    # If there are more than 10 qubits, we must use 2 spaces to display the qubit number.
    # We do not consider the case where there are more than 100 qubits.
    if num_qubits > 10:
        str_lines = ['{:2d}: '.format(i) + gate_strings['qw'] for i in range(num_qubits)]  # length: n
        str_longitude = [' ' * 4 + gate_strings['space']] * (num_qubits - 1)  # length: n - 1
    else:
        str_lines = ['{}: '.format(i) + gate_strings['qw'] for i in range(num_qubits)]
        str_longitude = [' ' * 3 + gate_strings['space']] * (num_qubits - 1)

    for layer in layers_print:
        # Tag quantum registers for every layer
        qreg_added = np.zeros(num_qubits)
        space_added = np.zeros(num_qubits - 1)
        layer_width = max(list(map(_width_of_gate, layer)))
        for cl in layer:
            qRegList = cl.qRegList
            qreg_added[qRegList] = 1
            # Controlled-Pauli gate on specified qubits
            if isinstance(cl.data, CPauliOP):
                # String for control qubits
                control_qubits = [qubit.index for qubit in cl.data.reg_c]
                bit_list = [b for b in np.binary_repr(cl.data.val, width=len(control_qubits))]
                for idx, bit in zip(control_qubits, bit_list):
                    str_lines[idx] += extend_gate_str(gate_strings[bit], layer_width)
                # String for target qubits
                target_qubits = [qubit.index for qubit in cl.data.reg_t]
                pauli_list = [*cl.data.pauli]
                for idx, pauli in zip(target_qubits, pauli_list):
                    str_lines[idx] += extend_gate_str(gate_strings[pauli], layer_width)
                # String for control spaces
                space_span = range(min(qRegList), max(qRegList))
                space_added[space_span] = 1  # tag drawn
                for idx in space_span:
                    str_longitude[idx] += extend_space_str(gate_strings['qwx'], layer_width)
            # Measurements on specified qubits
            elif isinstance(cl.data, QOperation.Measure.MeasureOP):
                for i in range(len(qRegList)):
                    str_lines[qRegList[i]] += extend_gate_str(gate_strings['meas'], layer_width)
                str_longitude = [str_longitude[i] + gate_strings['space'] for i in range(num_qubits - 1)]
            # Barrier gate on all qubits
            elif isinstance(cl.data, QOperation.Barrier.BarrierOP):
                str_lines = [str_lines[i] + gate_strings['barrier'] for i in range(num_qubits)]
                str_longitude = [str_longitude[i] + gate_strings['barrier'] for i in range(num_qubits - 1)]
                # Qubit and Space tags, indicating that the Barrier column is occupied
                qreg_added = np.ones(num_qubits)
                space_added = np.ones(num_qubits - 1)
            elif len(qRegList) == 1:
                str_lines[qRegList[0]] += extend_gate_str(single_qubit_gate_str(cl), layer_width)
            elif len(qRegList) == 2 or len(qRegList) == 3:
                # string for wires
                gstr_list = multi_qubit_gate_str(cl)
                for idx, gstr in zip(qRegList, gstr_list):
                    str_lines[idx] += extend_gate_str(gstr, layer_width)
                # String for control spaces
                space_span = range(min(qRegList), max(qRegList))
                space_added[space_span] = 1  # tag drawn
                for idx in space_span:
                    str_longitude[idx] += extend_space_str(gate_strings['qwx'], layer_width)
            else:
                raise TypeError('{} is not a supported gate type to print'.format(type(cl.data)))

        # Append quantum wires and spaces for qubits without quantum gates to keep the quantum circuit aligned
        for idx in np.where(qreg_added == 0)[0]:
            str_lines[idx] += '-' * layer_width
        for idx in np.where(space_added == 0)[0]:
            str_longitude[idx] += ' ' * layer_width

        str_lines = [str_lines[i] + gate_strings['qw'] for i in range(num_qubits)]
        str_longitude = [str_longitude[i] + gate_strings['space'] for i in range(num_qubits - 1)]

    # Add colors to qubits if the @colors parameter is set
    if colors is not None:
        for color_name, qubits in colors.items():
            color_name = color_name.lower()
            if color_name not in COLOR_TABLE:
                raise ArgumentError("print_circuit(): {} is not a supported color name "
                                    "in the COLOR_TABLE!".format(color_name))
            # Only color those qubits that appear in the quantum circuit
            indices = [idx for idx in qubits if idx < num_qubits]
            for idx in indices:
                str_lines[idx] = COLOR_TABLE[color_name] + str_lines[idx] + COLOR_TABLE['end']

    # Concat and print (optional)
    str_print = [[str_lines[i], str_longitude[i]] for i in range(num_qubits - 1)]
    str_print.append([str_lines[-1]])  # each element is a list
    str_print = sum(str_print, [])
    str_print = '\n'.join(str_print)
    if show:
        print(str_print)
    return str_print


def contain_measurement(qp: Union[QCompute.QEnv, List[CircuitLine]]) -> bool:
    r"""Checks if the quantum circuit contains measurement.

    :param qp: a `QProgram` instance or a quantum circuit data field
    :return: bool type
    """
    circuit = qp.circuit if isinstance(qp, QCompute.QEnv) else qp
    if not circuit:
        return False
    else:
        return isinstance(circuit[-1].data, QOperation.Measure.MeasureOP)


def remove_measurement(qp: Union[QCompute.QEnv, List[CircuitLine]]) -> Optional[CircuitLine]:
    r"""Remove measurement operations from the circuit.

    :param qp: Union[QCompute.QEnv, List[CircuitLine]], a `QProgram` instance or a quantum circuit data field
    :return: Optional[CircuitLine], represents the removed measurement operation.
            If there is no measurement operation in the quantum circuit, return ``None``
    """
    circuit = qp.circuit if isinstance(
        qp, QCompute.QEnv) else qp  # List[CircuitLine]

    if contain_measurement(circuit):
        return circuit.pop()
    else:
        return None


def append_measurement(qp: Union[QCompute.QEnv, List[CircuitLine]], measurement_cl: CircuitLine = None) -> None:
    r"""Append the measurement to the end of the quantum circuit.

    :param qp: a quantum circuit whose measurement operation will be removed
    :param measurement_cl: a `CircuitLine` instance, the removed measurement operation instance to be added
    """
    if contain_measurement(qp):
        print("The quantum program already contains measurement, cannot add measurement operation!")
    elif measurement_cl is None:
        raise ArgumentError("The measurementOp is None!")
    else:
        if isinstance(qp, QCompute.QEnv):
            qp.circuit.append(measurement_cl)
        else:
            qp.append(measurement_cl)


def add_barrier(qp: Union[QCompute.QEnv, List[CircuitLine]]) -> None:
    r"""Add a barrier to the end of the quantum circuit.

    :param qp: QProgram, a quantum program to which we append a barrier in the end
    """
    # construct a Barrier-type CircuitLine
    if isinstance(qp, QCompute.QEnv):
        cl = CircuitLine(data=QOperation.Barrier.BarrierOP(),
                         qRegList=qp.Q.toListPair()[1])
        qp.circuit.append(cl)
    else:
        num_qubits = num_qubits_of_circuit(qp)
        cl = CircuitLine(data=QOperation.Barrier.BarrierOP(),
                         qRegList=list(range(num_qubits)))
        qp.append(cl)


def remove_barrier(qp: Union[QCompute.QEnv, List[CircuitLine]]) -> None:
    r"""Remove all barriers from the quantum circuit.

    :param qp: QProgram, a quantum program for which we will remove all barriers
    """
    tag = 'barrier'
    circuit = qp.circuit if isinstance(qp, QCompute.QEnv) else qp

    for i, cl in enumerate(circuit):
        # tag barrier operation
        if isinstance(cl.data, QOperation.Barrier.BarrierOP):
            circuit[i] = tag
    # remove all tagged barrier operations
    while tag in circuit:
        circuit.remove(tag)


def layer_to_unitary(layer: List[CircuitLine], n: int) -> np.ndarray:
    r"""Compute the unitary matrix of a quantum layer.

    A quantum layer is described by a list of ``CircuitLine`` instances,
    each ``CircuitLine`` describes a quantum gate and the qubit(s) it operates.
    If the given quantum layer contains the measurement operations, we will remove it.

    .. note::

        Notice that when computing the unitary,
        we assume the LSB (the least significant bit) mode, i.e., the right-most qubit represents q[0].
        That is, the two-qubit gate :math:`CX_{0\to 1}` (q[0] controls q[1]) has the matrix representation:

        .. math:: CX_{0\to 1}
            = I\otimes \vert0\rangle\!\langle0\vert + X\otimes \vert1\rangle\!\langle1\vert
            = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}.

        Similarly, the gate :math:`CX_{1\to 0}` (q[1] controls q[0]) has the matrix representation:

        .. math:: CX_{1\to 0}
            = \vert0\rangle\!\langle0\vert\otimes I + \vert1\rangle\!\langle1\vert\otimes X
            = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}.

    :param layer: List[CircuitLine], a quantum layer composed of ``CircuitLine`` instances
    :param n: int, the number of qubits of the all Hilbert space
    :return: np.ndarray, the unitary matrix of the layer

    **Example**

    >>> from QCompute import *
    >>> layer = [CircuitLine(CX, qRegList=[0, 1])]
    >>> layer_to_unitary(layer, 2)
    [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
     [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
     [0.+0.j 0.+0.j 1.+0.j 0.+0.j]
     [0.+0.j 1.+0.j 0.+0.j 0.+0.j]]
    """
    # If there is a quantum measurement operation in the end, remove it
    remove_measurement(layer)

    _TWO_QUBIT_GATESET = {'CX', 'CY', 'CZ', 'CH', 'SWAP'}
    _THREE_QUBIT_GATESET = {'CCX', 'CSWAP'}

    U = np.identity(2 ** n, dtype='complex')
    for cl in layer:
        indices = cl.qRegList
        local_u = cl.data.getMatrix()
        # In QCompute, all multi-qubit gates' matrix is fixed as qubit i controls qubit i-1,
        # we thus must analyze the control qubit and permute the system.
        if (cl.data.name in _TWO_QUBIT_GATESET) and (indices[0] < indices[1]):
            local_u = permute_systems(local_u, [1, 0])
        elif (cl.data.name in _THREE_QUBIT_GATESET) and (indices[0] < indices[2]):
            local_u = permute_systems(local_u, [2, 1, 0])
        g_u = expand(local_u, indices, n)
        # Notice that the expand function assume the local systems order [0, ..., n-1],
        # we thus have to reverse the qubits to match the LSB assumption: [n-1, ..., 0]
        g_u = permute_systems(g_u, perm=list(reversed(range(n))))
        U = g_u @ U

    return U


def circuit_to_unitary(qp: QCompute.QEnv, qubits: List[int] = None) -> np.ndarray:
    r"""Compute the unitary matrix of the quantum circuit.

    A quantum circuit is described by a list of ``CircuitLine`` instances,
    each ``CircuitLine`` describes a quantum gate and the qubit(s) it operates.
    If the given quantum circuit contains the measurement operations, we will remove it.

    .. note::

        Notice that when computing the unitary,
        we assume the LSB (the least significant bit) mode, i.e., the right-most qubit represents q[0].
        That is, the two-qubit gate :math:`CX_{0\to 1}` (q[0] controls q[1]) has the matrix representation:

        .. math:: CX_{0\to 1}
            = I\otimes \vert0\rangle\!\langle0\vert + X\otimes \vert1\rangle\!\langle1\vert
            = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}.

        Similarly, the gate :math:`CX_{1\to 0}` (q[1] controls q[0]) has the matrix representation:

        .. math:: CX_{1\to 0}
            = \vert0\rangle\!\langle0\vert\otimes I + \vert1\rangle\!\langle1\vert\otimes X
            = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}.

    :param qp: Union[QProgram, List[CircuitLine]], quantum program for which we compute its unitary representation.
    :param qubits: List[int], target qubit(s)
    :return: np.ndarray, the target unitary

    **Examples**

        We use the Bell state preparation quantum circuit as example to compute the unitary matrix of the circuit.

        >>> qp = QEnv()
        >>> qp.Q.createList(2)
        >>> H(qp.Q[0])
        >>> CX(qp.Q[0], qp.Q[1])
        >>> MeasureZ(*qp.Q.toListPair())
        >>> circuit_to_unitary(qp)
        [[ 0.70710678+0.j  0.70710678+0.j  0.        +0.j  0.        +0.j]
         [ 0.        +0.j  0.        +0.j  0.70710678+0.j -0.70710678+0.j]
         [ 0.        +0.j  0.        +0.j  0.70710678+0.j  0.70710678+0.j]
         [ 0.70710678+0.j -0.70710678+0.j  0.        +0.j  0.        +0.j]]
    """
    # Deep copy the quantum program and remove the measurement if there exists any
    qp = copy.deepcopy(qp)
    remove_measurement(qp)
    if qubits is None:
        qubits = [i for i in range(len(qp.Q.registerMap.keys()))]

    _TWO_QUBIT_GATESET = {'CX', 'CY', 'CZ', 'CH', 'SWAP'}
    _THREE_QUBIT_GATESET = {'CCX', 'CSWAP'}

    # Number of qubits in the quantum program
    n = len(qubits)

    U = np.identity(2 ** n, dtype='complex')
    # Process the circuit layer by layer, each layer corresponds to a :math:`2^n\times 2^n` unitary matrix
    for circuit in qp.circuit:
        indices = [qubits.index(i) for i in circuit.qRegList]
        local_u = circuit.data.getMatrix()
        # In QCompute, all multi-qubit gates' matrix is fixed as qubit i controls qubit i-1,
        # we thus must analyze the control qubit and permute the system.
        if (circuit.data.name in _TWO_QUBIT_GATESET) and (indices[0] < indices[1]):
            local_u = permute_systems(local_u, [1, 0])
        elif (circuit.data.name in _THREE_QUBIT_GATESET) and (indices[0] < indices[2]):
            local_u = permute_systems(local_u, [2, 1, 0])
        V_i = expand(local_u, indices, n)
        V_i = permute_systems(V_i, perm=list(reversed(range(n))))
        U = V_i @ U
    return U


def circuit_to_state(qp: Union[QCompute.QEnv, List[CircuitLine]], vector: bool = False,
                     qubits: List[int] = None) -> np.ndarray:
    r"""Compute the quantum state generated by the quantum circuit.

    When computing  quantum state, we assume the initial input state is the all-zero state: :math:`\vert 0\cdots 0\rangle`.
    We first compute the unitary matrix of the quantum circuit using ``circuit_to_unitary``,
    and then use it to left multiply the all-zero state vector.

    :param qp: Union[QProgram, List[CircuitLine]], the state generating quantum circuit
    :param vector: bool, if ``True``, the return value is a state vector (a :math:`2^n\times 1` matrix),
                         where :math:`n` is the number of qubits;       
                         if ``False``, the return value is a density operator (a :math:`2^n\times 2^n` matrix)
    :param qubits: List[int], the target qubit(s), default to None
    :return: np.ndarray, the target quantum state (state vector or density operator)

    **Examples**

        We use the Bell state preparation as example.

        >>> qp = QEnv()
        >>> qp.Q.createList(2)
        >>> H(qp.Q[0])
        >>> CX(qp.Q[0], qp.Q[1])
        >>> circuit_to_state(qp, vector=True, qubits=[0])
        [[0.70710678+0.j]
         [0.        +0.j]
         [0.        +0.j]
         [0.70710678+0.j]]
    """
    # Deep copy the quantum program and remove the measurement if there exists any
    qp = copy.deepcopy(qp)
    remove_measurement(qp)

    if qubits is None:
        qubits = [i for i in range(len(qp.Q.registerMap.keys()))]

    _TWO_QUBIT_GATESET = {'CX', 'CY', 'CZ', 'CH', 'SWAP'}
    _THREE_QUBIT_GATESET = {'CCX', 'CSWAP'}

    n = len(qubits)
    state = basis(n, 0)
    for circuit in qp.circuit:
        indices = [qubits.index(i) for i in circuit.qRegList]
        local_u = circuit.data.getMatrix()
        # In QCompute, all multi-qubit gates' matrix is fixed as qubit i controls qubit i-1,
        # we thus must analyze the control qubit and permute the system.
        if (circuit.data.name in _TWO_QUBIT_GATESET) and (indices[0] < indices[1]):
            local_u = permute_systems(local_u, [1, 0])
        elif (circuit.data.name in _THREE_QUBIT_GATESET) and (indices[0] < indices[2]):
            local_u = permute_systems(local_u, [2, 1, 0])
        V_i = expand(local_u, indices, n)
        V_i = permute_systems(V_i, perm=list(reversed(range(n))))
        state = V_i @ state

    if vector:
        return state
    else:
        return state @ dagger(state)


def group_gate_indices(circuit: List[CircuitLine]) -> Tuple[List[int], List[int]]:
    r"""Group the gate indices of a quantum circuit into single-qubit and multi-qubits quantum gates.

    The quantum circuit is described by a sequential list of ``CircuitLine`` objects.
    Each ``CircuitLine`` represents a quantum gate and is indexed from 0.
    We scan the ``CircuitLine`` instances and group their indices
    by the number of qubits that the quantum gate operates.
    Notice that Measurement and Barrier gates are excluded.

    :param circuit: List[CircuitLine], a quantum circuit
    :return: Tuple, a tuple of two lists: a list of ``one_qubit_gate_indices``
                and a list of ``multi_qubit_gate_indices`` in the given quantum circuit

    **Examples**

        >>> qp = QEnv()
        >>> qp.Q.createList(2)
        >>> H(qp.Q[0])
        >>> CX(qp.Q[0], qp.Q[1])
        >>> S(qp.Q[0])
        >>> CZ(qp.Q[0], qp.Q[1])
        >>> MeasureZ(*qp.Q.toListPair())
        >>> one_qubit_gate_indices, multi_qubit_gate_indices = group_gate_indices(qp.circuit)
        >>> print_circuit(qp.circuit)
        0: ---H---@---S---@---MEAS---
                  |       |
        1: -------X-------Z---MEAS---
        >>> one_qubit_gate_indices
        [0, 2]
        >>> multi_qubit_gate_indices
        [1, 3]
    """
    circuit_raw = copy.deepcopy(circuit)
    # remove measurement and barriers from the circuit
    remove_measurement(circuit_raw)
    remove_barrier(circuit_raw)

    one_qubit_gate_indices = []  # store single-qubit gate indices
    multi_qubit_gate_indices = []  # store multi-qubit gate indices
    for i, cl in enumerate(circuit_raw):
        # get the number of qubits in current gate
        num_of_qubits = len(cl.qRegList)
        if num_of_qubits == 1:
            one_qubit_gate_indices.append(i)
        else:
            multi_qubit_gate_indices.append(i)

    return one_qubit_gate_indices, multi_qubit_gate_indices


def random_circuit(qubits: List[int],
                   cycle: int,
                   single: List[str] = None,
                   multi: List[str] = None,
                   neighboring: bool = True) -> QProgram:
    r"""Construct a random circuit of given cyles.

    Construct a random circuit of given cyles, each cycle is composed of two layers:

    + The first layer is composed of single-qubit gates randomly chosen from the set @single.
      For each qubit, randomly choose a single-qubit gate and operate.
      The supported single-qubit gate names are: `['X', 'Y', 'Z', 'ID', 'H', 'T', 'S', 'TDG', 'SDG', 'U']`,
      where 'U' stands for the 'U3' gate. If ``single=None``, use `['U']` by default.

    + The second layer is composed of multi-qubit gates randomly chosen from the set @multi.
      First randomly choose the qubit indices and then randomly choose a multi-qubit gate to operate.
      The supported multi-qubit gate names are: `['CX', 'CZ', 'CH', 'CY']`.
      If ``multi=None``, use `['CZ']` by default.

    **Usage**

    .. code-block:: python
        :linenos:

        qp = random_circuit(qubits=[0, 1], cycle=5)
        qp = random_circuit(qubits=[0, 2], cycle=5, single=['X','H','S','U'])
        qp = random_circuit(qubits=[0, 2], cycle=5, single=['X','H','S','U'], multi=['CZ'])

    :param qubits: List[int], the list of qubits
    :param cycle: int, the number of cycles
    :param single: List[str], list of single-qubit gate names
    :param multi: List[str], list of multi-qubit qubit gate names
    :param neighboring: bool, default to `True`, indicating that only neighbor qubits can perform two-qubit gates
    :return: QProgram, the constructed random quantum circuit

    **Examples**

        >>> from qcompute_qep.utils import circuit
        >>> qubits = [0, 1]
        >>> qp = circuit.random_circuit(qubits=qubits, cycle=2)
        >>> circuit.print_circuit(qp.circuit)
        0: ---U3(+1.13,+1.71,-2.81)---X---U3(+3.05,+0.15,-1.72)---X---MEAS---
                                      ｜                          ｜
        1: ---U3(-2.14,+1.98,-0.17)---@---U3(+1.53,-2.60,-2.63)---@---MEAS---
    """

    _SINGLE_QUBIT_GATE_SET = {'X', 'Y', 'Z', 'ID', 'H', 'T', 'S', 'TDG', 'SDG', 'RX', 'RY', 'RZ', 'U'}
    _MULTI_QUBIT_GATE_SET = {'CX', 'CZ', 'CH', 'CY'}
    if single:
        if all((gate in _SINGLE_QUBIT_GATE_SET) for gate in single):
            _single_qubit_gateset = single
        else:
            raise ArgumentError("in random_circuit(): "
                                "Supported single-qubit gates are: {}.".format(_SINGLE_QUBIT_GATE_SET))
    else:
        _single_qubit_gateset = ['U']

    if multi:
        if all((gate in _MULTI_QUBIT_GATE_SET) for gate in multi):
            _multi_qubit_gateset = multi
        else:
            raise ArgumentError("in random_circuit(): "
                                "Supported multi-qubit gates are {}.".format(_MULTI_QUBIT_GATE_SET))
    else:
        _multi_qubit_gateset = ['CZ']

    # Create environment
    env = QCompute.QEnv()

    # Initialize the circuit
    num_qubits = max(qubits) + 1
    q = env.Q.createList(num_qubits)

    def per_cycle_gate(n_qubits: int):
        # single-qubit gate layer in current cycle
        for i in range(n_qubits):
            gate_name = np.random.choice(_single_qubit_gateset)
            if gate_name == 'U':
                # randomly generate a 2*2 unitary matrix and use YZY decomposition
                u = unitary_group.rvs(2)
                alpha, theta, phi, lam = decompose_yzy(u)
                angles = [phi, theta, lam]
                gate = RotationGate.createRotationGateInstance('U', *angles)
            elif gate_name == 'RX' or gate_name == 'RY' or gate_name == 'RZ':
                # randomly generate a rotation angle in :math:`[0, 2\pi]`
                theta = 2 * np.pi * random.random()
                gate = RotationGate.createRotationGateInstance(gate_name, theta)
            else:
                gate = FixedGate.getFixedGateInstance(gate_name)
            gate(q[qubits[i]])

        # multi-qubit gate layer in current cycle
        if not neighboring:
            # Recursively and randomly generate all exclusive qubit pairs
            def qubit_pairs_exclusive(qubits_list: List[int]) -> List[Tuple[int, int]]:
                if len(qubits_list) <= 1:
                    return []
                else:
                    qubit_pair = random.sample(qubits_list, 2)
                    return [qubit_pair] + qubit_pairs_exclusive(qubits_list=[idx for idx in qubits_list
                                                                             if idx not in qubit_pair])

            qubit_pairs = qubit_pairs_exclusive(copy.deepcopy(qubits))
        else:  # only neighbor qubits can execute two-qubit gates
            # Sort the qubits
            qubits.sort()

            # Recursively and randomly generate all qubit neighboring pairs
            def qubit_pairs_neighboring(qubits_list: List[int]) -> List[Tuple[int, int]]:
                if len(qubits_list) <= 1:
                    return []
                else:
                    idx_c = random.randrange(0, len(qubits_list) - 1, 1)

                    if abs(qubits_list[idx_c] - qubits_list[idx_c + 1]) <= 1:
                        qubit_pair = [qubits_list[idx_c], qubits_list[idx_c + 1]]
                    else:
                        qubit_pair = None

                    # Remove the chosen qubits
                    qubits_list.pop(idx_c)
                    qubits_list.pop(idx_c)

                    if qubit_pair is None:
                        return qubit_pairs_neighboring(qubits_list=qubits_list)
                    else:
                        return [qubit_pair] + qubit_pairs_neighboring(qubits_list=qubits_list)

            qubit_pairs = qubit_pairs_neighboring(copy.deepcopy(qubits))

        for pair in qubit_pairs:
            # pick a two qubit gate from the list randomly
            gate = FixedGate.getFixedGateInstance(np.random.choice(_multi_qubit_gateset))
            gate(q[pair[0]], q[pair[1]])

    for m_cyc in range(cycle):
        per_cycle_gate(len(qubits))

    # Add measurements
    if not contain_measurement(env):
        index_list: list[int]
        qreg_list, index_list = env.Q.toListPair()
        QCompute.MeasureZ(qRegList=[qreg_list[x] for x in qubits],
                          cRegList=[index_list[x] for x in qubits])
    return env


def ideal_probability(qp: QProgram, bitstring: str) -> float:
    r"""Compute the theoretical output probability of a given bitstring generated by a quantum circuit.

    We aim to compute theoretically the probability of measuring bitstring
    when evolving from the computational basis state :math:`\vert 0\cdots 0\rangle` by the quantum program.
    Let :math:`U` be the quantum program and let :math:`x` be the bitstring,
    then the theoretical probability of obtaining :math:`x` is given by

    .. math:: P(x) = \vert \langle x \vert U \vert 0\cdots 0\rangle\vert^2.

    :param qp: QProgram, describes the quantum program
    :param bitstring: str, specifies the outcome bitstring
    :return: float, the ideal probability of obtaining the given outcome bitstring
    """
    prob = np.abs(circuit_to_state(qp, vector=True)) ** 2
    index = int(bitstring, 2)
    if index >= len(prob):
        raise ArgumentError("the given bitstring {} is wrong.".format(bitstring))
    return prob[index]


def map_qubits(qp: QProgram, qubits: List[int]) -> QProgram:
    r"""Map the qubits in `qp` based on the qubit order `qubits`.

    :param qp: QProgram, the quantum program whose qubits to be mapped
    :param qubits: List[int], the list of target qubits
    :return: QProgram, the quantum program whose qubits are mapped
    """
    n = len(qp.Q.registerMap.keys())

    # Create a new quantum program with maximal number of qubits
    new_qp = QCompute.QEnv()
    new_qp.Q.createList(max(qubits) + 1 if n < max(qubits) + 1 else n)
    qubits.sort()  # The list of qubit indices must be sorted

    # Map the qubit indices for each CircuitLine in qp
    for cl in qp.circuit:
        qreg_list = [qubits[i] for i in cl.qRegList]
        creg_list = [qubits[i] for i in cl.cRegList] if cl.cRegList is not None else None
        new_qp.circuit.append(CircuitLine(data=cl.data, qRegList=qreg_list, cRegList=creg_list))

    # Map the measured qubits. This is very important!
    measured_qreg = qp.measuredQRegSet if qp.measuredQRegSet is not None else None
    if measured_qreg is not None:
        indices = [list(qp.Q.registerMap.keys()).index(i) for i in measured_qreg]
        measured_qreg = [qubits[i] for i in indices]
    new_qp.measuredQRegSet = set(measured_qreg)

    measured_creg = qp.measuredCRegSet if qp.measuredCRegSet is not None else None
    if measured_creg is not None:
        indices = [list(qp.Q.registerMap.keys()).index(i) for i in measured_creg]
        measured_creg = [qubits[i] for i in indices]
    new_qp.measuredCRegSet = set(measured_creg)

    return new_qp


def enlarge_circuit(qp: QProgram, extra: int) -> QProgram:
    r"""Enlarge the quantum circuit by adding extra number of qubits.

    The enlarging procedure will create a new quantum program with :math:`n+k` qubits,
    where :math:`n` is the number of qubits of @qp and :math:`k` is the number of extra qubits.
    The first :math:`n` qubits are evolved in the same way as that of @qp.

    :param qp: QProgram, the quantum program to be enlarged
    :param extra: int, the extra number of qubits
    :return: QProgram, the enlarged quantum program, with :math:`n+k` qubits
    """
    if extra == 0:
        return copy.deepcopy(qp)

    new_qp = QCompute.QEnv()
    new_qp.Q.createList(number_of_qubits(qp) + extra)
    # Copy the relevant properties
    new_qp.Parameter = copy.deepcopy(qp.Parameter)
    new_qp.circuit = qp.circuit
    new_qp.procedureMap = qp.procedureMap
    new_qp.program = qp.program
    new_qp.backendName = qp.backendName
    new_qp.backendArgument = qp.backendArgument
    new_qp.shots = qp.shots
    new_qp.usingModuleList = qp.usingModuleList
    new_qp.usedServerModuleList = qp.usedServerModuleList
    new_qp.noiseDefineMap = qp.noiseDefineMap

    return new_qp


def execute(qp: Union[QProgram, List[QProgram]],
            qc: QComputer, **kwargs) -> Union[Dict[str, int], List[Dict[str, int]]]:
    r"""Execute a quantum program or a list of quantum programs on a quantum computer.

    This method supports the following two scenarios:

    + Run a single quantum program on the quantum computer (single model),
      and return the measurement outcomes in a dictionary.
    + Run a list of quantum programs on the quantum computer (batch model),
      and return the measurement outcomes in a list of dictionaries.

    Specifically, there exists a one-one mapping between the quantum programs and dictionaries:

    + [qp_1, qp_2, qp_3, ...]
    + [counts_1, counts_2, counts_3, ...]

    where each @qp_k is a quantum program, each @counts_k is a dictionary of `Dict[str, int]`,
    which corresponds to the measurement outcomes obtained by running @qp_k on the given quantum computer.

    Currently, supported ``QProgram`` and ``QComputer`` combinations are:

    + ``isinstance(qp, QCompute.QEnv)`` and ``isinstance(qc, QCompute.BackendName)``;
    + ``isinstance(qp, QCompute.QEnv)`` and ``isinstance(qc, qiskit.providers.Backend)``;
    + ``isinstance(qp, QCompute.QEnv)`` and ``isinstance(qc, Callable)``; and
    + ``isinstance(qp, qiskit.QuantumCircuit)`` and ``isinstance(qc, qiskit.providers.Backend)``.

    .. admonition:: Novel

        The quantum computer can be a Callable type, i.e., any interface implemented as follows

        .. code-block:: python
            :linenos:

            def quantum_computer(qps: List[QCompute.QEnv], **kwargs) -> List[Dict[str, int]]

        can serve as a valid quantum computer. In our `execute()` function, we can use the *callable*
        quantum computer to run the quantum programs. For detailed usage, see the Examples below.

    We support the following keyword argument:

    + `shots`: default to :math:`1024`, the number of shots each measurement should carry out
    + ``optimization_level``, defaults to :math:`0`, indicates the optimization level
      that the quantum circuit will be compiled. This parameter will be used when the ``qc``
      is an instance of ``qiskit.providers.Backend``.

    :param qp: QProgram, the target quantum program or a list of quantum programs
    :param qc: QComputer, the target quantum computer on which the quantum program will be executed
    :return: Union[Dict[str, int], List[Dict[str, int]]], a dictionary or a list of dictionaries
                        recording the measurement counts

    **Examples**

        >>> from QCompute import *
        >>> # qp1 creates a Bell state and measure in Z basis
        >>> qp1 = QEnv()
        >>> qp1.Q.createList(2)
        >>> H(qp1.Q[0])
        >>> CX(qp1.Q[0], qp1.Q[1])
        >>> MeasureZ(*qp1.Q.toListPair())
        >>> # qp2 creates a three-qubit GHZ state and measure in Z basis
        >>> qp2 = QEnv()
        >>> qp2.Q.createList(3)
        >>> H(qp2.Q[0])
        >>> CX(qp2.Q[0], qp2.Q[1])
        >>> CX(qp2.Q[0], qp2.Q[2])
        >>> MeasureZ(*qp2.Q.toListPair())
        >>> # Case 1: Execute the quantum programs in a batch model
        >>> counts_list = execute(qp=[qp1, qp2], qc=BackendName.LocalBaiduSim2, shots=1024)
        >>> for counts in counts_list:
        >>>     print(counts)
        {'11': 515, '00': 509}
        {'111': 485, '000': 539}
        >>> # Case 2: Execute the quantum programs in a callable quantum computer. We first define a callable interface
        >>> def quantum_computer(qps: List[QCompute.QEnv], **kwargs) -> List[Dict[str, int]]:
        >>>     qc_name = BackendName.LocalBaiduSim2
        >>>     shots: int = kwargs.get('shots', 1024)
        >>>     res = []
        >>>     for qp in qps:
        >>>         qp.backend(qc_name)
        >>>         result = qp.commit(shots=shots, fetchMeasure=True)
        >>>         res.append(result["counts"])
        >>>     return res
        >>> counts_list = execute(qp=[qp1, qp2], qc=quantum_computer, shots=2048)
        >>> for counts in counts_list:
        >>>     print(counts)
        {'11': 992, '00': 1056}
        {'111': 1015, '000': 1033}
    return res
    """
    # Record if the input @qp is a QuantumProgram or a list of QuantumPrograms
    _IS_QP_LIST = isinstance(qp, list)
    # If the input @qp is just a QuantumProgram, convert it to a `List[QuantumProgram]` type
    qps = qp if _IS_QP_LIST else [qp]

    # Replace self-defined high-level controlled Pauli gates with native gates if there is any
    # !WARNING! This piece of code must be reserved to decompose the `CPauliOP` gate since it is self-defined in QEP.
    for qp in qps:
        old_circuit = copy.deepcopy(qp.circuit)
        qp.circuit = []
        for i, cl in enumerate(old_circuit):
            cl = copy.deepcopy(cl)
            if isinstance(cl.data, CPauliOP):
                qp.circuit.extend(decompose_CPauli(cl.data))
            else:
                qp.circuit.append(cl)

    # Handle the computing task case by case based on the type of the quantum computer.
    # Case 1. the quantum computer is a QCompute.BackendName instance
    if isinstance(qc, QCompute.BackendName):
        # Each QuantumProgram must be a QCompute.QEnv instance, otherwise cannot be executed in QCompute.BackendName
        if any(not isinstance(qp, QCompute.QEnv) for qp in qps):
            raise ArgumentError("in execute(): since the quantum computer is a 'QCompute.BackendName' instance, "
                                "all Quantum Programs must be the 'QCompute.QEnv' type!")

        res = _execute_in_qcompute(qps=qps, qc=qc, **kwargs)
    # Case 2. the quantum computer is a QCompute.BackendName instance
    elif isinstance(qc, qiskit.providers.Backend):
        converted_qps = []
        # Convert quantum programs to the qiskit.QuantumCircuit type
        for qp in qps:
            if isinstance(qp, qiskit.QuantumCircuit):
                converted_qps.append(qp)
            elif isinstance(qp, QCompute.QEnv):
                converted_qps.append(conversion.from_qcompute_to_qiskit(qp))
            else:
                raise ArgumentError("in execute(): since the quantum computer is a 'qiskit.providers.Backend' "
                                    "instance, all Quantum Programs must be either "
                                    "the 'QCompute.QEnv' type or the 'qiskit.QuantumCircuit' type!")

        res = _execute_in_qiskit(qps=converted_qps, qc=qc, **kwargs)
    # Case 3. the quantum computer is a Callable instance
    elif isinstance(qc, Callable):
        res = _execute_in_callable(qps=qps, qc=qc, **kwargs)
    else:  # Currently, other quantum computer instances are not supported
        raise ArgumentError("in execute(): currently this method does not support specifying {} as "
                            "a 'QuantumComputer' instance!".format(qc))

    # If @qp is a QuantumProgram, convert the output to a dictionary of counts (but not a list of dictionaries)
    if not _IS_QP_LIST:
        assert len(res) == 1, "The number of input quantum programs is 1, " \
                              "so the number of output count dictionary must be 1!"
        res = res[0]
    return res


def _execute_in_qcompute(qps: List[QCompute.QEnv],
                         qc: QCompute.QPlatform.BackendName, **kwargs) -> List[Dict[str, int]]:
    r"""Execute quantum programs on a quantum computer from QCompute.

    The measurement outcomes will be returned in a list of dictionaries, each corresponds to a quantum program.
    Specifically, there exists a one-one mapping between the quantum programs and dictionaries:

    + [qp_1, qp_2, qp_3, ...]
    + [counts_1, counts_2, counts_3, ...]

    where each @qp_k is a quantum program, each @counts_k is a dictionary of `Dict[str, int]`,
    which corresponds to the measurement outcomes obtained by running @qp_k on the given quantum computer.

    Support the following keyword argument:

    + `shots`: default to :math:`1024`, the number of shots each measurement should carry out.

    :param qp: List[QCompute.QEnv], a list of quantum programs, each is a QCompute.QEnv instance
    :param qc: QCompute.QPlatform.BackendName, the target quantum computer
    :return: List[Dict[str, int]], a list of dictionaries recording the measurement counts for each quantum program
    """
    shots: int = kwargs.get('shots', 1024)
    res = []

    for qp in qps:
        # Set the backend
        qp.backend(qc)
        # Remove barrier gates, forbid mapping and enable unroll
        remove_barrier(qp)
        if qc == QCompute.BackendName.CloudIoPCAS:
            qp.serverModule(QCompute.ServerModule.MappingToIoPCAS, {'disable': False})
            qp.serverModule(QCompute.ServerModule.UnrollCircuitToIoPCAS, {'disable': False})
        elif qc == QCompute.BackendName.CloudBaiduQPUQian:
            qp.serverModule(QCompute.ServerModule.MappingToBaiduQPUQian, {'disable': False})
            qp.serverModule(QCompute.ServerModule.UnrollCircuitToBaiduQPUQian, {'disable': False})
        elif qc == QCompute.BackendName.CloudIonAPM:
            qp.serverModule(QCompute.ServerModule.UnrollCircuitToIonAPM, {'disable': False})
        else:
            pass

        # Commit the computation task and fetch the results
        result = qp.commit(shots=shots, fetchMeasure=True)
        # Obtain the 'counts' information for the computation result
        res.append(result["counts"])

    return res


def _execute_in_qiskit(qps: List[qiskit.QuantumCircuit],
                       qc: qiskit.providers.Backend, **kwargs) -> List[Dict[str, int]]:
    r"""Execute quantum programs on a quantum computer from qiskit.

    The measurement outcomes will be returned in a list of dictionaries, each corresponds to a quantum program.
    Specifically, there exists a one-one mapping between the quantum programs and dictionaries:

    + [qp_1, qp_2, qp_3, ...]
    + [counts_1, counts_2, counts_3, ...]

    where each @qp_k is a quantum program, each @counts_k is a dictionary of `Dict[str, int]`,
    which corresponds to the measurement outcomes obtained by running @qp_k on the given quantum computer.

    Support the following keyword argument:

    + `shots`: default to :math:`1024`, the number of shots each measurement should carry out.
    + ``optimization_level``, defaults to :math:`0`, indicates the optimization level
      that the quantum circuit will be compiled. This parameter will be used when the ``qc``
      is an instance of ``qiskit.providers.Backend``.

    :param qp: List[qiskit.QuantumCircuit], a list of quantum programs, each is a qiskit.QuantumCircuit instance
    :param qc: QCompute.QPlatform.BackendName, the target quantum computer
    :return: List[Dict[str, int]], a list of dictionaries recording the measurement counts for each quantum program
    """
    shots: int = kwargs.get('shots', 1024)
    # If the optimization level is set by the user, apply it
    optimization_level: int = kwargs.get('optimization_level', 0)

    res = []
    if isinstance(qc, IBMQBackend):
        qps = transpile(qps, backend=qc)
        job_manager = IBMQJobManager()
        # Run quantum program in qiskit quantum computer and get outcomes
        results = job_manager.run(qps, backend=qc, shots=shots).results()
        for i in range(len(qps)):
            res.append(results.get_counts(i))
    else:
        res = qiskit.execute(qps, qc, shots=shots, optimization_level=optimization_level).result().get_counts()
        if len(qps) == 1:
            res = [res]
    return res


def _execute_in_callable(qps: List[QCompute.QEnv], qc: Callable, **kwargs) -> List[Dict[str, int]]:
    r"""Execute quantum programs on a callable quantum computer.

    Unlike `_execute_in_qcompute()` and `_execute_in_qiskit()`, where the quantum computers are from
    known manufacturers, we allow the users to use Callable quantum computer interface and run the
    quantum programs. In this way, we have the advantage that the users can hide their hardware information
    when using the functionalities supported by QEP. The interface must be called as

    .. code-block:: python

        qc(qps, **kwargs)

    where @qps is a list of quantum programs and each quantum program is `QCompute.QEnv` type.
    We require that the return values of the interface be `List[Dict[str, int]]`, which
    records the measurement outcomes, each dictionary corresponds to a quantum program.
    Specifically, there exists a one-one mapping between the quantum programs and dictionaries:

    + [qp_1, qp_2, qp_3, ...]
    + [counts_1, counts_2, counts_3, ...]

    where each @qp_k is a quantum program, each @counts_k is a dictionary of `Dict[str, int]`,
    which corresponds to the measurement outcomes obtained by running @qp_k on the given quantum computer.

    Support the following keyword argument:

    + `shots`: default to :math:`1024`, the number of shots each measurement should carry out.
    + ``optimization_level``, defaults to :math:`0`, indicates the optimization level
      that the quantum circuit will be compiled. This parameter will be used when the ``qc``
      is an instance of ``qiskit.providers.Backend``.

    :param qp: List[qiskit.QuantumCircuit], a list of quantum programs, each is a qiskit.QuantumCircuit instance
    :param qc: Callable, a callable quantum computer
    :return: List[Dict[str, int]], a list of dictionaries recording the measurement counts for each quantum program
    """
    return qc(qps, **kwargs)
