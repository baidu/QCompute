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

"""Conversion functions between different quantum programming platforms
supported in the `qcompute_qep` module."""
import copy
import numpy as np
import datetime
from io import StringIO
from typing import List, Tuple
import qiskit

import QCompute
from QCompute.QPlatform.QOperation import CircuitLine
from QCompute.QPlatform import QOperation
from qcompute_qep.utils import limit_angle
import qcompute_qep.utils.circuit

# Currently supported quantum gates in OpenQASM 2.0
__QASM_SUPPORTED_GATES__ = {
    "barrier",
    "measure",
    "reset",
    "u3",
    "u2",
    "u1",
    "cx",
    "id",
    "u0",
    "u",
    "p",
    "x",
    "y",
    "z",
    "h",
    "s",
    "sdg",
    "t",
    "tdg",
    "rx",
    "ry",
    "rz",
    "sx",
    "sxdg",
    "cz",
    "cy",
    "swap",
    "ch",
    "ccx",
    "cswap",
    "crx",
    "cry",
    "crz",
    "cu1",
    "cp",
    "cu3",
    "csx",
    "cu",
    "rxx",
    "rzz",
    "rccx",
    "rc3x",
    "c3x",
    "c3sx",
    "c4x",
}


class QASMStringIO(StringIO):
    """Class implemented from io.StringIO class."""

    def __init__(self, *args, **kwargs):
        super(QASMStringIO, self).__init__(*args, **kwargs)

    def write(self, __s: str) -> int:
        """Automatically add a newline character after given text content.

        :param __s: string type, text content to be written into the current StringIO instance
        :return: number of characters
        """
        val_content = super(QASMStringIO, self).write(__s)
        val_tail = super(QASMStringIO, self).write(';\n')
        return val_content + val_tail

    def write_operation(self, opr: str, qreg_name: str, *args) -> int:
        """Write the command of one quantum operation.

        :param opr: string type, quantum operation name
        :param qreg_name: quantum register name. e.g. 'q' or 'qreg'
        :param args: indices of qubits
        """
        if len(args) == 0:
            line = opr + ' ' + qreg_name + ';\n'
        else:
            line_list_qubits = []
            for idx in args:
                line_list_qubits.append(qreg_name + '[{}]'.format(idx))
            line = opr + ' ' + ', '.join(line_list_qubits) + ';\n'
        n = super(QASMStringIO, self).write(line)
        return n

    def write_line_gap(self, n: int = 1) -> int:
        """Write one or more blank line(s)."""
        n = super(QASMStringIO, self).write('\n' * n)
        return n

    def write_comment(self, __s: str) -> int:
        """Conveniently add comment into QASM string."""
        n = super(QASMStringIO, self).write('// ' + __s + '\n')
        return n

    def write_header(self) -> int:
        """Write file header, including information of our QEP project and
        predefined information for QASM text."""
        n1 = super(QASMStringIO, self).write('// Generated from QEP v0.1.0\n')
        n2 = super(QASMStringIO, self).write('// Time: {}\n'.format(datetime.datetime.now()))
        n3 = super(QASMStringIO, self).write('\n')
        n4 = super(QASMStringIO, self).write('OPENQASM 2.0;\n')
        n5 = super(QASMStringIO, self).write('include "qelib1.inc";\n')
        n6 = super(QASMStringIO, self).write('\n' * 2)
        return n1 + n2 + n3 + n4 + n5 + n6


def parse_to_tuples(circuit: List[CircuitLine]) -> List[Tuple[str, List[int]]]:
    """Parse a quantum circuit into a list including information of gate
    operation and qubits. e.g. Barrier(q[0], q[1], q[2]), X(q[0]) -->

    [('barrier', [0, 1, 2]), ('X', [0])]

    :param circuit: supported quantum circuit instance in Quantum Leaf
    :return: a list whose elements is a tuple of gate name and quantum register indices
    """
    parsed_list = []
    for cl in circuit:
        gname = cl.data.name.lower()
        if gname not in __QASM_SUPPORTED_GATES__:
            raise ValueError('{} is not a supported gate type in OpenQSAM 2.0'.format(gname))
        if isinstance(cl.data, QOperation.FixedGate.FixedGateOP):
            parsed_list.append((gname, cl.qRegList))
        elif isinstance(cl.data, QOperation.RotationGate.RotationGateOP):
            if gname in {'rx', 'ry', 'rz', 'crx', 'cry', 'crz'}:
                angle = limit_angle(cl.data.argumentList[0])
                opr = '{}({:.2f})'.format(gname, angle)
                parsed_list.append((opr, cl.qRegList))
            else:  # u3, cu3
                gname += '3'
                angles = list(map(limit_angle, cl.data.argumentList))
                opr = '{}({:.2f}, {:.2f}, {:.2f})'.format(gname, *angles)
                parsed_list.append((opr, cl.qRegList))
        elif isinstance(cl.data, QOperation.Barrier.BarrierOP):
            parsed_list.append((gname, cl.qRegList))
        else:
            raise TypeError('{} is not a supported gate type to print'.format(type(cl.data)))

    return parsed_list


def to_qasm(circuit: List[CircuitLine], fname: str = None) -> str:
    """Convert a quantum circuit into QASM string/text representation.

    :param circuit: List[CircuitLine], supported quantum circuit instance in Quantum Leaf
    :param fname: str, name for saving a text-type file
    :return: str, a string describing the QASM representation of the quantum circuit
    """
    circuit = copy.deepcopy(circuit)
    output = QASMStringIO()  # StringIO stream in memory

    # Write version, time, and header information to the qasm file
    output.write_header()

    # Number of qubits in the quantum circuit
    n = qcompute_qep.utils.circuit.num_qubits_of_circuit(circuit)
    qubit_indices = []
    for cl in circuit:
        qubit_indices.extend(cl.qRegList)
    qubit_indices = np.unique(qubit_indices)
    num_of_register_qubits = max(x for x in qubit_indices) + 1

    # Pop the last measurement operation if exists
    measurement = qcompute_qep.utils.circuit.remove_measurement(circuit)

    if measurement is not None:
        output.write_comment('Qubits: {}, Bits: {}'.format(qubit_indices, list(range(n))))
        output.write('qreg q[{}]'.format(num_of_register_qubits))
        output.write('creg c[{}]'.format(n))
    else:
        output.write_comment('Qubits: {}'.format(qubit_indices))
        output.write('qreg q[{}]'.format(num_of_register_qubits))
    output.write_line_gap()

    # Add barrier and computational gates
    tuples_parsed = parse_to_tuples(circuit)
    output.write_comment('Quantum gate operations')
    for opr, idx in tuples_parsed:
        if idx == n:
            output.write_operation(opr, 'q')
        else:
            output.write_operation(opr, 'q', *idx)

    # Add measurement back if we have popped out before
    if measurement is not None:
        output.write_line_gap()
        output.write_comment('Final measurement')
        for i in range(len(qubit_indices)):
            output.write('measure q[{}] -> c[{}]'.format(qubit_indices[i], i))

    qasm_str = output.getvalue()
    output.close()

    if fname is not None:
        with open(fname, 'w') as f:
            f.write(qasm_str)

    return qasm_str


def from_qcompute_to_qiskit(qp: QCompute.QEnv) -> qiskit.QuantumCircuit:
    """Convert a quantum program written in QCompute to qiskit.QuantumCircuit
    type.

    :param qp: QCompute.QEnv, a quantum program written in QCompute
    :return: qiskit.QuantumCircuit, the equivalent quantum program written in qiskit.QuantumCircuit
    """
    qasm_str = to_qasm(qp.circuit)
    qiskit_qp = qiskit.circuit.QuantumCircuit.from_qasm_str(qasm_str)
    return qiskit_qp
