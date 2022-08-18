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
In the ``qcompute_qep`` module, the quantum programs are represented by the `QProgram` data structure
and the quantum computers are represented by the `QComputer` data structure. Currently,

+ `QProgram` supports ``qcompute.qenv`` and ``qiskit.QuantumCircuit`` data types.

+ `QComputer` supports ``qcompute.BackendName`` and ``qiskit.providers.Backend`` quantum computers.

Supports for more types of quantum programs and quantum computers are scheduled.
"""
import re
import qiskit
from typing import Union, Any, get_args, Type
import QCompute

from qcompute_qep.exceptions.QEPError import ArgumentError

QProgram: Type[Union[QCompute.QEnv, qiskit.QuantumCircuit]] = Union[QCompute.QEnv, qiskit.QuantumCircuit]
r"""The Quantum Program data type in ``qcompute_qep``.
"""

QComputer: Type[Union[str, QCompute.QPlatform.BackendName, qiskit.providers.Backend]] \
    = Union[str, QCompute.QPlatform.BackendName, qiskit.providers.Backend]
r"""The Quantum Computer data type in ``qcompute_qep``.
"""

__SUPPORTED_BACKENDS__ = []
for backend in QCompute.QPlatform.BackendName.__members__.values():
    name = backend.__str__()
    __SUPPORTED_BACKENDS__.append(name[name.index('.')+1:])

__SUPPORTED_BACKENDS__ += ["Santiago", "Quito", "Vigo", "Yorktown", "Montreal", "Aer_Simulator"]
r"""The supported quantum backends in ``qcompute_qep``.
"""


def _is_supported_qp_instance(qp: Any) -> bool:
    r"""Check if the given input is a supported quantum program instance.

    Current quantum program supported instances are:

    1. Instances of ``QCompute.QEnv``, and

    2. Instances of ``qiskit.QuantumCircuit``.

    :param qp: Any, the target input quantum program
    :return: bool, if the input is a valid quantum program instance, return True; otherwise, return False
    """
    if isinstance(qp, get_args(QProgram)):
        return True
    else:
        return False


def number_of_qubits(qp: QProgram = None) -> int:
    r"""Number of working qubits of the given quantum program instance.

    Current quantum program supported instances are:

    1. Instances of ``QCompute.QEnv``, and

    2. Instances of ``qiskit.QuantumCircuit``.

    :param qp: QProgram, the target quantum program
    :return: int, the number of qubits involved in the quantum program
    """
    if isinstance(qp, QCompute.QEnv):
        n = len(qp.Q.registerMap.keys())
    elif isinstance(qp, qiskit.QuantumCircuit):
        n = qp.num_qubits
    else:
        raise ArgumentError("in number_of_qubits(): the input {} is not a supported quantum program!".format(qp))
    return n


def _is_valid_qc_instance(qc: Any) -> bool:
    r"""Check if the given input is a supported quantum computer instance.

    Current supported instances are:

    1. Instances of ``QCompute.BackendName``, and

    4. Instances of ``qiskit.providers.Backend``.

    :param qc: Any, the target input quantum program
    :return: bool, if the input is a valid quantum program instance, return True; otherwise, return False
    """
    if isinstance(qc, get_args(QComputer)):
        return True
    else:
        return False


def get_qc_name(qc: Any) -> str:
    r"""Name of the given quantum computer.

    Supported quantum computer names are stored in the global constant: ``__SUPPORTED_BACKENDS__``

    :param qc: Any, the target input quantum program
    :return: str, if the input is a valid quantum program instance, return its name; otherwise, return None
    """
    # if not _is_valid_qc_instance(qc):
    #     raise ArgumentError("in get_qc_name(): the input {} is not an valid quantum computer!".format(qc))

    name = None
    qc_str = qc.__str__().lower()
    for backend in __SUPPORTED_BACKENDS__:
        if backend.lower() in qc_str:
            name = backend
            # Remove underscore if exists
            name = re.sub(r'_', '', name)
            if "fake" in qc_str:
                name = "Fake" + name
            break
    return name
