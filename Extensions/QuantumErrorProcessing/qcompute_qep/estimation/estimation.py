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
This file implements the Abstract Class of the `Quantum Estimation` method.
"""
import abc
from typing import Any, List

from qcompute_qep.utils.types import QProgram, QComputer


class Estimation(abc.ABC):
    """The Estimation abstract class.

    Quantum estimation is an experimental procedure to estimate the quality of a quantum system
    from measurement outcomes of a specific set of experiments.

    The implementations---``DFEState``, ``DFEProcess``, ``DFEMeasurement``,
    and ``CPEState``---must inherit this abstract class.
    """
    def __init__(self, qp: QProgram = None, qc: QComputer = None, **kwargs: Any):
        r"""Init function of the Quantum Estimation class.

        Optional keywords list are:

            + `qubits`: default to the complete set of qubits, specify the set of qubits to be estimated
            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out

        :param qp: QProgram, quantum program for creating the target quantum state
        :param qc: QComputer, the quantum computer
        """
        self._qp: QProgram = qp
        self._qc: QComputer = qc
        self._qubits: List[int] = None
        self._shots: int = None

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.estimate(*args, **kwargs)

    @abc.abstractmethod
    def estimate(self, qp: QProgram = None, qc: QComputer = None, **kwargs) -> float:
        r"""Every implementation of the Estimation class must inherit the `estimate` function.

        The `estimate` function aims to estimate the fidelity of the target quantum object.

        Optional keywords list are:

            + `qubits`: default to the complete set of qubits, specify the set of qubits to be estimated
            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out

        :param qp: QProgram, quantum program for creating the target quantum state
        :param qc: QComputer, the quantum computer instance
        :return: float, the estimated fidelity
        """
        raise NotImplementedError
