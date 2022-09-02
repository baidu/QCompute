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
The abstract `RandomizedBenchmarking` class (`Randomized Benchmarking`) for the randomized benchmarking module.
The implementations---`StandardRB`, `UnitarityRB`, and `CrossEntropyRB`---must inherit this abstract class.
"""
import abc
from typing import Any, Tuple
import numpy as np
import QCompute
import functools

from qcompute_qep.utils.types import QProgram, QComputer, number_of_qubits
import qcompute_qep.utils.circuit as circuit


class RandomizedBenchmarking(abc.ABC):
    r"""The Randomized Benchmarking Abstract Class.

    Each inherited class must implement the ``benchmark`` method.
    """

    def __init__(self, **kwargs: Any):
        r"""
        The init function of the `RandomizedBenchmarking` class. Optional keywords list are:

        + ``method``: default to 'inverse', specify the tomography method
        + ``shots``: default to :math:`4096`, the number of shots each measurement should carry out
        + ``ptm``: default to 'False', the quantum object should be in the Pauli transfer matrix form

        :param qp: QProgram, quantum program for creating the target quantum state
        :param qc: QComputer, the quantum computer
        """
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.benchmark(*args, **kwargs)

    @property
    def results(self):
        return

    @property
    def params(self):
        return

    @abc.abstractmethod
    def _fit_func(self, **kwargs) -> np.ndarray:
        r"""The target fit function used in randomized benchmarking.

        A commonly used exponential fit function is as follows:

        .. math:: f(x) = A f^x + B

        where :math:`x` is the estimated expectation value and :math:`f` is the target fidelity parameter.
        """
        pass

    @abc.abstractmethod
    def plot_results(self, **kwargs):
        r"""Plot randomized benchmarking results.

        Commonly, we would visualize the sampled and averaged expectation values for each given length,
        the fitted function.
        """
        pass

    @abc.abstractmethod
    def benchmark(self, **kwargs: Any) -> Any:
        r"""Abstract benchmark method of the Randomized Benchmarking Abstract Class.

        Every implementation of the `RandomizedBenchmarking` class must implement the `benchmark` function,
        which aims to randomized benchmark the target quantum device.
        """
        raise NotImplementedError


def default_prep_circuit(qp: QProgram) -> QProgram:
    r"""Modify the quantum circuit to prepare the initial state.

    This initial quantum state should be prepared by a quantum program.
    For each preparation operator in the complete preparation basis,
    decorate the given quantum program by adding the preparation quantum circuit
    to the beginning of the original quantum program.
    We assume the LSB (the least significant bit) mode, i.e., the right-most bit represents q[0]:

    ::

        "X        Y        I"
        q[2]    q[1]      q[0]

    This assumption is important when constructing the preparation circuits.

    .. admonition:: Examples

        If the original quantum program is the single-qubit H gate

            0: ---H---

        then the four decorated quantum programs are:

            0: ---H---              Prepare the `0` state :math:`\vert 0\rangle`

            0: ---X---H---          Prepare the `1` state :math:`\vert 1\rangle`

            0: ---H---H---          Prepare the `A` state :math:`\frac{\vert 0\rangle+\vert 1\rangle}{\sqrt2}`

            0: ---H---S---H---      Prepare the `L` state :math:`\frac{\vert 0\rangle+\vert i\rangle)}{\sqrt2}`

    :param qp: QProgram, the original quantum program
    :return: QProgram, decorated from the original quantum program by adding the
                        quantum state preparation circuit to the beginning
    """
    return qp


def default_meas_circuit(qp: QProgram) -> Tuple[QProgram, np.ndarray]:
    r"""Modify the quantum circuit to measure a given basis.

    Specify the quantum observable and its quantum measurement of the randomized benchmarking protocol.
    For each measurement operator in the complete measurement basis,
    decorate the given quantum program (without measurement) by adding the measurement specified
    by the operator to the end and construct the corresponding quantum observable.
    We assume the LSB (the least significant bit) mode, i.e., the right-most bit represents q[0]:

    ::

        "X        Y        I"
        q[2]    q[1]      q[0]

    This assumption is important when constructing the measurement circuits.

    .. admonition:: Examples

        Since the qubit is measured in `Z` basis by default,
        if we aim to measure the qubit in `X` basis, we can modify the quantum program as follows:

            0: ---H---MEAS---

    :param qp: QProgram, the original quantum program (without measurement)
    :return: Tuple[QProgram, np.ndarray], the modified quantum program with the target quantum measurement
                appended to the end and its corresponding quantum observable
    """
    # If the given quantum program does not contain a measurement, measure it in the Z basis
    qubit_indices = []
    for cl in qp.circuit:
        qubit_indices.extend(cl.qRegList)
    qubit_indices = np.unique(qubit_indices)

    if not circuit.contain_measurement(qp):
        qreglist, indexlist = qp.Q.toListPair()
        QCompute.MeasureZ(qRegList=[qreglist[x] for x in qubit_indices],
                          cRegList=[indexlist[x] for x in qubit_indices])

    # Set the quantum observable observable to :math:`\vert 0\cdots 0\rangle\!\langle 0\cdots 0\vert`
    n = len(qubit_indices)
    proj0 = np.array([[1, 0], [0, 0]]).astype(float)
    A = functools.reduce(np.kron, [proj0] * n)
    return qp, A
