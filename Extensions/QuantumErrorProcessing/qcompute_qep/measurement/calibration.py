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
Calibration aims at learning parameters of the noise matrix from the experimental data.
A single calibration round initializes the n-qubit register in a basis state :math:`\vert x\rangle`
and performs a noisy measurement of each qubit, keeping the record of the measured outcome :math:`y`.

**Examples preparation**

    >>> import json
    >>> import unittest
    >>> import qiskit
    >>> from QCompute import Define
    >>> from QCompute import BackendName
    >>>
    >>> from qiskit.providers.fake_provider import FakeSantiago
    >>> from qcompute_qep.measurement.calibration import CompleteCalibrator
    >>> from qcompute_qep.measurement.calibration import TPCalibrator
    >>> from qcompute_qep.measurement.calibration import init_complete_cal_circuits
    >>> from qcompute_qep.measurement.calibration import init_tp_cal_circuits
    >>> from qcompute_qep.measurement.calibration import extract_cal_data
    >>> from qcompute_qep.measurement.calibration import load_cal_data
    >>> from qcompute_qep.utils.circuit import print_circuit
    >>>
    >>> # Set the default maximal number of measurement shots
    >>> MAX_SHOTS = 4096
    >>> qc_ideal = BackendName.LocalBaiduSim2
    >>> qc_noisy = BackendName.CloudBaiduQPUQian
    >>> qubits = [1, 2]
"""
import os
import abc
import json
from abc import ABC
from builtins import str
import copy
from typing import Any, Dict, List, Tuple
import numpy as np
from itertools import combinations
from tqdm import tqdm

from QCompute import *
from QCompute.Calibration import CalibrationUpdate, CalibrationReadData
from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.measurement.utils import extract_substr, init_cal_data
from qcompute_qep.utils.types import QProgram, QComputer, get_qc_name
from qcompute_qep.utils.linalg import tensor, normalize, partial_trace
from qcompute_qep.utils.circuit import execute


class Calibrator(ABC):
    """
    Abstract interface `Calibrator` for the Measurement Error Mitigation Calibration.
    Concrete measurement error calibration methods must inherit this abstract class and implement the `calibrate`
    method.
    """

    @abc.abstractmethod
    def __init__(self):
        self._qc: QComputer = None
        self._cal_data: Dict[str, Dict[str, int]] = None
        self._qubits: List[int] = None
        self._cal_matrix: np.ndarray = None
        self._noise_resist: float = None
        self._sync: bool = True

    @property
    def cal_matrix(self):
        if self._cal_matrix is None:
            self.calibrate(qc=self._qc, cal_data=self._cal_data, qubits=self._qubits)
        return self._cal_matrix

    @property
    def noise_resist(self):
        return self.get_noise_resistance()

    @property
    def name(self):
        return self.__class__.__name__

    def __call__(self=None, **kwargs: Any) -> Any:
        return self.calibrate(**kwargs)

    def readout_fidelities(self) -> List[float]:
        r"""
        Compute the readout fidelities. For a basis state :math:`\vert x \rangle`, its readout fidelity
        is defined to be the probability of obtaining outcome x when :math:`\vert x \rangle` as input.

        :return: a list of float numbers representing the readout fidelities
        """

        return list(np.diagonal(self._cal_matrix))

    def average_readout_fidelity(self) -> float:
        r"""
        Compute the average readout fidelity, defined to be the average of the diagonal elements of the
        calibration matrix.
        The average readout fidelity quantifies the average probability that we obtain the outcome x
        when we use basis state :math:`\vert x \rangle` as input.

        :return: a float number representing the average readout fidelity
        """
        diag = np.diagonal(self._cal_matrix)
        return sum(diag) / len(diag)

    @abc.abstractmethod
    def calibrate(self, **kwargs: Any) -> Any:
        """
        Use the calibration data to estimate the calibration matrix.

        The calibration data is a dictionary of dictionary and has type Dict[str, Dict[str, int]], for which

        + the keyword represents the input computational basis state, and
        + the dictionary value represents the counts of the output computational basis states.

        .. code-block:: python

            {
            "00": {
                "00": N_00,
                "01": N_10,
                "10": N_20,
                "11": N_30
                },
            "01": {
                "00": N_01,
                "01": N_11,
                "10": N_21,
                "11": N_31
                },
            "10": {
                "00": N_02,
                "01": N_12,
                "10": N_22,
                "11": N_32
                },
            "11": {
                "00": N_03,
                "01": N_13,
                "10": N_23,
                "11": N_33
                }
            }
        """
        raise NotImplementedError

    def get_noise_resistance(self):
        min_diag = min(np.diagonal(self.cal_matrix))
        if min_diag < 0.5:
            raise ArgumentError(
                "The calibration matrix is invalid! Minimal diagonal element should bigger than 0.5!")

        return 2 * (1 - min_diag)

    def _preprocess_inputs(self):
        """
        Preprocess the input data: self._qc, self._cal_data, and self._qubits to make them consistent.
        More precisely, the number of qubits specified by @self._qubits must be consistent
        with the number of qubits specified by @self._cal_data. If they are not consistent, make them so.
        """
        if (self._qc is None) and (self._cal_data is None):
            raise ArgumentError("In Calibrator._preprocess_inputs(): neither @qc nor @cal_data has been set!")

        if (self._qc is None) and (self._cal_data is not None):
            n = len(next(iter(self._cal_data)))
            # the number of qubits in @qubits and @cal_data mismatch, we should extract the calibration data
            if self._qubits is not None:
                if len(self._qubits) <= n:
                    self._cal_data = extract_cal_data(self._cal_data, self._qubits)
                else:
                    raise ArgumentError("In Calibrator._preprocess_inputs(): "
                                        "the number of qubits in @qubits is larger than "
                                        "the number of qubits in @cal_data!")
            else:  # If @self._qubits is not set, set the default qubits list
                self._qubits = list(range(n))

        # If @self._qc is set, use its internal calibration data instead of the input calibration data
        if self._qc is not None:
            # Must initialize self._qubits when calibrate quantum computer
            if self._qubits is None:
                raise ArgumentError("In Calibrator._preprocess_inputs(): specify the number of qubits in @qubits!")
            else:
                if self.__class__.__name__ == 'CompleteCalibrator':
                    self._cal_data = load_cal_data(qc=self._qc, qubits=self._qubits, sync=self._sync)
                elif self.__class__.__name__ == 'TPCalibrator':
                    self._cal_data = load_cal_data(qc=self._qc, qubits=self._qubits, sync=self._sync, method='tp')


class CompleteCalibrator(Calibrator):
    """
    Calibrator based on complete model.
    """

    def __init__(self,
                 qc: QComputer = None,
                 cal_data: Dict[str, Dict[str, int]] = None,
                 qubits: List[int] = None,
                 sync: bool = True):
        """
        The init function of the Complete Calibrator.

        :param qc: QComputer, the quantum computer whose measurement device is to be calibrated.
        :param cal_data: Dict[str, Dict[str, int]], a dictionary of the calibration data
        :param qubits: List[int], the qubits list, composed of integers.
        """
        super(CompleteCalibrator, self).__init__()
        self._qc = qc
        self._cal_data = cal_data
        self._qubits = qubits
        self._sync = sync

        # Initialize the calibration matrix
        self.calibrate(qc=self._qc, cal_data=self._cal_data, qubits=self._qubits)

    def __str__(self):
        return "Complete Calibrator"

    def calibrate(self, **kwargs) -> Any:
        r"""The calibrate method of the CompleteCalibrator.

        Supported keywords in the list are:

        + ``qc``: QComputer, the quantum computer whose measurement device is to be calibrated.

        + ``cal_data``: Dict[str, Dict[str, int]], a dictionary of the calibration data.

        + ``qubits``: List[int], the qubits list that are calibrated, composed of integers.

        If these parameters are not set, use the default arguments set by the init function.

        If @qc is set, we load calibration data from file and update the @cal_data parameter.

        If @qc is not set, we use calibration data stored in @cal_data.

        Usage:

            .. code-block:: python
                :linenos:

                cc = CompleteCalibrator(qc=qc, cal_data=cal_data, qubits=qubits)
                cc.calibrate()
                cc.calibrate(cal_data=cal_data)
                cc.calibrate(cal_data=cal_data, qubits=qubits)
                cc.calibrate(qc=qc, qubits=qubits)
                cc.calibrate(qc=qc, cal_data=cal_data, qubits=qubits)

        **Examples**

            >>> cc_ideal = CompleteCalibrator(qc=qc_ideal, qubits=qubits)
            >>> cc_noisy = CompleteCalibrator(qc=qc_noisy, qubits=qubits)
            >>> print('complete_ideal\n', cc_ideal.cal_matrix)
            complete_ideal
            [[1. 0. 0. 0.]
             [0. 1. 0. 0.]
             [0. 0. 1. 0.]
             [0. 0. 0. 1.]]
            >>> print('complete_noisy\n', cc_noisy.cal_matrix)
            complete_noisy
            [[0.97729516 0.01964637 0.02946955 0.        ]
             [0.0088845  0.96561886 0.         0.02169625]
             [0.01382034 0.         0.95874263 0.01577909]
             [0.         0.01473477 0.01178782 0.96252465]]
        """
        # Parse the arguments. If not set, use the default arguments set by the init function.
        self._qc = kwargs.get('qc', self._qc)
        self._cal_data = kwargs.get('cal_data', self._cal_data)
        self._qubits = kwargs.get('qubits', self._qubits)
        self._sync = kwargs.get('sync', self._sync)

        self._preprocess_inputs()

        # Construct the calibration matrix
        n = len(self._qubits)
        dim = 2 ** n
        self._cal_matrix = np.zeros((dim, dim), dtype=float)

        # Learn the elements of the calibration matrix from the calibration data
        pbar = tqdm(total=100, desc='Step 3/3: Constructing calibration matrix!', ncols=80)
        for x, output_info in self._cal_data.items():  # iterate over input states
            pbar.update(100 / len(self._cal_data))
            for y, cnt in output_info.items():  # iterate over output states
                self._cal_matrix[int(y, 2), int(x, 2)] += cnt
        # Normalize along the row to make it column stochastic
        self._cal_matrix = normalize(self._cal_matrix, axis=0)
        return self._cal_matrix


class TPCalibrator(Calibrator):
    """
    Calibrator based on tensor product model.
    """

    def __init__(self,
                 qc: QComputer = None,
                 cal_data: Dict[str, Dict[str, int]] = None,
                 qubits: List[int] = None,
                 sync: bool = True):
        """
        The init function of the Complete Calibrator.

        :param qc: QComputer, the quantum computer whose measurement device is to be calibrated.
        :param cal_data: Dict[str, Dict[str, int]], a dictionary of the calibration data
        :param qubits: List[int], the qubits list, composed of integers.
        """
        super(TPCalibrator, self).__init__()
        self._qc = qc
        self._cal_data = cal_data
        self._qubits = qubits
        self._sync = sync
        self._cal_matrices = []

        # Initialize the calibration matrix
        self.calibrate(cal_data=self._cal_data, qubits=self._qubits)

    def __str__(self):
        return "Tensor Product Calibrator"

    def calibrate(self, **kwargs) -> Any:
        r"""
        The calibrate method of the TPCalibrator. Supported keywords in the list are:

        + ``qc``: QComputer, the quantum computer whose measurement device is to be calibrated.
        + ``cal_data``: Dict[str, Dict[str, int]], a dictionary of the calibration data.
        + ``qubits``: List[int], the qubits list that are calibrated, composed of integers.

        If these parameters are not set, use the default arguments set by the init function.

        If @qc is set, load calibration data from file and update the @cal_data parameter.

        If @qc is not set, use calibration data stored in @cal_data.

        Usage:

        .. code-block:: python
            :linenos:

            tp = TPCalibrator(qc=qc, cal_data=cal_data, qubits=qubits)
            tp.calibrate()
            tp.calibrate(cal_data=cal_data)
            tp.calibrate(cal_data=cal_data, qubits=qubits)
            tp.calibrate(qc=qc, qubits=qubits)
            tp.calibrate(qc=qc, cal_data=cal_data, qubits=qubits)

        **Examples**

            >>> tp_ideal = TPCalibrator(qc=qc_ideal, qubits=qubits)
            >>> tp_noisy = TPCalibrator(qc=qc_noisy, qubits=qubits)
            >>> print('tp_ideal\n', tp_ideal.cal_matrix)
            tp_ideal
            [[1. 0. 0. 0.]
             [0. 1. 0. 0.]
             [0. 0. 1. 0.]
             [0. 0. 0. 1.]]
            >>> print('tp_noisy\n', tp_noisy.cal_matrix)
            tp_noisy
            [[9.76718748e-01 2.77204459e-02 2.09163347e-02 5.93630586e-04]
             [1.13171442e-02 9.60315446e-01 2.42355515e-04 2.05650596e-02]
             [1.18270686e-02 3.35666348e-04 9.67629482e-01 2.74624816e-02]
             [1.37039082e-04 1.16284413e-02 1.12118278e-02 9.51378828e-01]]
        """
        # Parse the arguments. If not set, use the default arguments set by the init function.
        self._qc = kwargs.get('qc', self._qc)
        self._cal_data = kwargs.get('cal_data', self._cal_data)
        self._qubits = kwargs.get('qubits', self._qubits)
        self._preprocess_inputs()

        # Construct the calibration matrix
        n = len(self._qubits)
        # local_cal_matrices = []

        # TODO: The following learning procedure is brute-force. Need more efficient algorithm.
        # Learn the local calibration matrices one by one from the calibration data
        pbar = tqdm(total=100, desc='Step 3/3: Constructing calibration matrix!', ncols=80)
        for k in range(n):
            pbar.update(100 / n)
            local_A = np.zeros((2, 2), dtype=float)

            for x, output_info in self._cal_data.items():  # iterate over input states
                for y, cnt in output_info.items():  # iterate over output states
                    x_e, x_r = extract_substr(x, indices=[k])
                    y_e, y_r = extract_substr(y, indices=[k])
                    local_A[int(y_e, 2), int(x_e, 2)] += cnt

            # Normalize along the row to make it column stochastic
            local_A = normalize(local_A, axis=0)
            self._cal_matrices.append(local_A)

        self._cal_matrix = tensor(self._cal_matrices)
        return self._cal_matrix

    def readout_fidelity(self, qubit: int) -> (float, float):
        r"""
        Compute the readout fidelity of the given qubit.
        The readout fidelity of a qubit is characterized by two parameters:

        + the state :math:`\vert 0\rangle` readout fidelity, which is the probability of input :math:`\vert 0\rangle`
          state and obtain outcome 0, and

        + the state :math:`\vert 1\rangle` readout fidelity, which is the probability of input :math:`\vert 1\rangle`
          state and obtain outcome 1.

        :param qubit: int, the qubit index whose readout fidelity will be computed
        :return: A tuple of two float values, representing states 0 and 1 readout fidelities, respectively
        """

        return tuple(np.diagonal(self._cal_matrices[qubit]))

    def local_cal_matrix(self, qubit: int) -> np.ndarray:
        r"""
        Return the local calibration matrix of the given qubit.

        :param qubit: int, the qubit index whose local calibration matrix will be returned.
        :return: np.ndarray, a :math:`2 \times 2` matrix.
        """
        if qubit not in self._qubits:
            raise ArgumentError("The input qubit {} is not in the qubits list!".format(qubit))

        for element in self.local_cal_matrices():
            if element[0] == qubit:
                return element[1]

    def local_cal_matrices(self) -> List[Tuple[int, Any]]:
        r"""
        Return all the local calibration matrix stored in list.

        :return: List[Tuple[int, Any]], a list of :math:`2 \times 2` matrices.
        """
        # !!!DO NOT FORGET!!! to deep copy the self._qubits,
        # otherwise indices is a reference, and we actually sort @qubit from self._qubits.
        indices = copy.deepcopy(self._qubits)
        indices.sort()
        return list(zip(indices, self._cal_matrices))


def init_complete_cal_circuits(qubits: List[int] = None) -> Dict[str, QProgram]:
    r"""
    Initialize the measurement calibration circuits for the complete calibration model.
    Assume the circuit contains :math:`n` qubits (that is, there are :math:`n` in @qubits list).
    We initialize :math:`2^n` calibration circuits in total, each prepares a basis state.

    :param qubits: List[int], the qubits list, composed of integers.
    :return: Dict[str: QProgram], state labels and the corresponded QProgram objects which are the calibration circuits.

    **Examples**

        >>> init_cp_cir = init_complete_cal_circuits(qubits=qubits)
        >>> print(init_cp_cir)
        >>> for key, value in init_cp_cir.items():
        >>>    print_circuit(init_cp_cir[key].circuit)
        {'00': <QCompute.QPlatform.QEnv.QEnv object at 0x12d125e80>,
        '01': <QCompute.QPlatform.QEnv.QEnv object at 0x12d1394c0>,
        '10': <QCompute.QPlatform.QEnv.QEnv object at 0x12d139dc0>,
        '11': <QCompute.QPlatform.QEnv.QEnv object at 0x12d139c40}
        0: -------
        1: ---MEAS---
        2: ---MEAS---
        0: -------
        1: ---X---MEAS---
        2: -------MEAS---
        0: -------
        1: -------MEAS---
        2: ---X---MEAS---
        0: -------
        1: ---X---MEAS---
        2: ---X---MEAS---
    """
    target_qubit = len(qubits)
    size = max(qubits) + 1
    result = {}
    # Generate list of indices of qubits to be flipped for each circuit.
    # If qubits = [1, 2, 4], state_all would be [[], [1], [2], [4], [1, 2], [1, 4], [2, 4], [1, 2, 4]].
    # The first element corresponds to state |00000>.
    state_all = []
    for i in range(target_qubit + 1):
        for a in combinations(qubits, r=i):
            state_all.append(list(a))

    # Circuit of state |0...0>.
    state_str_0 = (str(bin(0)[2:])).zfill(size)
    pbar = tqdm(total=100, desc='Step 1/3: Constructing calibration circuit!', ncols=80)
    for state in range(0, 2 ** target_qubit):
        pbar.update(100 / (2 ** target_qubit))
        # Compute the binary string and reverse the order: We assume the LSB represents q[0], i.e.,
        #                   "1        0        1"
        #                   q[2]    q[1]      q[0]

        if state == 0:
            state_str = state_str_0
        else:
            state_str_temp = []
            for idx, val in enumerate(state_str_0[::-1]):
                # Flip the qubit(s) according to the list of indices.
                if idx in state_all[state]:
                    val = '1'
                # Collect the qubits state.
                state_str_temp.append(val)
            # Reverse the qubits state.
            state_str_temp.reverse()
            state_str = "".join(state_str_temp)

        # Step 1. Setup the calibration quantum circuit for the basis state @i
        qp = QEnv()
        qp.Q.createList(size)
        state_final = []
        for idx, val in enumerate(state_str[::-1]):
            if val == '1':
                X(qp.Q[idx])
            if idx in qubits:
                state_final.append(val)
        state_final.reverse()
        state_final = "".join(state_final)

        qreglist, indexlist = qp.Q.toListPair()
        MeasureZ(qRegList=[qreglist[x] for x in qubits],
                 cRegList=[indexlist[x] for x in qubits])

        result[state_final] = qp
    return result


def init_tp_cal_circuits(qubits: List[int] = None) -> Dict[str, QProgram]:
    r"""
    Initialize the measurement calibration circuits for the tensor product calibration model.
    In the tensor product model, no matter how many qubits the circuit contains,
    we initialize :math:`2` calibration circuits,
    one prepares the :math:`\vert 0\cdots 0\rangle` basis state, and
    another prepares the :math:`\vert 1\cdots 1\rangle` basis state.

    :param qubits: List[int], the qubits list, composed of integers.
    :return: Dict[str: QProgram], state labels and the corresponded QProgram objects which are the calibration circuits.

    **Examples**

        >>> init_tp_cir = init_tp_cal_circuits(qubits=qubits)
        >>> print(init_tp_cir)
        >>> for key, value in init_tp_cir.items():
        >>>    print_circuit(init_tp_cir[key].circuit)
        {'00': <QCompute.QPlatform.QEnv.QEnv object at 0x129d3dfa0>,
        '11': <QCompute.QPlatform.QEnv.QEnv object at 0x12daa80a0>}
        0: -------
        1: ---MEAS---
        2: ---MEAS---
        0: -------
        1: ---X---MEAS---
        2: ---X---MEAS---
    """
    size = max(qubits) + 1
    result = {}
    # Generate list of indices of qubits to be flipped for each circuit.
    # If qubits = [1, 2, 4], state_all would be [[], [1], [2], [4], [1, 2], [1, 4], [2, 4], [1, 2, 4]].
    # The first element corresponds to state |00000>.
    state_all = [[]]
    state_all.append(qubits)

    # # Circuit of state |0...0>.
    state_str_0 = (str(bin(0)[2:])).zfill(size)
    pbar = tqdm(total=100, desc='Step 1/3: Constructing calibration circuit!', ncols=80)
    for state in range(0, 2):
        pbar.update(100 / 2)
        # Compute the binary string and reverse the order: We assume the LSB represents q[0], i.e.,
        #                   "1        0        1"
        #                   q[2]    q[1]      q[0]

        if state == 0:
            state_str = state_str_0
        else:
            state_str_temp = []
            for idx, val in enumerate(state_str_0[::-1]):
                # Flip the qubit(s) according to the list of indices.
                if idx in state_all[state]:
                    val = '1'
                # Collect the qubits state.
                state_str_temp.append(val)
            # Reverse the qubits state.
            state_str_temp.reverse()
            state_str = "".join(state_str_temp)

        # Step 1. Setup the calibration quantum circuit for the basis state @i
        qp = QEnv()
        qp.Q.createList(size)
        state_final = []
        for idx, val in enumerate(state_str[::-1]):
            if val == '1':
                X(qp.Q[idx])
            if idx in qubits:
                state_final.append(val)
        state_final.reverse()
        state_final = "".join(state_final)

        qreglist, indexlist = qp.Q.toListPair()
        MeasureZ(qRegList=[qreglist[x] for x in qubits],
                 cRegList=[indexlist[x] for x in qubits])

        result[state_final] = qp

    return result


def extract_cal_data(raw_cal_data: Dict[str, Dict[str, int]], qubits: List[int] = None) -> Dict[str, Dict[str, int]]:
    """
    Extract calibration data of @qubits for the original calibration data.
    If @qubits is None, load the calibration data for all qubits by default.
    If the input cal_data is incomplete, we make it complete.

    :param raw_cal_data: Dict[str, Dict[str, int]], the original calibration data
    :param qubits: List[int], the qubits list whose calibration data is loaded, composed of integers.
    :return: Dict[str, Dict[str, int]], dictionary of the calibration data corresponding to @qubits.
    """
    if qubits is None:
        n = len(next(iter(raw_cal_data)))
        qubits = list(range(n))

    cal_data = init_cal_data(n=len(qubits), layer=2, init_value=0)
    pbar = tqdm(total=100, desc='Extracting calibration data', ncols=80)
    for x, output_info in raw_cal_data.items():
        pbar.update(100 / len(raw_cal_data))
        x_e, x_r = extract_substr(x, qubits)
        # `x_e` represents the key in the @cal_data, while `x_r` represents the remaining qubits
        for y, cnt in output_info.items():  # iterate over output states
            y_e, y_r = extract_substr(y, qubits)
            cal_data[x_e][y_e] += cnt

    return cal_data


def load_cal_data(qc: QComputer = None,
                  qubits: List[int] = None,
                  sync: bool = True,
                  method: str = 'complete') -> Dict[str, Dict[str, int]]:
    """
    Load calibration data of @qubits for the Quantum Computer specified by @qc.
    By default **sync = True**, the program would generate and run calibration circuits to obtain calibration data.

    .. note::

        For VIP users, we offer a faster way to obtain calibration data. Set the **sync = False**, the program would
        access the server through **TOKEN**, and download the latest **10** calibration data.
        For convenience, we only return the latest calibration data.
        If more calibration data is desired, please refer to `QCompute.Calibration`.

    If @qubits is None, load the calibration data for all qubits by default.

    :param qc: QComputer, the quantum computer whose measurement device is to be calibrated
    :param qubits: List[int], the qubits list whose calibration data is loaded, composed of integers
    :param method: string, decide which kind of calibration to be used, by default is complete model
    :param sync: bool, indicates whether instantaneous calibration procedure should be carried out or not
    :return: Dict[str, Dict[str, int]], dictionary of the calibration data corresponding to @qubits

    **Examples**

        >>> ideal_cp = load_cal_data(qc=qc_ideal, qubits=qubits)
        >>> Define.hubToken = ""
        >>> noisy_cp = load_cal_data(qc=qc_noisy, qubits=qubits, sync=False)
        >>> print('ideal_cp: ', ideal_cp)
        ideal_cp:  {'00': {'00': 1024, '01': 0, '10': 0, '11': 0}, '01': {'00': 0, '01': 1024, '10': 0, '11': 0},
        '10': {'00': 0, '01': 0, '10': 1024, '11': 0}, '11': {'00': 0, '01': 0, '10': 0, '11': 1024}}
        >>> print('noisy_cp: ', noisy_cp)
        noisy_cp:  {'00': {'00': 996, '01': 9, '10': 15, '11': 0}, '01': {'00': 18, '01': 989, '10': 0, '11': 13},
        '10': {'00': 18, '01': 1, '10': 988, '11': 11}, '11': {'00': 2, '01': 29, '10': 19, '11': 964}}
    """
    if sync is True:
        if method == 'complete':
            cir = init_complete_cal_circuits(qubits=qubits)
        else:
            cir = init_tp_cal_circuits(qubits=qubits)
        cal_data = {}
        pbar = tqdm(total=100, desc='Step 2/3: Collecting calibration data!', ncols=80)
        for key, value in cir.items():
            pbar.update(100 / len(cir))
            cal_data[key] = execute(qp=value, qc=qc, shots=1024)
    else:
        qc_name = qc.name.lower()
        if qc_name.endswith('iopcas'):
            qc_name = 'iopcas'
            # Update local calibration data.
            CalibrationUpdate(qc_name)
            # Load raw calibration data.
            cal_data_list = CalibrationReadData(qc_name)

            # Obtain the latest calibration data, and extract the desired calibration data.
            cal_data = extract_cal_data(cal_data_list[0].readData(), qubits)
        else:
            raise ArgumentError("Invalid quantum machine!")

    return cal_data
