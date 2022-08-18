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
In this module, we implement the measurement correction procedure.
It aims to correct the classical output probability distribution of a noisy quantum device.
Different kinds of correction procedures will be implemented:

1. The inverse approach,

2. The least square approach,

3. The iterative Bayesian unfolding approach, and

4. The truncated Neumann series approach.

.. note::

    We will give an example how these methods are used by constructing the GHZ state and calculate
    its expectation value (with noise and without noise) of a specific quantum observable.


**Examples preparation**

    >>> import copy
    >>> import unittest
    >>> import qiskit
    >>> import functools
    >>> import numpy as np
    >>> from QCompute import Define
    >>> from QCompute import BackendName
    >>> from QCompute import QEnv
    >>> from QCompute import MeasureZ
    >>> from QCompute import H
    >>> from QCompute import CX
    >>>
    >>> from qiskit.providers.fake_provider import FakeSantiago
    >>> from qcompute_qep.measurement.correction import InverseCorrector
    >>> from qcompute_qep.measurement.correction import LeastSquareCorrector
    >>> from qcompute_qep.measurement.correction import IBUCorrector
    >>> from qcompute_qep.measurement.correction import NeumannCorrector
    >>> from qcompute_qep.utils.types import get_qc_name
    >>> from qcompute_qep.utils.circuit import execute
    >>> from qcompute_qep.utils import expval_from_counts
    >>>
    >>> qc_ideal = BackendName.LocalBaiduSim2
    >>> qc_noisy = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())
    >>> #######################################################################################################
    >>> # Setup the quantum program for preparing the GHZ state and set the quantum observable.
    >>> #######################################################################################################
    >>> qp = QEnv()
    >>> n = 2
    >>> qp.Q.createList(n)
    >>> H(qp.Q[0])
    >>> for i in range(1, n):
    >>>    CX(qp.Q[0], qp.Q[i])
    >>> MeasureZ(*qp.Q.toListPair())
    >>> # Set the ideal quantum computer (simulator)
    >>> ideal_qc = BackendName.LocalBaiduSim2
    >>> # Set the noisy quantum computer
    >>> qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())
    >>> noisy_qc_name = get_qc_name(qc)
    >>>
    >>> # Set the quantum observable: :math:`O = |0\cdots 0><0\cdots 0| + |1\cdots 1><1\cdots 1|`
    >>> proj0 = np.array([[1, 0], [0, 0]]).astype(complex)
    >>> proj1 = np.array([[0, 0], [0, 1]]).astype(complex)
    >>> O = functools.reduce(np.kron, [proj0] * n) + functools.reduce(np.kron, [proj1] * n)
    >>>
    >>> # Ideal case
    >>> counts_ideal = execute(qp=copy.deepcopy(qp), qc=ideal_qc, shots=MAX_SHOTS)
    >>> # Compute the expectation value from counts
    >>> val_ideal = expval_from_counts(O, counts_ideal)
    >>>
    >>> # Noisy case
    >>> counts_noisy = execute(qp=copy.deepcopy(qp), qc=qc, shots=MAX_SHOTS)
    >>> # Compute the expectation value from counts
    >>> val_noisy = expval_from_counts(O, counts_noisy)
    >>>
    >>> print("The ideal expectation value is: {}".format(val_ideal))
    The ideal expectation value is: 1.0
    >>> print("The noisy expectation value is: {}".format(val_noisy))
    The noisy expectation value is: 0.9681396484375
"""

import abc
from abc import ABC
from typing import Any, List, Union, Dict
import numpy as np
import scipy.linalg as la
from scipy.optimize import minimize
from scipy.special import binom
import math
import warnings

from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.measurement.calibration import Calibrator, CompleteCalibrator, TPCalibrator
from qcompute_qep.measurement.utils import dict2vector, vector2dict
from qcompute_qep.utils.types import QComputer
from qcompute_qep.utils.linalg import normalize


class Corrector(ABC):
    """Abstract interface `Corrector` for the Measurement Error Mitigation
    correction.

    Concrete measurement error correction methods must inherit this
    abstract class and implement the `correct` method.
    """
    def __init__(self,
                 qc: QComputer = None,
                 calibrator: Union[Calibrator, str] = 'complete',
                 qubits: List[int] = None,
                 **kwargs):
        """The init function of the Corrector.

        :param qc: QComputer, the quantum computer whose measurement device is to be calibrated
        :param calibrator: Optional[Calibrator, str], a calibrator instance, by default is 'complete'
        :param qubits: List[int], the qubits list, composed of integers
        """
        super(Corrector, self).__init__()
        self._qc = qc
        self._qubits = qubits

        # Parse the input calibrator type.
        # The calibrator can be initialized via string or Calibrator instance.
        if isinstance(calibrator, str):
            cal_data = kwargs.get('cal_data', None)
            if calibrator.lower() == 'complete':  # Case insensitive
                self._calibrator = CompleteCalibrator(qc=qc, cal_data=cal_data, qubits=self._qubits)
            elif calibrator.lower() == 'tp':  # Case insensitive
                self._calibrator = TPCalibrator(qc=qc, cal_data=cal_data, qubits=self._qubits)
            else:
                raise ArgumentError("Calibrator with name {} is not defined!".format(calibrator))
        else:
            self._calibrator = calibrator

    @property
    def calibrator(self):
        return self._calibrator

    @calibrator.setter
    def calibrator(self, val):
        self._calibrator = val

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.correct(*args, **kwargs)

    @abc.abstractmethod
    def correct(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


class InverseCorrector(Corrector):
    """Corrector based on Matrix Inversion."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def __str__(self):
        return "Inverse Corrector"

    def correct(self, raw_data: Union[dict, np.ndarray], **kwargs: Any) -> Union[dict, np.ndarray]:
        """Use the Matrix Inverse method to correct raw data.

        :param raw_data: Optional[dict, np.ndarray], the input raw data, can be a dictionary
                            or a (unnormalized) probability vector.
        :param kwargs: other optional key word arguments.
        :return: The corrected data, same type as the input noisy data.

        Usage:

        .. code-block:: python
            :linenos:

            inv = InverseCorrector(qc=qc, calibrator=calibrator, cal_data=cal_data, qubits=qubits)
            inv = InverseCorrector(qc=qc, qubits=qubits)
            inv = InverseCorrector(calibrator=calibrator)
            inv = InverseCorrector(calibrator='complete', cal_data=cal_data, qubits=qubits)
            inv.correct(raw_data)

        **Examples**

        >>> # tensor product calibration
        >>> corr_tp_inv = InverseCorrector(qc=qc, calibrator='tp', qubits=range(n))
        >>> counts_tp_inv = corr_tp_inv.correct(counts_noisy)
        >>> # Compute the expectation value from corrected counts
        >>> val_tp_inv = expval_from_counts(O, counts_tp_inv)
        >>>
        >>> # complete model calibration
        >>> corr_cp_inv = InverseCorrector(qc=qc, calibrator='complete', qubits=range(n))
        >>> counts_cp_inv = corr_cp_inv.correct(counts_noisy)
        >>> # Compute the expectation value from corrected counts
        >>> val_cp_inv = expval_from_counts(O, counts_cp_inv)
        >>>
        >>> print("The 'Tensor Product Calibrator + Inverse Corrector' "
        >>>  "mitigated expectation value is: {}".format(val_tp_inv))
        The 'Tensor Product Calibrator + Inverse Corrector' mitigated expectation value is:
        0.9958477633347107
        >>> print("The 'Complete Calibrator + Inverse Corrector' "
        >>>  "mitigated expectation value is: {}".format(val_cp_inv))
        The 'Complete Calibrator + Inverse Corrector' mitigated expectation value is:
        0.9993213980997113
        """

        # Check and record the input data format
        type_mark = type(raw_data)
        if isinstance(raw_data, dict):
            raw_data = dict2vector(raw_data)

        # Inverse Matrix method.
        try:
            cal_matrix_inv = la.pinv(self._calibrator.cal_matrix)

            for i in cal_matrix_inv:
                if (i < 0).any():
                    warnings.warn('There are negative values in the pseudoinverse matrix, may choose another method.')
            corrected_data = np.matmul(cal_matrix_inv, raw_data)
            # Convert the data back to its input format
            if issubclass(type_mark, dict):
                return vector2dict(corrected_data)
            else:
                return corrected_data
        except np.linalg.LinAlgError:
            print("There is no pseudoinverse! Please choose another method!")


class LeastSquareCorrector(Corrector):
    """Corrector based on Least Square."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._tol = kwargs.get('tol', 1e-06)
        self.__opt_info = {}

    def __str__(self):
        return "Least Square Corrector"

    @staticmethod
    def _opt_fun(x: np.ndarray, A: np.ndarray, b: np.ndarray) -> float:
        r"""
        Target optimization function for the least square method:

            :math:`f(x) = \sum_i \vert (Ax)[i] - b[i]\vert^2`

        :param x: np.ndarray, the optimization variables, is a :math:`N\times 1` vector
        :param A: np.ndarray, the coefficient matrix, is a :math:`N\times N` vector
        :param b: np.ndarray, the dependent variable values, is a :math:`N\times 1` vector
        :return: float, the Euclidean distance between the two vectors
        """
        return sum((np.ravel(b) - np.ravel(np.matmul(A, x)))**2)

    def correct(self, raw_data: Union[dict, np.ndarray], **kwargs: Any) -> Union[Dict, np.ndarray]:
        """Use the Least Square method to correct raw data.

        Supported `(key, value)` pairs in the keyworded variable is:

            'tol' = 1e-6: set the optimization error tolerance.

        :param raw_data: Optional[dict, np.ndarray], the input raw data, can be a dictionary
                            or a (unnormalized) probability vector
        :param kwargs: other optional key word arguments
        :return: The corrected data, same type as the input noisy data

        Usage:

        .. code-block:: python
            :linenos:

            lsc = LeastSquareCorrector(qc=qc, calibrator=calibrator, cal_data=cal_data, qubits=qubits)
            lsc = LeastSquareCorrector(qc=qc, qubits=qubits)
            lsc = LeastSquareCorrector(calibrator=calibrator)
            lsc = LeastSquareCorrector(calibrator='complete', cal_data=cal_data, qubits=qubits)
            lsc.correct(raw_data)

        **Examples**

        >>> # tensor product calibration
        >>> corr_tp_ls = LeastSquareCorrector(qc=qc, calibrator='tp', qubits=range(n))
        >>> counts_tp_ls = corr_tp_ls.correct(counts_noisy)
        >>> val_tp_ls = expval_from_counts(O, counts_tp_ls)
        >>>
        >>> # complete model calibration
        >>> corr_cp_ls = LeastSquareCorrector(qc=qc, calibrator='complete', qubits=range(n))
        >>> counts_cp_ls = corr_cp_ls.correct(counts_noisy)
        >>> # Compute the expectation value from corrected counts
        >>> val_cp_ls = expval_from_counts(O, counts_cp_ls)
        >>>
        >>> print("The 'Tensor Product Calibrator + Least Square Corrector' "
        >>> "mitigated expectation value is: {}".format(val_tp_ls))
        The 'Tensor Product Calibrator + Least Square Corrector' mitigated expectation value is:
        0.9929511373849917
        >>> print("The 'Complete Calibrator + Least Square Corrector' "
        >>> "mitigated expectation value is: {}".format(val_cp_ls))
        The 'Complete Calibrator + Least Square Corrector' mitigated expectation value is:
        0.9931843936631248
        """

        # Check and record the input data format
        type_mark = type(raw_data)
        if isinstance(raw_data, dict):
            raw_data = dict2vector(raw_data)

        # Least square
        nqubits = int(math.log2(raw_data.size))
        # Set the initial guess
        x0 = np.random.rand(2 ** nqubits)
        x0 = x0 / sum(x0)
        nshots = sum(raw_data)
        constraints = ({'type': 'eq', 'fun': lambda x: nshots - sum(x)})
        # Set bounds to each random variable x
        bounds = tuple((0, nshots) for x in x0)
        corrected_data = minimize(self._opt_fun, x0, method='SLSQP', constraints=constraints, bounds=bounds,
                                  tol=self._tol, args=(self.calibrator.cal_matrix, raw_data))
        self.__opt_info['number of iterations'] = corrected_data.nit
        self.__opt_info['tol'] = self._tol
        self.__opt_info['final value of objective function'] = corrected_data['fun']
        self.__opt_info['termination status'] = corrected_data['message']
        # Convert the data back to its input format
        if issubclass(type_mark, dict):
            return vector2dict(corrected_data.x)
        else:
            return corrected_data.x

    def get_opt_info(self):
        return self.__opt_info


class IBUCorrector(Corrector):
    """Corrector based on Iterative Bayesian Unfolding."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._k = kwargs.get('max_iter', 100)
        self._tol = kwargs.get('tol', 1e-6)
        self.__opt_info = {}

    def __str__(self):
        return "Iterative Bayesian Unfolding Corrector"

    @staticmethod
    def _fun(cal_matrix: np.ndarray, raw_data: np.ndarray, target: np.ndarray) -> np.ndarray:
        r"""
        Iterative bayes function for the IBU method:

        .. math::
                target[i]=\frac{\sum_j cal\_matrix[j][i]\times raw\_data[j]}{\sum_k cal\_matrix[j][k]\times target[k]}



        :param cal_matrix: np.ndarray, is a :math:`N\times N` vector
        :param raw_data: np.ndarray, is a :math:`N\times 1` vector
        :param target: np.ndarray, is a :math:`N\times 1` vector, the target to be iterated
        :return: np.ndarray, is a :math:`N\times 1` vector, the iterated target
        """
        # Since the multiplication depends highly on the shape of target, we enforce it to be a column type
        target = target.reshape((target.size, 1))
        # Compute the Bayesian transition matrix from the calibration matrix
        R = cal_matrix.T * target
        # Normalize the Bayesian transition matrix along column
        R = normalize(R, axis=0)
        # Iterate
        target = np.dot(R, raw_data)

        return target

    def correct(self, raw_data: Union[dict, np.ndarray], **kwargs: Any) -> Union[Dict, np.ndarray]:
        r"""
        Use the Iterative Bayesian Unfolding method to correct raw data.

        Supported `(key, value)` pairs in the keyworded variable is:

            'tol' = 1e-6: set the optimization error tolerance.


        :param raw_data: Optional[dict, np.ndarray], the input raw data, can be a dictionary
            or a (unnormalized) probability vector.
        :param kwargs: other optional key word arguments
        :return: The corrected data, same type as the input noisy data.

        Usage:

        .. code-block:: python
            :linenos:

            ibu = IBUCorrector(qc=qc, calibrator=calibrator, cal_data=cal_data, qubits=qubits)
            ibu = IBUCorrector(qc=qc, qubits=qubits)
            ibu = IBUCorrector(calibrator=calibrator)
            ibu = IBUCorrector(calibrator='complete', cal_data=cal_data, qubits=qubits)
            ibu.correct(raw_data)

        **Examples**

        >>> # tensor product calibration
        >>> corr_tp_ibu = IBUCorrector(qc=qc, calibrator='tp', qubits=range(n))
        >>> counts_tp_ibu = corr_tp_ibu.correct(counts_noisy)
        >>> # Compute the expectation value from corrected counts
        >>> val_tp_ibu = expval_from_counts(O, counts_tp_ibu)
        >>>
        >>> # complete model calibration
        >>> corr_cp_ibu = IBUCorrector(qc=qc, calibrator='complete', qubits=range(n))
        >>> counts_cp_ibu = corr_cp_ibu.correct(counts_noisy)
        >>> # Compute the expectation value from corrected counts
        >>> val_cp_ibu = expval_from_counts(O, counts_cp_ibu)
        >>>
        >>> print("The 'Tensor Product Calibrator + Iterative Bayesian Unfolding Corrector' "
        >>> "mitigated expectation value is: {}".format(val_tp_ibu))
        The 'Tensor Product Calibrator + Iterative Bayesian Unfolding Corrector' mitigated expectation value is:
        0.9942077464684559
        >>> print("The 'Complete Calibrator + Iterative Bayesian Unfolding Corrector' "
        >>> "mitigated expectation value is: {}".format(val_cp_ibu))
        The 'Complete Calibrator + Iterative Bayesian Unfolding Corrector' mitigated expectation value is:
        0.9952241705397734
        """

        # Check and record the input data format
        type_mark = type(raw_data)
        if isinstance(raw_data, dict):
            raw_data = dict2vector(raw_data)

        # IBU method.
        size = len(raw_data)
        corrected_data = np.ones(size, dtype=float)*(sum(raw_data)/size)
        gap = 1
        nit = 0
        # Iterate the IBU function according to the max_iter and tol.
        for i in range(self._k):
            nit += 1
            if gap > self._tol:
                corrected_data_before = corrected_data.copy()
                corrected_data = self._fun(self.calibrator.cal_matrix, raw_data, corrected_data)
                gap = np.linalg.norm(corrected_data - corrected_data_before)
                continue
            else:
                break

        self.__opt_info['number of iterations'] = nit
        self.__opt_info['tol'] = self._tol
        self.__opt_info['final gap'] = gap
        # Convert the data back to its input format
        if issubclass(type_mark, dict):
            return vector2dict(corrected_data)
        else:
            return corrected_data

    def get_opt_info(self):
        return self.__opt_info


class NeumannCorrector(Corrector):
    """Corrector based on Truncated Neumann Series."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._tol = kwargs.get('tol', 1e-6)

    def __str__(self):
        return "Neumann Corrector"

    def correct(self, raw_data: Union[dict, np.ndarray], **kwargs: Any) -> Union[dict, np.ndarray]:
        r"""
        Use the Truncated Neumann Series method to correct raw data.

        Supported `(key, value)` pairs in the keyworded variable is:

            'tol' = 1e-6: set the optimization error tolerance.


        :param raw_data: Optional[dict, np.ndarray], the input raw data, can be a dictionary
            or a (unnormalized) probability vector.
        :return: The corrected data, same type as the input noisy data.

        Usage:

        .. code-block:: python
            :linenos:

            neu = NeumannCorrector(qc=qc, calibrator=calibrator, cal_data=cal_data, qubits=qubits)
            neu = NeumannCorrector(qc=qc, qubits=qubits)
            neu = NeumannCorrector(calibrator=calibrator)
            neu = NeumannCorrector(calibrator='complete', cal_data=cal_data, qubits=qubits)
            neu.correct(raw_data)

        **Examples**

        >>> # tensor product calibration
        >>> corr_tp_neu = NeumannCorrector(qc=qc, calibrator='tp', qubits=range(n))
        >>> counts_tp_neu = corr_tp_neu.correct(counts_noisy)
        >>> # Compute the expectation value from corrected counts
        >>> val_tp_neu = expval_from_counts(O, counts_tp_neu)
        >>>
        >>> # complete model calibration
        >>> corr_cp_neu = NeumannCorrector(qc=qc, calibrator='complete', qubits=range(n))
        >>> counts_cp_neu = corr_cp_neu.correct(counts_noisy)
        >>> # Compute the expectation value from corrected counts
        >>> val_cp_neu = expval_from_counts(O, counts_cp_neu)
        >>>
        >>> print("The 'Tensor Product Calibrator + Truncated Neumann Series Corrector' "
        >>> "mitigated expectation value is: {}".format(val_tp_neu))
        The 'Tensor Product Calibrator + Truncated Neumann Series Corrector' mitigated expectation value is:
        1.0014897858810545
        >>> print("The 'Complete Calibrator + Truncated Neumann Series Corrector' "
        >>> "mitigated expectation value is: {}".format(val_cp_neu))
        The 'Complete Calibrator + Truncated Neumann Series Corrector' mitigated expectation value is:
        0.992090581402768
        """

        # Check and record the input data format
        type_mark = type(raw_data)
        if isinstance(raw_data, dict):
            raw_data = dict2vector(raw_data)

        # Initialize the inverse calibration matrix
        cal_matrix_inv = np.identity(self.calibrator.cal_matrix.shape[0])

        # Truncated Neumann Series method.
        if not np.isclose(self.calibrator.noise_resist, 0, rtol=1e-05, atol=1e-08, equal_nan=False):
            optimal_truncated_num = math.ceil(math.log(self._tol)/math.log(self.calibrator.noise_resist)-1)
            for k in range(optimal_truncated_num+1):
                c_k = (-1)**k * binom(optimal_truncated_num+1, k+1)
                cal_matrix_inv += c_k * np.linalg.matrix_power(self.calibrator.cal_matrix, k)

        # Correct data
        corrected_data = np.matmul(cal_matrix_inv, raw_data)

        # Convert the data back to its input format
        if issubclass(type_mark, dict):
            return vector2dict(corrected_data)
        else:
            return corrected_data
