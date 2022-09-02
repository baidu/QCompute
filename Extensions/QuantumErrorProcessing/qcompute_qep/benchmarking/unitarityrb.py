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

"""Unitarity Randomized Benchmarking.

A scalable and robust algorithm for benchmarking the unitarity of the
Clifford gates by a single parameter called UPC (unitarity per Clifford)
using randomization techniques.
"""
import itertools
import numpy as np
from tqdm import tqdm
import scipy.stats as st
from typing import List
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit
from copy import deepcopy

import QCompute
from qcompute_qep.utils.linalg import tensor
from qcompute_qep.utils import expval_from_counts, execute, circuit, str_to_state
from qcompute_qep.quantum import clifford
import qcompute_qep.exceptions.QEPError as QEPError
import qcompute_qep.benchmarking as rb
from qcompute_qep.utils.types import QComputer, get_qc_name, QProgram, number_of_qubits
from QCompute import *
from QCompute.QPlatform.QOperation import CircuitLine
from qcompute_qep.exceptions.QEPError import ArgumentError

try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import pylab

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class UnitarityRB(rb.RandomizedBenchmarking):
    """The Unitarity Randomized Benchmarking class.

    Aim to benchmark the coherence(unitarity) noise of a complete set of Cliffords.
    """

    def __init__(self, qc: QComputer = None, qubits: List[int] = None, **kwargs):
        r"""init function of the Unitarity Randomized Benchmarking class.

        Optional keywords list are:

        + ``qubits2``: List[int], default to None. For example, if using `method=1`, qubits1=[0, 1],
                default setting for qubits2 will be [2, 3]. Otherwise, qubits2 should be given.
        + ``seq_lengths``: List[int], default to :math:`[1, 10, 20, 50, 75, 100]`, a list of sequence lengths
        + ``repeats``: int, default to :math:`6`, the number of repetitions of each sequence length
        + ``shots``: int, default to :math:`4096`, the number of shots each measurement carries out to estimate value
        + ``prep_circuit``: default to `default_prep_circuit`, prepares the initial
                    quantum state :math:`\vert 0\cdots 0 \rangle`
        + ``meas_circuit``: default to `default_meas_circuit`, add the Z basis measurement to
                the end of the RB circuits and set the quantum observable to
                :math:`\vert 0\cdots 0\rangle\!\langle 0\cdots 0\vert`

        :param qc: QComputer, the quantum computer on which the RB carries out
        :param qubits: List[int], the qubits who will be benchmarked
        """
        # Initialize the URB parameters. If not set, use the default parameters
        super().__init__(**kwargs)
        self._qc = qc
        self._qubits = qubits
        self._qubits2 = kwargs.get('qubits2', None)
        self._seq_lengths = kwargs.get('seq_lengths', [1, 10, 20, 50, 75, 100])
        self._repeats = kwargs.get('repeats', 6)
        self._shots = kwargs.get('shots', 4096)
        self._prep_circuit = kwargs.get('prep_circuit', rb.default_prep_circuit)
        self._meas_circuit = kwargs.get('meas_circuit', rb.default_meas_circuit)

        # Store the URB results. Initialize to an empty dictionary
        self._results = dict()

        # Store the URB parameters. Initialize to an empty dictionary
        self._params = dict()

    @property
    def params(self) -> dict:
        r"""
        Return the used parameters in unitarity randomized benchmarking in a dictionary
        """
        if not self._params:
            urb_params = dict()
            urb_params['qc'] = get_qc_name(self._qc)
            urb_params['qubits'] = self._qubits
            urb_params['seq_lengths'] = self._seq_lengths
            urb_params['repeats'] = self._repeats
            urb_params['shots'] = self._shots
            urb_params['prep_circuit'] = self._prep_circuit
            urb_params['meas_circuit'] = self._meas_circuit
            self._params = urb_params

        return self._params

    @property
    def results(self) -> dict:
        """Return the unitarity randomized benchmarking results in a
        dictionary."""
        # If the randomized benchmarking results have not been generated yet,
        # call the benchmark function to generate the results using the default parameters
        if (self._results is None) or (bool(self._results) is False):
            self.benchmark(self._qc, self._qubits)

        return self._results

    def _fit_func(self, m: np.ndarray, u: float, A: float, B: float) -> np.ndarray:
        r"""The fit function used in the unitarity randomized benchmarking.

        The used fit function is an exponential function in the input and is defined as follows:

        .. math:: p(x) = A u^{m-1} + B,

        where

        + :math:`m` is the sequence length, i.e., the number of Cliffords in the sequence.
        + :math:`u` is the unitarity of the Cliffords.
        + :math:`A` and :math:`B` absorb the state preparation and measurement errors (SPAM).

        :param m: int, corresponds to the sequence length.
        :param u: float, the unitarity of the noise.
        :param A: float, a parameter that absorbs the state preparation and measurement errors.
        :param B: float, another parameter that absorbs the state preparation and measurement errors.
        :return: np.ndarray, the estimated expectation value.
        """
        return A * u ** (m - 1) + B

    def benchmark(self, qc: QComputer, qubits: List[int], method=1, **kwargs) -> dict:
        r"""Execute the unitarity randomized benchmarking procedure on the quantum computer.

        The parameters `qc` and `qubits` must be set either by the init() function or here,
        otherwise the unitarity randomized benchmarking procedure will not carry out.

        Optional keywords list are

        + ``qubits2``: List[int], default to None. For example, if using `method=1`, qubits1=[0, 1],
                default setting for qubits2 will be [2, 3]. Otherwise, qubits2 should be given.
        + ``seq_lengths``: List[int], default to :math:`[1, 10, 20, 50, 75, 100]`, the list of sequence lengths.
        + ``repeats``: int, default to :math:`6`, the number of repetitions of each sequence length.
        + ``shots``: int, default to :math:`4096`, the number of shots each measurement should carry out.
        + ``prep_circuit``: default to `default_prep_circuit`, prepares the initial quantum state
                :math:`\vert 0\cdots 0\rangle`.
        + ``meas_circuit``: default to `default_meas_circuit`, add the Z basis measurement and
            set the quantum observable to :math:`\vert 0\cdots 0\rangle\!\langle 0\cdots 0\vert`.
        + ``method``: int, default to `method=1`. There are three methods to implement the URB.
            if `method = 1`, using two copies of the experiment
            that are run in parallel and a SWAP gate applied prior to measurement.
            It took the shortest time and with good performance, but it needs 2*n qubits.

            if `method = 2`, using only one copy, it adds up the expecatation values over measurement
            over all non-identity pauli basis for the same sequence. the consuming time is longer then `method=1`
            and variance is large, but it only needs n qubits.

            if `method = 3` , using "single copy implementation",
            which has small variance, but it took the longest time and only valid for single qubit.


        **Usage**

        .. code-block:: python
            :linenos:

            urb_results = urb.benchmark(qubits=[1], qc=qc)
            urb_results = urb.benchmark(qubits=[1], qc=qc, seq_lengths=[1,10,50,100])
            urb_results = urb.benchmark(qubits=[1], qc=qc, seq_lengths=[1,10,50,100], repeats=10, shots=1024)

            u = urb_results['u']  # the estimated unitarity
            u_err = urb_results['u_err']  # the standard deviation error of the estimation

        :return: dict, the randomized benchmarking results

        **Examples**

            >>> import qiskit
            >>> from qiskit.providers.fake_provider import FakeSantiago
            >>> from qcompute_qep.benchmarking.unitarityrb import UnitarityRB
            >>> qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())
            >>> urb = UnitarityRB(qubits=[0], qc=qc)
            >>> urb_results = urb.benchmark()
            >>> urb.plot_results()
        """
        # Parse the arguments from the key list. If not set, use default arguments from the init function
        self._qc = qc if qc is not None else self._qc
        self._qubits = qubits if qubits is not None else self._qubits
        self._qubits2 = kwargs.get('qubits2', self._qubits2)
        self._seq_lengths = kwargs.get('seq_lengths', self._seq_lengths)
        self._repeats = kwargs.get('repeats', self._repeats)
        self._shots = kwargs.get('shots', self._shots)
        self._prep_circuit = kwargs.get('prep_circuit', self._prep_circuit)
        self._meas_circuit = kwargs.get('meas_circuit', self._meas_circuit)

        if method == 1:
            if self._qubits2 is None:
                if self._qubits != list(range(len(self._qubits))):
                    raise ArgumentError('The qubits2 should be given.')
                else:
                    self._qubits2 = [x + len(self._qubits) for x in self._qubits]
            else:
                if len(self._qubits2) != len(self._qubits):
                    raise ArgumentError('The qubits2 should have same lengths with qubits1')

        if self._qc is None:
            raise QEPError.ArgumentError("URB: the quantum computer for benchmarking is not specified!")
        if self._qubits is None:
            raise QEPError.ArgumentError("URB: the qubits for benchmarking are not specified!")
        if method == 3:
            if len(self._qubits) > 1:
                raise QEPError.ArgumentError("URB: For now, single copy implementation is invalid for multi-qubit!")

        ###############################################################################################################
        # Step 1. Data Collection Phase.
        #   First construct the list of benchmarking quantum circuits.
        #   Then for each RB quantum circuit, evaluate its expectation value.
        ###############################################################################################################
        # Store the estimated expectation values, which is a :math:`R \times M` array,
        # where :math:`R` is the number of repeats and :math:`M` is the number of sequences
        expvals = np.empty([len(self._seq_lengths), self._repeats], dtype=float)

        n = len(self._qubits)  # number of qubits that we actually use
        d = 2 ** n
        # number of qubits that we need to register
        if method == 1:
            num_of_register_qubits = max(x for x in self._qubits + self._qubits2) + 1
        else:
            num_of_register_qubits = max(x for x in self._qubits) + 1

        # the progress bar
        pbar = tqdm(total=100, desc='Step 1/1 : Implement the URB...', ncols=100)
        for m, seq_m in enumerate(self._seq_lengths):
            for r in range(self._repeats):
                cliffords = clifford.random_clifford(n, seq_m)
                if method == 1:
                    # Using swap gate
                    # This method has best performance for now.
                    expval = self._swap_gate_method(num_of_register_qubits, cliffords, **kwargs)
                    expvals[m, r] = (d * expval - 1) / (d - 1)
                elif method == 2:
                    # Measuring in the non-identity Pauli basis
                    expval = self._pauli_basis_method(num_of_register_qubits, cliffords, **kwargs)
                    expvals[m, r] = expval * d / (d - 1)
                elif method == 3:
                    # using simple copy implementation method
                    # for more details see reference
                    # "Efficient unitarity randomized benchmarking of few-qubit Clifford gates"
                    expval = self._single_copy_implementation_method(num_of_register_qubits, cliffords, **kwargs)
                    expvals[m, r] = expval / (2 * (d * d - 1))
                else:
                    raise ArgumentError('The method is not supported in UnitarityRB')

                # update the progress bar
                pbar.update((100 / (self._repeats * len(self._seq_lengths))))
        pbar.close()

        ###############################################################################################################
        # Step 2. Data Processing Phase.
        #   Fit the list of averaged expectation values to the exponential model and extract the fitting results.
        ###############################################################################################################

        # For single_copy_implementation
        # it has different fitting model.
        if method == 3:
            # Set the bounds for the parameters tuple: :math:`(u, A, B)`
            bounds = ([0, 0, ], [1, 1, ])
            # Use scipy's non-linear least squares to fit the data
            xdata = self._seq_lengths
            ydata = np.log(np.mean(expvals, axis=1))
            sigma = np.std(expvals, axis=1)
            if len(sigma) - np.count_nonzero(sigma) > 0:
                sigma = None

            p0 = [0.99, 0.95, ]

            alpha_guess = 0
            count = 0
            for j in range(1, len(xdata)):
                dx = (xdata[j] - xdata[0])
                dy = (ydata[j] - ydata[0])
                alpha_guess += (dy / dx)
                count += 1
            alpha_guess = np.exp(alpha_guess / count)
            if alpha_guess < 1.0:
                p0[0] = alpha_guess

            tmp = 0
            count = 0
            for j in range(len(ydata)):
                tmp += (ydata[j] - np.log(p0[0]) * (xdata[j] - 1))
                count += 1
            p0[1] = np.exp(tmp / count)

            def func(m, u, A):
                return (m - 1) * np.log(u) + np.log(A)

            popt, pcov = curve_fit(func, xdata, ydata,
                                   p0=p0, sigma=sigma,
                                   bounds=bounds, method='dogbox')

            # Store the randomized benchmarking results
            params_err = np.sqrt(np.diag(pcov))
            self._results['expvals'] = expvals
            self._results['u'] = popt[0]
            self._results['A'] = popt[1]
            self._results['B'] = 0
            self._results['u_err'] = params_err[0]
        else:
            bounds = ([0, 0, 0], [1, 1, 1])
            # Use scipy's non-linear least squares to fit the data
            xdata = self._seq_lengths
            ydata = np.mean(expvals, axis=1)
            sigma = np.std(expvals, axis=1)
            if len(sigma) - np.count_nonzero(sigma) > 0:
                sigma = None

            p0 = [0.99, 0.95, 0]

            alpha_guess = []
            for j in range(1, len(xdata)):
                if ydata[j] > p0[2]:
                    dx = (xdata[j] - xdata[0])
                    dy = ((ydata[j] - p0[2]) / (ydata[0] - p0[2]))
                    alpha_guess.append(dy ** (1 / dx))
            if alpha_guess:
                if np.mean(alpha_guess) < 1.0:
                    p0[0] = np.mean(alpha_guess)

            tmp = []
            for j in range(len(ydata)):
                if ydata[j] > p0[2]:
                    tmp.append((ydata[j] - p0[2]) / p0[0] ** (xdata[j] - 1))
            if tmp:
                if np.mean(tmp) < 1.0:
                    p0[1] = np.mean(tmp)

            popt, pcov = curve_fit(self._fit_func, xdata, ydata,
                                   p0=p0, sigma=sigma,
                                   bounds=bounds, method='dogbox')

            # Store the randomized benchmarking results
            params_err = np.sqrt(np.diag(pcov))
            self._results['expvals'] = expvals
            self._results['u'] = popt[0]
            self._results['A'] = popt[1]
            self._results['B'] = popt[2]
            self._results['u_err'] = params_err[0]

        return self._results

    def plot_results(self, show: bool = True, fname: str = None):
        r"""Plot unitarity randomized benchmarking results.

        Commonly, we visualize the sampled and averaged expectation values for each given length,
        the fitted function, and the estimated unitarity.

        :param show: bool, default to True, show the plot figure or not
        :param fname: figure name for saving. If fname is None, do not save the figure
        """

        if not HAS_MATPLOTLIB:
            raise ImportError('Function "plot_results" requires matplotlib. Run "pip install matplotlib" first.')

        fig, ax = plt.subplots(figsize=(12, 8))

        xdata = self._seq_lengths
        expvals = self.results['expvals']

        # Plot the repeated estimates for each sequence
        ax.plot(xdata, expvals, color='gray', linestyle='none', marker='x')

        # Plot the mean of the estimated expectation values
        ax.plot(xdata, np.mean(expvals, axis=1), color='blue', linestyle='none', marker='v', markersize=13)

        # Plot the confidence interval of the fitting curve
        low_CI_bound, high_CI_bound = st.t.interval(0.95, len(xdata), loc=np.mean(expvals, axis=1),
                                                    scale=st.sem(expvals, axis=1))
        plt.fill_between(xdata, y1=low_CI_bound, y2=high_CI_bound, color='cornflowerblue', alpha=0.3, )

        # Plot the fitting function
        # ydata = [self._fit_func(x, self.results['u'], self.results['A'], self.results['B']) for x in xdata]
        fitting_curve_xdata = np.arange(xdata[0], xdata[-1] + 1, 1)
        ydata = [self._fit_func(x, self.results['u'], self.results['A'], self.results['B']) for x in
                 fitting_curve_xdata]
        ax.plot(fitting_curve_xdata, ydata, color='blue', linestyle='-', linewidth=2, label='fitting curve')
        ax.tick_params(labelsize='medium')

        # Set the labels
        ax.set_xlabel('Clifford Length', fontsize='large')
        ax.set_ylabel('Expectation Value', fontsize='large')
        ax.grid(True)

        # Shows the legend
        plt.legend(loc='lower left', fontsize=16)

        # Add the estimated fidelity and EPC parameters
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
        ax.text(0.85, 0.9,
                "Unitarity: {:.3f}({:.1e}) \n".format(self.results['u'],
                                                      self.results['u_err']),
                ha="center", va="center", fontsize=12, bbox=bbox_props, transform=ax.transAxes)

        # Set the x-axis locator always be integer
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save the figure if `fname` is set
        if fname is not None:
            plt.savefig(fname, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)
        # Show the figure if `show==True`
        if show:
            plt.show()

    def _single_copy_implementation_method(self, n, cliffords, **kwargs):
        expval = 0
        init_qp1 = QCompute.QEnv()
        init_qp2 = QCompute.QEnv()
        q1 = init_qp1.Q.createList(n)
        q2 = init_qp2.Q.createList(n)

        # Apply the clifford gates
        for c in cliffords:
            c(q1, self._qubits)
            c(q2, self._qubits)

        for prep_ch in itertools.product([('0', '1'), ('A', 'D'), ('L', 'R')], repeat=n):
            ch1, ch2 = zip(*prep_ch)
            rb_qp1 = _prep_circuit(ch1, init_qp1, self._qubits)
            rb_qp2 = _prep_circuit(ch2, init_qp2, self._qubits)

            # The same state measuring in complete Pauli basis,
            # For n qubits, there are totally 4^n-1 Pauli basis
            # subtracted off the identity Pauli basis
            for pauli_str in itertools.product(['I', 'X', 'Y', 'Z'], repeat=n):
                p = list(pauli_str)
                if all(x == 'I' for x in p):
                    continue
                meas_qp1, meas_ob1 = _meas_circuit(p, rb_qp1, self._qubits)
                counts1 = execute(qp=meas_qp1, qc=self._qc, **kwargs)
                meas_qp2, meas_ob2 = _meas_circuit(p, rb_qp2, self._qubits)
                counts2 = execute(qp=meas_qp2, qc=self._qc, **kwargs)
                a = counts_list(meas_ob1, counts1)
                b = counts_list(meas_ob2, counts2)
                x = a - b
                # unbiased estimator
                unbiased_estimator = 1 / self._shots * np.var(x, ddof=1)

                value = np.mean(x)
                expval += value * value - unbiased_estimator
        return expval

    def _swap_gate_method(self, n, cliffords, **kwargs):
        qubits1 = self._qubits
        qubits2 = self._qubits2
        num_qubits = len(qubits1)
        # Using two copies of the experiment (with the same sequence) that are run in parallel
        # and a SWAP gate applied immediately prior to measurement.
        rb_qp = QCompute.QEnv()
        q = rb_qp.Q.createList(n)

        # Apply the clifford gates
        for c in cliffords:
            c(q, qubits1)
            c(q, qubits2)

        meas_qp = _swap_meas_circuit(rb_qp, qubits1, qubits2)

        counts = execute(qp=meas_qp, qc=self._qc, **kwargs)

        total = 0
        for k, v in counts.items():
            cbits = 0
            for i in range(num_qubits):
                cbits ^= int(k[i]) & int(k[i + num_qubits])
            if cbits:
                total += v
        expval = (1 - 2 * total / self._shots)
        return expval

    def _pauli_basis_method(self, n, cliffords, **kwargs):
        expval = 0
        rb_qp = QCompute.QEnv()
        q = rb_qp.Q.createList(n)
        for c in cliffords:
            c(q, self._qubits)

        # The same state measuring in complete Pauli basis,
        # For n qubits, there are totally 4^n-1 Pauli basis
        # subtracted off the identity Pauli basis
        for pauli_str in itertools.product(['I', 'X', 'Y', 'Z'], repeat=n):
            p = list(pauli_str)
            if all(x == 'I' for x in p):
                continue
            meas_qp, meas_ob = _meas_circuit(p, rb_qp, self._qubits)
            counts = execute(qp=meas_qp, qc=self._qc, **kwargs)
            value = expval_from_counts(meas_ob, counts)
            expval += value * value
        return expval


def counts_list(A: np.ndarray, counts: dict) -> np.ndarray:
    r"""It is only used in class UnitarityRB.

    :param A: np.ndarray, a Hermitian operator that is diagonalized in the measurement basis
    :param counts: dict, dict-type counts data, means result of shot measurements,
            e.g. ``{'000000': 481, '111111': 519}``
    :return: np.ndarray, the expectation value list
    """
    expects = []
    if list(counts.keys())[0][:2].lower() == '0x':
        bits = len(bin(max(map(lambda x: int(x, 16), counts.keys())))[2:])
    else:
        bits = None
    for k, v in counts.items():
        state = str_to_state(k, bits=bits)
        if state.shape != A.shape:
            raise ValueError("Shapes of density matrix and operator are not equal!")
        expects.extend([np.real(np.trace(state @ A))] * v)
    return np.array(expects)


def _prep_circuit(ch, qp, qubits) -> QProgram:
    """The function is testing for class UnitarityRB.

    Will be updated in the future. For more information, see
    qcompute_qep.tomography.basis.PauliPrepBasis.
    """
    if isinstance(qp, QCompute.QEnv):
        n = number_of_qubits(qp)
        prep_qp = deepcopy(qp)
        for i in range(len(ch)):
            idx = n - i - 1
            c = ch[i]
            qubit_idx = qubits[idx]
            if c == '0':
                pass
            elif c == '1':  # Execute X on the target qubit to the beginning of the quantum program
                clX = CircuitLine(data=X, qRegList=[qubit_idx])
                prep_qp.circuit = [clX] + prep_qp.circuit
            elif c == 'A':  # Execute H on the target qubit to the beginning of the quantum program
                clH = CircuitLine(data=H, qRegList=[qubit_idx])
                prep_qp.circuit = [clH] + prep_qp.circuit
            elif c == 'D':  # Execute X and H on the target qubit to the beginning of the quantum program
                clX = CircuitLine(data=X, qRegList=[qubit_idx])
                clH = CircuitLine(data=H, qRegList=[qubit_idx])
                prep_qp.circuit = [clX, clH] + prep_qp.circuit
            elif c == 'L':  # Execute H and S on the target qubit to the beginning of the quantum program
                clH = CircuitLine(data=H, qRegList=[qubit_idx])
                clS = CircuitLine(data=S, qRegList=[qubit_idx])
                prep_qp.circuit = [clH, clS] + prep_qp.circuit
            elif c == 'R':  # Execute X, H and S on the target qubit to the beginning of the quantum program
                clX = CircuitLine(data=X, qRegList=[qubit_idx])
                clH = CircuitLine(data=H, qRegList=[qubit_idx])
                clS = CircuitLine(data=S, qRegList=[qubit_idx])
                prep_qp.circuit = [clX, clH, clS] + prep_qp.circuit
        return prep_qp


def _meas_circuit(pauli, qp, qubits):
    """This function is only used in class UnitarityRB.

    Will be updated in the future. For more information, see
    qcompute_qep.tomography.basis.PauliMeasBasis.
    """
    if isinstance(qp, QCompute.QEnv):
        eigs = []
        n = len(qubits)
        meas_qp = deepcopy(qp)
        q = meas_qp.Q
        for idx in range(len(pauli)):
            i = n - idx - 1
            P = pauli[i]
            if P == 'X':
                QCompute.H(q[qubits[i]])
            elif P == 'Y':
                QCompute.SDG(q[qubits[i]])
                QCompute.H(q[qubits[i]])
            else:
                pass
            eigs.append(np.diag([1, 1]) / np.sqrt(2) if P == 'I' else np.diag([1, -1]) / np.sqrt(2))

        # If the given quantum program does not contain a measurement, measure it in the Z basis
        if not circuit.contain_measurement(qp):
            qreglist, indexlist = qp.Q.toListPair()
            QCompute.MeasureZ(qRegList=[qreglist[x] for x in qubits],
                              cRegList=[indexlist[x] for x in qubits])

        meas_ob = tensor(eigs)
        return meas_qp, meas_ob


def _swap_meas_circuit(qp, qubits1, qubits2):
    """This function is only used in class UnitarityRB.

    Will be updated in the future.
    """

    n = len(qubits1)
    meas_qp = qp
    q = meas_qp.Q

    for i in range(n):
        CX(q[qubits2[i]], q[qubits1[i]], )
        H(q[qubits2[i]])
    # If the given quantum program does not contain a measurement, measure it in the Z basis
    if not circuit.contain_measurement(qp):
        qreglist, indexlist = qp.Q.toListPair()
        QCompute.MeasureZ(qRegList=[qreglist[x] for x in qubits1 + qubits2],
                          cRegList=[indexlist[x] for x in qubits1 + qubits2])

    return meas_qp
