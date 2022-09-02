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
Interleaved Randomized Benchmarking.
A scalable and robust algorithm for benchmarking the specific individual Clifford gate
by a single parameter called error rate using randomization techniques.
"""
from typing import List
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator
import scipy.stats as st
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import pylab

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

import QCompute
from qcompute_qep.utils import expval_from_counts, execute
from qcompute_qep.quantum import clifford
from qcompute_qep.utils.types import QComputer, get_qc_name
import qcompute_qep.exceptions.QEPError as QEPError
import qcompute_qep.benchmarking as rb


class InterleavedRB(rb.RandomizedBenchmarking):
    r"""
    The Interleaved Randomized Benchmarking class.
    Aim to benchmark the specific individual Clifford gate by average error-rate.
    """

    def __init__(self, target_gate: clifford.Clifford = None, qc: QComputer = None, qubits: List[int] = None, **kwargs):
        r"""init function of the Standard Randomized Benchmarking class.

        Optional keywords list are:

        + ``target_gate``: class Clifford, the target clifford gate that we want to benchmark.
        + ``seq_lengths``: List[int], default to :math:`[1, 10, 20, 50, 75, 100]`, a list of sequence lengths
        + ``repeats``: int, default to :math:`6`, the number of repetitions of each sequence length
        + ``shots``: int, default to :math:`4096`, the number of shots each measurement carries out to estimate value
        + ``prep_circuit``: default to `default_prep_circuit`, prepares the initial quantum state
                                :math:`\vert 0\cdots 0\rangle`
        + ``meas_circuit``: default to `default_meas_circuit`, add the Z basis measurement to the end of
                                the IRB circuits and set the quantum observable to
                                :math:`\vert 0\cdots 0\rangle\!\langle 0\cdots 0\vert`

        for ``prep_circuit`` and ``meas_circuit`` see more details in benchmarking.utils.default_prep_circuit
                                and benchmarking.utils.default_meas_circuit

        :param qc: QComputer, the quantum computer on which the RB carries out
        :param qubits: List[int], the qubits who will be benchmarked
        """
        # Initialize the Standard RB parameters. If not set, use the default parameters
        super().__init__(**kwargs)

        self._target_gate = target_gate
        self._qc = qc
        self._qubits = qubits
        self._seq_lengths = kwargs.get('seq_lengths', [1, 10, 20, 50, 75, 100])
        self._repeats = kwargs.get('repeats', 6)
        self._shots = kwargs.get('shots', 4096)
        self._prep_circuit = kwargs.get('prep_circuit', rb.default_prep_circuit)
        self._meas_circuit = kwargs.get('meas_circuit', rb.default_meas_circuit)

        # Store the standard randomized benchmarking results. Initialize to an empty dictionary
        self._results = dict()

        # Store the standard randomized benchmarking parameters. Initialize to an empty dictionary
        self._params = dict()

    @property
    def target_gate(self) -> clifford.Clifford:
        """
        Return the target gate of Interleaved randomized benchmarking.
        """
        if self._target_gate is None:
            raise QEPError.ArgumentError('Interleaved RB: The target gate is not set yet')
        return self._target_gate

    @property
    def results(self) -> dict:
        """
        Return the randomized benchmarking results in a dictionary.

        In InterleavedRB, the results contains three more dictionary.
        which is ['StandardRB'], ['InterleavedRB'], ['target_gate'].

        **Usage**

        .. code-block:: python
            :linenos:

            # the estimated error-rate parameter of target gate
            error-rate = irb_results[target_gate]['r']
            # the standard deviation error of the estimation error-rate
            error-rate_err = irb_results[target_gate]['r_err']
            # the estimated average gate fidelity parameter of StandardRB
            SRB_fidelity = irb_results['StandardRB']['f']
            SRB_EPC = irb_results['StandardRB']['epc'] # the estimated EPC parameter of StandardRB
            # the estimated average gate fidelity parameter of InterleavedRB
            IRB_fidelity = irb_results['InterleavedRB']['f']
            IRB_EPC = irb_results['InterleavedRB']['epc'] # the estimated EPC parameter of InterleavedRB
        """
        # If the randomized benchmarking results have not been generated yet,
        # call the benchmark function to generate the results using the default parameters
        if (self._results is None) or (bool(self._results) is False):
            self.benchmark(self._target_gate, self._qc, self._qubits)

        return self._results

    @property
    def params(self) -> dict:
        r"""Return the used parameters in randomized benchmarking in a dictionary.
        """
        if not self._params:
            rb_params = dict()
            rb_params['qc'] = get_qc_name(self._qc)
            rb_params['qubits'] = self._qubits
            rb_params['seq_lengths'] = self._seq_lengths
            rb_params['repeats'] = self._repeats
            rb_params['shots'] = self._shots
            rb_params['prep_circuit'] = self._prep_circuit
            rb_params['meas_circuit'] = self._meas_circuit
            rb_params['target_gate'] = self._target_gate
            self._params = rb_params

        return self._params

    def _fit_func(self, x: np.ndarray, f: float, A: float, B: float) -> np.ndarray:
        r"""The fit function used in the Interleaved randomized benchmarking.

        The used fit function is an exponential function in the input and is defined as follows:

        .. math:: p(x) = A f^{2x-1} + B,

        where

        + :math:`x` is the sequence length, i.e., the number of the composite gates Clifford gates in the sequence,
        + :math:`f` is the fidelity parameter of the twirled depolarizing channel,
        + :math:`A` and :math:`B` absorb the State Preparation and Measurement errors (SPAM).

        note that we simply apply the inverse of every Clifford gates on the circuit,
        so the function turn to be like this.

        :param x: int, corresponds to the sequence length
        :param f: float, the fidelity parameter of the depolarizing channel
        :param A: float, a parameter that absorbs the state preparation and measurement errors
        :param B: float, another parameter that absorbs the state preparation and measurement errors
        :return: np.ndarray, the estimated expectation value
        """
        return A * f ** (2 * x - 1) + B

    def benchmark(self, target_gate: clifford.Clifford, qc: QComputer, qubits: List[int], **kwargs) -> dict:
        r"""Execute the randomized benchmarking procedure on the quantum computer.
        For more details of estimating target gate error-rate, see these Ref. [1].

        The parameters `target_gate`, `qc` and `qubits` must be set either by the init() function or here,
        otherwise the randomized benchmarking procedure will not carry out.

        Optional keywords list are:

        + ``target_gate``: class Clifford, the target clifford gate that we want to benchmark.
        + ``seq_lengths``: List[int], default to :math:`[1, 10, 20, 50, 75, 100]`, the list of sequence lengths
        + ``repeats``: int, default to :math:`6`, the number of repetitions of each sequence length
        + ``shots``: int, default to :math:`4096`, the number of shots each measurement should carry out
        + ``prep_circuit``: default to `default_prep_circuit`, prepares the initial quantum state
                                :math:`\vert 0\cdots 0 \rangle`
        + ``meas_circuit``: default to `default_meas_circuit`, add the Z basis measurement
                                and set the quantum observable to :math:`\vert 0\cdots 0\rangle\!\langle 0\cdots 0\vert`

        Reference:
        [1]. Efficient measurement of quantum gate error by interleaved randomized benchmarking.
            https://arxiv.org/abs/1203.4550.

        **Usage**

        .. code-block:: python
            :linenos:

            irb_results = irb.benchmark(qubits=[1], qc=qc)
            irb_results = irb.benchmark(qubits=[1], qc=qc, seq_lengths=[1,10,50,100])
            irb_results = irb.benchmark(qubits=[1], qc=qc, seq_lengths=[1,10,50,100], repeats=10, shots=1024)


        :return: dict, the Interleaved randomized benchmarking results

        **Examples**

            >>> import qiskit
            >>> from qiskit.providers.fake_provider import FakeSantiago
            >>> from qcompute_qep.benchmarking.interleavedrb import InterleavedRB
            >>> from qcompute_qep.quantum.clifford import Clifford
            >>> qc = qiskit.providers.aer.AerSimulator.from_backend(FakeParis())
            >>> target_gate=Clifford(1)
            >>> irb = InterleavedRB()
            >>> irb_results = irb.benchmark(target_gate=target_gate,qubits=[1], qc=qc)
            >>> print(irb_results)
            >>> irb.plot_results()
        """
        # Parse the arguments from the key list. If not set, use default arguments from the init function
        self._target_gate = target_gate if target_gate is not None else self._target_gate
        self._qc = qc if qc is not None else self._qc
        self._qubits = qubits if qubits is not None else self._qubits
        self._seq_lengths = kwargs.get('seq_lengths', self._seq_lengths)
        self._repeats = kwargs.get('repeats', self._repeats)
        self._shots = kwargs.get('shots', self._shots)
        self._prep_circuit = kwargs.get('prep_circuit', self._prep_circuit)
        self._meas_circuit = kwargs.get('meas_circuit', self._meas_circuit)

        if self._target_gate is None:
            raise QEPError.ArgumentError('Interleaved RB: the target gate is not specified')
        else:
            if not isinstance(target_gate, clifford.Clifford):
                raise QEPError.ArgumentError(
                    'Interleaved RB: the target gate should be class Clifford, but received {}. For more details'
                    'see qcoumpute_qep.quantum.clifford.Clifford'.format(type(target_gate)))
        if self._qc is None:
            raise QEPError.ArgumentError("Interleaved RB: the quantum computer for benchmarking is not specified!")
        if self._qubits is None:
            raise QEPError.ArgumentError("Interleaved RB: the qubits for benchmarking are not specified!")
        print('There are two steps to run')
        # First implement the Standard Randomized Benchmarking
        srb = rb.StandardRB()
        srb.benchmark(qc=self._qc,
                      qubits=self._qubits,
                      **kwargs)
        self._results['StandardRB'] = srb.results

        ###############################################################################################################
        # Step 1. Data Collection Phase.
        #   First construct the list of benchmarking quantum circuits.
        #   Then for each RB quantum circuit, evaluate its expectation value.
        ###############################################################################################################
        # Store the estimated expectation values, which is a :math:`R \times M` array,
        # where :math:`R` is the number of repeats and :math:`M` is the number of sequences

        expvals = np.empty([len(self._seq_lengths), self._repeats], dtype=float)
        n = len(self._qubits)  # number of qubits
        num_of_register_qubits = max(x for x in self._qubits) + 1
        pbar = tqdm(total=100, desc='Step 2/2 : Implement the IRB...', ncols=100)
        for m, seq_m in enumerate(self._seq_lengths):
            for r in range(self._repeats):
                # Construct a random sequence of Clifford gates of length seq_m
                cliffords = clifford.random_clifford(n, seq_m)

                # Setup the randomized benchmarking quantum circuit
                rb_qp = QCompute.QEnv()
                q = rb_qp.Q.createList(num_of_register_qubits)
                # Create a list to store the inverse circuit of every Clifford gate
                inv_circuit = []
                for i, c in enumerate(cliffords):
                    self._target_gate(q, self._qubits)
                    inv_circuit.append(self._target_gate.get_inverse_circuit(self._qubits))
                    c(q, self._qubits)
                    inv_circuit.append(c.get_inverse_circuit(self._qubits))
                # Reverse the sequence of applied Clifford gates and operate them on the inputs
                rb_qp.circuit += sum(inv_circuit[::-1], [])
                # Prepare the input quantum state
                rb_qp = self._prep_circuit(rb_qp)
                # Add the desired measurement corresponds to the target quantum observable
                rb_qp, rb_ob = self._meas_circuit(rb_qp)

                # Run the RB quantum circuit, estimate the expectation value, and store the data
                counts = execute(qp=rb_qp, qc=self._qc, **kwargs)
                expval = expval_from_counts(A=rb_ob, counts=counts)
                expvals[m, r] = expval
                pbar.update((100 / (self._repeats * len(self._seq_lengths))))
        pbar.close()

        ###############################################################################################################
        # Step 2. Data Processing Phase.
        #   Fit the list of averaged expectation values to the exponential model and extract the fitting results.
        ###############################################################################################################
        # Set the bounds for the parameters tuple: :math:`(f, A, B)`
        bounds = ([0, 0, 1 / 2 ** n], [1, 1, 1])

        # Use scipy's non-linear least squares to fit the data
        xdata = self._seq_lengths
        ydata = np.mean(expvals, axis=1)
        sigma = np.std(expvals, axis=1)
        if len(sigma) - np.count_nonzero(sigma) > 0:
            sigma = None

        p0 = [0.99, 0.95, 1 / 2 ** n]

        alpha_guess = []
        for j in range(1, len(xdata)):
            if ydata[j] > p0[2]:
                dx = (xdata[j] - xdata[0])
                dy = ((ydata[j] - p0[2]) / (ydata[0] - p0[2]))
                alpha_guess.append(dy ** (1 / (2 * dx)))
        if alpha_guess:
            if np.mean(alpha_guess) < 1.0:
                p0[0] = np.mean(alpha_guess)

        tmp = []
        for j in range(len(ydata)):
            if ydata[j] > p0[2]:
                tmp.append((ydata[j] - p0[2]) / (p0[0] ** (2 * xdata[j])))

        if tmp:
            if np.mean(tmp) < 1.0:
                p0[1] = np.mean(tmp)

        popt, pcov = curve_fit(self._fit_func, xdata, ydata,
                               p0=p0, sigma=sigma, maxfev=500000,
                               bounds=bounds, method='dogbox')

        # Store the Interleaved randomized benchmarking results in results['InterleavedRB']
        params_err = np.sqrt(np.diag(pcov))
        _results = dict()
        _results['expvals'] = expvals
        _results['f'] = popt[0]
        _results['A'] = popt[1]
        _results['B'] = popt[2]
        _results['f_err'] = params_err[0]
        d = 2 ** len(self._qubits)
        _results['epc'] = (d - 1) / d * (1 - popt[0])
        _results['epc_err'] = (d - 1) / d * (1 - params_err[0])
        self._results['InterleavedRB'] = _results

        # target gate results
        _results = dict()
        alpha_c = self._results['InterleavedRB']['f']
        alpha_c_err = self._results['InterleavedRB']['f_err']
        alpha = self._results['StandardRB']['f']
        alpha_err = self._results['StandardRB']['f_err']

        # Calculate r_target (=r_c^est) Ref.[1] - Eq. (4):
        r_target = (d - 1) * (1 - alpha_c / alpha) / d

        # Calculate the systematic error bounds Ref.[1] - Eq. (5):
        bound_1 = (d - 1) * (abs(alpha - alpha_c / alpha) + (1 - alpha)) / d
        bound_2 = 2 * (d * d - 1) * (1 - alpha) / (alpha * d * d) \
                  + 2 * (np.sqrt(1 - alpha)) * (np.sqrt(d * d - 1)) / alpha

        bound = min(bound_1, bound_2)
        Lower = r_target - bound
        Upper = r_target + bound

        # Calculate r_target_error
        alpha_err_sq = (alpha_err / alpha) * (alpha_err / alpha)
        alpha_c_err_sq = (alpha_c_err / alpha_c) * (alpha_c_err / alpha_c)
        r_target_err = ((d - 1) / d) * (alpha_c / alpha) \
                       * (np.sqrt(alpha_err_sq + alpha_c_err_sq))

        # Store the target_gate results in results['target_gate']
        _results['r'] = r_target
        _results['r_err'] = r_target_err
        _results['r_lower_bound'] = Lower
        _results['r_upper_bound'] = Upper
        _results['bound'] = bound
        self._results['target_gate'] = _results
        return self._results

    def plot_results(self, show: bool = True, fname: str = None, ):
        r"""Plot randomized benchmarking results.

        Commonly, we visualize the sampled and averaged expectation values for each given length,
        the fitted function, and the estimated fidelity and Error Per Clifford.

        :param show: bool, default to True, show the plot figure or not
        :param fname: figure name for saving. If fname is None, do not save the figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError('Function "plot_results" requires matplotlib. Run "pip install matplotlib" first.')

        srb_results = self.results['StandardRB']
        irb_results = self.results['InterleavedRB']
        target_gate_results = self.results['target_gate']

        xdata = self._seq_lengths
        srb_expvals = srb_results['expvals']
        irb_expvals = irb_results['expvals']
        srb_mean = np.mean(srb_expvals, axis=1)
        irb_mean = np.mean(irb_expvals, axis=1)
        fig, ax = plt.subplots(figsize=(12, 8))

        # Plot the repeated estimates for each sequence
        ax.plot(xdata, srb_expvals, color='gray', linestyle='none', marker='x')
        ax.plot(xdata, irb_expvals, color='gray', linestyle='none', marker='o')

        # Plot the fitting function
        ydata_srb = [self._fit_func(x, srb_results['f'], srb_results['A'], srb_results['B']) for x in xdata]
        ydata_irb = [self._fit_func(x, irb_results['f'], irb_results['A'], irb_results['B']) for x in xdata]
        ax.plot(xdata, ydata_srb, color='blue', linestyle='-', linewidth=2, label='StandardRB fitting curve')
        ax.plot(xdata, ydata_irb, color='green', linestyle='-', linewidth=2, label='InterleavedRB fitting curve')

        # Plot the confidence interval of the fitting curve
        srb_low_CI_bound, srb_high_CI_bound = st.t.interval(0.95, len(xdata), loc=srb_mean,
                                                            scale=st.sem(srb_expvals, axis=1))
        irb_low_CI_bound, irb_high_CI_bound = st.t.interval(0.95, len(xdata), loc=irb_mean,
                                                            scale=st.sem(irb_expvals, axis=1))
        plt.fill_between(xdata, y1=srb_low_CI_bound, y2=srb_high_CI_bound, color='cornflowerblue', alpha=0.3, )
        plt.fill_between(xdata, y1=irb_low_CI_bound, y2=irb_high_CI_bound, color='lightgreen', alpha=0.3, )

        # Plot the mean data for each lengths
        ax.plot(xdata, srb_mean, color='blue', linestyle='none', marker='v', markersize=13)
        ax.plot(xdata, irb_mean, color='green', linestyle='none', marker='*', markersize=13)

        ax.tick_params(labelsize='medium')

        # Show the legend
        plt.legend(loc='lower left', fontsize=16)

        # Add the fidelity of StandardRB and InterleavedRB
        # and estimated EPC parameters of target gate
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
        ax.text(0.7, 0.9,
                "StandardRB average gate fidelity: {:.4f}({:.3e}) \n "
                "InterleavedRB average gate fidelity: {:.4f}({:.3e})\n "
                "Target gate error rate: {:.4f}({:.3e}) \n"
                "Systematic error bound: {:.4f}".format(srb_results['f'],
                                                        srb_results['f_err'],
                                                        irb_results['f'],
                                                        irb_results['f_err'],
                                                        target_gate_results['r'],
                                                        target_gate_results['r_err'],
                                                        target_gate_results['bound']),
                ha="center", va="center", fontsize=12, bbox=bbox_props, transform=ax.transAxes)

        # Set the labels
        ax.set_xlabel('Clifford Length', fontsize='large')
        ax.set_ylabel('Expectation Value', fontsize='large')
        ax.grid(True)

        # Set the x-axis locator always be integer
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save the figure if `fname` is set
        if fname is not None:
            plt.savefig(fname, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)
        # Show the figure if `show==True`
        if show:
            plt.show()
