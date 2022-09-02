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

"""Cross Entropy Benchmarking."""
from scipy.optimize import curve_fit
from typing import List
import numpy as np
from tqdm import tqdm
from matplotlib.ticker import MaxNLocator
import scipy.stats as st
import warnings

warnings.filterwarnings('ignore')

try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import pylab

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from qcompute_qep.utils.types import QProgram, QComputer, get_qc_name
import qcompute_qep.utils.circuit as circuit
import qcompute_qep.exceptions.QEPError as QEPError
import qcompute_qep.benchmarking as rb


class XEB(rb.RandomizedBenchmarking):
    """The Cross Entropy Benchmarking class.

    Aim to benchmark the quantum device by using cross entropy method.
    """

    def __init__(self, qc: QComputer = None, qubits: List[int] = None, **kwargs):
        r"""init function of the Cross Entropy Benchmarking class.

        Optional keywords list are:

        + ``seq_lengths``: List[int], default to :math:`[1, 10, 20, 50, 75, 100]`, a list of sequence lengths
        + ``repeats``: int, default to :math:`6`, the number of repetitions of each sequence length
        + ``shots``: int, default to :math:`4096`, the number of shots each measurement carries out to estimate value
        + ``entropy_type``: default to `linear`, the entropy type of XEB `linear` or `log`.
        + ``prep_circuit``: default to `default_prep_circuit`, prepares the initial
                            quantum state :math:`\vert 0\cdots 0 \rangle`
        + ``meas_circuit``: default to `default_meas_circuit`, add the Z basis measurement
                            to the end of the XEB circuits and set the quantum observable to
                            :math:`\vert 0\cdots 0\rangle\!\langle 0\cdots 0\vert`

        for `prep_circuit` and `meas_circuit`, see more details in benchmarking.utils.default_prep_circuit
                        and benchmarking.utils.default_meas_circuit.

        :param qc: QComputer, the quantum computer on which the XEB carries out
        :param qubits: List[int], the qubits who will be benchmarked
        """
        # Initialize the XEB parameters. If not set, use the default parameters
        super().__init__(**kwargs)
        self._qc = qc
        self._qubits = qubits if qubits is not None else list(range(2))
        self._seq_lengths = kwargs.get('seq_lengths', [1, 10, 20, 50, 75, 100, 125, 150, 175, 200])
        self._repeats = kwargs.get('repeats', 6)
        self._shots = kwargs.get('shots', 4096)
        self._entropy_type = kwargs.get('entropy_type', 'linear')

        # Store the cross entropy benchmarking results. Initialize to an empty dictionary
        self._results = dict()

        # Store the cross entropy benchmarking parameters. Initialize to an empty dictionary
        self._params = dict()

    @property
    def results(self) -> dict:
        """Return the cross entropy benchmarking results in a dictionary.

        **Usage**

        .. code-block:: python
            :linenos:

            lam = xeb_results['lambda']  # the estimated ENR
            lam_err = xeb_results['lambda_err']  # the standard deviation error of the estimation
        """
        # If the cross entropy benchmarking results have not been generated yet,
        # call the benchmark function to generate the results using the default parameters
        if (self._results is None) or (bool(self._results) is False):
            self.benchmark()

        return self._results

    @property
    def params(self) -> dict:
        """Return the used parameters in cross entropy benchmarking in a
        dictionary."""
        if not self._params:
            xeb_params = dict()
            xeb_params['qc'] = get_qc_name(self._qc)
            xeb_params['qubits'] = self._qubits
            xeb_params['seq_lengths'] = self._seq_lengths
            xeb_params['repeats'] = self._repeats
            xeb_params['shots'] = self._shots
            xeb_params['entropy_type'] = self._entropy_type
            self._params = xeb_params

        return self._params

    def _fit_func(self, d: int, lam: float, A: float) -> np.ndarray:
        r"""The fit function used in the standard cross entropy benchmarking.

        The used fit function is an exponential function in the input and is defined as follows:

        .. math:: p(x) = A e ^ { -\lambda d }

        where

        + :math:`\lambda` is the Effective Foise Rate (ENR),
        + :math:`d` is the sequence length, i.e., the number of cycles,
        + :math:`A` absorb the state preparation and measurement errors (SPAM).

        :param lam: int, corresponds to the effective noise rate
        :param d: int, corresponds to the sequence length
        :param A: float, a parameter that absorbs the State Preparation and Measurement errors

        :return: np.ndarray, the estimated expectation value
        """
        return A * np.exp(-lam * d)

    def cross_entropy(self, prob: float) -> float:

        if self._entropy_type == 'linear':
            return prob
        elif self._entropy_type == 'log':
            return - np.log2(prob)
        else:
            raise QEPError.ArgumentError("XEB: the cross entropy type {} is invalid!".format(self._entropy_type))

    def fidelity(self, average_entropy: float) -> float:
        # Number of qubits
        n = len(self._qubits)
        if self._entropy_type == 'linear':
            return 2 ** n * average_entropy - 1
        elif self._entropy_type == 'log':
            return n + np.euler_gamma - average_entropy
        else:
            raise QEPError.ArgumentError("XEB: the cross entropy type {} is invalid!".format(self._entropy_type))

    def benchmark(self, qc: QComputer = None, qubits: List[int] = None, single_gates: List[str] = None,
                  multi_gates: List[str] = None, **kwargs) -> dict:
        r"""Execute the cross entropy benchmarking procedure on the quantum computer.

        The parameters `qc` and `qubits` must be set either by the init() function or here,
        otherwise the cross entropy benchmarking procedure will not carry out.

        Optional keywords list are:

        + ``_seq_lengths``: List[int], default to :math:`[1, 10, 20, 50, 75, 100]`, the list of sequence lengths
        + ``repeats``: int, default to :math:`6`, the number of repetitions of each sequence length
        + ``shots``: int, default to :math:`4096`, the number of shots each measurement should carry out
        + ``single_gates``: List[str], default to None, while construct the random circuit by using `U3` gates
                            for single-qubit gates.
        + ``multi_gates``: List[str], default to None, while construct the random circuit by using `CNOT` gates
                            for multi-qubit gates.

        **Usage**

        .. code-block:: python
            :linenos:

            xeb_results = xeb.benchmark(qubits=[1], qc=qc,_seq_lengths=[1,10,50,100])
            xeb_results = xeb.benchmark(qubits=[1], qc=qc,_seq_lengths=[1,10,50,100], repeats=10, shots=1024)
            xeb_results = xeb.benchmark(qubits=[1], qc=qc,_seq_lengths=[1,10,50,100],
                                                    single_gates=['X','H','S','U'],multi_gates=['CZ'])

        :return: dict, the cross entropy benchmarking results

        **Examples**

            >>> import qiskit
            >>> from qiskit.providers.fake_provider import FakeSantiago
            >>> from qcompute_qep.benchmarking.xeb import XEB
            >>> qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())
            >>> xeb = XEB()
            >>> xeb_results = xeb.benchmark(qubits=[1], qc=qc)
            >>> xeb.plot_results()
            >>> print(xeb_results)
        """
        # Parse the arguments from the key list. If not set, use default arguments from the init function
        self._qc = qc if qc is not None else self._qc
        self._qubits = qubits if qubits is not None else self._qubits
        self._seq_lengths = kwargs.get('seq_lengths', self._seq_lengths)
        self._repeats = kwargs.get('repeats', self._repeats)
        self._shots = kwargs.get('shots', self._shots)

        if self._qc is None:
            raise QEPError.ArgumentError("XEB: the quantum computer for benchmarking is not specified!")
        if self._qubits is None:
            raise QEPError.ArgumentError("XEB: the qubits for benchmarking are not specified!")

        ###############################################################################################################
        # Step 1. Data Collection Phase.
        #   First construct the list of benchmarking quantum circuits.
        #   Then for each XEB quantum circuit, evaluate its expectation value.
        ###############################################################################################################
        # Store the estimated average fidelities, which is a :math:`M`-dimensional array,
        # where :math:`M` is the number of sequences
        fids = np.zeros((len(self._seq_lengths), self._repeats), dtype=float)
        pbar = tqdm(total=100, desc='Step 1/1 : Implement the XEB...', ncols=100)
        # Estimate the average cycle fidelity for each quantum circuit cycle
        for m, cyc_m in enumerate(self._seq_lengths):
            for r in range(self._repeats):
                # Generate a n-qubit random quantum circuit with number of cycles cyc_m
                xeb_qp = circuit.random_circuit(self._qubits, cyc_m, single=single_gates, multi=multi_gates)
                # circuit.print_circuit(xeb_qp.circuit)
                # Calculate the non-noise entropy
                ideal_entropy = np.sum((np.abs(circuit.circuit_to_state(xeb_qp, vector=True)) ** 2) ** 2)

                # Run the noisy XEB quantum circuit and record the measurement outcomes
                counts = circuit.execute(qp=xeb_qp, qc=self._qc, **kwargs)
                # Estimate the empirical average entropy from the measurement outcomes
                average_entropy = 0.0
                for bitstring, cnt in counts.items():
                    average_entropy += self.cross_entropy(_ideal_probability(xeb_qp, bitstring)) * cnt / self._shots

                # Compute the average unbiased XEB fidelity (F_uXEB) of current cycle
                fids[m][r] = self.fidelity(average_entropy) / self.fidelity(ideal_entropy)
                pbar.update((100 / (self._repeats * len(self._seq_lengths))))
        pbar.close()

        ###############################################################################################################
        # Step 2. Data Processing Phase.
        #   Fit the list of averaged expectation values to the exponential model and extract the fitting results.
        ###############################################################################################################
        bounds = ([0, 0], [1, 1])
        # Use scipy's non-linear least squares to fit the data
        xdata = self._seq_lengths
        ydata = np.mean(fids, axis=1)
        sigma = np.std(fids, axis=1)
        if len(sigma) - np.count_nonzero(sigma) > 0:
            sigma = None

        p0 = [0.99, 0.5, ]
        alpha_guess = 0
        count = 0
        for j in range(1, len(xdata)):
            dx = (xdata[j] - xdata[0])
            dy = (ydata[j] / ydata[0])
            if dy > 0:
                alpha_guess += -np.log(dy) / dx
                count += 1
        if count > 0:
            alpha_guess /= count
        if 0 < alpha_guess <= 1.0:
            p0[0] = alpha_guess

        tmp = 0
        count = 0
        for j in range(len(ydata)):
            tmp += (ydata[j] / np.exp(-p0[0] * xdata[j]))
            count += 1
        if count > 0:
            tmp /= count
        if 0 < tmp < 1.0:
            p0[1] = tmp

        popt, pcov = curve_fit(self._fit_func, xdata, ydata,
                               p0=p0, sigma=sigma, maxfev=500000,
                               bounds=bounds, method='dogbox')

        # Store the cross entropy benchmarking results
        params_err = np.sqrt(np.diag(pcov))
        self._results['fids'] = fids
        self._results['lambda'] = popt[0]
        self._results['A'] = popt[1]
        self._results['lambda_err'] = params_err[0]
        self._results['std'] = sigma

        return self._results

    def plot_results(self, show: bool = True, fname: str = None):
        r"""Plot cross entropy benchmarking results.

        Commonly, we visualize the sampled and averaged expectation values for each given length,
        the fitted function, and the estimated fidelity.

        :param show: bool, default to True, show the plot figure or not
        :param fname: figure name for saving. If fname is None, do not save the figure
        """
        if not HAS_MATPLOTLIB:
            raise ImportError('Function "plot_xeb_results" requires matplotlib. Run "pip install matplotlib" first.')

        fig, ax = plt.subplots(figsize=(12, 8))

        xdata = self._seq_lengths
        fids = self.results['fids']

        # Plot the repeated estimates for each sequence
        ax.plot(xdata, fids, color='gray', linestyle='none', marker='x')

        # Plot the mean of the estimated expectation values
        ax.plot(xdata, np.mean(fids, axis=1), color='blue', linestyle='none', marker='v', markersize=13)

        # Plot the confidence interval of the fitting curve
        low_CI_bound, high_CI_bound = st.t.interval(0.95, len(xdata), loc=np.mean(fids, axis=1),
                                                    scale=st.sem(fids, axis=1))
        plt.fill_between(xdata, y1=low_CI_bound, y2=high_CI_bound, color='cornflowerblue', alpha=0.3, )

        # Plot the fitting function
        ydata = [self._fit_func(x, self.results['lambda'], self.results['A'], )
                 for x in xdata]
        ax.plot(xdata, ydata, color='blue', linestyle='-', linewidth=2, label='fitting curve')
        ax.tick_params(labelsize='medium')

        # Set the labels
        ax.set_xlabel('Random Circuit Cycle', fontsize='large')
        ax.set_ylabel('Estimated Fidelity', fontsize='large')
        ax.grid(True)

        # Show the legend
        plt.legend(loc='lower left', fontsize=16)

        # Add the estimated fidelity and EPC parameters
        bbox_props = dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5)
        ax.text(0.85, 0.9,
                r"$\lambda$: {:.3f}({:.1e})".format(self.results['lambda'], self.results['lambda_err']),
                ha="center", va="center", fontsize=12, bbox=bbox_props, transform=ax.transAxes)

        # Set the x-axis locator always be integer
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Save the figure if `fname` is set
        if fname is not None:
            plt.savefig(fname, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)
        # Show the figure if `show==True`
        if show:
            plt.show()


def _ideal_probability(qp: QProgram, bitstring: str) -> float:
    r"""Compute the theoretical output probability of quantum circuit.

    We aim to compute theoretically the probability of measuring bitstring
    when evolving from the computational basis state :math:`\vert 0\cdots 0\rangle` by the quantum program.
    Let :math:`U` be the quantum program and let :math:`x` be the bitstring,
    then the theoretical probability of obtaining :math:`x` is given by

    .. math:: P(x) = \vert \langle x \vert U \vert 0\cdots 0\rangle\vert^2.

    :param qp: QProgram, describes the quantum program
    :param bitstring: str, specifies the outcome bitstring
    :return: float, the ideal probability of obtaining the given outcome bitstring
    """
    prob = np.abs(circuit.circuit_to_state(qp, vector=True)) ** 2
    index = int(bitstring, 2)
    if index >= len(prob):
        raise QEPError.ArgumentError("the given bitstring {} is wrong.".format(bitstring))
    return prob[index]
