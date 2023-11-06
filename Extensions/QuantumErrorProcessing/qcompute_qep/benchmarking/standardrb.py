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
Standard Randomized Benchmarking.
A scalable and robust algorithm for benchmarking the complete set of Clifford gates
by a single parameter called EPC (error per Clifford) using randomization techniques.
"""
from typing import List
from tqdm import tqdm
import numpy as np
from scipy.optimize import curve_fit
from matplotlib.ticker import MaxNLocator
import scipy.stats as st
import itertools
import warnings

import QCompute
from Extensions.QuantumErrorProcessing.qcompute_qep.utils import expval_from_counts, execute
from Extensions.QuantumErrorProcessing.qcompute_qep.quantum import clifford
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.types import QComputer, get_qc_name
import Extensions.QuantumErrorProcessing.qcompute_qep.exceptions.QEPError as QEPError
import Extensions.QuantumErrorProcessing.qcompute_qep.benchmarking as rb

warnings.filterwarnings("ignore")
try:
    from matplotlib import pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
except ImportError:
    raise ImportError('SRB requires matplotlib to visualize results. Run "pip install matplotlib" first.')


class StandardRB(rb.RandomizedBenchmarking):
    r"""The Standard Randomized Benchmarking class.

    Standard Randomized Benchmarking aims to benchmark the complete set of Clifford gates
    by a single parameter called EPC (Error Per Clifford).
    """

    def __init__(self, qc: QComputer = None, qubits: List[int] = None, **kwargs):
        r"""init function of the Standard Randomized Benchmarking class.

        Optional keywords list are:

        + ``seq_lengths``: List[int], default to :math:`[1, 10, 20, 50, 75, 100]`, a list of sequence lengths
        + ``repeats``: int, default to :math:`6`, the number of repetitions of each sequence length
        + ``shots``: int, default to :math:`4096`, the number of shots each measurement carries out to estimate value
        + ``prep_circuit``: default to `default_prep_circuit`, prepares the initial quantum state
                        :math:`\vert 0\cdots 0\rangle`
        + ``meas_circuit``: default to `default_meas_circuit`, add the Z basis measurement to the end
                        of the SRB circuits and set the quantum observable to
                        :math:`\vert 0\cdots 0\rangle\!\langle 0\cdots 0\vert`

        for `prep_circuit` and `meas_circuit` see more details in `benchmarking.utils.default_prep_circuit`
                        and `benchmarking.utils.default_meas_circuit`.

        :param qc: QComputer, the quantum computer on which the RB carries out
        :param qubits: List[int], the qubits who will be benchmarked
        """
        # Initialize the Standard RB parameters. If not set, use the default parameters
        super().__init__(**kwargs)
        self._qc = qc
        self._qubits = qubits
        self._seq_lengths = kwargs.get("seq_lengths", [1, 10, 20, 50, 75, 100])
        self._repeats = kwargs.get("repeats", 6)
        self._shots = kwargs.get("shots", 4096)
        self._prep_circuit = kwargs.get("prep_circuit", rb.default_prep_circuit)
        self._meas_circuit = kwargs.get("meas_circuit", rb.default_meas_circuit)

        # Store the standard randomized benchmarking results. Initialize to an empty dictionary
        self._results = dict()
        self._params = dict()

    @property
    def results(self) -> dict:
        """
        Return the randomized benchmarking results in a dictionary.

        **Usage**

        .. code-block:: python
            :linenos:

            f = rb_results['f']  # the estimated fidelity parameter
            f_err = rb_results['f_err']  # the standard deviation error of the estimation
            epc = rb_results['epc']  # the estimated EPC parameter

        """
        # If the randomized benchmarking results have not been generated yet,
        # call the benchmark function to generate the results using the default parameters
        if (self._results is None) or (bool(self._results) is False):
            self.benchmark(self._qc, self._qubits)

        return self._results

    @property
    def params(self) -> dict:
        r"""Parameters used in randomized benchmarking in a dictionary."""
        if not self._params:
            rb_params = dict()
            rb_params["qc"] = get_qc_name(self._qc)
            rb_params["qubits"] = self._qubits
            rb_params["seq_lengths"] = self._seq_lengths
            rb_params["repeats"] = self._repeats
            rb_params["shots"] = self._shots
            rb_params["prep_circuit"] = self._prep_circuit
            rb_params["meas_circuit"] = self._meas_circuit
            self._params = rb_params

        return self._params

    def _fit_func(self, x: np.ndarray, f: float, A: float, B: float) -> np.ndarray:
        r"""The fit function used in the standard randomized benchmarking.

        The used fit function is an exponential function in the input and is defined as follows:

        .. math:: p(x) = A f^{2x-1} + B,

        where

        + :math:`x` is the sequence length, i.e., the number of :math:`n`-qubit Clifford gates in the sequence,
        + :math:`f` is the fidelity parameter of the twirled depolarizing channel,
        + :math:`A` and :math:`B` absorb the State Preparation and Measurement errors (SPAM).

        Note that we simply apply the inverse of every Clifford gate to the end of the circuit,
        so the function has a scaling coefficient :math:`2x-1` instead of the commonly used :math:`x`.

        :param x: int, corresponds to the sequence length
        :param f: float, the fidelity parameter of the depolarizing channel
        :param A: float, a parameter that absorbs the state preparation and measurement errors
        :param B: float, another parameter that absorbs the state preparation and measurement errors
        :return: np.ndarray, the estimated expectation value
        """
        return A * f ** (2 * x - 1) + B

    def benchmark(self, qc: QComputer, qubits: List[int], **kwargs) -> dict:
        r"""Execute randomized benchmarking on the quantum computer.

        The parameters `qc` and `qubits` must be set either by the init() function or here,
        otherwise the randomized benchmarking procedure will not carry out.

        Optional keywords list are:

        + ``seq_lengths``: List[int], default to :math:`[1, 10, 20, 50, 75, 100]`, the list of sequence lengths
        + ``repeats``: int, default to :math:`6`, the number of repetitions of each sequence length
        + ``shots``: int, default to :math:`4096`, the number of shots each measurement should carry out
        + ``prep_circuit``: default to `default_prep_circuit`, prepares the initial quantum state
                                :math:`\vert 0\cdots 0 \rangle`
        + ``meas_circuit``: default to `default_meas_circuit`, add the Z basis measurement and
                                set the quantum observable to :math:`\vert 0\cdots 0\rangle\!\langle 0\cdots 0\vert`

        **Usage**

        .. code-block:: python
            :linenos:

            rb_results = benchmarking.benchmark(qubits=[1], qc=qc)
            rb_results = benchmarking.benchmark(qubits=[1], qc=qc, seq_lengths=[1,10,50,100])
            rb_results = benchmarking.benchmark(qubits=[1], qc=qc, seq_lengths=[1,10,50,100], repeats=10, shots=1024)

        :return: dict, the randomized benchmarking results

        **Examples**

            >>> import qiskit
            >>> from qiskit.providers.fake_provider import FakeSantiago
            >>> from Extensions.QuantumErrorProcessing.qcompute_qep.benchmarking.standardrb import StandardRB
            >>> qc = qiskit.providers.aer.AerSimulator.from_backend(FakeSantiago())
            >>> benchmarking = StandardRB()
            >>> rb_results = benchmarking.benchmark(qubits=[1], qc=qc)
            >>> print(rb_results)
            >>> benchmarking.plot_results()
        """
        # Parse the arguments from the key list. If not set, use default arguments from the init function
        self._qc = qc if qc is not None else self._qc
        self._qubits = qubits if qubits is not None else self._qubits
        self._seq_lengths = kwargs.get("seq_lengths", self._seq_lengths)
        self._repeats = kwargs.get("repeats", self._repeats)
        self._shots = kwargs.get("shots", self._shots)
        self._prep_circuit = kwargs.get("prep_circuit", self._prep_circuit)
        self._meas_circuit = kwargs.get("meas_circuit", self._meas_circuit)

        if self._qc is None:
            raise QEPError.ArgumentError("SRB: the quantum computer for benchmarking is not specified!")
        if self._qubits is None:
            raise QEPError.ArgumentError("SRB: the qubits for benchmarking are not specified!")

        ###############################################################################################################
        # Step 1. Construct a list of benchmarking quantum circuits.
        ###############################################################################################################
        pbar = tqdm(total=100, desc="SRB Step 1/4: Constructing benchmarking quantum circuits ...", initial=0)

        n = len(self._qubits)  # number of qubits
        num_of_register_qubits = max(x for x in self._qubits) + 1
        # Quantum programs and quantum observables list
        rb_qp_list = []
        rb_ob_list = []
        seq_lengths = list(
            itertools.chain.from_iterable(itertools.repeat(seq, self._repeats) for seq in self._seq_lengths)
        )
        for seq_m in seq_lengths:
            # Construct a random sequence of Clifford gates of length seq_m
            cliffords = clifford.random_clifford(n, seq_m)
            # Initialize the randomized benchmarking quantum circuit
            rb_qp = QCompute.QEnv()
            q = rb_qp.Q.createList(num_of_register_qubits)
            # Create a list to store the inverse circuit of every Clifford gate
            inv_circuit = []
            for i, c in enumerate(cliffords):
                c(q, self._qubits)
                inv_circuit.append(c.get_inverse_circuit(self._qubits))
            # Reverse the sequence of applied Clifford gates and operate them on the inputs
            rb_qp.circuit += sum(inv_circuit[::-1], [])
            # Prepare the input quantum state
            rb_qp = self._prep_circuit(rb_qp)
            # Add the desired measurement corresponds to the target quantum observable
            rb_qp, rb_ob = self._meas_circuit(rb_qp)

            rb_qp_list.append(rb_qp)
            rb_ob_list.append(rb_ob)

        ###############################################################################################################
        # Step 2. Run the quantum circuits in batch.
        ###############################################################################################################
        pbar.desc = "SRB Step 2/4: Running quantum circuits, which might be very time consuming ..."
        pbar.update(100 / 4)

        counts_list = execute(qp=rb_qp_list, qc=self._qc, **kwargs)
        ###############################################################################################################
        # Step 3. Estimate the expectation values from the measurement outcomes.
        ###############################################################################################################
        pbar.desc = "SRB Step 3/4: Estimating expectation values from the measurement outcomes ..."
        pbar.update(100 / 4)

        # Store the estimated expectation values, which is a :math:`M \times R` array,
        # where :math:`M` is the number of sequences and :math:`R` is the number of repeats of each sequence.
        expvals = np.empty((len(self._seq_lengths), self._repeats), dtype=float)
        for k in range(len(rb_qp_list)):
            m, r = divmod(k, self._repeats)
            expvals[m][r] = expval_from_counts(A=rb_ob_list[k], counts=counts_list[k])

        ###############################################################################################################
        # Step 4. Fit the expectation values to the exponential model.
        ###############################################################################################################
        pbar.desc = "SRB Step 4/4: Fitting expectation values to the exponential model ..."
        pbar.update(100 / 4)
        # Set the bounds for the parameters tuple: :math:`(f, A, B)`
        # min_eig = min(np.diag(rb_ob))
        # max_eig = max(np.diag(rb_ob))
        # bounds = ([0, min_eig - max_eig, min_eig], [1, max_eig - min_eig, max_eig])
        bounds = ([0, 0, 1 / 2**n], [1, 1, 1])

        # Use scipy's non-linear least squares to fit the data
        xdata = self._seq_lengths
        ydata = np.mean(expvals, axis=1)
        sigma = np.std(expvals, axis=1)
        if len(sigma) - np.count_nonzero(sigma) > 0:
            sigma = None

        p0 = [0.99, 0.95, 1 / 2**n]
        alpha_guess = []
        for j in range(1, len(xdata)):
            if ydata[j] > p0[2]:
                dx = xdata[j] - xdata[0]
                dy = (ydata[j] - p0[2]) / (ydata[0] - p0[2])
                alpha_guess.append(dy ** (1 / (2 * dx)))
        if alpha_guess:
            if np.mean(alpha_guess) < 1.0:
                p0[0] = np.mean(alpha_guess)

        tmp = []
        for j in range(len(ydata)):
            if ydata[j] > p0[2]:
                tmp.append((ydata[j] - p0[2]) / (p0[0] ** (2 * xdata[j])))

        if tmp and np.mean(tmp) < 1.0:
            p0[1] = np.mean(tmp)

        # Must call `curve_fit` with `*_` to avoid the error "ValueError: too many values to unpack"
        popt, pcov, *_ = curve_fit(
            f=self._fit_func,
            xdata=np.asarray(xdata),
            ydata=ydata,
            p0=np.asarray(p0),
            sigma=sigma,
            bounds=bounds,
            maxfev=500000,
            method="dogbox",
        )

        # Store the randomized benchmarking results
        params_err = np.sqrt(np.diag(pcov))
        self._results["expvals"] = expvals
        self._results["f"] = popt[0]
        self._results["A"] = popt[1]
        self._results["B"] = popt[2]
        self._results["f_err"] = params_err[0]
        d = 2 ** len(self._qubits)
        self._results["epc"] = (d - 1) / d * (1 - popt[0])
        self._results["epc_err"] = (d - 1) / d * (1 - params_err[0])

        pbar.desc = "SRB successfully finished!"
        pbar.update(100 - pbar.n)
        pbar.close()

        return self._results

    def plot_results(self, show: bool = True, fname: str = None):
        r"""Plot randomized benchmarking results.

        Commonly, we visualize the sampled and averaged expectation values for each given length,
        the fitted function, and the estimated fidelity and Error Per Clifford.

        :param show: bool, default to True, show the plot figure or not
        :param fname: figure name for saving. If fname is None, do not save the figure
        """
        fig, ax = plt.subplots(figsize=(12, 8))

        xdata = self._seq_lengths
        expvals = self.results["expvals"]

        # Plot the repeated estimates for each sequence
        ax.plot(xdata, expvals, color="gray", linestyle="none", marker="x")

        # Plot the mean of the estimated expectation values
        ax.plot(xdata, np.mean(expvals, axis=1), color="blue", linestyle="none", marker="v", markersize=13)

        # Plot the fitting function
        ydata = [self._fit_func(x, self.results["f"], self.results["A"], self.results["B"]) for x in xdata]
        ax.plot(xdata, ydata, color="blue", linestyle="-", linewidth=2, label="fitting curve")
        ax.tick_params(labelsize="medium")

        # Plot the confidence interval of the fitting curve
        low_CI_bound, high_CI_bound = st.t.interval(
            0.95, len(xdata), loc=np.mean(expvals, axis=1), scale=st.sem(expvals, axis=1)
        )
        plt.fill_between(
            xdata,
            y1=low_CI_bound,
            y2=high_CI_bound,
            color="cornflowerblue",
            alpha=0.3,
        )

        # Set the labels
        ax.set_xlabel("Clifford Length", fontsize="large")
        ax.set_ylabel("Expectation Value", fontsize="large")
        ax.grid(True)

        # Set the x-axis locator always be integer
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

        # Show the legend
        plt.legend(loc="lower left", fontsize="large")

        # Add the estimated fidelity and EPC parameters
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor="wheat", alpha=0.5)
        ax.text(
            0.8,
            0.9,
            "Average Gate Fidelity: {:.4f}({:.1e}) \n "
            "Error Per Clifford: {:.4f}({:.1e})".format(
                self.results["f"], self.results["f_err"], self.results["epc"], self.results["epc_err"]
            ),
            ha="center",
            va="center",
            fontsize="large",
            bbox=bbox_props,
            transform=ax.transAxes,
        )

        # Save the figure if `fname` is set
        if fname is not None:
            plt.savefig(fname, format="png", dpi=600, bbox_inches="tight", pad_inches=0.1)
        # Show the figure if `show` is True
        if show:
            plt.show()
