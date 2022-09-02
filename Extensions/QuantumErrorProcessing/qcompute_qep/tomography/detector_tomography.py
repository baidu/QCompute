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
This file aims to collect functions related to the Quantum Detector Tomography.
"""
import itertools
import functools
import math
import copy
from tqdm import tqdm
from typing import List, Union, Dict
import scipy.linalg as la
import numpy as np
from QCompute import *
from mpl_toolkits.axes_grid1 import make_axes_locatable

from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.tomography import Tomography, MeasurementBasis, init_measurement_basis
from qcompute_qep.tomography.basis import PauliPrepBasis
from qcompute_qep.utils.types import QComputer, QProgram, number_of_qubits
from qcompute_qep.utils.circuit import execute
from qcompute_qep.quantum.pauli import ptm_to_operator, operator_to_ptm
from qcompute_qep.quantum import pauli
from qcompute_qep.utils.utils import expval_from_counts
from qcompute_qep.measurement.utils import state_labels

try:
    from matplotlib import pyplot as plt
    import pylab

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas
    import seaborn as sns

    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


class DetectorTomography(Tomography):
    r"""The Quantum Detector Tomography class.

    Quantum detector tomography is the process by which a quantum measurement is reconstructed using measurements on an
    ensemble of identical quantum states.
    """

    def __init__(self, qc: QComputer = None, qp: QProgram = None, qubits: List[int] = None, **kwargs):
        r"""The init function of the Quantum Detector Tomography class.

        Optional keywords list are:

            + `method`: default to ``inverse``, specify the detector tomography method
            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out
            + `tol`: default to :math:`10^{-5}`, the precision
            + `ptm`: default to ``False``, if the measurement should be returned to the Pauli transfer matrix form

        :param qp: QProgram, quantum program for creating the target quantum measurement
        :param qc: QComputer, the quantum computer

        """
        super().__init__(qp, qc, **kwargs)
        self._povm_list = None
        self._qubits = qubits
        self._qp: QProgram = qp
        self._qc: QComputer = qc
        self._method: str = kwargs.get('method', 'inverse')
        self._shots: int = kwargs.get('shots', 4096)
        self._tol: float = kwargs.get('tol', 1e-5)
        self._ptm: bool = kwargs.get('ptm', False)

    def fit(self, qp: QProgram = None, qc: QComputer = None, **kwargs) -> Union[np.ndarray, List[np.ndarray]]:
        r"""Execute POVMs for the quantum measurement specified by @qp on the quantum computer @qc.

        Optional keywords list are:

            + `method`: default to ``inverse``, specify the detector tomography method. Current support:

                + ``inverse``: the inverse method;
                + ``lstsq``: the least square method;
                + ``mle``: the maximum likelihood estimation method.

            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out

            + `tol`: default to :math:`10^{-5}`, the precision

            + `ptm`: default to ``False``, if the measurement should be returned to the Pauli transfer matrix form

        :param qp: QProgram, quantum program for estimating POVM
        :param qc: QComputer, the quantum computer instance
        :return: Union[np.ndarray, List[np.ndarray]], the estimated POVM, in PTM presentation or list of ordered
         vector space form

        Usage:

        .. code-block:: python
            :linenos:

            meas = DetectorTomography.fit(qp=qp, qc=qc)
            meas = DetectorTomography.fit(qp=qp, qc=qc, method='inverse')
            meas = DetectorTomography.fit(qp=qp, qc=qc, method='lstsq', shots=4096)
            meas = DetectorTomography.fit(qp=qp, qc=qc, method='mle', shots=4096, tol=1e-5)

        **Examples**

            >>> import QCompute
            >>> from qcompute_qep.tomography.detector_tomography import DetectorTomography
            >>> qp = QCompute.QEnv()
            >>> qp.Q.createList(2)
            >>> qc = QCompute.BackendName.LocalBaiduSim2
            >>> detec = DetectorTomography()
            >>> meas = detec.fit(qp, qc, method='inverse', shots=4096)
            >>> fid = detec.fidelity
            >>> print('Measurement fidelity: F = {:.5f}'.format(fid))
            Fidelity between the ideal and noisy states: F = 1.00000
        """
        # Parse the arguments. If not set, use the default arguments set by the init function
        self._qp = qp if qp is not None else self._qp
        self._qc = qc if qc is not None else self._qc
        self._method = kwargs.get('method', self._method)
        self._shots = kwargs.get('shots', self._shots)
        self._tol = kwargs.get('tol', self._tol)
        self._ptm = kwargs.get('ptm', self._ptm)
        self._qubits: list = kwargs.get('qubits', self._qubits)

        # If the quantum program or the quantum computer is not set, the detector tomography cannot be executed
        if self._qc is None:
            raise ArgumentError("in DetectorTomography.fit(): the quantum computer is not set!")
        if self._qp is None and self._qubits is None:
            raise ArgumentError("in DetectorTomography.fit(): at least specify one of the quantum program and qubits!")
        elif self._qp is None and self._qubits is not None:
            set_list = set(self._qubits)
            if len(set_list) != len(self._qubits):
                raise ArgumentError(
                    "in DetectorTomography.fit(): repeated elements in qubits!")
            else:
                n_qubit = max(self._qubits)+1
                qp = QEnv()  # qp is short for "quantum program", instance of QProgram
                qp.Q.createList(n_qubit)
                n = len(self._qubits)
        elif self._qubits is None and self._qp is not None:
            n = number_of_qubits(qp)
            self._qubits = list(range(n))
        else:
            n_from_qp = number_of_qubits(qp)
            if n_from_qp != max(self._qubits)+1:
                raise ArgumentError(
                    "in DetectorTomography.fit(): number of qubits are not consistent between qp and qubits!")
            n = len(self._qubits)

        # Step 1. construct a list of tomographic quantum circuits from the quantum program
        prep_b = PauliPrepBasis()

        qreglist, indexlist = qp.Q.toListPair()
        MeasureZ(qRegList=[qreglist[x] for x in self._qubits],
                 cRegList=[indexlist[x] for x in self._qubits])

        prep_qps = prep_b.prep_circuits(qp, qubits=self._qubits)

        pro_expmnt = np.zeros((2 ** n, 4 ** n), dtype=float)
        meas_operator = [0] * (2 ** n)
        R = [0] * (2 ** n)
        pbar1 = tqdm(total=100, desc='Step 1/3 : Constructing quantum circuits...', ncols=80)
        for i in range(2 ** n):
            pbar1.update(100 / (2 ** n))
            meas_operator[i] = np.identity(2 ** n, dtype=float) / (2 ** n)
            R[i] = np.identity(2 ** n, dtype=float) / (2 ** n)
        pbar1.close()
        rho_ptm = np.real(prep_b.transition_matrix(n))

        # Step 2. run the tomographic quantum circuits on the quantum computer and estimate the expectation values
        # run circuit, get the estimation of << E_m | rho >>
        pbar2 = tqdm(total=100, desc='Step 2/3 : Collecting experiment results...', ncols=80)
        for i, prep_qp in enumerate(prep_qps):
            pbar2.update(100 / int(len(prep_qps)))
            counts = execute(prep_qp, self._qc, **kwargs)
            for key, value in counts.items():
                pro_expmnt[int(key, 2), i] += value/self._shots
        pbar2.close()

        # Step 3. perform the fitting procedure to estimate the quantum measurement
        gap = [1] * (2 ** n)
        if self._method.lower() == 'inverse':  # The naive inverse method
            # Compute the pseudoinverse of the transition matrix
            print('Step 3/3 : Working on INVERSE method...')
            meas_ptm = pro_expmnt @ np.linalg.pinv(rho_ptm.T)
        elif self._method.lower() == 'lstsq':
            # the ordinary least square method
            print('Step 3/3 : Working on LEAST SQUARE method...')
            meas_ptm = pro_expmnt @ np.conj(rho_ptm) @ la.pinv(np.transpose(rho_ptm) @ np.conj(rho_ptm))
        elif self._method.lower() == 'mle':
            # the maximum likelihood estimation
            pbar3 = tqdm(total=100, desc='Step 3/3 : Working hard on MLE method...May take a long time', ncols=80)
            while max(gap) > self._tol:
                pbar3.update(100/(2**n))
                for l in range(2 ** n):
                    meas_operator_previous = copy.deepcopy(meas_operator)
                    coe_1 = np.zeros((2 ** n, 2 ** n), dtype=complex)
                    coe_2 = np.zeros((2 ** n, 2 ** n), dtype=complex)
                    for j in range(4 ** n):
                        f_lj = pro_expmnt[l, j]
                        p_lj = np.trace(np.matmul(meas_operator[l], pauli.ptm_to_operator(rho_ptm[j, :])))
                        coe_2 += f_lj / p_lj * pauli.ptm_to_operator(rho_ptm[j, :])
                        for i in range(2 ** n):
                            for k in range(4 ** n):
                                f_ij = pro_expmnt[i, j]
                                f_ik = pro_expmnt[i, k]
                                p_ik = np.trace(np.matmul(meas_operator[i], pauli.ptm_to_operator(rho_ptm[k, :])))
                                p_ij = np.trace(np.matmul(meas_operator[i], pauli.ptm_to_operator(rho_ptm[j, :])))
                                coe = f_ij * f_ik / (p_ij * p_ik)

                                coe_1 += coe * pauli.ptm_to_operator(rho_ptm[j, :]) @ meas_operator[i] @ \
                                         pauli.ptm_to_operator(rho_ptm[k, :])

                    R[l] = la.fractional_matrix_power(coe_1, -0.5) @ coe_2

                    meas_operator[l] = R[l] @ meas_operator[l] @ R[l].T.conjugate()
                    gap[l] = np.linalg.norm(meas_operator[l] - meas_operator_previous[l])
            pbar3.close()
            # Convert to PTM for the sake of consistency.
            meas_ptm = []
            for i in meas_operator:
                meas_ptm.append(operator_to_ptm(i))
        else:
            raise ArgumentError("In DetectorTomography.fit(), unsupported tomography method '{}'".format(self._method))

        if self._ptm:
            return meas_ptm
        else:
            povm_list = []
            for i, t in enumerate(meas_ptm):
                povm_list.append(pauli.ptm_to_operator(meas_ptm[i]))
            self._povm_list = povm_list
            return povm_list

    @property
    def fidelity(self):
        r"""Calculate the measurement fidelity with respect to the computational basis.
        """
        fid = 0
        for i, povm_element in enumerate(self._povm_list):
            fid += np.diagonal(povm_element)[i]
        return fid.real / len(self._povm_list)


def visualization(element: np.ndarray, binary_labels: bool = True, **kwargs) -> None:
    r"""Visualize the POVM elements.

    Optional keywords list are:

        + ``fig_name``: String, default to None. If given, will store figure with given file name
        + ``labels``: List[str], default to None. A list of sequence string for labels.

    :param element: the POVM element to be visualized.
    :param binary_labels: bool, indicator for adding binary labels to the x axis or not.
            Notice that if counts is very large (more than 5 qubits), then it is meaningless to add the labels.
    """
    if not HAS_SEABORN:
        raise ImportError('Function "plot_histograms" requires pandas and seaborn. '
                          'Please run "pip install pandas, seaborn" first.')
    fig_name = kwargs.get('fig_name', None)
    labels = kwargs.get('labels', None)

    n_qubit = int(math.log(len(element), 2))
    # Set x axis labels
    if binary_labels:  # Compute the number of qubits and create the binary label list (ignore the @labels input)
        labels = state_labels(n_qubit)
    if labels is None:
        labels = [str(i) for i in range(n_qubit)]

    povm_r = element.real
    povm_i = element.imag

    fig, (ax, ax2) = plt.subplots(ncols=2, figsize=(2*(n_qubit+1), 2*(n_qubit+1)))
    vmin = min(povm_r.min(), povm_r.min())
    vmax = max(povm_r.max(), povm_r.max())

    sns.heatmap(povm_r, cmap="RdBu", ax=ax, vmin=vmin, vmax=vmax, cbar=False, square=True)

    ax.set_xticks([i + 0.5 for i in range(len(element))])
    ax.set_xticklabels(labels=labels, rotation=45)
    ax.set_yticks([i + 0.5 for i in range(len(element))])
    ax.set_yticklabels(labels=labels, rotation=45)
    ax.set_xlabel('real part')
    ax.xaxis.set_label_position('top')

    sns.heatmap(povm_i, cmap="RdBu", ax=ax2, vmin=vmin, vmax=vmax, cbar=False, yticklabels=False, square=True)
    ax2.set_xticks([i + 0.5 for i in range(len(element))])
    ax2.set_xticklabels(labels=labels, rotation=45)
    ax2.set_xlabel('imaginary part')
    ax2.xaxis.set_label_position('top')

    cax = fig.add_axes([ax2.get_position().x1+0.02,
                        ax2.get_position().y0,
                        0.02,
                        ax2.get_position().height])

    plt.colorbar(ax2.collections[0], cax=cax)

    if fig_name is not None:
        plt.savefig(fig_name, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)
        plt.show()
    else:
        plt.show()
