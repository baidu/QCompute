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
This file aims to collect functions related to the utility functions used in the `qcompute_qep.tomography` package.
"""
import copy
import json
import math
from typing import List, Dict, Union, Iterable, Tuple
import numpy as np

from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.quantum.pauli import complete_pauli_basis

try:
    from matplotlib import pyplot as plt
    from matplotlib import rc
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import pylab
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import pandas
    import seaborn
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def plot_process_ptm(ptm: np.ndarray,
                     show_labels: bool = False,
                     title: str = None,
                     fig_name: str = None) -> None:
    r"""
    Visualize the Pauli transfer matrix of the quantum process.

    :param ptm: np.ndarray, a :math:`4^n \times 4^n` Pauli transfer matrix.
    :param show_labels: bool, default to ``False``, indicator for adding labels to the x and y axes or not.
        Notice that if ptm is very large (more than 5 qubits), then it is meaningless to add the labels.
    :param title: str, default to None, a string that describes the data in @ptm
    :param fig_name: str, default to None, the file name for saving

    **Examples**

        >>> import QCompute
        >>> import qcompute_qep.tomography as tomography
        >>> qp = QCompute.QEnv()
        >>> qp.Q.createList(2)
        >>> QCompute.CZ(qp.Q[1], qp.Q[0])
        >>> qc = QCompute.BackendName.LocalBaiduSim2
        >>> st = tomography.ProcessTomography()
        >>> noisy_ptm = st.fit(qp, qc, prep_basis='Pauli', meas_basis='Pauli', method='inverse', shots=4096, ptm=True)
        >>> tomography.plot_process_ptm(ptm=noisy_ptm.data, show_labels=True, title='LocalBaiduSim2')

    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Function "plot_process_ptm" requires matplotlib. Please run "pip install matplotlib" first.')

    # Enforce the Pauli transfer matrix to be a real matrix
    ptm = np.real(ptm)
    # Compute the number of qubits
    n = int(math.log(ptm.shape[0], 4))
    cpb = complete_pauli_basis(n)
    # Create the label list
    labels = [pauli.name for pauli in cpb]

    fig, ax = plt.subplots(figsize=(12, 8))

    # Visualize the matrix
    im = ax.imshow(ptm, vmin=-1, vmax=1, cmap='RdBu')

    # Add the colorbar
    fig.colorbar(im, ax=ax)

    # Add ticklabels
    if show_labels:
        # We want to show all ticks and label them with the respective list entries
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        size = 'small' if n <= 2 else 'xx-small'
        ax.tick_params(axis='both', labelsize=size)
        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    else:
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    # Add minor ticks and use them to visualize gridlines
    ax.set_xticks(np.arange(-0.5, len(labels), 0.5), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 0.5), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    if title is not None:  # set figure title
        ax.set_title(title, fontsize='medium')
    if fig_name is not None:  # save figure
        plt.savefig(fig_name, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)

    plt.show()


def compare_process_ptm(ptms: List[np.ndarray],
                        titles: List[str] = None,
                        show_labels: bool = False,
                        fig_name: str = None) -> None:
    r"""
    Compare the Pauli transfer matrices of the quantum process, maybe obtained via different methods.

    :param ptms: List[np.ndarray], a list of Pauli transfer matrices of size :math:`4^n \times 4^n`
    :param titles: List[str], default to None, a list of strings that describes the data in @ptms
    :param show_labels: bool, default to None, indicator for adding labels to the x and y axes or not.
            Notice that if ptm is very large (more than 5 qubits), then it is meaningless to add the labels.
    :param fig_name: str, default to None, the file name for saving

    **Examples**

        >>> import QCompute
        >>> import qcompute_qep.tomography as tomography
        >>> from qcompute_qep.utils.circuit import circuit_to_unitary
        >>> import qcompute_qep.quantum.pauli as pauli
        >>> import qcompute_qep.utils.types as typing
        >>> qp = QCompute.QEnv()
        >>> qp.Q.createList(2)
        >>> QCompute.CZ(qp.Q[1], qp.Q[0])
        >>> ideal_cnot = circuit_to_unitary(qp)
        >>> ideal_ptm = pauli.unitary_to_ptm(ideal_cnot).data
        >>> qc = QCompute.BackendName.LocalBaiduSim2
        >>> qc_name = typing.get_qc_name(qc)
        >>> st = tomography.ProcessTomography()
        >>> noisy_ptm = st.fit(qp, qc, prep_basis='Pauli', meas_basis='Pauli', method='inverse', shots=4096, ptm=True)
        >>> diff_ptm = ideal_ptm - noisy_ptm.data
        >>> tomography.compare_process_ptm(ptms=[ideal_ptm, noisy_ptm.data, diff_ptm])

    """
    if not HAS_MATPLOTLIB:
        raise ImportError('Function "compare_process_ptm" requires matplotlib. '
                          'Please run "pip install matplotlib" first.')

    # Compute the number of qubits
    n = int(math.log(ptms[0].shape[0], 4))
    cpb = complete_pauli_basis(n)
    # Create the label list
    labels = [pauli.name for pauli in cpb]

    if (titles is not None) and (len(ptms) != len(titles)):
        raise ArgumentError("in compare_process_ptm(): the number of matrices and titles must the same!")

    # Visualize the PTM matrices
    fig, axs = plt.subplots(nrows=1, ncols=len(ptms), figsize=(12, 8))
    fontsize = 8 if n <= 2 else 3
    im = None

    for i, ptm in enumerate(ptms):
        # Enforce the Pauli transfer matrix to be a real matrix
        ptm = np.real(ptm)
        im = axs[i].imshow(ptm, vmin=-1, vmax=1, cmap='RdBu')
        if titles is not None:
            axs[i].set_title(titles[i], fontsize='medium')
        # Add ticklabels
        if show_labels:
            # We want to show all ticks and label them with the respective list entries
            axs[i].set_xticks(np.arange(len(labels)))
            axs[i].set_xticklabels(labels)
            axs[i].set_yticks(np.arange(len(labels)))
            axs[i].set_yticklabels(labels)
            axs[i].tick_params(axis='both', labelsize='small')
            # Rotate the tick labels and set their alignment.
            plt.setp(axs[i].get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        else:
            axs[i].axes.get_xaxis().set_visible(False)
            axs[i].axes.get_yaxis().set_visible(False)

        # Add minor ticks and use them to visualize gridlines
        axs[i].set_xticks(np.arange(-0.5, len(labels), 0.5), minor=True)
        axs[i].set_yticks(np.arange(-0.5, len(labels), 0.5), minor=True)
        axs[i].grid(which='minor', color='w', linestyle='-', linewidth=1)

    # Add the colorbar. Create new axes according to image position
    cax = fig.add_axes([axs[-1].get_position().x1+0.02,
                        axs[-1].get_position().y0,
                        0.02,
                        axs[-1].get_position().height])
    plt.colorbar(im, cax=cax)

    # Save the figure if needed
    if fig_name is not None:
        plt.savefig(fig_name, format='png', dpi=600, bbox_inches='tight', pad_inches=0.1)

    plt.show()
