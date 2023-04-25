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
This script collects classes and functions implementing the Quantum Gate Set Tomography (GST) [G15]_ [NGR+21]_.
In Quantum State Tomography (QST) and Quantum Process Tomography (QPT),
it is assumed that the initial states and final measurements are known and noiseless.
However, these states and measurements must be prepared using quantum gates which themselves may be faulty,
known as the state preparation and measurement (SPAM) errors.
This results in a self-consistency problem. GST can be viewed as a self-consistent extension of QPT,
by including the SPAM gates self-consistently in the gate set to be estimated.

Currently, we support the following GST method:

+ Linear inversion method, see Section 3.4 in [G15]_,

while the following GST method is under construction:

+ Maximum likelihood estimation method, see Section 3.5 in [G15]_.

What's more, we offer the ``GSTOptimizer`` abstract class, any optimization class inherits this
abstract class can be used as an optimizer in the ``GateSetTomography`` class.

References:

.. [G15] Greenbaum, Daniel. "Introduction to quantum gate set tomography." arXiv preprint arXiv:1509.02921 (2015).

.. [NGR+21] Nielsen, Erik, et al. "Gate set tomography." Quantum 5 (2021): 557.
"""
import numpy as np
from typing import Dict, List, Union
from tqdm import tqdm
import itertools
import warnings

from QCompute import *
from qcompute_qep.utils.types import QComputer
from qcompute_qep.utils.circuit import execute, map_qubits
from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.tomography import Tomography, GateSet, STD1Q_GATESET_RXRY, STD2Q_GATESET_RXRYCZ, \
    GSTOptimizer, LinearInversionOptimizer

# Stop print warning information
warnings.filterwarnings("ignore")


class GateSetTomography(Tomography):
    r"""The Quantum Gate Set Tomography class.

    """

    def __init__(self, qc: QComputer = None,
                 gate_set: GateSet = None,
                 qubits: List[int] = None,
                 optimizer: Union[str, GSTOptimizer] = 'linear_inverse',
                 **kwargs):
        r"""init function of the Quantum GateSet Tomography class.

        Optional keywords list are:

            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out

        :param qc: QComputer, default to None, the target quantum computer
        :param qubits: List[int], default to None, the index of target qubit(s) we want to tomography
        :param gate_set: GateSet, default to None, the gate set we want to estimate
        :param optimizer: Union[str, GSTOptimizer], default to 'linear_inverse',
                        the gateset tomography optimizer that is used
        """
        super().__init__(qc=qc, **kwargs)
        self._qc: QComputer = qc
        self._qubits: List[int] = qubits
        self._gateset: GateSet = gate_set
        self._shots: int = kwargs.get('shots', 8000)
        self._optimizer: GSTOptimizer = None
        self._initialize_optimizer(optimizer)

        # Number of working qubits. Notice that working qubits must count from 0 to the maximal index in the `qubits`
        self._n = max(self._qubits) + 1 if self._qubits is not None else None

        # Store the gate set tomography results. Initialize to an empty dictionary
        self._result = dict()

    def _initialize_optimizer(self, optimizer: Union[str, GSTOptimizer]):
        r"""Initialize the optimizer from name.

        :param optimizer: Union[str, GSTOptimizer], default to 'linear_inverse', the GST optimizer to be used
        """
        if isinstance(optimizer, str):
            if optimizer == 'linear_inverse':
                self._optimizer = LinearInversionOptimizer()
            elif optimizer == 'mle':
                # self._optimizer = MLEOptimizer()
                raise ArgumentError("in GateSetTomography(): The maximum likelihood estimation optimizer "
                                    "will be supported in the next version, "
                                    "Please use the linear inversion optimizer!")
            else:
                raise ArgumentError("in GateSetTomography(): undefined optimizer name {}. "
                                    "Supported optimizer names are: 'linear_inverse' and 'mle'.".format(optimizer))
        elif isinstance(optimizer, GSTOptimizer):
            self._optimizer = optimizer
        else:
            raise ArgumentError("in GateSetTomography: undefined optimizer type!")
        pass

    @property
    def result(self):
        return self._result

    def fit(self, qc: QComputer = None,
            gate_set: GateSet = None,
            qubits: List[int] = None,
            optimizer: Union[str, GSTOptimizer] = 'linear_inverse',
            **kwargs) -> GateSet:
        r"""Execute Gate Set Tomography on the quantum computer.

        Optional keywords list are:

            + `shots`: default to :math:`4096`, the number of shots each measurement should carry out

        If `gate_set` is 'None', we use standard gate sets in QEP:

            + For single-qubit GST, we use the standard gate set ``STD1Q_GATESET_RXRY``,
            + For two-qubit GST, we use the standard gate set ``STD2Q_GATESET_RXRYCZ``.
            + Currently, we do not support three- or more qubit GST.

        There are other standard gate sets in QEP: ``STD1Q_GATESET_RXRYRX``, ``STD1Q_GATESET_RXRYID``,
        ``STD2Q_GATESET_RXRYCX``, ``STD2Q_GATESET_RXRYSWAP``.
        What's more, the gate set ``STD1Q_GATESET_RXRYID`` is overcomplete with :math:`6` preparation and
        measurement quantum circuits.

        Optional choices for the `optimizer` parameter are:

        + '`linear`' for the linear inversion method (the default method), and
        + '`mle`' for the maximum likelihood estimation method.

        .. note::

            You can also use a user-defined optimizer which inherts the `GSTOptimizer` interface.

        :param qc: QComputer, the quantum computer
        :param qubits: List[int], default to None, the index of target qubit(s) we want to tomography
        :param gate_set: GateSet, default to None, the gate set we want to estimate
        :param optimizer: Union[str, GSTOptimizer], default to 'linear_inverse',
                        the gateset tomography optimizer that is used
        :return: GateSet, the updated gate set which contains the information of noisy gates, state, and measurement

        Usage:

        .. code-block:: python
            :linenos:

            gate_set = GateSetTomography.fit(qc, qubits=[0], gate_set=gate_set)
            gate_set = GateSetTomography.fit(qc, qubits=[0, 1], gate_set=gate_set, shots=2 ** 15)

        **Examples**

            >>> import QCompute
            >>> import numpy
            >>> from qcompute_qep.tomography import GateSetTomography, GateSet, STD1Q_GATESET_RXRY
            >>> from QCompute.QPlatform.QOperation import CircuitLine
            >>> gateset = GateSet(gates={'G_rx90': CircuitLine(QCompute.RX(numpy.pi / 2), [0]),
            >>>                          'G_ry90': CircuitLine(QCompute.RY(numpy.pi / 2), [0])},
            >>>                   prep_gates=[['G_rx90', 'G_rx90'], ['G_rx90'], ['G_ry90']],
            >>>                   meas_gates=[['G_rx90', 'G_rx90'], ['G_rx90'], ['G_ry90']])
            >>> tomo = GateSetTomography()
            >>> gateset1 = tomo.fit(QCompute.BackendName.LocalBaiduSim2, qubits=[0], gate_set=gateset)
            >>> # Alternatively, you can use standard gate set in QEP:
            >>> gateset2 = tomo.fit(QCompute.BackendName.LocalBaiduSim2, qubits=[0], gate_set=STD1Q_GATESET_RXRY)
        """
        # Init parameters
        self._qc = qc if qc is not None else self._qc
        self._qubits = qubits if qubits is not None else self._qubits
        self._gateset = gate_set if gate_set is not None else self._gateset
        self._method = kwargs.get('method', self._method)
        self._shots = kwargs.get('shots', self._shots)
        self._n = max(self._qubits) + 1 if self._qubits is not None else self._gateset.n
        # Set up the optimizer
        self._optimizer = optimizer if optimizer is not None else self._optimizer
        self._initialize_optimizer(self._optimizer)

        ###############################################################################################################
        # Step 0. Check the correctness and consistency of input parameters.
        ###############################################################################################################
        if self._qc is None:
            raise ArgumentError("in GateSetTomography.fit(): the quantum computer is not set!")
        # Assert the list of qubits to be estimated
        if self._qubits is None:
            raise ArgumentError("in GateSetTomography.fit(): the list of qubits to be estimated is not set!")
        if len(self._qubits) != 1 and len(self._qubits) != 2:
            raise ArgumentError("in GateSetTomography.fit(): too many qubits to be estimated! "
                                "Currently we only supports single- and two-qubit GST!")
        if len(set(self._qubits)) != len(self._qubits):
            raise ArgumentError("in GateSetTomography.fit(): the list of qubits contain repeated indices!")
        self._qubits.sort()

        # Assert the number of qubits and the given gate set must match
        # Case 1. gateset is not set. Set the default standard gate set
        if self._gateset is None:
            self._gateset = STD1Q_GATESET_RXRY if len(self._qubits) == 1 else STD2Q_GATESET_RXRYCZ
        # Case 2. gateset is set, but does not match the number of qubits to be estimated
        if len(self._qubits) != self._gateset.n:
            raise ArgumentError("in GateSetTomography.fit(): the number of qubits to be estimated in `qubits` "
                                "does not match the number of qubits in `gateset`!")
        # Case 3. gateset is set, but is not complete
        if not self._gateset.verify_completeness():
            raise ArgumentError("in GateSetTomography.fit(): the given gate set is not complete!")

        ###############################################################################################################
        # Step 1. Running tomographic circuits, collect the experimental data,
        #         and construct the experimental accessible gate set \tilde{G}, correspond to Eq. (3.15) in [G15]_.
        ###############################################################################################################
        pbar = tqdm(total=100, desc='GST Step 1/5: Constructing experimentally accessible gate set ...')
        gateset_exp = self._construct_gateset_exp(shots=self._shots, **kwargs)
        ###############################################################################################################
        # Step 2. Estimate the optimized gate set that matches the experimental data best
        ###############################################################################################################
        pbar.desc = "GST Step 5/5: Estimate the optimized gate set ..."
        pbar.update(80)
        opt_result = self._optimizer.optimize(gateset=self._gateset, gateset_exp=gateset_exp, shots=self._shots)
        # Merge optimization results result
        self._result.update(opt_result)

        pbar.desc = "Successfully finished GST!"
        pbar.update(100 - pbar.n)
        pbar.close()

        return self._gateset

    def _extract_expval_from_counts(self, counts: dict) -> float:
        r"""Extract expectation value of the measurement operator from counts.

        In GST, the measurement operator :math:`E` is fixed to :math:`\vert 0\cdots 0\rangle\!\langle 0\cdots 0\vert`.
        thus we estimate the expectation value from counts in the following way:

        .. math:: \text{Pr}(E) = \frac{N_0}{N},

        where :math:`N_0` is the number of occurrences of bitstring `0...0` and :math:`N` is the number of shots.

        :param counts: dict, a dictionary records the measurement outcomes and frequencies
        :return: float, the estimated expectation value
        """
        str_0 = '0'.zfill(len(self._qubits))
        count_0 = counts.pop(str_0, 0)
        return float(count_0 / self._shots)

    def _gate_exp(self, gate_name: str, **kwargs) -> np.ndarray:
        r"""Estimate the experimentally accessible quantum gate :math:`\tilde{G}_k`.

        In [G15]_, :math:`\tilde{G}_k` is defined in Eq. (3.15). It is obtained by running tomographic
        quantum circuits constructed as follows:

            1. Insert a preparation circuit :math:`P_j` in front of :math:`G_k` and
            2. Append a measurement circuit :math:`M_i` in the end of :math:`G_k`.
            3. Measure the qubit in the :math:`Z` basis.

        .. admonition:: Important!

            The `**kwargs` parameter must be reserved and is passed directly to the `execute()` function.

        :param gate_name: str, the name of gate whose experimentally accessible version we aim to estimate
        :return: np.ndarray, the PTM representation of the experimentally accessible :math:`\tilde{G}_k`
        """
        num_prep, num_meas = len(self._gateset.prep_gates), len(self._gateset.meas_gates)

        # The CircuitLine representation of current quantum gate
        gate_k = self._gateset.gates[gate_name]

        # Step 1. Construct a list of tomographic quantum circuits
        qp_list = []
        # Notice that here @i is the index for measurement and @j is the index for state
        for i, j in itertools.product(range(num_meas), range(num_prep)):
            qp_ij = QEnv()
            qp_ij.Q.createList(self._n)
            # Insert the state preparation quantum circuit
            qp_ij.circuit += self._gateset.create_prep_circuit(gate_idx=j)
            # Insert the target quantum circuit g_k
            if gate_k is not None:
                qp_ij.circuit.append(gate_k)
            # Insert the measurement quantum circuit
            qp_ij.circuit += self._gateset.create_meas_circuit(gate_idx=i)
            # Mapping the qubits in qp_ij to the target qubits specified by `self._qubits`
            qp_ij = map_qubits(qp_ij, self._qubits)
            # Add Z measurement on the target qubits
            qreglist, indexlist = qp_ij.Q.toListPair()
            MeasureZ(qRegList=[qreglist[x] for x in self._qubits], cRegList=[indexlist[x] for x in self._qubits])

            # Add current quantum program to the program list
            qp_list.append(qp_ij)

        # Step 2. Run the quantum circuits in batch.
        counts_list = execute(qp=qp_list, qc=self._qc, **kwargs)

        # Step 3. Estimate the probabilities from the measurement outcomes.
        p_ij = np.empty((num_meas, num_prep), dtype=float)
        for i, j in itertools.product(range(num_meas), range(num_prep)):
            k = i * num_meas + j
            p_ij[i, j] = self._extract_expval_from_counts(counts_list[k])
        return p_ij

    def _rho_exp(self, **kwargs) -> np.ndarray:
        r"""Estimate the experimentally accessible quantum state :math:`\tilde{\rho}`.

        In [G15]_, :math:`\tilde{\rho}` is defined in Eq. (3.19).
        It is obtained by running tomographic quantum circuits constructed as follows:

            1. Add a measurement circuit :math:`M_i`.
            2. Measure the qubit in the :math:`Z` basis.

        .. admonition:: Important!

            The `**kwargs` parameter must be reserved and is passed directly to the `execute()` function.

        :return: np.ndarray, the PTM representation of the experimentally accessible :math:`\tilde{\rho}`
        """
        num_meas = len(self._gateset.meas_gates)

        # Run measurement quantum circuits to estimate :math:`\tilde{\rho}`
        # Step 1. Construct a list of tomographic quantum circuits
        qp_list = []
        for i in range(num_meas):
            qp_rho = QEnv()
            qp_rho.Q.createList(self._n)
            # Measure the initial quantum state :math:`\rho` in a target quantum basis
            qp_rho.circuit += self._gateset.create_meas_circuit(gate_idx=i)
            # Mapping the qubits in qp_rho to the target qubits specified by `self._qubits`
            qp_rho = map_qubits(qp_rho, self._qubits)
            # Add Z measurement on the target qubits
            qreglist, indexlist = qp_rho.Q.toListPair()
            MeasureZ(qRegList=[qreglist[x] for x in self._qubits], cRegList=[indexlist[x] for x in self._qubits])

            # Add current quantum program to the program list
            qp_list.append(qp_rho)

        # Step 2. Run the quantum circuits in batch.
        counts_list = execute(qp=qp_list, qc=self._qc, **kwargs)

        # Step 3. Estimate the probabilities from the measurement outcomes.
        rho_exp = np.zeros((num_meas, 1), dtype=float)
        for i in range(num_meas):
            rho_exp[i, 0] = self._extract_expval_from_counts(counts_list[i])

        return rho_exp

    def _meas_exp(self, **kwargs) -> np.ndarray:
        r"""Estimate the experimentally accessible measurement operator :math:`\tilde{E}`.

        In [G15]_, :math:`\tilde{E}` is defined in Eq. (3.20).
        It is obtained by running tomographic quantum circuits constructed as follows:

            1. Add a state preparation circuit :math:`P_j`.
            2. Measure the qubit in the :math:`Z` basis.

        .. admonition:: Important!

            The `**kwargs` parameter must be reserved and is passed directly to the `execute()` function.

        :return: np.ndarray, the PTM representation of the experimentally accessible :math:`\tilde{E}`
        """
        num_prep = len(self._gateset.prep_gates)
        # Run state preparation quantum circuits to estimate E tilde Eq. (3.20).
        # Step 1. Construct a list of tomographic quantum circuits
        qp_list = []
        for j in range(num_prep):
            qp_meas = QEnv()
            qp_meas.Q.createList(self._n)
            # Insert the state preparation quantum circuit
            qp_meas.circuit += self._gateset.create_prep_circuit(gate_idx=j)
            # Mapping the qubits in qp_meas to the target qubits specified by `self._qubits`
            qp_meas = map_qubits(qp_meas, self._qubits)
            # Add Z measurement on the target qubits
            qreglist, indexlist = qp_meas.Q.toListPair()
            MeasureZ(qRegList=[qreglist[x] for x in self._qubits], cRegList=[indexlist[x] for x in self._qubits])

            # Add current quantum program to the program list
            qp_list.append(qp_meas)

        # Step 2. Run the quantum circuits in batch.
        counts_list = execute(qp=qp_list, qc=self._qc, **kwargs)

        # Execute the quantum program and analyze the results
        meas_exp = np.zeros((1, num_prep), dtype=float)
        for j in range(num_prep):
            meas_exp[0, j] = self._extract_expval_from_counts(counts_list[j])

        return meas_exp

    def _construct_gateset_exp(self, **kwargs) -> dict:
        r"""Construct the experimentally accessible gate set from experimental data.

        In the notation of [G15]_, this function constructs the following gate set:

        .. math::

            \tilde{\mathcal{G}} = \{ |\tilde{\rho}\rangle\!\rangle, \langle\!\langle\tilde{E}\vert,
                                        G_0, \tilde{G}_1, \cdots, \tilde{G}_K\}.

        The gate set is constructed by running various tomographic quantum circuits on the quantum computer
        and post-process the measurement data.
        """
        # Store the experimental accessible gate set :math:`\tilde{\mathcal{G}}`
        gateset_exp: Dict[str, np.ndarray] = dict()

        # Run tomographic quantum circuits to construct the experimental accessible quantum gate G_tilde_k
        pbar = tqdm(total=100, desc="GST Step 2/5: Running tomographic quantum circuits for gates, "
                                    "which is time consuming ...", initial=20)
        for g_k in self._gateset.gates.keys():
            gateset_exp.update({g_k: self._gate_exp(g_k, **kwargs)})

        # Run tomographic quantum circuit to construct the experimental accessible rho_tilde
        pbar.desc = "GST Step 3/5: Running tomographic quantum circuits for rho, which is time consuming ..."
        pbar.update(20)
        gateset_exp.update({'rho': self._rho_exp(**kwargs)})

        # Run tomographic quantum circuit to construct the experimental accessible E_tilde
        pbar.desc = "GST Step 4/5: Running tomographic quantum circuits for E, which is time consuming ..."
        pbar.update(20)
        gateset_exp.update({'E': self._meas_exp(**kwargs)})

        # Record the results
        self._result.update({'gateset exp': gateset_exp})
        self._result.update({'g': gateset_exp[GateSet.NULL_GATE_NAME]})

        pbar.close()
        return gateset_exp
