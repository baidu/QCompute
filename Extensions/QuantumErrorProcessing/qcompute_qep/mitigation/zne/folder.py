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
Define classes and functions for implementing unitary folding methods.
"""

import abc
from typing import Any, List, Tuple
import copy
import numpy as np

from qcompute_qep.utils.circuit import remove_measurement, remove_barrier, \
    append_measurement, circuit_to_layers, group_gate_indices, \
    inverse_layer, inverse_layers, layers_to_circuit, depth_of_circuit, print_circuit
from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.utils.types import QProgram

__SUPPORTED_FOLDERS__ = {'circuit', 'layer', 'gate'}
__SUPPORTED_METHODS__ = {'left', 'right', 'random'}


class Folder(abc.ABC):
    r"""The Abstract Basic Folder Class.

    Each inherited folder class must implement the ``fold`` method.
    """

    def __init__(self, **kwargs) -> None:
        r"""
        Pre-set necessary data fields of Folder instance.

        :param method: folding indices selecting method; type: str; optional: left, right, random
        :param seed: random seed for randomly selecting folding indices
        """
        qp = kwargs.get('qp', None)
        self._scale_factor = kwargs.get('scale_factor', 1.0)
        self._method = kwargs.get('method', 'right')
        self._seed = kwargs['seed'] if 'seed' in kwargs.keys() else None

        if self._scale_factor < 1:
            raise ArgumentError("in fold(): Requires scale_factor >= 1, but the input is {}".format(self._scale_factor))
        if self._method not in __SUPPORTED_METHODS__:
            raise ArgumentError("in fold(): Supported supported folding parameter generating methods"
                                "are {}, but the input is {}".format(__SUPPORTED_METHODS__, self._method))

        self._qp_origin = copy.deepcopy(qp)  # store the original quantum circuit for possible later reference
        self._qp_folded = None  # store the folded quantum circuit
        self._has_folded = False  # means it has the record of folding

    @property
    def qp_origin(self):
        return self._qp_origin

    @property
    def qp_folded(self):
        return self._qp_folded

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        self._method = value

    @property
    def name(self):
        return self.__class__.__name__

    @property
    def scale_factor(self):
        return self._scale_factor

    @scale_factor.setter
    def scale_factor(self, value):
        self._scale_factor = value

    @property
    def has_folded(self):
        return self._has_folded

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.fold(*args, **kwargs)

    @abc.abstractmethod
    def fold(self, *args: Any, **kwargs: Any) -> Any:
        r"""The abstract fold function.

        Each inherited class must implement this method.
        """
        raise NotImplementedError

    def _preprocess_params(self, **kwargs) -> None:
        r"""Preprocess the parameters before folding the quantum circuit.
        """
        qp = kwargs.get('qp')
        if qp is None:
            raise ArgumentError("in fold(): Input quantum circuit is None, cannot fold! Please check the input.")
        self._qp_origin = copy.deepcopy(qp)
        self._scale_factor = kwargs.get('scale_factor', self._scale_factor)
        self._method = kwargs.get('method', self._method)
        self._seed = kwargs.get('seed', self._seed)

        if self._scale_factor < 1:
            raise ArgumentError("in fold(): Requires scale_factor >= 1, but the input is {}".format(self._scale_factor))
        if self._method not in __SUPPORTED_METHODS__:
            raise ArgumentError("in fold(): Supported supported folding parameter generating methods"
                                "are {}, but the input is {}".format(__SUPPORTED_METHODS__, self._method))

        self._qp_folded = copy.deepcopy(qp)  # record the folded quantum circuit
        self._has_folded = True  # means it has the record of folding

        if self._scale_factor == 1:
            return self._qp_folded

    def __str__(self) -> str:
        s = 'A {} object. folding method: {}, has folding record: {}'.format(self.name, self.method, self.has_folded)
        if self.has_folded:
            s += ', with scaling factor: {}'.format(self.scale_factor)
        return s

    def _config_folding_method(self, method: str):
        if method not in __SUPPORTED_METHODS__:
            raise ArgumentError("'{}' is not a supported folding parameter generating method. "
                                "Should be in {}.".format(method, __SUPPORTED_METHODS__))
        else:
            self._method = method

    def compute_folding_parameters(self) -> Tuple[int, int, List[int]]:
        r"""Compute the folding parameters.

        Compute the folding parameters: the integer :math:`n`, the partial :math:`s` folding parts,
        and the corresponding partial folding indices,
        according to the scale factor :math:`\lambda` and the depth :math:`d`.
        The computation procedure is as follows.

        .. admonition:: Procedure

            1. Determine the closest integer :math:`k` to the quantity :math:`d(\lambda - 1)/2`.
            2. Perform an integer division of :math:`k` by :math:`d`. Set the quotient to :math:`n` and the reminder to :math:`s`.
            3. Generate the folding_indices set based on the method. The set is of size :math:`s`.

        :return: a tuple (n, s, indices), represent integer and partial foldings, and the folding indices
        """
        # If using the 'gate-level folding' method, we must use the number of multi-qubits gates as :math:`d`
        if isinstance(self, GateFolder):
            _, multi_qubit_gate_indices = group_gate_indices(self._qp_origin.circuit)
            d = len(multi_qubit_gate_indices)
        else:  # Else, use the number of layers in the quantum circuit as :math:`d`
            d = depth_of_circuit(self._qp_origin.circuit, measure=False)

        # In the very special case where the depth (or the number of multi-qubits) are 0, do not fold at all
        if d == 0:
            return 0, 0, None

        # folding parameters (integer folding and partial folding)
        k = round(d * (self._scale_factor - 1) / 2)
        n, s = divmod(k, d)

        # obtain folding indices set from the partial folding parameter :math:`s`
        indices = np.arange(d)

        if self._method == 'left':
            indices = indices[:s]
        elif self._method == 'right':
            indices = indices[d - s:]
        else:  # random
            rand_state = np.random.RandomState(self._seed)
            rand_state.shuffle(indices)
            indices = indices[:s]

        return int(n), int(s), indices.tolist()


class CircuitFolder(Folder):
    r"""The Circuit-level Folding Class.

    CircuitFolder folds a given quantum circuit in the circuit level, i.e., :math:`U \mapsto (UU^\dagger)U`.
    """

    def __init__(self, *args, **kwargs):
        super(CircuitFolder, self).__init__(*args, **kwargs)

    def fold(self, qp: QProgram, scale_factor: float, method: str = 'right', **kwargs: Any) -> QProgram:
        r"""Fold a given quantum circuit by the scale factor at the circuit level.

        Let :math:`U` represents the quantum circuit and assume its layer representation: :math:`U=[L_1,\cdots, L_d]`.
        The folding procedure carries out as follows.

        .. admonition:: Procedure

            + Step 1: Compute the folding parameters :math:`n` and @folding_indices from @scale_factor and @method.
            + Step 2: Initialize the folded quantum circuit :math:`V = U`, which will be finally returned.
            + Step 3: Fold the original circuit :math:`n` times and append it to :math:`V`, that is,

                .. math:: V \mapsto (U U^\dagger)^n V

            + Step 4: For each chosen layer :math:`L` in @folding_indices, fold it and append it to :math:`V`, that is,

                .. math:: V \mapsto (L_i L_i^\dagger)^{n} V

        :param qp: QProgram, the quantum circuit :math:`U` to be folded
        :param scale_factor: float, the scaling factor, which determines the parameters :math:`n` and @folding_indices
        :param method: str, folding indices selecting method. Optional: ['left', 'right', 'random']
        :return: QProgram, the folded quantum circuit :math:`V`

        **Examples**

            >>> qp = QEnv()
            >>> q = qp.Q.createList(2)
            >>> H(q[0])
            >>> CX(q[0], q[1])
            >>> MeasureZ(*qp.Q.toListPair())
            >>> print_circuit(qp.circuit)
            0: ---S---@---MEAS---
                      |
            1: -------X---MEAS---
            >>> folder = CircuitFolder()
            >>> qp_folded = folder.fold(qp, scale_factor=3)
            >>> print_circuit(qp_folded.circuit)
            0: ---S---@---@---SDG---S---@---MEAS---
                      |   |             |
            1: -------X---X-------------X---MEAS---
            >>> qp_folded = folder.fold(qp, scale_factor=4, method='left')
            >>> print_circuit(qp_folded.circuit)
            0: ---S---@---@---SDG---S---@---SDG---S---MEAS---
                      |   |             |
            1: -------X---X-------------X-------------MEAS---
            >>> qp_folded = folder.fold(qp, scale_factor=4, method='right')
            >>> print_circuit(qp_folded.circuit)
            0: ---S---@---@---SDG---S---@---@---@---MEAS---
                      |   |             |   |   |
            1: -------X---X-------------X---X---X---MEAS---
        """
        # Preprocess the parameters before folding
        self._preprocess_params(qp=qp, scale_factor=scale_factor, method=method, **kwargs)

        # Deep copy the original qp and modify the copied qp within the function
        qp_raw = copy.deepcopy(self._qp_origin)

        # Remove measurement and barrier gates if exist
        remove_measurement(qp_raw)
        measure_cl = remove_measurement(self._qp_folded)  # store temporarily the measurement operator
        remove_barrier(qp_raw)
        remove_barrier(self._qp_folded)

        # Generate the folding parameters according to the given indices generating method @method
        n, _, folding_indices = self.compute_folding_parameters()

        # Circuit-level folding, the folded quantum circuit will be stored in self._qp_folded
        # 1) whole circuit folding (circuit-wise)
        for i in range(n):
            self._qp_folded = _fold_and_append(self._qp_folded, qp_raw)

        # 2) fractional layers folding
        if folding_indices:
            self._qp_folded = _fold_and_append(self.qp_folded, qp_raw, folding_indices)

        # Append the popped measurements to the end of the folded quantum circuit
        if measure_cl is not None:
            append_measurement(self._qp_folded, measure_cl)

        return self._qp_folded


class LayerFolder(Folder):
    r"""The Layer-level Folding Class.

    LayerFolder folds a given quantum circuit in the layer level.
    """

    def __init__(self, *args, **kwargs):
        super(LayerFolder, self).__init__(*args, **kwargs)

    def fold(self, qp: QProgram, scale_factor: float, method: str = 'left', **kwargs: Any) -> QProgram:
        r"""Fold a given quantum circuit by the scale factor at the layer level.

        More precisely,
        Let :math:`U` represents the quantum circuit and assume its layer representation: :math:`U=[L_1,\cdots, L_d]`.
        The folding procedure carries out as follows.

        .. admonition:: Procedure

            + Step 1: Compute the folding parameters :math:`n` and @folding_indices from @scale_factor and @method
            + Step 2: Initialize the folded quantum circuit :math:`V` to identity, which will be finally returned
            + Step 3: For each layer :math:`L_i` in :math:`U`, fold and append it to :math:`V` via the following rule:

                + If :math:`i` belongs to @folding_indices, fold :math:`L_i` :math:`n+1` times and append, i.e.,

                    .. math:: V \mapsto (L_i L_i^\dagger)^{n+1} V

                + If :math:`i` does not belong to @folding_indices, fold :math:`L_i` :math:`n` times and append, i.e.,

                    .. math:: V \mapsto (L_i L_i^\dagger)^{n} V

        :param qp: QProgram, the quantum circuit to be folded
        :param scale_factor: float, the scaling factor, which determines the parameters :math:`n` and @folding_indices
        :param method: str, folding indices selecting method. Optional: ['left', 'right', 'random']
        :return: QProgram, the folded quantum circuit :math:`V`

        **Examples**

            >>> qp = QEnv()
            >>> q = qp.Q.createList(2)
            >>> H(q[0])
            >>> CX(q[0], q[1])
            >>> MeasureZ(*qp.Q.toListPair())
            >>> print_circuit(qp.circuit)
            0: ---S---@---MEAS---
                      |
            1: -------X---MEAS---
            >>> folder = LayerFolder()
            >>> qp_folded = folder.fold(qp, scale_factor=3)
            >>> print_circuit(qp_folded.circuit)
            0: ---S---SDG---S---@---@---@---MEAS---
                                |   |   |
            1: -----------------X---X---X---MEAS---
            >>> qp_folded = folder.fold(qp, scale_factor=4, method='left')
            >>> print_circuit(qp_folded.circuit)
            0: ---S---SDG---S---SDG---S---@---@---@---MEAS---
                                          |   |   |
            1: ---------------------------X---X---X---MEAS---
            >>> qp_folded = folder.fold(qp, scale_factor=4, method='right')
            >>> print_circuit(qp_folded.circuit)
            0: ---S---SDG---S---@---@---@---MEAS---
                                |   |   |
            1: -----------------X---X---X---MEAS---
        """
        # Preprocess the parameters before folding
        self._preprocess_params(qp=qp, scale_factor=scale_factor, method=method, **kwargs)

        # deep copy the original qp and modify the copied qp within the function
        qp_raw = copy.deepcopy(self._qp_origin)

        # remove measurement and barrier gates if exist
        remove_measurement(qp_raw)
        measure_cl = remove_measurement(self._qp_folded)  # store temporarily the measurement operator
        remove_barrier(qp_raw)
        remove_barrier(self._qp_folded)

        layers_raw = circuit_to_layers(qp_raw.circuit)

        # generate the folding parameters according to the given indices generating method @method
        n, _, folding_indices = self.compute_folding_parameters()

        # layer-level folding, the folded quantum circuit will be finally stored in self._qp_folded
        layers_folded = []
        for i, layer in enumerate(layers_raw):
            if i in folding_indices:
                # fold n + 1 times
                layers_folded.extend([layer] + [inverse_layer(layer), layer] * (n + 1))
            else:
                # fold n times
                layers_folded.extend([layer] + [inverse_layer(layer), layer] * n)

        self._qp_folded.circuit = layers_to_circuit(layers_folded)

        # append the popped measurements to the end of the folded quantum circuit
        if measure_cl is not None:
            append_measurement(self._qp_folded, measure_cl)

        return self._qp_folded


class GateFolder(Folder):
    r"""The Gate-level Folding Class.

    GateFolder folds a given quantum circuit in the multi-qubit gate level.
    """

    def __init__(self, *args, **kwargs):
        super(GateFolder, self).__init__(*args, **kwargs)

    def fold(self, qp: QProgram, scale_factor: float, method: str = 'left', **kwargs: Any) -> QProgram:
        r"""Fold a given quantum circuit by the scale factor at the gate level.

        More precisely,
        Let :math:`U` represents the quantum circuit and assume its gate representation: :math:`U=[G_1,\cdots, G_d]`,
        where each gate :math:`G_i` is an instance of the CircuitLine.
        The folding procedure carries out as follows.

        .. admonition:: Procedure

            + Step 1: Compute the folding parameters :math:`n` and @folding_indices from @scale_factor and @method.

            + Step 2: Initialize the folded quantum circuit :math:`V` to identity, which will be finally returned.

            + Step 3: For each gate :math:`G_i` in :math:`U`, fold and append it to :math:`V` via the following rule:

                + If :math:`G_i` is a single-qubit gate, does not fold and just append, i.e.,

                    .. math:: V \mapsto G_i V

                + If :math:`G_i` is a multi-qubit gate and belongs to @folding_indices, fold :math:`G_i` :math:`n+1` times and append, i.e.,

                    .. math:: V \mapsto (G_i G_i^\dagger)^{n+1} V

                + If :math:`G_i` is a multi-qubit gate and does not belong to @folding_indices, fold :math:`G_i` :math:`n` times and append, i.e.,

                    .. math:: V \mapsto (G_i G_i^\dagger)^{n} V

        :param qp: QProgram, the quantum circuit to be folded
        :param scale_factor: float, the scaling factor, which determines the parameters :math:`n` and @folding_indices
        :param method: str, folding indices selecting method. Optional: ['left', 'right', 'random']
        :return: QProgram, the folded quantum circuit :math:`V`

        **Examples**

            Construct three-qubit GHZ state preparation circuit and fold it accordingly.
            Notice that the last ``H`` is added for illustration purpose.

            >>> qp = QEnv()
            >>> q = qp.Q.createList(3)
            >>> H(q[0])
            >>> CX(q[0], q[1])
            >>> CX(q[0], q[2])
            >>> H(q[0])
            >>> MeasureZ(*qp.Q.toListPair())
            >>> print_circuit(qp.circuit)
            0: ---H---@---@---H---MEAS---
                      |   |
            1: -------X-----------MEAS---
                          |
            2: -----------X-------MEAS---
            >>> folder = GateFolder()
            >>> qp_folded = folder.fold(qp, scale_factor=3)
            >>> print_circuit(qp_folded.circuit)
            0: ---H---@---@---@---@---@---@---H---MEAS---
                      |   |   |   |   |   |
            1: -------X---X---X-------------------MEAS---
                                  |   |   |
            2: -------------------X---X---X-------MEAS---
            >>> qp_folded = folder.fold(qp, scale_factor=5)
            >>> print_circuit(qp_folded.circuit)
            0: ---H---@---@---@---@---@---@---@---@---@---@---H---MEAS---
                      |   |   |   |   |   |   |   |   |   |
            1: -------X---X---X---X---X---------------------------MEAS---
                                          |   |   |   |   |
            2: ---------------------------X---X---X---X---X-------MEAS---
        """
        # Preprocess the parameters before folding
        self._preprocess_params(qp=qp, scale_factor=scale_factor, method=method, **kwargs)

        # deep copy the original qp and modify the copied qp within the function
        qp_raw = copy.deepcopy(self._qp_origin)

        # remove measurement and barrier gates if exist
        remove_measurement(qp_raw)
        measure_cl = remove_measurement(self._qp_folded)  # store temporarily the measurement operator
        remove_barrier(qp_raw)
        remove_barrier(self._qp_folded)

        one_qubit_gate_indices, multi_qubit_gate_indices = group_gate_indices(qp_raw.circuit)

        # generate the folding parameters according to the given indices generating method @method
        n, _, folding_indices = self.compute_folding_parameters()

        # gate-level folding, the folded quantum circuit will be stored in self._qp_folded
        self._qp_folded.circuit = []
        for i, cl in enumerate(qp_raw.circuit):
            if i in one_qubit_gate_indices:  # current gate is a single-qubit gate, do not fold
                self._qp_folded.circuit.extend([cl])
            else:  # current gate is a multi-qubit gate
                # find the index of the multi-qubit gate index @i in the @multi_qubit_gate_indices list
                try:
                    k = multi_qubit_gate_indices.index(i)
                except ValueError:
                    raise ArgumentError("mulit-qubit gate index {} is not in the circuit list!".format(i))

                # if the current multi-qubit gate index is in the @folding_indices, fold it n+1 times
                if k in folding_indices:
                    self._qp_folded.circuit.extend([cl] + [cl.inverse(), cl] * (n + 1))
                else:  # else, fold it n times
                    self._qp_folded.circuit.extend([cl] + [cl.inverse(), cl] * n)

        # append the popped measurements to the end of the folded quantum circuit
        if measure_cl is not None:
            append_measurement(self._qp_folded, measure_cl)

        return self._qp_folded


def _fold_and_append(qp: QProgram, qp_inserted: QProgram = None, folding_indices: List[int] = None) -> QProgram:
    r"""Folding a quantum circuit by appending another circuit.

    Extend the given circuit @qp by folding and inserting some layers of @qp_inserted to the end.
    The chosen layers of @qp_inserted is specified folding_indices.

    .. note::

        Let :math:`U` represents @qp, and let :math:`[V_1, \cdots, V_d]` represents @qp_inserted.
        If the folding_indices are :math:`[1, 3]`, then the extended quantum program has the form

        .. math::     V_3 V_3^\dagger \cdot V_1 V_1^\dagger \cdot U

    :param qp: QProgram, the quantum program to be folded
    :param qp_inserted: QProgram, the quantum program to be inserted
    :param folding_indices: List[int], specifies the layers to be folded in the target circuit @qp_inserted.
                        If None, all layers of the target quantum circuit @qp_inserted
                        (the entire quantum circuit) will be folded and appended to @qp
    :return: QProgram, the final folded `QProgram` instance
    """
    if qp_inserted is None:
        qp_inserted = copy.deepcopy(qp)
    qp_folded = copy.deepcopy(qp)

    # remove measurement and barrier gates if exist
    remove_measurement(qp_inserted)
    remove_barrier(qp_inserted)

    # parse layers information from a quantum circuit data field
    layers_inserted = circuit_to_layers(qp_inserted.circuit)

    # identity insertion and append: U -> U (V^dag V)
    # U is the unitary corresponding to @qp, and
    # V can be either the entire quantum circuit or some chosen layers of qp_inserted determined by @folding_indices

    # determine the folding indices. If None, fold all layers in @qp_inserted
    if folding_indices is None:
        folding_indices = list(range(len(layers_inserted)))

    # collect the layers to be folded
    layers_selected = [layer for i, layer in enumerate(layers_inserted) if i in folding_indices]
    layers_selected_dag = inverse_layers(layers_selected)

    # append V^dag
    qp_folded.circuit.extend(layers_to_circuit(layers_selected_dag))

    # append V
    qp_folded.circuit.extend(layers_to_circuit(layers_selected))

    return qp_folded
