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

"""Implement the abstract Quantum Channel class and its inherited classes:

1. the PTM (Pauli Transfer Matrix) Representation,

2. the Chi (Process) Representation,

3. the Choi Representation, and

4. the Kraus Representation.

For more details on Quantum Channels, we suggest the references [NC10]_ and [G15]_.

References:

.. [NC10] Nielsen, Michael A., and Isaac L. Chuang.
    "Quantum Computation and Quantum Information: 10th Anniversary Edition."
    Cambridge University Press, 2010.

.. [G15] Greenbaum, Daniel. "Introduction to quantum gate set tomography." arXiv preprint arXiv:1509.02921 (2015).
"""
from __future__ import annotations

import abc
from typing import Union, List, Tuple
import numpy as np
import copy
from Extensions.QuantumErrorProcessing.qcompute_qep.exceptions.QEPError import ArgumentError
from Extensions.QuantumErrorProcessing.qcompute_qep.quantum.pauli import complete_pauli_basis, from_name_to_matrix
import Extensions.QuantumErrorProcessing.qcompute_qep.utils.linalg as la

# define the abstract quantum channel data type
QChannel = Union[np.ndarray, List[np.ndarray], List[Tuple[np.ndarray, np.ndarray]]]


class QuantumChannel(abc.ABC):
    """This is the abstract Quantum Channel class."""
    def __init__(self, data: QChannel = None, rep_type: str = None, shape: Tuple[int, int] = None, name: str = None):
        """The init function of quantum channel.

        :param data: Union[np.array, List[np.array], List[np.array, np.array]]
                    List[np.array] used to init the Kraus representation of quantum channel
                    List[np.array, np.array] used to init general Kraus representation
                    np.array use to init others representation
        :param rep_type: str the representation type of quantum channel, like "Kraus", "Chi"
        :param shape: List[int, int] the operator shape of channel [in_dim, out_dim]
                    for n-qubits system, the in_dim is  :math:`2^n` (the dimension of linear map).
        :param name: str user can define the representation name by self arbitrarily
        """
        self._data = data
        self._rep_type = rep_type
        self._shape = shape
        self._name = name

    @property
    def data(self):
        return self._data

    @property
    def rep_type(self):
        return self._rep_type

    @property
    def name(self):
        return self._name

    def is_cptp(self) -> bool:
        """Return True if completely-positive trace-preserving (CPTP)."""
        choi = qc_convert(self, "Choi")
        return choi.is_cp() and choi.is_tp()

    def is_cp(self) -> bool:
        """Return true if completely-positive."""
        choi = qc_convert(self, "Choi")
        return choi.is_cp()

    def is_tp(self) -> bool:
        """Return True if trace-preserving."""
        choi = qc_convert(self, "Choi")
        return choi.is_tp()

    def is_unital(self) -> bool:
        """Return True if unital."""
        # TODO qiskit 中转换为 operator
        pass

    @abc.abstractmethod
    def evolve(self, rho: np.array) -> np.array:
        """Evolve a quantum state \\rho by the quantum channel.

        :param rho: the input quantum state, a vector or density matrix
        :return: the final quantum state in terms of density matrix
        """
        pass

    @abc.abstractmethod
    def tensor(self, other: 'QuantumChannel' = None, **kwargs) -> 'QuantumChannel':
        """Tensor with other quantum channel. Optional keywords list are:

        + `k`: default to 1, the times of tensor operate

        :param other: QuantumChannel
        :return: QuantumChannel, Whole quantum channel after tensor
        """
        raise NotImplementedError

    @abc.abstractmethod
    def concat(self, other: 'QuantumChannel' = None, **kwargs) -> 'QuantumChannel':
        """Concatenate with other quantum channel.Optional keywords list are:

        + `k`: default to 1, the times of tensor operate

        :param other: QuantumChannel
        :return: QuantumChannel, Whole quantum channel after concatenate
        """
        raise NotImplementedError

    @abc.abstractmethod
    def inverse(self):
        """Inverse the quantum channel."""
        raise NotImplementedError


class PTM(QuantumChannel):
    """This is the Pauli Transfer Matrix Representation class."""
    def __init__(self, data, name: str = None):
        """The init function fo PTM representation class.

        :param data: a np.array or a list or a other representation type of quantum channel
        :param name: str user can define the representation name by self arbitrarily
        """
        if isinstance(data, (list, np.ndarray)):
            ptm = np.asarray(data, dtype=complex)
            odim, idim = ptm.shape
            in_dim = np.sqrt(idim)
            out_dim = np.sqrt(odim)
        else:  # TODO: use another QuantumChannel to init
            raise ArgumentError("In PTM.init(), unsupported input type")
        super().__init__(ptm, "PTM", (in_dim, out_dim), name)

    def __repr__(self):
        prefix = f"{self._rep_type}("
        pad = len(prefix) * " "
        return "{}{},\n{}input_dims={}, output_dims={})".format(
            prefix,
            np.array2string(np.asarray(self.data), separator=", ", prefix=prefix),
            pad,
            self._shape[0],
            self._shape[1],
        )

    def is_tp(self) -> bool:
        """Return True if trace-preserving (TP).

        The first row of the PTM is one and all zeros.
        """
        return self.data[0, 0] == 1 and np.all(self.data[0, 1:] == 0)

    def is_unital(self) -> bool:
        """Return True if unital The fist column of the PTM is one and all
        zeros."""
        return self.data[0, 0] == 1 and np.all(self.data[1:, 0] == 0)

    def evolve(self, rho: np.array) -> np.array:
        r"""
        TODO: add a optional key word to decide the return type

        :param rho: the input quantum state, a vector or a density matrix
        :return: the quantum after evolve, the return type is density matrix
        """
        pass

    def tensor(self, other: QuantumChannel = None, **kwargs) -> QuantumChannel:
        """Tensor with other quantum channel. Optional keywords list are:

        + `k`: default to 1, the times of tensor operate

        :param other: QuantumChannel
        :return: QuantumChannel, Whole quantum channel after tensor
        """
        if other is None:
            other = copy.copy(self)
        if not isinstance(other, type(self)):  # TODO: convert to PTM first, and use optional key words
            other = PTM(other)
        times = kwargs.get('k', 1)  # default 1
        if times <= 0:
            raise ArgumentError("in channel.PTM.tensor(): the k has to > 0!")
        ret = np.kron(self.data, other.data)
        for _ in range(times-1):
            ret = np.kron(ret, other.data)
        return PTM(ret)

    def concat(self, other: QuantumChannel = None, **kwargs) -> QuantumChannel:
        """Concatenate with other quantum channel.Optional keywords list are:

        + `k`: default to 1, the times of tensor operate

        :param other: QuantumChannel
        :return: QuantumChannel, Whole quantum channel after concatenate
        """
        if other is None:
            other = copy.copy(self)
        if not isinstance(other, type(self)):  # TODO: convert to PTM first, and use optional key words
            other = PTM(other)
        times = kwargs.get('k', 1)  # default 1
        if times <= 0:
            raise ArgumentError("in channel.PTM.concat(): the k has to > 0!")
        ret = np.dot(other.data, self.data)
        for _ in range(times - 1):
            ret = np.dot(other.data, ret)
        return PTM(ret)

    def inverse(self):
        return PTM(np.invert(self.data))


class Chi(QuantumChannel):
    """This is the Process Matrix Representation class."""

    def __init__(self, data, name: str = None):
        pass

    def evolve(self, rho: np.array) -> np.array:
        """Evolve a quantum state \\rho by the quantum channel.

        :param rho: the input quantum state, a vector or density matrix
        :return: the final quantum state in terms of density matrix
        """
        pass

    def tensor(self, other: 'QuantumChannel' = None, **kwargs) -> 'QuantumChannel':
        """Tensor with other quantum channel. Optional keywords list are:

        + `k`: default to 1, the times of tensor operate

        :param other: QuantumChannel
        :return: QuantumChannel, Whole quantum channel after tensor
        """
        pass

    def concat(self, other: 'QuantumChannel' = None, **kwargs) -> 'QuantumChannel':
        """Concatenate with other quantum channel.Optional keywords list are:

        + `k`: default to 1, the times of tensor operate

        :param other: QuantumChannel
        :return: QuantumChannel, Whole quantum channel after concatenate
        """
        pass

    def inverse(self):
        """Inverse the quantum channel."""
        pass


class Choi(QuantumChannel):
    """This is the Choi Representation class."""
    def __init__(self, data, name: str = None):
        pass

    def evolve(self, rho: np.array) -> np.array:
        """Evolve a quantum state \\rho by the quantum channel.

        :param rho: the input quantum state, a vector or density matrix
        :return: the final quantum state in terms of density matrix
        """
        pass

    def tensor(self, other: 'QuantumChannel' = None, **kwargs) -> 'QuantumChannel':
        """Tensor with other quantum channel. Optional keywords list are:

        + `k`: default to 1, the times of tensor operate

        :param other: QuantumChannel
        :return: QuantumChannel, Whole quantum channel after tensor
        """
        pass

    def concat(self, other: 'QuantumChannel' = None, **kwargs) -> 'QuantumChannel':
        """Concatenate with other quantum channel.Optional keywords list are:

        + `k`: default to 1, the times of tensor operate

        :param other: QuantumChannel
        :return: QuantumChannel, Whole quantum channel after concatenate
        """
        pass

    def inverse(self):
        """Inverse the quantum channel."""
        pass


class Kraus(QuantumChannel):
    """This is the Kraus Representation class."""

    def __init__(self, data, name: str = None):
        pass

    def evolve(self, rho: np.array) -> np.array:
        """Evolve a quantum state \\rho by the quantum channel.

        :param rho: the input quantum state, a vector or density matrix
        :return: the final quantum state in terms of density matrix
        """
        pass

    def tensor(self, other: 'QuantumChannel' = None, **kwargs) -> 'QuantumChannel':
        """Tensor with other quantum channel. Optional keywords list are:

        + `k`: default to 1, the times of tensor operate

        :param other: QuantumChannel
        :return: QuantumChannel, Whole quantum channel after tensor
        """
        pass

    def concat(self, other: 'QuantumChannel' = None, **kwargs) -> 'QuantumChannel':
        """Concatenate with other quantum channel.Optional keywords list are:

        + `k`: default to 1, the times of tensor operate

        :param other: QuantumChannel
        :return: QuantumChannel, Whole quantum channel after concatenate
        """
        pass

    def inverse(self):
        """Inverse the quantum channel."""
        pass


###############################################################################
# In the following, we implement a set of functions for the transformation
# between different types of quantum channel representations.
###############################################################################
def qc_convert(in_qc: QuantumChannel = None, tar_type: str = None) -> QuantumChannel:
    """Transfer the representation of quantum channel to target type quantum
    channel.

    :param in_qc: QuantumChannel, the representation of quantum channel before transformation
    :param tar_type: str, the target type representation of quantum channel
    :return: QuantumChannel, the new quantum channel after transformation
    """
    if tar_type == "Choi" or tar_type == "choi":
        return Choi(_to_choi(in_qc.data, in_qc.rep_type))


def op_convert(in_op: QChannel = None, in_type: str = None, tar_type: str = None) -> QChannel:
    """Transfer the representation of quantum channel to target type quantum
    channel.

    :param in_op: QuantumChannel, the representation of quantum channel before transformation
    :param in_type: the type representation of input channel data
    :param tar_type: str, the target type representation of quantum channel
    :return:
    TODO: only transfer the operator between the representation
    """
    if tar_type == "Choi":
        return _to_choi(in_op, in_type)


def _to_choi(data: QChannel, rep_type: str) -> np.array:
    """Convert all representation to choi matrix.

    :param data: the quantum channel data before transformation
    :param rep_type: the representation type of quantum channel
    :return: a choi-matrix
    TODO
    """
    if rep_type == "PTM" or rep_type == "ptm":
        natural = _ptm_to_natural(data)
        return _natural_to_choi(natural)


def _to_ptm(data: QChannel, rep_type: str) -> np.array:
    """Convert all representation to ptm matrix.

    :param data: the quantum channel data before transformation
    :param rep_type: the representation type of quantum channel
    :return: a ptm-matrix
    TODO
    """
    pass


###############################################################################
# In the following, we implement one to one type transformation
###############################################################################
def _ptm_to_natural(data: np.ndarray = None) -> np.ndarray:
    """Convert PTM representation to Natural representation."""
    pass


def _natural_to_choi(data: np.ndarray = None) -> np.ndarray:
    """Convert Natural representation to Choi representation.

    .. math:: \\mathcal{N}_{vn,um} = \\Lambda_{mn,uv}
    """
    pass


###############################################################################
# In the following, we implement some mathematical support
###############################################################################
def _transform_from_pauli(data: QChannel = None, num_qubits: int = 0) -> QChannel:
    """Transform the basis of operator from Pauli basis.

    :param data: QChannel, the operator in terms of Pauli basis
    :param num_qubits: the number of qubits
    :return:
    """
    pass


def _transform_to_pauli(data: QChannel = None, num_qubits: int = 0):
    """Transform the basis of operator to Pauli basis.

    :param data: QChannel, the operator in terms of Pauli basis
    :param num_qubits: the number of qubits
    :return:
    """
    pass


def _reshuffle(mat, shape):
    """Reshuffle the indices of a bipartite matrix A[ij,kl] -> A[lj,ki]."""
    return np.reshape(
        np.transpose(np.reshape(mat, shape), (3, 1, 2, 0)),
        (shape[3] * shape[1], shape[0] * shape[2]),
    )


def unitary_to_ptm(unitary: np.ndarray) -> QuantumChannel:
    r"""From a process unitary to its Pauli transfer matrix.

    Convert a process unitary in the matrix representation to the Pauli transfer matrix (PTM) representation.
    Assume the unitary has :math:`n` qubits, then the corresponding PTM is of size :math:`4^n\times 4^n`.

    **Examples**

        In the following, we compute the PTM of the :math:`X` gate.

        >>> X = np.array([[0, 1], [1, 0]]).astype(complex) / np.sqrt(2)
        >>> ptm = unitary_to_ptm(X)
        >>> print(ptm.data)
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j -1.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j -1.+0.j]]

    :param unitary: np.ndarray, the matrix representation of the quantum process unitary
    :return: np.ndarray, the Pauli transfer matrix representation of the unitary
    """
    if not np.log2(unitary.shape[0]).is_integer():
        raise ArgumentError("in unitary_to_ptm(): the dimensions of the unitary must be the power of 2!")
    # Number of qubits
    n = int(np.log2(unitary.shape[0]))
    cpb = complete_pauli_basis(n)
    ptm = np.zeros((len(cpb), len(cpb)), dtype=float)
    for j, pauli_j in enumerate(cpb):
        for k, pauli_k in enumerate(cpb):
            ptm[j, k] = np.real(np.trace(pauli_j.matrix @ unitary @ pauli_k.matrix @ la.dagger(unitary)))
    return PTM(ptm)


def ptm_to_process(ptm: QuantumChannel, type: str = 'kraus') -> QuantumChannel:
    r"""From the Pauli transfer matrix of a quantum process to the given representation.

    Convert a quantum process in the Pauli transfer matrix representation to the target representation.
    Candidate target representations are:

    + str = ``superoperator``, the superoperator representation, also known as the natural representation;
    + str = ``choi``, the Choi representation;
    + str = ``kraus``, the Kraus operator representation, also known as the operator-sum representation;
    + str = ``chi``, the chi matrix representation, aka. the process matrix representation.

    :param ptm: np.ndarray, the Pauli transfer matrix representation of the quantum process
    :param type: str, default to 'kraus', type: str = 'kraus'
    :return: QuantumChannel, a quantum channel class instance
    """
    return qc_convert(ptm, type)
