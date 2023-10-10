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
The `BasicCode` class for the quantum error correction module.
The implementations---`BitFlipCode`, `PhaseFlipCode`, `FourQubitCode`,
`FiveQubitCode`, `SteaneCode`, and `ShorCode`---all must inherit this class.

References:

.. [CYK+22] Chen, Edward H., et al.
    "Calibrated decoders for experimental quantum error correction."
    Physical Review Letters 128.11 (2022): 110504.

.. [R19] Joschka Roffe.
    "Quantum error correction: an introductory guide."
    Contemporary Physics 60.3 (2019): 226-245.

.. [NC10] Nielsen, Michael A., and Isaac L. Chuang.
    "Quantum Computation and Quantum Information: 10th Anniversary Edition."
    Cambridge University Press, 2010.

.. [G97] Daniel Gottesman.
    "Stabilizer Codes and Quantum Error Correction."
    PhD thesis, California Institute of Technology (1997).
"""
import copy
import pprint
from typing import Any, List, Union, Callable

import numpy as np
from QCompute import *

from Extensions.QuantumErrorProcessing.qcompute_qep.exceptions import ArgumentError
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.types import QProgram, number_of_qubits
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.gate import CPauli
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.circuit import enlarge_circuit, print_circuit
from Extensions.QuantumErrorProcessing.qcompute_qep.quantum.pauli import pauli2bsf, bsf2pauli, bsp, mutually_commute
from Extensions.QuantumErrorProcessing.qcompute_qep.correction.stabilizer import StabilizerCode
from Extensions.QuantumErrorProcessing.qcompute_qep.correction.utils import pauli_list_to_check_matrix, check_matrix_to_standard_form
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.utils import COLOR_TABLE


class BasicCode(StabilizerCode):
    r"""The basic stabilizer error correction code class.

    .. admonition:: Note on Stabilizers and Standard From

        One should be very careful on the difference between the two
        properties---``self._stabilizers`` and ``self._standard_form``---defined for a stabilizer code.
        The former is specified by the user,
        while the latter is derived from the former by constructing a *standard form*, using Gaussian elimination.
        Both of them characterize a complete list of stabilizer generators of the same stabilizer code.
        See Section 4.1 of [G97]_ for the detailed description of standard form of a stabilizer code.

        We use the standard form, instead of the original stabilizer generators that the users specified,
        to construct the syndrome dictionary, to encode, correct, and decode the logical states.
        In other words, the original stabilizer generators specified by the users are used **only** to
        construct the standard form, and does not play roles in other functions and procedures.

    .. admonition:: Note on Qubits Order

        In general, a stabilizer code encodes a :math:`k`-qubit state into :math:`n` physical qubits,
        using :math:`n-k` ancilla qubits to implement the stabilizer measurements.
        Thus, a stabilizer code has :math:`2n-k` qubits in total.
        In our implementation, we assume that :math:`k`-qubit state is stored in
        the first :math:`k` qubits of the :math:`n` physical qubits,
        while the ancilla qubits are stored in the last :math:`n-k` qubits:

            ::

                 qubit indices: [q[0] ... q[k-1] |    q[k] ... q[n-1]    | q[n] ... q[2n-k-1]]
              number of qubits:        k         |         n-k           |        k
                     partition: original qubits  | other physical qubits |  ancilla qubits

    """
    def __init__(self, stabilizers: List[str], error_types: List[str], **kwargs: Any):
        r"""init function of the `BasicCode` class.

        The ``BasicCode`` class must be instantiated with two required arguments: `stabilizers` and `error_types`.

        The `stabilizers` argument specifies the set of stabilizer generators. For example,
        the stabilizer generators of the five-qubit code are:

        ::

            stabilizers=['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'].

        The `error_types` argument specifies
        the set of detectable error types. Current supported error types are: {'X', 'Y', 'Z'},
        where 'X' means that single-qubit X-type error is detectable, similarly for 'Y' and 'Z'. For example,

        + `error_types=['X']` means that arbitrary single-qubit X-type error can be detected.

        + `error_types=['X', 'Z']` means that both single-qubit X-type and Z-type errors can be detected.


        Optional keywords list are:

            + `name`: str, default to `None`, the name of the stabilizer code

        :param stabilizers: List[str], a list of stabilizer generators, each is a Pauli string
        :param error_types: List[str], a list of detectable error types
        """
        super().__init__(stabilizers=stabilizers, **kwargs)
        self._stabilizers = stabilizers         # A list of stabilizer generators of the code
        self._error_types = error_types         # A list of detectable error types by the code
        self._name = kwargs.get('name', None)   # The name of the code
        # Number of physical qubits
        self._n = len(self._stabilizers[0])
        # Number of logical qubits, equals to the number of physical qubits minus the number of stabilizer generators
        self._k = self._n - len(self._stabilizers)

        # Construct the check matrix from stabilizer generators
        self._check_matrix = pauli_list_to_check_matrix(self._stabilizers)

        # Convert the check matrix to standard form and construct logical operators
        self._standard_form, self._logical_xs, self._logical_zs, self._r = \
            check_matrix_to_standard_form(self._check_matrix)

        # After successfully generating logical operators, perform **Sanity Check** on the stabilizer code.
        # Quantum Error Correction can be achieved only if these sanity checks are satisfied.
        self.sanity_check()

        # Compute the code distance and update the parameter `self._d`
        self._compute_code_distance()

        # Construct the set of detectable errors and update the parameter `self._detectable_errors`
        self._construct_detectable_errors()

        # Construct syndrome lookup dictionary and update the syndrome dictionary `self._syndrome_dict`
        self._construct_syndrome_dict()

        pass

    @property
    def n(self):
        r"""Number of physical qubits
        """
        return self._n

    @property
    def k(self):
        r"""Number of logical qubits
        """
        return self._k

    @property
    def d(self):
        r"""Distance of the stabilizer code.
        """
        return self._d

    @property
    def n_k_d(self):
        r"""The triple [[n,k,d]] of the stabilizer code.
        """
        return self._n, self._k, self._d

    @property
    def r(self):
        r"""Rank of the X portion of the check matrix
        """
        return self._r

    @property
    def detectable_errors(self):
        r"""The set of detectable errors by the stabilizer code.
        """
        return self._detectable_errors

    @property
    def syndrome_dict(self):
        r"""The syndrome dictionary of the stabilizer code.
        """
        return self._syndrome_dict

    @property
    def stabilizers(self):
        r"""The list of stabilizer generators in Pauli string, specified by the users.
        """
        return self._stabilizers

    @property
    def error_types(self):
        r"""The detectable error types, specified by the users.
        """
        return self._error_types

    @property
    def check_matrix(self):
        r"""The check matrix of the stabilizer generators.
        """
        return self._check_matrix

    @property
    def name(self):
        r"""The name of the stabilizer code.
        """
        if self._name is None:
            return "[[{}, {}, {}]] Stabilizer Code".format(self._n, self._k, self._d)
        else:
            return self._name

    @property
    def info(self):
        r"""Detailed information of the stabilizer code.
        """
        info_str = ["*******************************************************************************",
                    "Code Name: {}".format(self.name),
                    "Parameters: {}".format(self.n_k_d),
                    "Stabilizers: {}".format(self.stabilizers),
                    "Standard Form: {}".format(self.standard_form(form='str')),
                    "Rank: {}".format(self.r),
                    "Logical X(s): {}".format(self.logical_xs(form='str')),
                    "Logical Z(s): {}".format(self.logical_zs(form='str')),
                    "Detectable Errors: {}".format(self.detectable_errors),
                    "Syndrome Dictionary: ", pprint.pformat(self.syndrome_dict),
                    "NOTE: For a Pauli string, the right most qubit represents 'q0'.",
                    "*******************************************************************************"]
        return "\n".join(info_str)

    def __str__(self):
        return self.info

    def standard_form(self, form: str = 'bsf') -> Union[List[str], np.ndarray]:
        r"""Return the standard form the check matrix.

        The form of the returned standard form is determined by the argument `@form`.

        + | ``form='str'`` indicates that the operators specified by the standard form
          | are in string form; correspondingly the return value is a list of strings.

        + | ``form='bsf'`` indicates that the operators specified by the standard form
          | are in binary symplectic form; correspondingly the return value is numpy array.
          | In this case, directly return ``self._standard_form``.

        :param form: str, default to 'bsf', indicates which form the standard form should be returned.

        **Examples**

            >>> five_qubit_code = FiveQubitCode()
            >>> print("Standard Form in String: {}".format(five_qubit_code.standard_form(form='str')))
            Standard Form in string: ['YZIZY', 'IXZZX', 'ZZXIX', 'ZIZYY']
            >>> print("Standard Form in BSF:")
            >>> print(five_qubit_code.standard_form(form='bsf'))
            Standard Form in BSF:
            [[1 0 0 0 1 1 1 0 1 1]
             [0 1 0 0 1 0 0 1 1 0]
             [0 0 1 0 1 1 1 0 0 0]
             [0 0 0 1 1 1 0 1 1 1]]
        """
        if form == 'str':
            return [bsf2pauli(row) for row in self._standard_form]
        elif form == 'bsf':
            return self._standard_form
        else:
            raise ArgumentError("in standard_form(): undefined Pauli form! Supported "
                                "forms are 'str' (string form) and 'bsf' (binary symplectic form)")

    def logical_xs(self, form: str = 'bsf') -> Union[List[str], np.ndarray]:
        r"""Return the set of logical X operators with given form.

        The form the returned logical X operators is determined by the argument `@form`.

        + | ``form='str'`` indicates that the logical X operators are in string form;
          | correspondingly the return value is a list of strings.

        + | ``form='bsf'`` indicates that the logical X operators are in binary symplectic form;
          | correspondingly the return value is numpy array. In this case, directly return ``self._logical_xs``.

        :param form: str, default to 'bsf', indicates which form the logical X operators should be returned.

        **Examples**

            >>> five_qubit_code = FiveQubitCode()
            >>> print("Logical X in Pauli String: {}".format(five_qubit_code.logical_xs(form='str')))
            Logical X in Pauli String: ['ZIIZX']
            >>> print("Logical X in BSF: {}".format(five_qubit_code.logical_xs(form='bsf')))
            Standard Form in BSF: [[0 0 0 0 1 1 0 0 1 0]]
        """
        if form == 'str':
            return [bsf2pauli(row) for row in self._logical_xs]
        elif form == 'bsf':
            return self._logical_xs
        else:
            raise ArgumentError("in logical_xs(): undefined Pauli form! Supported "
                                "forms are 'str' (string form) and 'bsf' (binary symplectic form)")

    def logical_zs(self, form: str = 'bsf') -> Union[List[str], np.ndarray]:
        r"""Return the set of logical Z operators with given form.

        The form the returned logical Z operators is determined by the argument `@form`.

        + | ``form='str'`` indicates that the logical Z operators are in string form;
          | correspondingly the return value is a list of strings.

        + | ``form='bsf'`` indicates that the logical Z operators are in binary symplectic form;
          | correspondingly the return value is numpy array. In this case, directly return ``self._logical_zs``.

        :param form: str, default to 'bsf', indicates which form the logical Z operators should be returned.

        **Examples**

            >>> five_qubit_code = FiveQubitCode()
            >>> print("Logical Z in Pauli String: {}".format(five_qubit_code.logical_zs(form='str')))
            Logical Z in Pauli String: ['ZZZZZ']
            >>> print("Logical Z in BSF: {}".format(five_qubit_code.logical_zs(form='bsf')))
            Logical Z in BSF: [[0 0 0 0 0 1 1 1 1 1]]
        """
        if form == 'str':
            return [bsf2pauli(row) for row in self._logical_zs]
        elif form == 'bsf':
            return self._logical_zs
        else:
            raise ArgumentError("in logical_zs(): undefined Pauli form! Supported "
                                "forms are 'str' (string form) and 'bsf' (binary symplectic form)")

    def sanity_check(self) -> bool:
        r"""Check if the stabilizer code is well-defined.

        A stabilizer code is *well-defined* if it satisfies the following conditions:

        1. The list of stabilizers must mutually commute;

        2. The list of logical X operators must mutually commute;

        3. The list of logical Z operators must mutually commute;

        4. The list of stabilizers must mutually commute with the logical X operators;

        5. The list of stabilizers must mutually commute with the logical Z operators;

        6. The logical X and Z operators operating on the **same** logical qubit must anti-commute.

        If **any** of the above conditions fail, we raise an ``ArgumentError`` indicating that the stabilizer
        code does not pass the sanity check. Return *True* if all the six conditions are satisfied.

        :return: bool, *True* if all the six conditions are satisfied;
                        If any of the conditions fail, a corresponding ``ArgumentError`` is raised.
        """
        # Condition 1: The list of stabilizers must mutually commute.
        if not mutually_commute(self._stabilizers):
            raise ArgumentError('{} Code: Stabilizers must mutually commute!'.format(self._name))
        # Conditions 2 and 4: The list of logical X operators must mutually commute and
        # the list of stabilizers must mutually commute with the logical X operators.
        # if not mutually_commute(self._stabilizers + self.logical_xs(form='str')):
        #     raise ArgumentError('{} Code: Stabilizers must commute with logical X operators!'.format(self._name))
        # Conditions 3 and 5: The list of logical Z operators must mutually commute and
        # the list of stabilizers must mutually commute with the logical Z operators.
        # if not mutually_commute(self._stabilizers + self.logical_zs(form='str')):
        #     raise ArgumentError('{} Code: Stabilizers must commute with logical Z operators!'.format(self._name))
        # Condition 6: The logical X and Z operators operating on the same logical qubit must anti-commute.
        for logical_x, logical_z in zip(self.logical_xs(form='str'), self.logical_zs(form='str')):
            if mutually_commute([logical_x, logical_z]):
                raise ArgumentError('{} Code: Logical X and Z operators '
                                    'on the same qubit must anticommute!'.format(self._name))

    def _compute_code_distance(self):
        r"""Compute the code distance of the stabilizer code.

        The distance of a stabilizer code is defined to be the *minimum weight* of
        the logical operators of the stabilizer code.
        The weight of a Pauli operator is defined to be the number of single-qubit Pauli operators
        that are unequal to 'I'. For example, the weight of the Pauli operator ``'XXIXZ'`` is :math:`4`.

        #TODO# Compute the minimum distance by considering all logical operators.
        """
        # Construct the set of logical operators
        logical_ops = self.logical_xs(form='str') + self.logical_zs(form='str')
        # Find the minimum weight
        self._d = min([len(op) - op.upper().count('I') for op in logical_ops])

    def _construct_detectable_errors(self):
        r"""Compute the set of detectable errors of the stabilizer code.

        This function computes the set of detectable errors (in Pauli strings) of the stabilizer code,
        from the ``self._error_types`` parameter.
        This parameter is given by the user and specifies what kind of errors can be detected.
        Current supported error types are: {'X', 'Y', 'Z'}. For example,

        + `self._error_types=['X']` means that single-qubit X-type error can be detected.

        + `self._error_types=['X', 'Z']` means that both single-qubit X-type and Z-type errors can be detected.

        #TODO# add more error types. For example, X-type error on one qubit and a Z-type error on another.
        """
        self._detectable_errors = []

        for error in self._error_types:
            # Detectable single-qubit Pauli error
            if error in ['X', 'Y', 'Z']:
                for pos in range(self.n):
                    error_str = ['I'] * self._n
                    error_str[self.n - 1 - pos] = error
                    self._detectable_errors.append("".join(map(str, error_str)))
            else:
                raise ArgumentError("BasicCode(): undefined error type. Supported error types are: {'X', 'Y', 'Z'}.")

    def _construct_syndrome_dict(self):
        r"""Compute the syndrome dictionary of the stabilizer code.

        The syndrome dictionary is a dictionary of the form:

        ::

            {'IIIIX': '1100', 'IIIXI': '0110', 'IIXII': '0011', ...}

        where each key represents a detectable error and the corresponding value represents the syndrome,
        i.e., the commutativity relation of the error with the list of stabilizer generators.
        Each error must have :math:`n` single-qubit Pauli operators, and each syndrome must have size :math:`n-k`.

        Consider the (key, value) pair ``('IIIIX': '1100')``, it means that the
        single-qubit :math:`X` error on qubit :math:`0` will yield syndrome :math:`1100`.
        Likewise, ``('IIIXI': '0110')`` means that the single-qubit :math:`X` error on qubit :math:`1`
        will yield syndrome :math:`0110`.

        .. note::

            We assume the LSB (the least significant bit) mode when defining the error operator.
            That is,  the right-most bit of string represents q[0]:

            ::

                name:           'I      I     I     I      X'

                qubits:         q[4]   q[3]  q[2]  q[1]   q[0]

        .. note::

            When computing the syndrome, we assume it is ordered as :math:`s_l \cdots s_2s_1s_0`,
            where :math:`s_i\in\{0,1\}` is the measurement outcome of the :math:`i`-th stabilizer generator,
            with respect to the stabilizer generator order specified by the variable ``self._stabilizers``.

        .. admonition:: Example

            Consider the five-qubit stabilizer code, whose stabilizer generators are given by

            .. math:: \mathcal{S} = \{S_0=XZZXI, S_1=IXZZX, S_2=XIXZZ, S_3=ZXIXZ\}.

            The five-qubit stabilizer code can detect arbitrary single-qubit X, Y, and Z errors.
            Its syndrome dictionary is computed as follows:

            ::

                {'IIIIX': '1100', 'IIIXI': '0110', 'IIXII': '0011', 'IXIII': '0001', 'XIIII': '1000',
                 'IIIIY': '1110', 'IIIYI': '1111', 'IIYII': '0111', 'IYIII': '1011', 'YIIII': '1101',
                 'IIIIZ': '0010', 'IIIZI': '1001', 'IIZII': '0100', 'IZIII': '1010', 'ZIIII': '0101'}

            where the syndrome is ordered as :math:`s_3s_2s_1s_0` and :math:`s_i\in\{0,1\}` is the measurement
            outcome of the :math:`i`-th stabilizer generator listed in :math:`\mathcal{S}` above.
        """
        self._syndrome_dict = {}
        # For each detectable error (described in Pauli string), compute its syndrome by
        # checking the commutativity relation with the stabilizer generators.
        # Thus, the syndrome of each detectable error is a :math:`n-k` binary string.
        for error in self._detectable_errors:
            syndrome_list = [bsp(row, pauli2bsf(error)) for row in self._standard_form]
            # Reverse the syndrome list so that :math:`s_0` is stored in the least significant bit
            syndrome_list.reverse()
            # Record the syndrome for current error in a binary string
            self._syndrome_dict[error] = "".join(map(str, syndrome_list))

        # Sort the syndrome_dict based on the values if necessary
        # self._syndrome_dict = dict(sorted(self._syndrome_dict.items(), key=lambda x: int(x[1], 2)))

    def encode(self, qp: QProgram, **kwargs) -> QProgram:
        r"""Append the encoding circuit to the quantum program.

        This function encodes the quantum state :math:`\vert\psi\rangle`, specified by the quantum circuit,
        using ``self._n`` number of physical qubits. Intuitively, the encoding procedure goes as follows:

        + First prepare the logical state :math:`\vert 0_L\rangle`, and

        + | Then use the logical X operators, controlled by :math:`\vert\psi\rangle`, to :math:`\vert 0_L\rangle`,
          | to prepare the final logical state :math:`\vert\psi_L\rangle`.

        As suggested in [G97]_, we can safely switch the above two steps for more efficient and elegant quantum circuit.
        Please refer to  Section 4.2 of [G97]_ for more details on the encoding procedure.

        .. note::

            One should be careful with the mapping between the order in stabilizers and the order of qubits,
            which is very important in defining the encoding procedure.

            ::

                stabilizer (str):    X      Z       X

                stabilizer (bsf):   (1      0       1       |   0       1       0)

                qubit order:       q[2]   q[1]     q[0]        q[2]    q[1]    q[0]

        .. admonition:: Qubits Order

            As emphasized in the *Note on Qubits Orders* block of ``BasicCode``'s docstring,
            the stabilizer code uses :math:`2n-k` qubits, and
            we follow the convention that original quantum state is stored
            in the first :math:`k` qubits of the :math:`n` physical qubits:

            ::

                 qubit indices: [q[0] ... q[k-1] |    q[k] ... q[n-1]    | q[n] ... q[2n-k-1]]
              number of qubits:        k         |         n-k           |        k
                     partition: original qubits  | other physical qubits |  ancilla qubits

        :param qp: QProgram, a quantum program to be encoded by the stabilizer code, which has :math:`k` qubits
        :return: QProgram, a new quantum program with encoding circuit, which has :math:`n` qubits
        """
        assert self.k == number_of_qubits(qp), "encode(): The input quantum circuit for appending the encoding " \
                                               "circuit must have {} qubits!".format(self.k)
        # Step 0. Enlarge the original quantum circuit with :math:`n` physical qubits
        enc_qp = enlarge_circuit(qp=copy.deepcopy(qp), extra=self._n-self._k)

        # Step 1. First prepare the logical 0 state. Based on Section 4.2 of [G97]_.
        for i in range(self.r):
            # Map the i-th bit in the BSF to the (n-1-i)-th physical qubit
            idx = self._n - 1 - i
            # first apply Hadamard
            H(enc_qp.Q[idx])
            # If there exists a Z_i factor in the i-th stabilizer, apply S gate
            if self._standard_form[i, self.n + i] == 1:
                S(enc_qp.Q[idx])
            # Apply the i-th stabilizer generator conditioned on qubit idx.
            # Note that the idx-th qubit must be removed from the qubit list in order to achieve conditional Pauli
            pauli = bsf2pauli(self._standard_form[i])
            pauli = pauli[:i] + pauli[i+1:]  # Remove the i-th bit (correspondingly, the idx-th qubit)
            indices = list(reversed(range(self._n)))
            del indices[i]                   # Remove the i-th bit (correspondingly, the idx-th qubit)
            CPauli(reg_c=[enc_qp.Q[idx]], reg_t=[enc_qp.Q[j] for j in indices], val=1, pauli=pauli)

        # Step 2. Conditionally flip the logical state using logical X operators.
        for i, row in enumerate(self.logical_xs()):
            # One should keep in mind that we assume the right-most qubit is q[0],
            # this means that the j-th bit in the BSF maps to the (n-1-j)-th qubit.
            # Here we reverse the range order, so that controlled X on qubits with small indices first.
            for j in reversed(range(self.r, self._n - self._k)):
                if row[j] == 1:
                    CX(enc_qp.Q[i], enc_qp.Q[self._n - 1 - j])

        return enc_qp

    def detect(self, qp: QProgram, **kwargs) -> QProgram:
        r"""Append the error detection circuit to the quantum program.

        In quantum error correction, error detection is done by performing stabilizer measurements
        (also known as the parity measurement) of different stabilizer generators.
        Since there are :math:`n-k` stabilizer generators,
        we need to introduce :math:`n-k` extra ancilla qubits, each controlling a stabilizer measurement.

        We implement the stabilizer measurements in a coherent version, in which
        the measurements are moved from an intermediate stage of the quantum circuit to the end of the circuit;
        this is known as the *Principle of Deferred Measurement*.

        For more details on the error detection procedure, see Section 10.5.8 of [NC10]_.

        .. admonition:: Ancilla Qubits Order

            As emphasized in the *Note on Qubits Order* block of ``BasicCode``'s docstring,
            the stabilizer code uses :math:`2n-k` qubits, and
            we follow the convention that ancilla qubits are stored in the last :math:`n-k` qubits:

            ::

                 qubit indices: [q[0] ... q[k-1] |    q[k] ... q[n-1]    | q[n] ... q[2n-k-1]]
              number of qubits:        k         |         n-k           |        k
                     partition: original qubits  | other physical qubits |  ancilla qubits

        .. admonition:: Explanation

            In the standard terminology of quantum error correction, *correction* is composed of two
            steps: 1. detecting the error, and 2) correct the error.
            However, there are many stabilizer codes (for example the :math:`[[4, 2, 2]]` code)
            can only detect but cannot correct the errors.
            That is, it can tell you that some errors occur but does not know exactly the error type.
            As so, we introduce the ``detect()`` function, which will
            append the error detection circuit (a list of stabilizer measurements) to the quantum program,
            without error correction conditioned on the measurement results.

        :param qp: QProgram, a quantum program to be detected by the stabilizer code, which has :math:`n` qubits
        :return: QProgram, a new quantum program with error detection circuit appended, which has :math:`2n-k` qubits
        """
        assert self.n == number_of_qubits(qp), "detect(): The input quantum circuit for appending error detecting " \
                                               "circuit must have {} qubits!".format(self.n)

        # Step 0. Enlarge the input quantum circuit by appending :math:`n-k` extra ancilla qubits.
        det_qp = enlarge_circuit(qp=copy.deepcopy(qp), extra=self._n-self._k)

        # Measure all stabilizer generators, each controlled by an ancilla qubit.
        for i in range(self.n - self.k):
            anc_qubit = det_qp.Q[self.n + i]
            # Apply H gate
            H(anc_qubit)
            # Apply the conditional stabilizer generator
            # **DO** remember that we adopt the LSB assumption that the rightmost qubit represents qubit 0
            reg_t = [det_qp.Q[j] for j in range(self.n)][::-1]
            pauli = bsf2pauli(self._standard_form[i])
            CPauli(reg_c=[anc_qubit], reg_t=reg_t, val=1, pauli=pauli)
            # Apply H gate again
            H(anc_qubit)

        return det_qp

    def correct(self, qp: QProgram, **kwargs) -> QProgram:
        r"""Append the error correction circuit to the quantum program.

        This function implements the **error correction** step in QEC code without involving error detection。
        It checks the syndrome dictionary to identify the recovery operation for the error syndrome,
        and then append it to the quantum circuit constructed by the ``detect()`` function.

        For more details on the error correction procedure, see Section 10.5.8 of [NC10]_.

        .. note::

                Note that the recovery operations depend on the classical error syndromes,
                which requires that we can incorporate classical computing during the coherence time of the
                qubits to perform quantum operations. However, ``QCompute`` and most quantum hardware platforms
                currently do not support this kind of *dynamic circuits*.
                As so, we implement the correction circuit in a coherent version,
                that is, the recovery operations are fully quantum and the measurement are moved to
                the very end of the quantum circuit.
                This is possible thanks to the *Principle of Deferred Measurement*.

        .. admonition:: Ancilla Qubits Order

            As emphasized in the *Note on Qubits Order* block of ``BasicCode``'s docstring,
            the stabilizer code uses :math:`2n-k` qubits, and
            we follow the convention that ancilla qubits are stored in the last :math:`n-k` qubits:

            ::

                 qubit indices: [q[0] ... q[k-1] |    q[k] ... q[n-1]    | q[n] ... q[2n-k-1]]
              number of qubits:        k         |         n-k           |        k
                     partition: original qubits  | other physical qubits |  ancilla qubits

        :param qp: QProgram, a quantum circuit to be corrected by the stabilizer code, which has :math:`n` qubits
        :return: QProgram, a new quantum circuit with correction circuit appended, which has :math:`2n-k` qubits
        """
        assert self.n * 2 - self.k == number_of_qubits(qp), "correct(): The input quantum circuit for appending "\
                                                            "error correcting circuit must have {} " \
                                                            "qubits!".format(self.n * 2 - self.k)
        #TODO# Check if the stabilizer codes supports error correction
        cor_qp = copy.deepcopy(qp)

        # Apply correction operations based on syndromes.
        # Note that we implement a coherent version of correction, thus there are no measurements on ancilla qubits.
        for error, syndrome in self._syndrome_dict.items():
            # Control qubits are the ancilla qubits.
            # The :math:`i`-th ancilla qubit controls the :math:`i`-th stabilizer,
            # and we assume that the syndrome is ordered as :math:`s_{n-k-1}\cdots s_2s_1s_0`,
            # where :math:`s_i` represents the measurement outcome of the :math:`i`-th stabilizer.
            reg_c = [cor_qp.Q[self.n + i] for i in range(self.n - self.k)][::-1]
            # Target qubits are the working qubits
            reg_t = [cor_qp.Q[j] for j in range(self.n)][::-1]
            CPauli(reg_c=reg_c, reg_t=reg_t, val=int(syndrome, 2), pauli=error)

        return cor_qp

    def detect_and_correct(self, qp: QProgram, **kwargs) -> QProgram:
        r"""Append both detection and correction circuits to the quantum program.

        This function is composed of two steps: **error detection** and **error correction**.
        In the **error detection** step, we call the ``detect()`` function,
        which will give us an error syndrome of :math:`n-k` bits, indicating which error has occurred.
        In the **error correction** step, we call the ``correct()`` function,
        which will check the syndrome dictionary to identify the recovery operations.

        For more details on the detection and correction procedure, see Section 10.5.8 of [NC10]_.

        .. note::
            In many theoretical analysis and physical implementations, **error detection and error correction**
            together is called **error correction**.
            In our implementation, we separate detection from correction, since some error correction
            codes are only capable of detecting quantum errors.

        :param qp: QProgram, a quantum circuit to be corrected by the stabilizer code, which has :math:`n` qubits
        :return: QProgram, a new circuit with detection and correction circuits appended, which has :math:`2n-k` qubits
        """
        assert self.n == number_of_qubits(qp), "detect_and_correct(): The input quantum circuit for appending " \
                                               "error detecting and correcting circuits " \
                                               "must have {} qubits!".format(self.n)

        det_qp = self.detect(qp=qp)
        cor_qp = self.correct(qp=det_qp)

        return cor_qp

    def decode(self, qp: QProgram, **kwargs) -> QProgram:
        r"""Append the decoding circuit to the quantum program.

        In our implementation, we adopt a naive approach where decoding is just the inverse procedure of encoding.
        That is to say, all we need to do is to reverse the entire quantum circuit in the ``encode()`` function.

        The number of qubits of the input quantum circuit must be at least :math:`n`.
        If the number of qubits is larger than :math:`n`, the first :math:`n` qubits will be
        treated as the physical qubits and will be decoded.

        .. note::

            There exist more efficient decoding procedures, by exploring the unique structure
            of the stabilizer code and by taking the target logical state into consideration.
            We refer to Section 4.3 of [G97]_ for more details on other decoding methods.

        #TODO# Implement more efficient decoding methods.

        :param qp: QProgram, a quantum program to be decoded by the stabilizer code,
                    which must have at least :math:`n` qubits
        :return: QProgram, a new quantum program with decoding circuit, which has the same number of
                    qubits as the input quantum circuit
        """
        assert number_of_qubits(qp) >= self.n, "decode(): The input quantum circuit for appending " \
                                               "the decoding circuit must have at least {} qubits!".format(self.n)

        # We do not change the original quantum circuit
        dec_qp = copy.deepcopy(qp)

        # Step 1. Reverse Step 2 in the encoding procedure
        for i, row in reversed(list(enumerate(self.logical_xs()))):
            # One should keep in mind that we assume the right-most qubit is q[0],
            # this means that the j-th bit in the BSF maps to the (n-1-j)-th qubit.
            # Here we do not reverse the range order, so that controlled X on qubits with large indices first.
            # This precisely reverse the encoding procedure.
            for j in range(self.r, self._n - self._k):
                if row[j] == 1:
                    CX(dec_qp.Q[i], dec_qp.Q[self._n - 1 - j])

        # Step 2. Reverse Step 1 in the encoding procedure
        for i in reversed(range(self.r)):
            # Map the i-th bit in the BSF to the (n-1-i)-th physical qubit
            idx = self._n - 1 - i
            # Apply the i-th stabilizer generator conditioned on qubit idx.
            # Note that the idx-th qubit must be removed from the qubit list in order to achieve conditional Pauli
            pauli = bsf2pauli(self._standard_form[i])
            pauli = pauli[:i] + pauli[i+1:]  # Remove the i-th bit (correspondingly, the idx-th qubit)
            indices = list(reversed(range(self._n)))
            del indices[i]                   # Remove the i-th bit (correspondingly, the idx-th qubit)
            CPauli(reg_c=[dec_qp.Q[idx]], reg_t=[dec_qp.Q[j] for j in indices], val=1, pauli=pauli)

            # If there exists a Z_i factor in the i-th stabilizer, apply SDG gate
            if self._standard_form[i, self.n + i] == 1:
                SDG(dec_qp.Q[idx])

            # Apply the Hadamard gate
            H(dec_qp.Q[idx])
        return dec_qp

    def _print_qec_circuit(self, method: Callable, method_name: str, n: int) -> str:
        r"""Print the encoding/correcting/decoding quantum circuit in text string style.

        :param method: Callable, the function interface that implements the encoding/correcting/decoding
        :param method_name: str, name of the method. Options: 'Encoding', 'Detecting',
                                                              'Correcting', 'Detecting-and-Correcting', 'Decoding'
        :param n: int, number of qubits for the quantum circuit that serves as the input to `method`
        :return: str, the encoding/correcting/decoding quantum circuit in simple text string style
        """
        qp = QEnv()
        qp.Q.createList(n)
        # Construct the corresponding circuit string for the circuit
        circuit_str = print_circuit(method(qp).circuit,
                                    show=False,
                                    colors={'red': list(range(self.k)),
                                            'blue': list(range(self.k, self.n)),
                                            'yellow': list(range(self.n, self.n * 2 - self.k))})

        sep_str = "**********************************************"
        type_str = "{} Circuit of the '{}':".format(method_name, self.name)
        info_str = "Qubits Category: [{}][{}][{}]\n".format(COLOR_TABLE['yellow'] + "Ancilla" + COLOR_TABLE['end'],
                                                            COLOR_TABLE['blue'] + "Physical" + COLOR_TABLE['end'],
                                                            COLOR_TABLE['red'] + "Original" + COLOR_TABLE['end'])

        # Concatenate the strings
        circuit_str = [type_str, sep_str, circuit_str, sep_str, info_str]
        return '\n'.join(circuit_str)

    def print_encode_circuit(self, show=True) -> str:
        r"""Print the encoding quantum circuit in text string style.

        **Examples**

            >>> five_qubit_code = FiveQubitCode()
            >>> five_qubit_code.print_encode_circuit()
            >>> # Note that the quantum circuit will be printed in color in the terminal
            Encoding Circuit of the 'Five-Qubit Code':
            **********************************************
            0: -----------Y-------X-------X-----------Y---
                          |       |       |           |
            1: -----------Z-------Z-------I---H---S---●---
                          |       |       |           |
            2: -----------I-------Z---H---●-----------Z---
                          |       |       |           |
            3: -----------Z---H---●-------Z-----------I---
                          |       |       |           |
            4: ---H---S---●-------I-------Z-----------Z---
            **********************************************
            Qubits Category: [Ancilla][Physical][Original]

        :param show: bool, if the encoding circuit will be shown in the terminal in form of texts
        :return: str, the encoding quantum circuit in simple text string style
        """
        circuit_str = self._print_qec_circuit(method=self.encode, method_name='Encoding', n=self.k)
        if show:
            print(circuit_str)

        return circuit_str

    def print_detect_circuit(self, show=True) -> str:
        r"""Print the detecting quantum circuit in text string style.

        **Examples**

            >>> five_qubit_code = FiveQubitCode()
            >>> five_qubit_code.print_detect_circuit()
            >>> # Note that the quantum circuit will be printed in color in the terminal
            Detecting Circuit of the 'Five-Qubit Code':
            **********************************************
            0: -------Y---X-------X-------Y-----------
                      |   |       |       |
            1: -------Z---Z-------I-------Y-----------
                      |   |       |       |
            2: -------I---Z-------X-------Z-----------
                      |   |       |       |
            3: -------Z---X-------Z-------I-----------
                      |   |       |       |
            4: -------Y---I-------Z-------Z-----------
                      |   |       |       |
            5: ---H---●-------H-----------------------
                          |       |       |
            6: ---H-------●-----------H---------------
                                  |       |
            7: ---H---------------●-----------H-------
                                          |
            8: ---H-----------------------●-------H---
            **********************************************
            Qubits Category: [Ancilla][Physical][Original]

        :param show: bool, if the detecting circuit will be shown in the terminal in form of texts
        :return: str, the detecting quantum circuit in simple text string style
        """
        circuit_str = self._print_qec_circuit(method=self.detect, method_name='Detecting', n=self.n)
        if show:
            print(circuit_str)

        return circuit_str

    def print_correct_circuit(self, show=True) -> str:
        r"""Print the correcting quantum circuit in text string style.

        **Examples**

            >>> five_qubit_code = FiveQubitCode()
            >>> five_qubit_code.print_correct_circuit()
            >>> # Note that the quantum circuit will be printed in color in the terminal
            Correcting Circuit of the 'Five-Qubit Code':
            **********************************************
            0: ---X---I---I---I---I---Y---I---I---I---I---Z---I---I---I---I---
                  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
            1: ---I---X---I---I---I---I---Y---I---I---I---I---Z---I---I---I---
                  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
            2: ---I---I---X---I---I---I---I---Y---I---I---I---I---Z---I---I---
                  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
            3: ---I---I---I---X---I---I---I---I---Y---I---I---I---I---Z---I---
                  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
            4: ---I---I---I---I---X---I---I---I---I---Y---I---I---I---I---Z---
                  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
            5: ---●---●---○---●---●---○---●---○---●---○---●---○---○---○---●---
                  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
            6: ---○---●---●---○---○---●---●---●---●---○---●---○---○---●---○---
                  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
            7: ---○---○---○---●---●---●---○---●---●---●---●---○---●---○---○---
                  |   |   |   |   |   |   |   |   |   |   |   |   |   |   |
            8: ---●---●---●---○---●---○---○---●---○---●---●---●---○---○---○---
            **********************************************
            Qubits Category: [Ancilla][Physical][Original]

        :param show: bool, if the correcting circuit will be shown in the terminal in form of texts
        :return: str, the correcting quantum circuit in simple text string style
        """
        circuit_str = self._print_qec_circuit(method=self.correct, method_name='Correcting', n=self.n*2 - self.k)
        if show:
            print(circuit_str)

        return circuit_str

    def print_decode_circuit(self, show=True) -> str:
        r"""Print the decoding quantum circuit in text string style.

        **Examples**

            >>> five_qubit_code = FiveQubitCode()
            >>> five_qubit_code.print_decode_circuit()
            >>> # Note that the quantum circuit will be printed in color in the terminal
            Decoding Circuit of the 'Five-Qubit Code':
            **********************************************
            0: ---Y-------------X-------X-------Y-------------
                  |             |       |       |
            1: ---●---SDG---H---I-------Z-------Z-------------
                  |             |       |       |
            2: ---Z-------------●---H---Z-------I-------------
                  |             |       |       |
            3: ---I-------------Z-------●---H---Z-------------
                  |             |       |       |
            4: ---Z-------------Z-------I-------●---SDG---H---
            **********************************************
            Qubits Category: [Ancilla][Physical][Original]

        :param show: bool, if the decoding quantum circuit will be shown in the terminal in form of texts
        :return: str, the decoding quantum circuit in simple text string style
        """
        circuit_str = self._print_qec_circuit(method=self.decode, method_name='Decoding', n=self.n)
        if show:
            print(circuit_str)

        return circuit_str

    def print_encode_decode_circuit(self, show=True) -> str:
        r"""Print the 'encoding➜decoding' quantum circuit in text string style.

        **Examples**

            >>> five_qubit_code = FiveQubitCode()
            >>> five_qubit_code.print_encode_decode_circuit()
            >>> # Note that the quantum circuit will be printed in color in the terminal
            Encoding➜Decoding Circuit of the 'Five-Qubit Code':
            **********************************************
            0: -----------Y-------X-------X-----------Y---≡---Y-------------X-------X-------Y-------------
                          |       |       |           |   ≡   |             |       |       |
            1: -----------Z-------Z-------I---H---S---●---≡---●---SDG---H---I-------Z-------Z-------------
                          |       |       |           |   ≡   |             |       |       |
            2: -----------I-------Z---H---●-----------Z---≡---Z-------------●---H---Z-------I-------------
                          |       |       |           |   ≡   |             |       |       |
            3: -----------Z---H---●-------Z-----------I---≡---I-------------Z-------●---H---Z-------------
                          |       |       |           |   ≡   |             |       |       |
            4: ---H---S---●-------I-------Z-----------Z---≡---Z-------------Z-------I-------●---SDG---H---
            **********************************************
            Qubits Category: [Ancilla][Physical][Original]

        :param show: bool, if true the 'encoding➜decoding' quantum circuit will be shown in the terminal
        :return: str, the 'encoding➜decoding' quantum circuit in simple text string style
        """
        def _encode_decode(qp: QProgram) -> QProgram:
            enc_qp = self.encode(qp)
            Barrier(*enc_qp.Q.registerMap.values())   # Add barrier for separation
            dec_qp = self.decode(enc_qp)
            return dec_qp

        circuit_str = self._print_qec_circuit(method=_encode_decode,
                                              method_name='Encoding➜Decoding', n=self.k)

        if show:
            print(circuit_str)

        return circuit_str

    def print_detect_correct_circuit(self, show=True) -> str:
        r"""Print the 'detecting➜correcting' quantum circuit in text string style.

        **Examples**

            >>> five_qubit_code = FiveQubitCode()
            >>> five_qubit_code.print_detect_correct_circuit()
            >>> # Note that the quantum circuit will be printed in color in the terminal
            Encoding➜Decoding Circuit of the 'Five-Qubit Code':
            **********************************************
            0: -----------Y-------X-------X-----------Y---≡---Y-------------X-------X-------Y-------------
                          |       |       |           |   ≡   |             |       |       |
            1: -----------Z-------Z-------I---H---S---●---≡---●---SDG---H---I-------Z-------Z-------------
                          |       |       |           |   ≡   |             |       |       |
            2: -----------I-------Z---H---●-----------Z---≡---Z-------------●---H---Z-------I-------------
                          |       |       |           |   ≡   |             |       |       |
            3: -----------Z---H---●-------Z-----------I---≡---I-------------Z-------●---H---Z-------------
                          |       |       |           |   ≡   |             |       |       |
            4: ---H---S---●-------I-------Z-----------Z---≡---Z-------------Z-------I-------●---SDG---H---
            **********************************************
            Qubits Category: [Ancilla][Physical][Original]

        :param show: bool, if true the 'detecting➜correcting' quantum circuit will be shown in the terminal
        :return: str, the 'detecting➜correcting' quantum circuit in simple text string style
        """
        def _detect_correct(qp: QProgram) -> QProgram:
            det_qp = self.detect(qp)
            Barrier(*det_qp.Q.registerMap.values())   # Add barrier for separation
            cor_qp = self.correct(det_qp)
            return cor_qp

        circuit_str = self._print_qec_circuit(method=_detect_correct,
                                              method_name='Detecting➜Correcting',
                                              n=self.n)

        if show:
            print(circuit_str)

        return circuit_str

    def print_encode_detect_correct_decode_circuit(self, show=True) -> str:
        r"""Print the 'encoding➜detecting➜correcting➜decoding' quantum circuit in text string style.

        **Examples**

            >>> five_qubit_code = FiveQubitCode()
            >>> five_qubit_code.print_encode_detect_correct_decode_circuit()
            >>> # Note that the quantum circuit will be printed in color in the terminal
            >>> # The entire quantum circuit is too long to be shown here

        :param show: bool, if true the 'encoding➜detecting➜correcting➜decoding' quantum circuit
                            will be shown in the terminal in form of texts
        :return: str, the 'encoding➜detecting➜correcting➜decoding' quantum circuit in simple text string style
        """
        def _encode_detect_correct_decode(qp: QProgram) -> QProgram:
            enc_qp = self.encode(qp)
            Barrier(*enc_qp.Q.registerMap.values())     # Add barrier for separation
            det_qp = self.detect(enc_qp)
            Barrier(*det_qp.Q.registerMap.values())     # Add barrier for separation
            cor_qp = self.correct(det_qp)
            Barrier(*cor_qp.Q.registerMap.values())     # Add barrier for separation
            dec_qp = self.decode(cor_qp)
            return dec_qp

        circuit_str = self._print_qec_circuit(method=_encode_detect_correct_decode,
                                              method_name='Encoding➜Detecting➜Correcting➜Decoding', n=self.k)

        if show:
            print(circuit_str)

        return circuit_str


class BitFlipCode(BasicCode):
    r"""The three-qubit bit-flip quantum error detection code class.

        The stabilizer generators of the three-qubit bit-flip code are:

        .. math:: \mathcal{S}_{\rm bf} = \{S_0=IZZ, S_1=ZZI\}.

        The bit-flip code can detect and correct arbitrary single-qubit X errors, but it cannot detect Y and Z errors.
    """
    def __init__(self, **kwargs: Any):
        r"""init function of the `BitFlipCode` class.

        .. note::

            We assume the LSB (the least significant bit) mode when defining the stabilizer generators.
            That is, the right-most bit of string represents q[0]:

            ::

                name:           'X      I     Y      Z'

                qubits:         q[3]  q[2]   q[1]   q[0]
        """
        super().__init__(stabilizers=['IZZ', 'ZZI'], error_types=['X'], name='Bit-Flip Code', **kwargs)


class PhaseFlipCode(BasicCode):
    r"""The three-qubit phase-flip quantum error detection code class.

        The stabilizer generators of the three-qubit phase-flip code are:

        .. math:: \mathcal{S}_{\rm pf} = \{S_0=IXX, S_1=XXI\}.

        The phase-flip code can detect and correct arbitrary single-qubit Z errors, but it cannot detect X and Y errors.
    """
    def __init__(self, **kwargs: Any):
        r"""init function of the `PhaseFlipCode` class.

        .. note::

            We assume the LSB (the least significant bit) mode when defining the stabilizer generators.
            That is, the right-most bit of string represents q[0]:

            ::

                name:           'X      I     Y      Z'

                qubits:         q[3]  q[2]   q[1]   q[0]
        """
        super().__init__(stabilizers=['IXX', 'XXI'], error_types=['Z'], name='Phase-Flip Code', **kwargs)


class FourOneTwoCode(BasicCode):
    r"""The :math:`[[4, 1, 2]]` quantum error detection code class.

        The stabilizer generators of the :math:`[[4, 1, 2]]` code are:

        .. math::

            S_0=IZIZ, S_1=XXXX, S_2=ZIZI.

        This code can detect but cannot correct arbitrary single-qubit X, Y, and Z errors.

        See the Supplementary Material of [CYK+22]_ for more details on the :math:`[[4, 1, 2]]` code.
    """
    def __init__(self, **kwargs: Any):
        r"""init function of the `FourOneTwoCode` class.

        .. note::

            We assume the LSB (the least significant bit) mode when defining the stabilizer generators.
            That is, the right-most bit of string represents q[0]:

            ::

                name:           'X      I     Y      Z'

                qubits:         q[3]  q[2]   q[1]   q[0]
        """
        super().__init__(stabilizers=['IZIZ', 'XXXX', 'ZIZI'],
                         error_types=['X', 'Y', 'Z'],
                         name='[[4, 1, 2]] Code', **kwargs)


class FourTwoTwoCode(BasicCode):
    r"""The :math:`[[4, 2, 2]]` quantum error detection code class.

        The stabilizer generators of the :math:`[[4, 2, 2]]` code are:

        .. math::

            S_0=XXXX, S_1=ZZZZ.

        This code can detect but cannot correct arbitrary single-qubit X, Y, and Z errors.

        See Section 4.3 of [R19]_ for more details on the :math:`[[4, 2, 2]]` error detection code.
    """
    def __init__(self, **kwargs: Any):
        r"""init function of the `FourOneTwoCode` class.

        .. note::

            We assume the LSB (the least significant bit) mode when defining the stabilizer generators.
            That is, the right-most bit of string represents q[0]:

            ::

                name:           'X      I     Y      Z'

                qubits:         q[3]  q[2]   q[1]   q[0]
        """
        super().__init__(stabilizers=['XXXX', 'ZZZZ'],
                         error_types=['X', 'Y', 'Z'],
                         name='[[4, 2, 2]] Code', **kwargs)


class FiveQubitCode(BasicCode):
    r"""The five-qubit quantum error correction code class.

        The stabilizer generators of the five-qubit code are:

        .. math:: \mathcal{S}_{\rm 5q} = \{S_0=XZZXI, S_1=IXZZX, S_2=XIXZZ, S_3=ZXIXZ\}.

        The five-qubit code can detect and correct arbitrary single-qubit X, Y, and Z errors.
    """
    def __init__(self, **kwargs: Any):
        r"""init function of the `FiveQubitCode` class.

        .. note::

            We assume the LSB (the least significant bit) mode when defining the stabilizer generators.
            That is, the right-most bit of string represents q[0]:

            ::

                name:           'X      I     Y      Z'

                qubits:         q[3]  q[2]   q[1]   q[0]
        """
        super().__init__(stabilizers=['XZZXI', 'IXZZX', 'XIXZZ', 'ZXIXZ'],
                         error_types=['X', 'Y', 'Z'],
                         name='Five-Qubit Code', **kwargs)


class SteaneCode(BasicCode):
    r"""The seven-qubit Steane quantum error correction code class.

        The stabilizer generators of the Steane code are:

        .. math::

            S_0=XXXXIII, S_1=XXIIXXI, S_2=XIXIXIX,

            S_3=ZZZZIII, S_4=ZZIIZZI, S_5=ZIZIZIZ.

        The Steane code can detect and correct arbitrary single-qubit X, Y, and Z errors.
    """
    def __init__(self, **kwargs: Any):
        r"""init function of the `SteaneCode` class.

        .. note::

            We assume the LSB (the least significant bit) mode when defining the stabilizer generators.
            That is, the right-most bit of string represents q[0]:

            ::

                name:           'X      I     Y      Z'

                qubits:         q[3]  q[2]   q[1]   q[0]
        """
        super().__init__(stabilizers=['XXXXIII', 'XXIIXXI', 'XIXIXIX', 'ZZZZIII', 'ZZIIZZI', 'ZIZIZIZ'],
                         error_types=['X', 'Y', 'Z'],
                         name='Steane Code', **kwargs)


class ShorCode(BasicCode):
    r"""The nine-qubit Shor quantum error correction code class.

        The stabilizer generators of the Shor code are:

        .. math::

            S_0=IIIIIIIZZ, S_1=IIIIIIZZI,

            S_2=IIIIZZIII, S_3=IIIZZIIII,

            S_4=IZZIIIIII, S_5=ZZIIIIIII,

            S_6=IIIXXXXXX, S_7=XXXXXXIII.

        The Shor code can detect and correct arbitrary single-qubit X, Y, and Z errors.
    """
    def __init__(self, **kwargs: Any):
        r"""init function of the `ShorCode` class.

        .. note::

            We assume the LSB (the least significant bit) mode when defining the stabilizer generators.
            That is, the right-most bit of string represents q[0]:

            ::

                name:           'X      I     Y      Z'

                qubits:         q[3]  q[2]   q[1]   q[0]
        """
        super().__init__(stabilizers=['IIIIIIIZZ', 'IIIIIIZZI',
                                      'IIIIZZIII', 'IIIZZIIII',
                                      'IZZIIIIII', 'ZZIIIIIII',
                                      'IIIXXXXXX', 'XXXXXXIII'],
                         error_types=['X', 'Y', 'Z'],
                         name='Shor Code', **kwargs)
