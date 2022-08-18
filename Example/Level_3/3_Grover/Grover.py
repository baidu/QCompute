#!/usr/bin/python3
# -*- coding: utf8 -*-

# Copyright (c) 2020 Baidu, Inc. All Rights Reserved.
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
Grover's Algorithm
This is an instance for Grover's search algorithm,
where the Grover oracle encoding the search target is implemented directly.
More information refers to the tutorial.

Reference
[1] Grover, Lov K. "A fast quantum mechanical algorithm for database search." Proceedings of the 28th Annual ACM
    Symposium on Theory of Computing (https://dl.acm.org/doi/10.1145/237814.237866). 1996.
[2] 百度量子计算研究所, "格罗弗算法." 量易简 (https://qulearn.baidu.com/textbook/chapter3/格罗弗算法.html), 2022.
[3] Barenco, Adriano, et al. "Elementary gates for quantum computation." Physical review A
    (https://journals.aps.org/pra/abstract/10.1103/PhysRevA.52.3457) 52.5 (1995): 3457.
[4] Gidney, Craig. "Constructing Large Controlled Nots."
    (https://algassert.com/circuits/2015/06/05/Constructing-Large-Controlled-Nots.html), Web. 05 Jun. 2015.
"""

import sys
import numpy
from random import randint
sys.path.append('../..')  # "from QCompute import *" requires this
from QCompute import *

# matchSdkVersion('Python 2.0.1')


def func_find_borrowable_qubits(reg_work, num_qubits_borrowed=1):
    """
    Find several qubits not used (so they are borrowable) and form a new register.
    :param reg_work: a quantum register which should be disjoint with borrowed qubits
    :param num_qubits_borrowed: the number of borrowable qubits we need
    :return: a quantum register containing n_q_b borrowable qubits
    """
    reg_borrowed = []
    assert len(reg_work) > 0  # reg_work must be none-empty
    env_quantum = reg_work[0].env
    idx_qubit = 0
    while len(reg_borrowed) < num_qubits_borrowed:
        if env_quantum.Q[idx_qubit] not in reg_work:
            reg_borrowed += [env_quantum.Q[idx_qubit]]
        idx_qubit += 1
    return reg_borrowed


def circ_multictrl_X(qubit_target, reg_ctrlling, reg_borrowed=None):
    """
    An implement for multi-control X gate (abbreviated as CnX) with several borrowed qubits, whose cost will be O(n).
    The decomposition algorithm refers to [3] and [4].
    Here CnX(|c>|t>) == |c> X|t>, if |c> == |11...1>;
                        |c>|t>,   else.
    :param qubit_target: |t>, the target qubit of the CnX gate
    :param reg_ctrlling: |c>, a quantum register containing several qubits as the controlling qubits for the CnX gate.
                              It is noted that r_c may be an empty register, i.e. r_c == [].
    :param reg_borrowed: |b>, a quantum register containing several qubits as the borrowed qubits for the CnX gate.
                              It is noted that r_b may also be an empty register, i.e. r_b == [].
    :return: Set n = the number of qubits in |c>.
             If n != 0, the state |b>|c>|t> will change into |b> CnX(|c>|t>) after this quantum circuit;
             Else if n == 0, the state |b>|t> will change into |b> X|t> after this quantum circuit.
             It is noted that the state |b> will always stay unchanged, so called borrowed qubits.
    """
    if reg_borrowed is None:
        reg_borrowed = []
    num_qubit_ctrlling = len(reg_ctrlling)  # count the number n in reg_ctrlling
    if num_qubit_ctrlling == 0:  # for the case n == 0, it just |t> -> X|t>
        X(qubit_target)
    elif num_qubit_ctrlling == 1:  # for the case n == 1, it just |c>|t> -> CX|c>|t>
        CX(reg_ctrlling[0], qubit_target)
    elif num_qubit_ctrlling == 2:  # for the case n == 2, it just |c0>|c1>|t> -> CCX|c0>|c1>|t>
        CCX(reg_ctrlling[0], reg_ctrlling[1], qubit_target)
    else:  # for the case n >= 3
        if len(reg_borrowed) == 0:  # if there is no borrowable qubit, we need to find one
            # here the reg_work in following function is all qubits used, i.e. [qubit_target] + reg_ctrlling
            reg_borrowed = func_find_borrowable_qubits([qubit_target] + reg_ctrlling)
        if len(reg_borrowed) < num_qubit_ctrlling - 2:  # if there are several but not enough borrowed qubits,
            # here is a decomposition: C^n(X) -> several C^(n/2)(X) gates
            qubit_borrowed = reg_borrowed[0]
            num_qubit_ctrlling_half = (num_qubit_ctrlling + 1) // 2
            reg_ctrlling_half_1 = reg_ctrlling[:num_qubit_ctrlling_half]
            reg_ctrlling_half_2 = reg_ctrlling[num_qubit_ctrlling_half:] + [qubit_borrowed]
            # the following is exactly the decomposition
            circ_multictrl_X(qubit_borrowed, reg_ctrlling_half_1, reg_borrowed=reg_ctrlling_half_2)
            circ_multictrl_X(qubit_target, reg_ctrlling_half_2, reg_borrowed=reg_ctrlling_half_1)
            circ_multictrl_X(qubit_borrowed, reg_ctrlling_half_1, reg_borrowed=reg_ctrlling_half_2)
            circ_multictrl_X(qubit_target, reg_ctrlling_half_2, reg_borrowed=reg_ctrlling_half_1)
        else:  # if there are enough borrowed qubits, here is another decomposition: C^n(X) -> several CCX gates
            # this decomposition repeats the following circuit twice
            for iterator in range(2):
                CCX(reg_ctrlling[0], reg_borrowed[0], qubit_target)
                for idx in range(1, num_qubit_ctrlling - 2):
                    CCX(reg_ctrlling[idx], reg_borrowed[idx], reg_borrowed[idx - 1])
                CCX(reg_ctrlling[-2], reg_ctrlling[-1], reg_borrowed[num_qubit_ctrlling - 3])
                for idx in range(num_qubit_ctrlling - 3, 0, -1):
                    CCX(reg_ctrlling[idx], reg_borrowed[idx], reg_borrowed[idx - 1])


def circ_multictrl_Z(qubit_target, reg_ctrlling, reg_borrowed=None):
    """
    An implement for multi-control Z gate (abbreviated as CnZ) with several borrowed qubits, whose cost will be O(n).
    Here CnZ(|c>|t>) == |c>Z|t>, if |c> == |11...1>;
                        |c>|t>,  else.
    :param qubit_target: |t>, the target qubit of the CnZ gate
    :param reg_ctrlling: |c>, a quantum register containing several qubits as the controlling qubits for the CnZ gate.
                              It is noted that r_c may be an empty register, i.e. r_c == [].
    :param reg_borrowed: |b>, a quantum register containing several qubits as the borrowed qubits for the CnZ gate.
                              It is noted that r_b may also be an empty register, i.e. r_b == [].
    :return: Set n = the number of qubits in |c>.
             If n != 0, the state |b>|c>|t> will change into |b> CnZ(|c>|t>);
             Else if n == 0, the state |b>|t> will change into |b> X|t>
             It is noted that the state |b> will stay unchanged.
    """
    num_qubit_ctrlling = len(reg_ctrlling)  # count the number n in reg_ctrlling
    if num_qubit_ctrlling == 0:  # for the case n == 0, it just |t> -> Z|t>
        Z(qubit_target)
    elif num_qubit_ctrlling == 1:  # for the case n == 1, it just |c>|t> -> CZ|c>|t>
        CZ(reg_ctrlling[0], qubit_target)
    else:  # for the case n >= 2, we implement CnZ by CnX and H gates
        H(qubit_target)
        circ_multictrl_X(qubit_target, reg_ctrlling, reg_borrowed=reg_borrowed)
        H(qubit_target)


def circ_Grover_oracle(reg_sys, int_target):
    """
    This function give a circuit to implement the Grover oracle in Grover's algorithm.
    Generally, the search target should be unknown, and encoded by Grover oracle.
    However, in this implement we suppose the search target is known, such that we can implement an oracle to encode it.
    :param reg_sys: |s>, the system register to operate the Grover oracle
    :param int_target: t, the search target we want.
    :return: GO == I - 2|t><t|,
             GO |s> == -|s>, if s == t;
                       |s>,  if <t|s> == 0.
    """
    num_qubit = len(reg_sys)
    # Since CnZ == I - 2|11...1><11...1|, we can flip CnZ into GO by two layers of X gates.
    # Meanwhile, those X gates encode the search target s.
    # the first layer of X gates encoding the search target s
    for int_k in range(num_qubit):
        if (int_target >> int_k) % 2 == 0:
            X(reg_sys[-1 - int_k])

    Barrier(*reg_sys)
    # the multictrl gate CnZ
    circ_multictrl_Z(reg_sys[-1], reg_sys[:-1])

    Barrier(*reg_sys)
    # the second layer of X gates encoding the search target s
    for int_k in range(num_qubit):
        if (int_target >> int_k) % 2 == 0:
            X(reg_sys[-1 - int_k])


def circ_diffusion_operator(reg_sys):
    """
    This function give a circuit to implement the diffusion operator in Grover's algorithm.
    The diffusion operator flip the phase along the state |++...+>,
    which could be implemented by CnZ and two layers of H gates and two layers of X gates.
    :param reg_sys: |s>, the system register to operate the diffusion operator
    :return: DO == I - 2|++...+><++...+|,
             DO |s> == -|s>, if |s> == |++...+>;
                       |s>,  if <++...+|s> == 0.
    """
    # the first layer of H gates and the first layer of X gates
    for qbit in reg_sys:
        H(qbit)
        X(qbit)

    Barrier(*reg_sys)
    # the multictrl gate CnZ
    circ_multictrl_Z(reg_sys[-1], reg_sys[:-1])

    Barrier(*reg_sys)
    # the second layer of X gates and the second layer of H gates
    for qbit in reg_sys:
        X(qbit)
        H(qbit)


def Grover(num_qubit, int_target=None):
    """
    :param num_qubit: n, the number of qubits which will encode the database to search
    :param int_target: t, the index of the search target, defaulted to be generated randomly
    """
    # create environment
    env = QEnv()
    # choose backend Baidu Local Quantum Simulator-Sim2
    env.backend(BackendName.LocalBaiduSim2)
    # create the quantum register encoding the database
    reg_sys = env.Q.createList(num_qubit)
    # generate the search target randomly if unspecified
    if int_target is None:
        int_target = randint(0, 2 ** num_qubit - 1)
    else:
        assert int_target < 2 ** num_qubit
    # prepare the initial state in Grover's algorithm
    for idx_qubit in reg_sys:
        H(idx_qubit)
    # Alternate the Grover oracle and diffusion operator for certain times,
    # which only depends on the size of the database.
    for iterator in range(round(numpy.pi / (4 * numpy.arcsin(2 ** (-num_qubit / 2))) - 1 / 2)):
        Barrier(*reg_sys)
        Barrier(*reg_sys)
        # Call the Grover oracle
        circ_Grover_oracle(reg_sys, int_target)

        Barrier(*reg_sys)
        Barrier(*reg_sys)
        # Call the diffusion operator
        circ_diffusion_operator(reg_sys)

    # Finally, we measure reg_sys to verify Grover's algorithm works correctly.
    # Here the result of measurement is shown in positive sequence.
    MeasureZ(reg_sys, range(num_qubit - 1, -1, -1))
    # Commit the quest to the cloud
    env.commit(16000, downloadResult=False)


if __name__ == '__main__':
    # searching the state |3> on a 4-qubit circuit
    Grover(6, int_target=3)
