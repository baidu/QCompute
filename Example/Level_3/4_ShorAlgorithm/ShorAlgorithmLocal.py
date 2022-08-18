#!/usr/bin/python3
# -*- coding: utf8 -*-

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
This code is an implement for the paper
Circuit for Shor’s algorithm using 2n+3 qubits
by St´ephane Beauregard
https://arxiv.org/pdf/quant-ph/0205095.pdf

In this code ‘QFT’ or means the Quantum Fourier Transform without the swap step.

The quantum backend choosing and token information refers to the function func_quantum_order_finding
"""

import math
import sys
import numpy as np
from numpy import pi
from sympy.ntheory import isprime
from sympy.ntheory.residue_ntheory import n_order
from random import randint

sys.path.append('../..')  # "from QCompute import *" requires this
from QCompute import *

matchSdkVersion('Python 3.0.0')


def CU1(q1, q2, float_theta):
    """
    a single-parameter two-qubit gate, which is the ctrl version for U1 gate,
    the matrix form of CU1(theta) is the diagonal matrix {1,1,1,e^{i theta}}
    in fact we do not distinguish which qubit is the ctrlling or ctrlled qubit
    :param q1: a qubit
    :param q2: another qubit
    :param float_theta: the rotation angle
    :return: |q1>|q2> -> <0|q1>*|q1>|q2> + <1|q1>*|q1>U1(theta)|q2>
    """
    CU(0, 0, float_theta)(q1, q2)


def CCU1(q1, q2, q3, float_theta):
    """
    a single-parameter three-qubit gate, which is the double-ctrl version for U1 gate,
    the matrix form of CCU1(theta) is the diagonal matrix {1,1,1,1,1,1,1,e^{i theta}}
    in fact we do not distinguish which qubit is the ctrlling or ctrlled qubit
    :param q1: a qubit
    :param q2: another qubit
    :param q3: a third qubit
    :param float_theta: the rotation angle
    :return: |q1>|q2>|q3> -> <0|q1*q2*q3>*|q1>|q2>|q3> + e^{i theta}*<1|q1*q2*q3>*|q1>|q2>|q3>
    """
    float_theta_half = float_theta / 2
    CU(0, 0, float_theta_half)(q2, q3)
    CX(q1, q2)
    CU(0, 0, -float_theta_half)(q2, q3)
    CX(q1, q2)
    CU(0, 0, float_theta_half)(q1, q3)


def func_qftadd(reg_system, int_adder):
    """
    a circuit implement the addition under the Fourier bases
    :param reg_system: QFT|s>, we write the state as a image of the Fourier transform
    :param int_adder: a
    :return: a circuit which implement the map: QFT|s> -> QFT|s+a>
    """
    for idx_qubit in range(len(reg_system)):  # For each qubit in the reg_s, we operate a U1 gate on it
        U(2 * pi * (int_adder % 2 ** (idx_qubit + 1)) / (2 ** (idx_qubit + 1)))(
            reg_system[-1 - idx_qubit])


def func_qftadd_inverse(reg_system, int_adder):
    """
    the inverse circuit for func_qftadd
    :param reg_system: QFT|s+a>
    :param int_adder: a
    :return: a circuit which implement the map: QFT|s+a> -> QFT|s>
    """
    for idx_qubit in range(len(reg_system)):  # For each qubit in the reg_s, we operate a U1 gate on it
        U(-2 * pi * (int_adder % 2 ** (idx_qubit + 1)) / (2 ** (idx_qubit + 1)))(
            reg_system[-1 - idx_qubit])


def func_ctrl_qftadd(reg_system, qubit_ctrlling, int_adder):
    """
    the ctrl version for func_qftadd
    :param qubit_ctrlling: the ctrlling qubit |c>
    :param reg_system: QFT|s>
    :param int_adder: a
    :return: |c>QFT|s> -> <0|c>*|c>QFT|s> + <1|c>*|c>QFT|s+a>
    """
    for idx_qubit in range(len(reg_system)):  # For each qubit in the reg_s, we operate a CU1 gate on |c> and it
        CU1(qubit_ctrlling, reg_system[-1 - idx_qubit],
            2 * pi * np.mod(int_adder, 2 ** (idx_qubit + 1)) / (2 ** (idx_qubit + 1)))


def func_ctrl_qftadd_inverse(reg_system, qubit_ctrlling, int_adder):
    """
    the inverse circuit for func_ctrl_qftadd; also the ctrl version for func_qftadd_inverse
    :param qubit_ctrlling: |c>
    :param reg_system: QFT|s+a>
    :param int_adder: a
    :return: |c>QFT|s+a> -> <0|c>*|c>QFT|s+a> + <1|c>*|c>QFT|s>
    """
    for idx_qubit in range(len(reg_system)):  # For each qubit in the reg_s, we operate a CU1 gate on |c> and it
        CU1(qubit_ctrlling, reg_system[-1 - idx_qubit],
            -2 * pi * np.mod(int_adder, 2 ** (idx_qubit + 1)) / (2 ** (idx_qubit + 1)))


def func_double_ctrl_qftadd(reg_system, reg_ctrlling, int_adder):
    """
    the double-ctrl version for func_qftadd
    :param reg_ctrlling: a list of two ctrlling qubit [|c_1>,|c_2>]
    :param reg_system: QFT|s>
    :param int_adder: a
    :return: |c_1>|c_2>QFT|s> -> <0|c_1*c_2>*|c_1>|c_2>QFT|s> + <1|c_1*c_2>*|c_1>|c_2>QFT|s+a>
    """
    for idx_qubit in range(len(reg_system)):  # For each qubit in the reg_s, we operate a CCU1 gate on |c_1>|c_2> and it
        CCU1(reg_ctrlling[0], reg_ctrlling[1], reg_system[-1 - idx_qubit],
             2 * pi * np.mod(int_adder, 2 ** (idx_qubit + 1)) / (2 ** (idx_qubit + 1)))


def func_double_ctrl_qftadd_inverse(reg_system, reg_ctrlling, int_adder):
    """
    the inverse version for func_double_qftadd; also the double-ctrl version for func_qftadd_inverse
    :param reg_ctrlling: a list of two qubit [|c_1>,|c_2>]
    :param reg_system: QFT|s+a>
    :param int_adder: a
    :return: |c_1>|c_2>QFT|s+a> -> <0|c_1*c_2>*|c_1>|c_2>QFT|s+a> + <1|c_1*c_2>*|c_1>|c_2>QFT|s>
    """
    for idx_qubit in range(len(reg_system)):  # For each qubit in the reg_s, we operate a CCU1 gate on |c_1>|c_2> and it
        CCU1(reg_ctrlling[0], reg_ctrlling[1], reg_system[-1 - idx_qubit],
             -2 * pi * np.mod(int_adder, 2 ** (idx_qubit + 1)) / (2 ** (idx_qubit + 1)))


def func_swap_in_qft(reg_system):
    """
    Reverse the reg |q_0>|q_1>...|q_n> -> |q_n>...|q_1>|q_0> by using SWAP gates
    :param reg_system: |q_0>|q_1>...|q_n> a quantum register
    :return: |q_0>|q_1>...|q_n> -> |q_n>...|q_1>|q_0>
    """
    number_qubit = len(reg_system)
    for idx_qubit in range(0, number_qubit // 2):  # SWAP the i-th qubit and the (n-i-1)-th qubit
        SWAP(reg_system[idx_qubit], reg_system[number_qubit - idx_qubit - 1])


def func_qft_without_swap(reg_system):
    """
    Quantum Fourier Transform without the swap step, |s> -> QFT|s>
    :param reg_system: |s>
    :return:
    """
    number_qubit = len(reg_system)
    for idx1_qubit in range(0, number_qubit - 1):  # The outer loop
        H(reg_system[idx1_qubit])  # Operate a H gate on the idx1-th qubit
        for idx2_qubit in range(2, number_qubit - idx1_qubit + 1):  # The inner loop
            # where we will operate a CU1 gate in each loop
            idx3_qubit = idx1_qubit + idx2_qubit - 1  # idx3 is the idx for the ctrlling qubit
            # idx1 is the ctrlled qubit and idx2 is related to the rotation angle
            CU(0, 0, 2 * pi / pow(2, idx2_qubit))(reg_system[idx3_qubit], reg_system[idx1_qubit])
    H(reg_system[number_qubit - 1])  # Do not forget there is a H gate operating on the last qubit


def func_qft_without_swap_inverse(reg_system):
    """
    the inverse of Quantum Fourier Transform without the swap step, QFT|s> -> |s>
    :param reg_system: |s>
    :return:
    """
    number_qubit = len(reg_system)
    for idx1_qubit in range(0, number_qubit - 1):  # The outer loop
        H(reg_system[number_qubit - idx1_qubit - 1])
        for idx2_qubit in range(0, idx1_qubit + 1):  # The inner loop where we will operate a CU1 gate in each loop
            CU(0, 0, -pi / 2 ** (idx1_qubit - idx2_qubit + 1))(reg_system[number_qubit - idx2_qubit - 1], reg_system[
                number_qubit - idx1_qubit - 2])
    H(reg_system[0])


def func_double_ctrl_qftaddmod(reg_system, reg_ctrlling, qubit_zeroed, int_adder, int_divisor):
    """
    CC-qftadd(int_adder)mod(int_divisor)
    |c_1>|c_2>QFT|s> -> <0|c_1*c_2>*|c_1>|c_2>QFT|s> + <1|c_1*c_2>*|c_1>|c_2>QFT|s+a mod d>
    the complement comes from the Figure 5 in arXiv quant-ph/0205095
    :param reg_system: QFT|s> with s < d
    :param reg_ctrlling: [|c_1>,|c_2>]
    :param qubit_zeroed: |0>
    :param int_adder: a with a < d
    :param int_divisor: d
    :return: |c_1>|c_2>QFT|s>|0> -> <0|c_1*c_2>*|c_1>|c_2>QFT|s>|0> + <1|c_1*c_2>*|c_1>|c_2>QFT|s+a mod d>|0>
    """
    func_double_ctrl_qftadd(reg_system, reg_ctrlling, int_adder)
    func_qftadd_inverse(reg_system, int_divisor)
    func_qft_without_swap_inverse(reg_system)
    CX(reg_system[0], qubit_zeroed)
    func_qft_without_swap(reg_system)
    func_ctrl_qftadd(reg_system, qubit_zeroed, int_divisor)
    func_double_ctrl_qftadd_inverse(reg_system, reg_ctrlling, int_adder)
    func_qft_without_swap_inverse(reg_system)
    X(reg_system[0])
    CX(reg_system[0], qubit_zeroed)
    X(reg_system[0])
    func_qft_without_swap(reg_system)
    func_double_ctrl_qftadd(reg_system, reg_ctrlling, int_adder)


def func_double_ctrl_qftaddmod_inverse(reg_system, reg_ctrlling, qubit_zeroed, int_adder, int_divisor):
    """
    the inverse of CC-qftadd(int_adder)mod(int_divisor)
    :param reg_system: QFT|s> with s < d
    :param reg_ctrlling: [|c_1>,|c_2>]
    :param qubit_zeroed: |0>
    :param int_adder: a with a < d
    :param int_divisor: d
    :return: |c_1>|c_2>QFT|s>|0> -> <0|c_1*c_2>*|c_1>|c_2>QFT|s>|0> + <1|c_1*c_2>*|c_1>|c_2>QFT|s-a mod d>|0>
    """
    func_double_ctrl_qftadd_inverse(reg_system, reg_ctrlling, int_adder)
    func_qft_without_swap_inverse(reg_system)
    X(reg_system[0])
    CX(reg_system[0], qubit_zeroed)
    X(reg_system[0])
    func_qft_without_swap(reg_system)
    func_double_ctrl_qftadd(reg_system, reg_ctrlling, int_adder)
    func_ctrl_qftadd_inverse(reg_system, qubit_zeroed, int_divisor)
    func_qft_without_swap_inverse(reg_system)
    CX(reg_system[0], qubit_zeroed)
    func_qft_without_swap(reg_system)
    func_qftadd(reg_system, int_divisor)
    func_double_ctrl_qftadd_inverse(reg_system, reg_ctrlling, int_adder)


def func_ctrl_addprodmod(reg_system, reg_factor_1, qubit_ctrlling, qubit_zeroed, int_factor_2, int_divisor):
    """
    :param reg_system: |s> with s < d
    :param reg_factor_1: |f_1>, a quantum state encoding the factor_1
    :param qubit_ctrlling: |c>
    :param qubit_zeroed: |0>
    :param int_factor_2: f_2, a classical data
    :param int_divisor: d
    :return: |c>|f_1>|s> -> <0|c>|c>|f_1>|s> + <1|c>|c>|f_1>|s+ f_1*f_2 mod d>
    the complement comes from the Figure 6 in arXiv quant-ph/0205095
    """
    func_qft_without_swap(reg_system)
    for idx_qubit in range(len(reg_factor_1)):  # For each qubit in reg_f_1, we will operate a CC-qftaddmod gate where
        # regarding idx_qubit as one of the two ctrlling qubit
        func_double_ctrl_qftaddmod(reg_system, [qubit_ctrlling, reg_factor_1[-1 - idx_qubit]], qubit_zeroed,
                                   (int_factor_2 * (2 ** idx_qubit)) % int_divisor, int_divisor)
    func_qft_without_swap_inverse(reg_system)


def func_ctrl_addprodmod_inverse(reg_system, reg_factor_1, qubit_ctrlling, qubit_zeroed, int_factor_2, int_divisor):
    """
    the inverse for func_ctrl_addprodmod
    :param reg_system:
    :param reg_factor_1:
    :param qubit_ctrlling:
    :param qubit_zeroed:
    :param int_factor_2:
    :param int_divisor:
    :return:
    """
    func_qft_without_swap(reg_system)
    for idx_qubit in range(len(reg_factor_1)):  # For each qubit in reg_f_1, we will operate a CC-qftaddmod gate where
        # regarding idx_qubit as one of the two ctrlling qubit
        func_double_ctrl_qftaddmod_inverse(reg_system, [qubit_ctrlling, reg_factor_1[-1 - idx_qubit]], qubit_zeroed,
                                           (int_factor_2 * (2 ** idx_qubit)) % int_divisor, int_divisor)
    func_qft_without_swap_inverse(reg_system)


def func_ctrl_multmod(reg_system, reg_zeroed, qubit_ctrlling, qubit_zeroed, int_factor, int_divisor):
    """
    |c>|s> -> <0|c>|c>|s>  + <1|c>|c>|s * f mod d>
    the complement comes from the Figure 7 in arXiv quant-ph/0205095
    :param reg_system: |s>
    :param reg_zeroed: |0*>, a register initialled into |0*>
    :param qubit_ctrlling: |c>
    :param qubit_zeroed: |0>, a qubit at state |0>
    :param int_factor: f
    :param int_divisor: d
    :return: |c>|s>|0*>|0> -> <0|c>|c>|s>|0*>|0>  + <1|c>|c>|s * f mod d>|0*>|0>
    """
    func_ctrl_addprodmod(reg_zeroed, reg_system, qubit_ctrlling, qubit_zeroed, int_factor, int_divisor)
    for idx_qubit in range(min(len(reg_system), len(reg_zeroed))):  # We CSWAP the corresponding qubit in those two reg
        # from the end since maybe the two reg has different length
        CSWAP(qubit_ctrlling, reg_system[-1 - idx_qubit], reg_zeroed[-1 - idx_qubit])
    func_ctrl_addprodmod_inverse(reg_zeroed, reg_system, qubit_ctrlling, qubit_zeroed, pow(int_factor, -1, int_divisor),
                                 int_divisor)


def func_from_fraction_continued(list_int_generator):
    """
    a int list such as [0,1,2,3] -> the denominator 7 of 0+1/(1+1/(2+1/3)))=10/7
    :param list_int_generator: a int list such as [0,1,2,3]
    :return: the denominator of the fraction generated by the list
    even though the func returns the variable int_numerator
    """
    int_denominator = 1
    int_numerator = 0
    while len(list_int_generator) != 0:  # We use a stack data structure to compute the denominator of continued
        # fraction, and the fraction is stored as two int: the denominator and the numerator
        int_denominator_new = list_int_generator.pop() * int_denominator + int_numerator
        int_numerator = int_denominator
        int_denominator = int_denominator_new
        # The above swapping step implicate computing the inverse of a fraction
    return int_numerator  # We return the denominator even though the func returns a variable called int_numerator


def func_result_to_order(int_numerator, number_qubit, int_max_denominator):
    """
    :param int_numerator: N, may come from a quantum state
    :param number_qubit: n, number of qubit, corresponding to a denominator 2^n
    :param int_max_denominator: d, we want to find the order which < d
    :return: the order as a denominator of some truncate of the continued fraction N/(2^n) which is maximal and < d
    """
    list_int_generator_fraction = []
    int_denominator = 2 ** number_qubit
    while int_denominator != 0:  # By Euclidean algorithm we compute the representation in continued fraction
        list_int_generator_fraction.append(int_numerator // int_denominator)
        int_remainder = int_numerator % int_denominator
        int_numerator = int_denominator
        int_denominator = int_remainder
    # Thus the coefficients form list_int_gen._frac.
    for idx_int in range(len(list_int_generator_fraction)):  # From front to back, from short to long,
        # we truncate the continued fraction
        if func_from_fraction_continued(list_int_generator_fraction[:idx_int + 1]) >= int_max_denominator:
            return func_from_fraction_continued(list_int_generator_fraction[:idx_int])
    return func_from_fraction_continued(list_int_generator_fraction)


def func_result_to_order_list(int_numerator, number_qubit, int_max_denominator):
    """
    For the case that the number of ancilla qubits is not enough, we need to traverse all probable denominator in the
    step of continued fraction expansions. Thus this func returns all possible orders.
    :param int_numerator: N, may come from a quantum state
    :param number_qubit: n, number of qubit, corresponding to a denominator 2^n
    :param int_max_denominator: d, we want to find the order which < d
    :return: the list of orders as a denominator of some truncate of the continued fraction N/(2^n) which is < d
    """
    list_int_generator_fraction = []
    int_denominator = 2 ** number_qubit
    while int_denominator != 0:  # By Euclidean algorithm we compute the representation in continued fraction
        list_int_generator_fraction.append(int_numerator // int_denominator)
        int_remainder = int_numerator % int_denominator
        int_numerator = int_denominator
        int_denominator = int_remainder
    # Thus the coefficients form list_int_gen._frac.
    list_int_order = []
    for idx_int in range(len(list_int_generator_fraction)):  # From front to back, from short to long,
        # we truncate the continued fraction
        int_order_maybe = func_from_fraction_continued(list_int_generator_fraction[:idx_int + 1])
        if int_order_maybe < int_max_denominator:
            list_int_order += [int_order_maybe]
        else:
            break
    return list_int_order


def func_qof_data_processing(int_divisor, number_qubit_ancilla, dict_task_result_counts):
    """
    :param int_divisor: d
    :param number_qubit_ancilla: the number of qubits for the estimating the phase
    :param dict_task_result_counts: a dict storing the counts data in the task result
    :return: a dict {"order":shots} storing the order and the shots, such as {"2":5,"4":7} means that 5 shots indicate
             the order may be 2 and 7 shots indicate the order may be 4.
    dict_task_result_counts of form {"quantum_output":shots} is a quantum output from an estimation for the fraction
    r/ord(f,d)/2^t, where ord(f,d) is the order of f mod d, r is a random in {0,1,...,ord(f,d)-1}, and
    t = number_qubit_ancilla as following corresponding to the precision.
    For the case that the ancilla is enough we compute the maximal denominator, and for the case that the ancilla is not
    enough we compute all the possible denominators.
    """
    dict_order = {}
    # The case that the number of ancilla is enough
    if number_qubit_ancilla >= math.log(int_divisor, 2) * 2:
        for idx_key in dict_task_result_counts.keys():  # We need to transform the key in dict_task
            if dict_task_result_counts[idx_key] <= 0:  # Skip the measurement results with counts <= 0
                continue
            # From a numerator to the order by calling func_result_to_order
            int_order_maybe = func_result_to_order(int(idx_key[::-1], 2), number_qubit_ancilla, int_divisor)
            str_int_order_maybe = "{0}".format(int_order_maybe)
            if str_int_order_maybe not in dict_order.keys():
                dict_order[str_int_order_maybe] = dict_task_result_counts[idx_key]
            else:
                dict_order[str_int_order_maybe] += dict_task_result_counts[idx_key]
    else:  # The case that the number of ancilla is not enough
        for idx_key in dict_task_result_counts.keys():  # We need to transform the key in dict_task
            # from a numerator to the order by calling func_result_to_order
            if dict_task_result_counts[idx_key] <= 0:  # Skip the measurement results with counts <= 0
                continue
            list_int_order_maybe = func_result_to_order_list(int(idx_key[::-1], 2), number_qubit_ancilla, int_divisor)
            for int_order_maybe in list_int_order_maybe:
                str_int_order_maybe = "{0}".format(int_order_maybe)
                if str_int_order_maybe not in dict_order.keys():
                    dict_order[str_int_order_maybe] = dict_task_result_counts[idx_key]
                else:
                    dict_order[str_int_order_maybe] += dict_task_result_counts[idx_key]
    return dict_order


def func_quantum_order_finding(int_factor, int_divisor, int_shots, number_qubit_ancilla):
    """
    :param int_factor: f
    :param int_divisor: d
    :param int_shots: the shots number for each quantum circuit
    :param number_qubit_ancilla: the number of qubits for the estimating the phase
    :return: an estimation for the fraction r/ord(f,d)/2^t, where ord(f,d) is the order of f mod d, r is a random in
             {0,1,...,ord(f,d)-1}, and t = number_qubit_ancilla as following corresponding to the precision.
    """
    # Create the quantum environment
    env = QEnv()
    Define.hubToken = ''
    # Choose backend
    env.backend(BackendName.LocalBaiduSim2)
    # env.backend(BackendName.CloudBaiduSim2Water)

    # Decide the number of qubit which will be used to encode the eigenstate
    number_qubit_system = int(math.ceil(math.log(int_divisor, 2)))

    # Create the quantum register
    # The ancilla qubit used for phase estimation
    reg_ancilla = [env.Q[idx_qubit] for idx_qubit in range(0, number_qubit_ancilla)]
    number_qubit_part2 = number_qubit_ancilla + number_qubit_system
    # The system register holding the eigenstate
    reg_system = [env.Q[idx_qubit] for idx_qubit in range(number_qubit_ancilla, number_qubit_part2)]
    number_qubit_part3 = number_qubit_ancilla + 2 * number_qubit_system + 1
    # The zeroed register used in the circuit of func_ctrl_multmod
    reg_zeroed = [env.Q[idx_qubit] for idx_qubit in range(number_qubit_part2, number_qubit_part3)]
    qubit_zeroed = env.Q[number_qubit_part3]  # The other zeroed qubit used in the circuit of func_ctrl_multmod

    # Initialise the state |0...01> as a superposition of concerned eigenstates
    X(reg_system[-1])

    # The following is the quantum phase estimation algorithm
    for idx_qubit in range(len(reg_ancilla)):
        H(reg_ancilla[idx_qubit])
        func_ctrl_multmod(reg_system, reg_zeroed, reg_ancilla[idx_qubit], qubit_zeroed, pow(int_factor, 2 ** idx_qubit,
                                                                                            int_divisor), int_divisor)
    func_qft_without_swap_inverse(reg_ancilla)

    # We only measure the reg_ancilla, which gives the estimation of the phase
    MeasureZ(reg_ancilla, range(number_qubit_ancilla))

    env.module(CompositeGateModule())

    return func_qof_data_processing(int_divisor, number_qubit_ancilla, env.commit(
        int_shots, fetchMeasure=True)["counts"])


def func_Shor_algorithm(int_divisor, number_qubit_ancilla=None, int_shots=2, int_factor=None):
    """
    We want to factor the int int_divisor
    :param int_divisor: d, which we want to factor
    :param number_qubit_ancilla: the number of qubit which will be used for quantum phase estimation
    :param int_shots: the number of shots whose default value is 2; when int_shots > 2 means we want to know the
                      distribution of the state after quantum phase estimation
    :param int_factor: an integer whose order will be computed by the quantum order finding algorithm
    :return: a factor of d
    here it will print the computation process such as ord(4 mod 15) = 2 and the factorization such as 15 = 3 * 5,
    where "not quantum" means the current factorization comes from a classical part of Shor's algorithm.
    For int_shots > 2, we will print the number of shots where we obtain a correct factorization.
    """
    # Some classical cases
    if int_divisor < 0:
        int_divisor = -int_divisor
    if int_divisor == 0:
        print("{0} is zero.".format(int_divisor))
        return 0
    if int_divisor == 1:
        print("{0} is unit.".format(int_divisor))
        return 1
    elif isprime(int_divisor):
        print("{0} is prime.".format(int_divisor))
        return 1
    else:
        # The case that d is a power of some prime
        for idx_int in range(2, int(math.floor(math.log(int_divisor, 2))) + 1):
            if pow(int(pow(int_divisor, 1 / idx_int)), idx_int) == int_divisor:
                print("{0[0]} is a power of {0[1]}.".format([int_divisor, int(pow(int_divisor, 1 / idx_int))]))
                return idx_int
        # The case that d is even
        if int_divisor % 2 == 0:
            print("{0[0]} = {0[1]} * {0[2]}".format([int_divisor, 2, int_divisor // 2]))
            return 2
        else:
            # Generate a random f (int_factor) which can be assigned when called
            if int_factor is None:
                int_factor = randint(2, int_divisor - 2)
            # If the input int_factor is invalid to introduce the factorization, we reset the value of int_factor
            elif int_factor % int_divisor == 1 or int_factor % int_divisor == int_divisor - 1 or \
                    int_factor % int_divisor == 0:
                print('The value of int_factor is invalid!')
                return func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla, int_shots=int_shots)
            int_gcd = math.gcd(int_factor, int_divisor)
            # The case that f and d are not co-prime
            if int_gcd != 1:
                print('{0[0]} = {0[1]} * {0[2]}, not quantum'.format([int_divisor, int_gcd, int_divisor // int_gcd]))
                return int_gcd
            else:
                # From now on, we entry the quantum part
                number_qubit_system = int(math.ceil(math.log(int_divisor, 2)))
                # The number_qubit_ancilla used for phase estimation should be 2 * n_q_s + 4 such that the successful
                # probability of the phase estimation will > 98.6%
                if number_qubit_ancilla is None:
                    number_qubit_ancilla = 2 * number_qubit_system + 4
                # The case that the number of ancilla is enough
                if number_qubit_ancilla >= math.log(int_divisor, 2) * 2:
                    # A dict storing the possible order and its corresponding number of shots
                    dict_order = func_quantum_order_finding(int_factor, int_divisor, int_shots, number_qubit_ancilla)
                    # The list of possible order
                    list_order = [int(idx_key) for idx_key in dict_order.keys()]
                    int_order = 0
                    if int_shots == 1 or int_shots == 2 or len(list_order) == 1:
                        if int_shots == 1 or len(list_order) == 1:
                            int_order = list_order[0]
                        elif int_shots == 2 and len(list_order) == 2:
                            # For two shots, we compute the least common multiple as the order
                            int_order = int(list_order[0] * list_order[1] / math.gcd(list_order[0], list_order[1]))
                        int_pow_half = pow(int_factor, int_order // 2, int_divisor)
                        # To check whether int_factor and its order can introduce the factorization
                        if pow(int_factor, int_order, int_divisor) == 1 and int_order % 2 == 0 and int_pow_half != \
                                int_divisor - 1 and int_pow_half != 1:
                            print('ord({0[0]} mod {0[1]}) = {0[2]}'.format([int_factor, int_divisor, int_order]))
                            # An f which satisfies some appropriate conditions will give a factorization of d
                            print('{0[0]} = {0[1]} * {0[2]}'.format(
                                [int_divisor, math.gcd(int_divisor, int_pow_half - 1),
                                 math.gcd(int_divisor, int_pow_half + 1)]))
                            return math.gcd(int_divisor, int_pow_half - 1)
                        else:
                            # Maybe we compute a wrong order, maybe f and its order cannot give the factorization of d
                            print('Perhaps ord({0[0]} mod {0[1]}) = {0[2]},\n'.format([int_factor, int_divisor,
                                                                                       int_order]))
                            print('but it cannot give the factorization of {0}.'.format(int_divisor))
                            # We haven't compute the correct order and need to recompute
                            func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                int_shots=int_shots)
                    elif int_shots > 2:
                        # Here we use sympy to compute the order to confirm which possible order is correct
                        int_order_true = n_order(int_factor, int_divisor)
                        if int_order_true in list_order:
                            print('We obtain ord({0[0]} mod {0[1]}) = {0[2]} for {0[3]} of {0[4]} times.'.format(
                                [int_factor, int_divisor, int_order_true, dict_order["{0}".format(int_order_true)],
                                 int_shots]))
                            int_pow_half_true = pow(int_factor, int_order_true // 2, int_divisor)
                            # To check whether int_factor and its order can introduce the factorization
                            if pow(int_factor, int_pow_half_true, int_divisor) == 1 and int_order_true % 2 == 0 and \
                                    int_pow_half_true != int_divisor - 1:
                                print('{0[0]} = {0[1]} * {0[2]}'.format([int_divisor, math.gcd(
                                    int_divisor, int_pow_half_true - 1), math.gcd(int_divisor, int_pow_half_true + 1)]))
                                return math.gcd(int_divisor, int_pow_half_true - 1)
                            else:  # int_factor cannot introduce the factorization of d
                                print('But it cannot give the factorization of {0}.'.format(int_divisor))
                                func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                    int_shots=int_shots)
                        else:
                            # We haven't compute the correct order and need to recompute
                            print("we haven't computed the correct order of {0[0]} mod {0[1]}.".format(
                                [int_factor, int_divisor]))
                            func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                int_shots=int_shots)
                else:  # The case that the number of ancilla is not enough, where the cost is more.
                    print("Since the ancilla qubits are not enough to estimate the order with a high probability,")
                    print("we need to traverse all probable denominator in the step of continued fraction expansions.")
                    print("Thus some order may be obtained for twice as many times as the number of shots.\n")
                    # A dict storing the possible order and its corresponding number of shots
                    dict_order = func_quantum_order_finding(int_factor, int_divisor, int_shots, number_qubit_ancilla)
                    # The list of possible order
                    list_order = [int(idx_key) for idx_key in dict_order.keys()]
                    int_order = 0
                    if int_shots == 1 or len(list_order) == 1:  # For the case 1-shot
                        for int_order_maybe in list_order:  # Check "maybe" is the correct order one by one
                            if int_order_maybe != 0 and pow(int_factor, int_order_maybe, int_divisor) == 1:
                                int_order = int_order_maybe  # If correct, recorded in int_order
                                break
                        int_pow_half = pow(int_factor, int_order // 2, int_divisor)
                        # To check whether int_factor and its order can introduce the factorization
                        if pow(int_factor, int_order, int_divisor) == 1 and int_order % 2 == 0 and int_pow_half != \
                                int_divisor - 1 and int_pow_half != 1:
                            print("ord({0[0]} mod {0[1]}) = {0[2]}".format([int_factor, int_divisor, int_order]))
                            # An f which satisfies some appropriate conditions will give a factorization of d
                            print("{0[0]} = {0[1]} * {0[2]}".format(
                                [int_divisor, math.gcd(int_divisor, int_pow_half - 1),
                                 math.gcd(int_divisor, int_pow_half + 1)]))
                            return math.gcd(int_divisor, int_pow_half - 1)
                        else:
                            # Maybe we compute a wrong order, maybe f and its order cannot give the factorization of d
                            if int_order == 0:  # We haven't computed the correct order of int_factor
                                print("We haven't computed the correct order of {0[0]} mod {0[1]}.\n".format(
                                    [int_factor, int_divisor]))
                            else:  # int_factor cannot introduce the factorization of d
                                print("Perhaps ord({0[0]} mod {0[1]}) = {0[2]},".format(
                                    [int_factor, int_divisor, int_order]))
                                print("but it cannot give the factorization of {0}.".format(int_divisor))
                            # And we need to recompute
                            func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                int_shots=int_shots)
                    elif int_shots > 1:  # For the case int_shots > 1
                        # Here we use sympy to compute the order to confirm which possible order is correct
                        int_order_true = n_order(int_factor, int_divisor)
                        if int_order_true in list_order:
                            print("We obtain ord({0[0]} mod {0[1]}) = {0[2]} for {0[3]} of {0[4]} times.".format(
                                [int_factor, int_divisor, int_order_true, dict_order["{0}".format(int_order_true)],
                                 int_shots]))
                            int_pow_half_true = pow(int_factor, int_order_true // 2, int_divisor)
                            # To check whether int_factor and its order can introduce the factorization
                            if pow(int_factor, int_pow_half_true, int_divisor) == 1 and int_order_true % 2 == 0 and \
                                    int_pow_half_true != int_divisor - 1:
                                print("{0[0]} = {0[1]} * {0[2]}".format([int_divisor, math.gcd(
                                    int_divisor, int_pow_half_true - 1), math.gcd(int_divisor, int_pow_half_true + 1)]))
                                return math.gcd(int_divisor, int_pow_half_true - 1)
                            else:  # int_factor cannot introduce the factorization of d
                                print("But it cannot give the factorization of {0}.".format(int_divisor))
                                func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                    int_shots=int_shots)
                        else:
                            # We haven't compute the correct order and need to recompute
                            print("we haven't computed the correct order of {0[0]} mod {0[1]}.\n".format(
                                [int_factor, int_divisor]))
                            func_Shor_algorithm(int_divisor, number_qubit_ancilla=number_qubit_ancilla,
                                                int_shots=int_shots)


if __name__ == "__main__":
    func_Shor_algorithm(15, number_qubit_ancilla=8, int_shots=2, int_factor=2)
