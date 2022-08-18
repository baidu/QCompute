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
VQE
"""
from copy import copy
import multiprocessing as mp
import pickle
import random
from functools import reduce

import numpy as np
import scipy
import scipy.linalg
from matplotlib import pyplot as plt
import time
import os

import sys

sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

matchSdkVersion('Python 3.0.0')

# Hyper-parameter setting
shots = 1024
n = 4  # n must be larger than or equal to 2; n is the size of our quantum system
assert n >= 2
L = 2  # L is the number of layers
iteration_num = 20
experiment_num = 4  # That's the number of parallel experiments we will run;
# it indicates the number of processes we will use.
# Don't stress your computer too much.
learning_rate = 0.3
delta = np.pi / 2  # Calculate analytical derivative
SEED = 36  # This number will determine what the final Hamiltonian is. It is also
# used to make sure Mac and Windows behave the same using multiprocessing module.
K = 3  # k is the number of local Hamiltonian in H
N = 3 * n * L  # N is the number of parameters needed for the circuit
random.seed(SEED)


def random_pauli_generator(l):
    """
    The following functions are used to generate random Hamiltonian
    """

    s = ''
    for i in range(l):
        s = s + random.choice(['i', 'x', 'y', 'z'])
    return s


def random_H_generator(n, k):
    """
    n is the number of qubits, k is the number of local Hamiltonian in H
    """

    H = []
    for i in range(k):
        H.append([random.random(), random_pauli_generator(n)])
    return H


Hamiltonian = random_H_generator(n, K)  # Our Hamiltonian H


# From Paddle_quantum package
def NKron(AMatrix, BMatrix, *args):
    """
    Recursively execute kron n times. This function at least has two matrices.
    :param AMatrix: First matrix
    :param BMatrix: Second matrix
    :param args: If have more matrix, they are delivered by this matrix
    :return: The result of tensor product.
    """

    return reduce(
        lambda result, index: np.kron(result, index),
        args,
        np.kron(AMatrix, BMatrix), )


def ground_energy(Ha):
    """
    It returns the ground energy of Hamiltonian Ha,
    which looks like [[12, 'xyiz'], [21, 'zzxz'], [10, 'iixy']].
    """

    # It is a local function
    def my_f(s):
        s = s.lower()

        I = np.eye(2) + 0j
        X = np.array([[0, 1], [1, 0]]) + 0j
        Y = np.array([[0, -1j], [1j, 0]])
        Z = np.array([[1, 0], [0, -1]]) + 0j

        if s == 'x':
            return X
        elif s == 'y':
            return Y
        elif s == 'z':
            return Z
        else:
            return I

    # It is a local function
    def my_g(s_string):
        H = []
        for ele in s_string:
            H.append(my_f(ele))
        return NKron(*H)

    sum = 0
    for ele in Ha:
        sum += ele[0] * my_g(ele[1])

    eigen_vector = np.sort(scipy.linalg.eig(sum)[0])
    return eigen_vector[0].real


def fig_name():
    """
    Generate a title of figure with time.
    """

    return os.path.dirname(__file__) + '/VQE' + time.strftime("%Y%m%d%H%M%S", time.localtime(time.time())) + '.png'


def eigen_plot(eigenv_list, actual_eigenv):
    """
    This is the plot function of actual loss over iterations.
    """

    for ele in eigenv_list:
        plt.plot(list(range(1, len(ele) + 1)), ele, linewidth=4)

    plt.axhline(y=actual_eigenv, color='black', linestyle='-.')
    plt.xlabel('iteration')
    plt.ylabel('loss')
    plt.title('Actual Loss Over Iteration')
    plt.savefig(fig_name())
    # plt.show()


def prob_calc(data_dic):
    """
    Measure the first (ancillary) qubit. Return the value
    of 'the probability of getting 0' minus 'the probability of getting 1'.
    """

    sum_0 = 0
    sum_1 = 0
    for key, value in data_dic.items():
        if int(list(key)[-1], 16) % 2 == 0:
            sum_0 += value
        else:
            sum_1 += value
    return (sum_0 - sum_1) / shots


# TODO: Add entangled layer

def add_block(q, loc, para):
    """
    Add a RzRyRz gate block. Each block has 3 parameters.
    """

    RZ(para[0])(q[loc])
    RY(para[1])(q[loc])
    RZ(para[2])(q[loc])


def add_layer(para, q):
    """
    Add a layer, which has 3*n parameters. para is a 2-D numpy array
    """

    for i in range(1, n + 1):
        add_block(q, i, para[i - 1])
    for i in range(1, n):
        CX(q[i], q[i + 1])
    CX(q[n], q[1])


def self_defined_circuit(para, hamiltonian):
    """
    H is a list, for example, if H = 12*X*Y*I*Z + 21*Z*Z*X*Z + 10* I*I*X*Y,
    then Hamiltonian is [[12, 'xyiz'], [21, 'zzxz'], [10, 'iixy']](upper case or lower case are all fine).
    It returns the expectation value of H.
    """

    env = QEnv()
    env.backend(BackendName.LocalBaiduSim2)

    # The first qubit is ancillary
    q = env.Q.createList(n + 1)

    hamiltonian = [symbol.lower() for symbol in hamiltonian]
    high_D_para = para.reshape(L, n, 3)  # Change 1-D numpy array to a 3-D numpy array

    # Set up our parameterized circuit
    for i in range(1, n + 1):
        H(q[i])

    # Add parameterized circuit
    for i in range(L):
        add_layer(high_D_para[i], q)

    for i in range(n):
        # Set up Pauli measurement circuit
        if hamiltonian[i] == 'x':
            H(q[i + 1])
            CX(q[i + 1], q[0])

        elif hamiltonian[i] == 'z':
            CX(q[i + 1], q[0])

        elif hamiltonian[i] == 'y':
            RZ(-np.pi / 2)(q[i + 1])
            H(q[i + 1])
            CX(q[i + 1], q[0])

    # Measurement result
    MeasureZ(*env.Q.toListPair())
    taskResult = env.commit(shots, fetchMeasure=True)
    return prob_calc(taskResult['counts'])


def diff_fun(f, para):
    """
    It calculates the gradient of f on para,
    update parameters according to the gradient, and return the updated parameters.
    'para' is a np.array.
    """

    para_length = len(para)
    gradient = np.zeros(para_length)

    for i in range(para_length):
        para_copy_plus = copy(para)
        para_copy_minus = copy(para)
        para_copy_plus[i] += delta
        para_copy_minus[i] -= delta

        gradient[i] = (f(para_copy_plus) - f(para_copy_minus)) / 2

    new_para = copy(para)
    res = new_para - learning_rate * gradient
    return res


def loss_fun(para):
    """
    This is the loss function.
    """

    res = sum([ele[0] * self_defined_circuit(para, ele[1]) for ele in Hamiltonian])
    return res


def multi_process_fun(j):
    """
    This function runs one experiment, parameter j indicates it is the j-th experiment.
    """

    np.random.seed()
    para = np.random.rand(N) * np.pi
    para_list = [para]

    for i in range(iteration_num):
        para_list.append(diff_fun(loss_fun, para_list[i]))

    with (outputDirPath / f"para{j}.pickle").open(mode='wb') as fp:
        pickle.dump(para_list, fp)


def main():
    """
    main
    """
    
    pool = mp.Pool(experiment_num)
    pool.map(multi_process_fun, range(experiment_num))
    loss_list = []

    for _ in range(experiment_num):
        actual_loss = []
        with (outputDirPath / f"para{_}.pickle").open(mode='rb') as fp:
            new_para_list = pickle.load(fp)

        for j in range(iteration_num):
            actual_loss.append(loss_fun(new_para_list[j]))

        loss_list.append(actual_loss)

    eigen_plot(loss_list, ground_energy(Hamiltonian))


if __name__ == '__main__':
    main()
