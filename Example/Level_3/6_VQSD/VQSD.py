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
VQSD
"""
import copy

import numpy as np

import sys

sys.path.append('../../..')  # "from QCompute import *" requires this
from QCompute import *

matchSdkVersion('Python 3.0.0')

shots = 100000
n = 2  # n-qubit
delta = np.pi / 2  # calculate derivative
learning_rate = 0.5  # learning rate
N = 15  # number of parameters
para = np.random.rand(N) * 2 * np.pi  # initial parameters


def state_prepare(q, i):
    """
    This function is used to prepare state
    """

    RX(0.1)(q[i])
    RZ(0.4)(q[i + 1])
    CX(q[i], q[i + 1])
    RY(0.8)(q[i])
    RZ(1.2)(q[i])


def universal_cir(q, i, para):
    """
    This function builds a 15-parameterized circuit, which is
    enough to simulate any 2-qubit Unitaries
    """

    RZ(para[0])(q[i])
    RY(para[1])(q[i])
    RZ(para[2])(q[i])

    RZ(para[3])(q[i + 1])
    RY(para[4])(q[i + 1])
    RZ(para[5])(q[i + 1])

    CX(q[i + 1], q[i])

    RZ(para[6])(q[i])
    RY(para[7])(q[i + 1])

    CX(q[i], q[i + 1])

    RY(para[8])(q[i + 1])

    CX(q[i + 1], q[i])

    RZ(para[9])(q[i])
    RY(para[10])(q[i])
    RZ(para[11])(q[i])

    RZ(para[12])(q[i + 1])
    RY(para[13])(q[i + 1])
    RZ(para[14])(q[i + 1])


def my_cir(para):
    """
    This function returns the measurement result
    """

    env = QEnv()
    env.backend(BackendName.LocalBaiduSim2)
    q = env.Q.createList(2 * n)

    # Prepare a state
    for i in range(2):
        state_prepare(q, 2 * i)

    # Add parameterized circuit
    for i in range(2):
        universal_cir(q, 2 * i, para)

    # DIP test
    for i in range(2):
        CX(q[i], q[i + n])

    MeasureZ(*env.Q.toListPair())
    taskResult = env.commit(shots, fetchMeasure=True)

    return taskResult['counts']


def data_processing(data_dic):
    """
    This function returns the frequency of getting 00xx
    """

    sum_0 = 0
    for key, value in data_dic.items():
        if int(list(key)[0]) + int(list(key)[1]) == 0:
            sum_0 += value
    return sum_0 / shots


def loss_fun(para):
    """
    This is the loss function
    """

    return -data_processing(my_cir(para))


def diff_fun(f, para):
    """
    It returns a updated parameter set, para is a np.array
    """

    para_length = len(para)
    gradient = np.zeros(para_length)

    for i in range(para_length):
        para_copy_plus = copy.copy(para)
        para_copy_minus = copy.copy(para)
        para_copy_plus[i] += delta
        para_copy_minus[i] -= delta

        gradient[i] = (f(para_copy_plus) - f(para_copy_minus)) / 2

    new_para = copy.copy(para)
    res = new_para - learning_rate * gradient
    return res


def main():
    """
    Now we perform eigenvalues readout
    """

    para_list = [para]
    loss_list = []

    for i in range(30):
        para_list.append(diff_fun(loss_fun, para_list[i]))
        loss_list.append(loss_fun(para_list[i]))

    env = QEnv()
    env.backend(BackendName.LocalBaiduSim2)

    q = env.Q.createList(n)

    state_prepare(q, 0)
    universal_cir(q, 0, para_list[-1])

    MeasureZ(*env.Q.toListPair())
    taskResult = env.commit(shots, fetchMeasure=True)
    print(taskResult['counts'])


if __name__ == '__main__':
    main()
