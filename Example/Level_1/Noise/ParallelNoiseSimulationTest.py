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
This is a complex noisy circuit to test how well bulid-in multi-process works 
"""
from QCompute import *
import sys
import random
import time

from QCompute.Define import Settings

Settings.outputInfo = False

sys.path.append('../..')

matchSdkVersion('Python 3.3.2')


def self_defined_noisy_circuit() -> 'QEnv':
    """
    A self defined noisy random H + CX + RX circuit
    """
    # Create environment
    env = QEnv()
    # Choose backend Baidu local simulator
    env.backend(BackendName.LocalBaiduSim2)

    # Number of qubits, no larger than 20  
    num_qubit = 13
    # Number of gates in each for loop
    gate_num = 3  # Depth of circuit = num_qubit * gate_num

    assert num_qubit > 2
    assert gate_num > 2

    # Initialize a QCompute circuit
    q = env.Q.createList(num_qubit)

    # A noisy random H + CX + RX circuit
    for i in range(num_qubit - 1):
        H(q[i])
        CX(q[i], q[i + 1])
        # Random rotation angles
        rotation_list = [random.uniform(0, 6.28) for _ in range(gate_num - 2)]
        # random quantum registers
        qreg_list = [random.randint(0, num_qubit - 1) for _ in range(gate_num - 2)]
        for i in range(gate_num - 2):
            RX(rotation_list[i])(q[qreg_list[i]])

    # Measure with the computational basis
    MeasureZ(*env.Q.toListPair())

    # Define noise instances  
    # Define a Bit Flip noise instance
    bfobj = BitFlip(0.1)
    # Define a 2-qubit Depolarizing noise instance
    dpobj = Depolarizing(2, 0.1)

    # Add noises
    env.noise(['H', 'RX'], [bfobj])
    env.noise(['CX'], [dpobj])

    return env


def main():
    """
    main
    """

    # Ture off multi-process in noisy simulator for comparison
    # Settings.noiseMultiprocessingSimulator = False

    env = self_defined_noisy_circuit()

    # Commit the task with 1024 shots
    env.commit(1024, fetchMeasure=True)


if __name__ == '__main__':
    Settings.noiseMultiprocessingSimulator = False
    start = time.time()
    main()
    print('Noise single processing simulator time costs:', time.time() - start, 's')

    Settings.noiseMultiprocessingSimulator = True
    start = time.time()
    main()
    print('Noise multi processing simulator time costs:', time.time() - start, 's')
