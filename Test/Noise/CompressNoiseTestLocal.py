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
This is a simple case of using CompressNoiseModule.
"""
import sys
import time
import random
from QCompute import *

from QCompute.Define import Settings

sys.path.append('../..')

def self_defined_noisy_circuit(qubits: int, gates: int) -> 'QEnv':
    """
    A self defined noisy random H + CX + RX circuit

    :param qubits: the maximum qubits
    :param gates: the number of gates on each qubit
    :return: a QCompute environment
    """

    # Create environment
    env = QEnv()
    # Choose backend Baidu cloud simulator
    # env.backend(BackendName.CloudBaiduSim2Water)
    env.backend(BackendName.LocalBaiduSim2)

    # env.backend(BackendName.CloudBaiduSim2Wind)

    # Number of qubits  
    num_qubit = qubits
    # Number of gates in each for loop
    gate_num = gates  # Depth of circuit = num_qubit * gate_num

    assert num_qubit > 2

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
        for k in range(gate_num - 2):
            RX(rotation_list[k])(q[qreg_list[k]])


    # Measure with the computational basis
    MeasureZ(*env.Q.toListPair()) 

    # Define noise instances
    # Define a 2-qubit noise instance
    dpobj = Depolarizing(2, 0.1)

    # Add noises
    env.noise(['CX'], [dpobj])

    return env


def main(qubits: int, gates: int, compress_or_not: bool) -> None:
    """
    main test.

    :param qubits: the maximum qubits
    :param gates: the number of gates on each qubit
    :param compress_or_not: use CompressNoiseModule or not
    :return: a QCompute environment
    """

    env = self_defined_noisy_circuit(qubits, gates)
    if compress_or_not:
        # Case1: CompressedNoiseModule
        start = time.time()
        env.commit(1024, fetchMeasure=True)
        print('CompressNoiseModule in single-processing simulator time costs', time.time() - start, 's')
    else:
        # Case2: No CompressedNoiseModule
        start = time.time()
        env.module(CompressNoiseModule({'disable': True}))
        env.commit(1024, fetchMeasure=True)
        print('No CompressNoiseModule in single-processing simulator time costs', time.time() - start, 's')


if __name__ == '__main__':
    qubitsList = [4] 
    gatesList = [50]

    Settings.outputInfo = False

    for qubits in qubitsList:
        for gates in gatesList:

            Settings.noiseMultiprocessingSimulator = False

            # CompressNoiseModule
            main(qubits, gates, True)

            # Disable CompressNoiseModule
            main(qubits, gates, False)
            
    