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
There is a simple case of simulating a circuit with 2-qubit noise.
Real value after a noisy circuit is calculated for comparison.
"""
from QCompute import *
from typing import List, Dict
import sys
import numpy as np

sys.path.append('../..')

matchSdkVersion('Python 3.3.3')

noiseType = 'Depolarizing'


def test_circuit_2qubit(noises: List[noiseType], bool_gate_dict: Dict[int, List[bool]]) -> 'QEnv':
    """
    This function gives a QCompute environment to test noise on 2-qubit gates,
    - "state preparation circuit"  -  CX - CX - noise.
    Here the "state preparation circuit" generates a basis of states in two-dimensional Hilbert space,
    which is composed by X and H gates on two qubits.

    :param noises: a list of QCompute noise instances
    :param bool_gate_dict: {0: [bool_Xgate_0, bool_Hgate_0], 1: [boo_Xgate_1, bool_Hgate_1]}
        where bool_Xgate_index: true for inserting an X gate on the qubit q[index], false for None
        and bool_Hgate_index: true for inserting a H gate on the qubit q[index], false for None
    :return: a QCompute environment 
    """
    # Create environment
    env = QEnv()
    # Choose backend Baidu local simulator with noise
    env.backend(BackendName.LocalBaiduSim2)

    q = env.Q.createList(2)

    # Initialize a state preparation circuit
    for position_index, bool_gate_list in bool_gate_dict.items():
        if bool_gate_list[0]:
            X(q[position_index])

        if bool_gate_list[1]:
            H(q[position_index])

    # Initialize a identity circuit to verify the effect of pure noise
    CX(q[0], q[1])
    CX(q[0], q[1])

    # add noises after above circuit
    for noise_instance in noises:
        env.noise(gateNameList=['CX'], noiseList=[
                  noise_instance], positionList=[1])

    # Measure with the computational basis
    MeasureZ(*env.Q.toListPair())

    return env


def test_real_value(noises: List[noiseType], bool_gate_dict: Dict[int, List[bool]], shots: int) -> Dict[str, int]:
    """
    This function calculates the output of a sequential of noises for a circuit
    which is composed of X and H gates on two qubits and its description is written in bool_gate_dict.
    :param noises: a list of QCompute noise instances
    :param bool_gate_dict: {0: [bool_Xgate_0, bool_Hgate_0], 1: [boo_Xgate_1, bool_Hgate_1]}
        where bool_Xgate_index: true for inserting an X gate on the qubit q[index], false for None
        and bool_Hgate_index: true for inserting a H gate on the qubit q[index], false for None
    :param shots: the shots after measuring the output state in Z basis
    """
    # Initialize state before any gate
    in_state = np.array([1.0, 0.0, 0.0, 0.0])

    # Initialize a 1-qubit identity matrix dict
    matrix_dict = {0: np.array([[1.0, 0.0], [0.0, 1.0]]), 1: np.array([
        [1.0, 0.0], [0.0, 1.0]])}

    # Collect the effective gate matrix on each qubit under bool_gate_dict
    for position_index, bool_gate_list in bool_gate_dict.items():
        if bool_gate_list[0]:
            matrix_dict[position_index] = np.dot(
                X.getMatrix(), matrix_dict[position_index])

        if bool_gate_list[1]:
            matrix_dict[position_index] = np.dot(
                H.getMatrix(), matrix_dict[position_index])

    # Multiply the effective gate matrix on two qubits
    matrix = np.kron(matrix_dict[0], matrix_dict[1])

    # Get the input state before noise
    in_state = np.dot(matrix, in_state)

    # Apply noises on current state, and get the output state
    out_state = apply_noise(noises, in_state)

    # Counts after measureZ
    counts = {}
    for i in range(2):
        for j in range(2):
            counts[str(i) + str(j)
                   ] = round(abs(out_state[2 * i + j, 2 * i + j]) * shots)

    counts['11'] = shots - counts['00'] - counts['01'] - counts['10']
    return counts


# Calculate the real output state after noise by mathematical calculation
def apply_noise(noises: List[noiseType], state: np.ndarray) -> np.ndarray:
    """
    This function calculates the output of a sequential of noises for any input state,
    :param noises: a list of QCompute noise instances
    :param state: the input state of a sequential of noises
    """
    density_matrix = np.outer(state, state.T.conjugate())
    for noise in noises:
        matrix_temp = 0.0 + 0.0j
        for index in range(len(noise.krauses)):
            kraus_temp = noise.krauses[index].reshape(4, 4)
            if noise.noiseClass == 'mixed_unitary_noise':
                matrix_temp += noise.probabilities[index] * \
                    kraus_temp@density_matrix@kraus_temp.T.conjugate()
            else:
                matrix_temp += kraus_temp@density_matrix@kraus_temp.T.conjugate()
        density_matrix = matrix_temp

    return density_matrix


def main():
    # Define a 2-qubit Depolarizing noise instance
    dpobj_2_01 = Depolarizing(bits=2, probability=0.1)

    # Initialize a test circuit
    bool_0 = [True, False]
    bool_1 = [True, True]
    bool_dict = {0: bool_0, 1: bool_1}
    noise_list = [dpobj_2_01]

    env = test_circuit_2qubit(noises=noise_list, bool_gate_dict=bool_dict)

    # Commit the task with 1000 shots
    shots = 1000
    env.commit(shots=shots, fetchMeasure=True)

    # Real value from mathematical calculation
    realValue = test_real_value(
        noises=noise_list, bool_gate_dict=bool_dict, shots=shots)
    print('The real value should be:', realValue)


if __name__ == '__main__':
    main()
