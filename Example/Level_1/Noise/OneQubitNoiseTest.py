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
There is a simple case of simulating a circuit with 1-qubit noise.
Real values after noisy circuits are calculated for comparison.
"""
import sys
from multiprocessing import freeze_support
from typing import List, Union, Dict

import numpy as np

from QCompute import *

sys.path.append('../..')

matchSdkVersion('Python 3.3.5')

noiseType = Union['AmplitudeDamping', 'BitFlip', 'CustomizedNoise', 'BitPhaseFlip',
                  'Depolarizing', 'PauliNoise', 'PhaseDamping', 'PhaseFlip', 'ResetNoise']


def test_circuit_1qubit(noises: List[noiseType], bool_Xgate: bool, bool_Hgate: bool) -> 'QEnv':
    """
    This function gives a QCompute environment to test noise on 1-qubit gate,

    - "state preparation circuit"  -  S - Z - S - noise.

    Here the "state preparation circuit" generates a basis of states in one-dimensional Hilbert space,

    which is composed by X and H gates on one qubit.

    :param noises: a list of QCompute noise instances

    :param bool_Xgate: true for inserting an X gate, false for None

    :param bool_Hgate: true for inserting a H gate, false for None

    :return: a QCompute environment 
    """
    # Create environment
    env = QEnv()
    # Choose backend Baidu local simulator with noise
    env.backend(BackendName.LocalBaiduSim2)

    q = env.Q.createList(1)

    # Initialize a state preparation circuit
    if bool_Xgate:
        X(q[0])

    if bool_Hgate:
        H(q[0])

    # Initialize an identity circuit to verify the effect of pure noise
    S(q[0])
    Z(q[0])
    S(q[0])

    # add noise after above circuit
    for noise_instance in noises:
        env.noise(gateNameList=['S'], noiseList=[
            noise_instance], positionList=[1])

    # Measure with the computational basis
    MeasureZ(*env.Q.toListPair())

    return env


def test_real_value(noises: List[noiseType], bool_Xgate: bool, bool_Hgate: bool, shots: int) -> Dict[str, int]:
    """
    This function calculates the output of a sequential of noises for a circuit

    which is determined by bool_Xgate and bool_Hgate.

    :param noises: a list of QCompute noise instances

    :param bool_Xgate: true for inserting an X gate, false for None

    :param bool_Hgate: true for inserting a H gate, false for None

    :param shots: the shots after measuring the output state in Z basis
    """
    # Initial state before any gate
    in_state = np.array([1.0, 0.0])

    if bool_Xgate:
        in_state = np.dot(X.getMatrix(), in_state)

    if bool_Hgate:
        in_state = np.dot(H.getMatrix(), in_state)

    # Apply noises on current state
    out_state = apply_noise(noises, in_state)

    # Counts after measureZ
    counts = {'0': round(abs(out_state[0, 0]) * shots)}
    counts['1'] = shots - counts['0']
    return counts


def apply_noise(noises: List[noiseType], state: np.ndarray) -> np.ndarray:
    """
    This function calculate the output of a sequential of noises for any input state,

    :param noises: a list of QCompute noise instances

    :param state: the input state before a sequential of noises
    """
    density_matrix = np.outer(state, state.T.conjugate())
    for noise in noises:
        matrix_temp = 0.0 + 0.0j
        for index in range(len(noise.krauses)):
            if noise.noiseClass == 'mixed_unitary_noise':
                matrix_temp += noise.probabilities[index] * \
                               noise.krauses[index] @ density_matrix @ noise.krauses[index].T.conjugate()
            else:
                matrix_temp += noise.krauses[index] @ density_matrix @ noise.krauses[index].T.conjugate()
        density_matrix = matrix_temp

    return density_matrix


def main():
    # Define QCompute noise instances
    # Define a BitFlip noise
    bfobj_01 = BitFlip(probability=0.1)
    # Define an AmplitudeDamping noise
    adobj_005 = AmplitudeDamping(probability=0.05)
    # Define a Customized noise
    kraus_list = [np.array([[np.sqrt(1 - 0.1), 0.0], [0.0, 1.0]]),
                  np.array([[0.0, 0.0], [np.sqrt(0.1), 0.0]])
                  ]
    cnobj = CustomizedNoise(krauses=kraus_list)
    # Define a BitPhaseFlip noise
    bpfobj_01 = BitPhaseFlip(probability=0.1)
    # Define a Depolarizing noise
    dpobj_1_01 = Depolarizing(bits=1, probability=0.1)
    # Define a PauliNoise noise
    pnobj_005_01_015 = PauliNoise(
        probability1=0.05, probability2=0.1, probability3=0.15)
    # Define a PhaseDamping noise
    pdobj_01 = PhaseDamping(probability=0.1)
    # Define a PhaseFlip noise
    pfobj_01 = PhaseFlip(probability=0.1)
    # Define a ResetNoise noise
    rnobj_01_01 = ResetNoise(probability1=0.1, probability2=0.1)

    # Initialize a test circuit with noise
    bool_list = [True, False]
    noise_list = [bfobj_01, adobj_005, cnobj, bpfobj_01, dpobj_1_01,
                  pnobj_005_01_015, pdobj_01, pfobj_01, rnobj_01_01]

    env = test_circuit_1qubit(
        noises=noise_list, bool_Xgate=bool_list[0], bool_Hgate=bool_list[1])

    # Commit the task with 1000 shots
    shots = 1000
    env.commit(shots=shots, fetchMeasure=True)

    # Real value from mathematical calculation
    realValue = test_real_value(
        noises=noise_list, bool_Xgate=bool_list[0], bool_Hgate=bool_list[1], shots=shots)
    print('The real value should be:', realValue)


if __name__ == '__main__':
    # For multiprocess
    # Must use `if __name__ == '__main__':`
    # And use `freeze_support()` in it.
    freeze_support()

    main()
