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
This is an example to calculate a specific noisy circuit.
"""
from QCompute import *
import sys

sys.path.append('../..')

def self_define_circuit(backend_argument: str = None) -> 'QResult':
    """
    A self defined circuit
    """
    # Create environment
    env = QEnv()
    # Choose backend Baidu local simulator
    if not backend_argument:
        env.backend(BackendName.LocalBaiduSim2)
    elif backend_argument == 'Dense_Matmul_Probability':
        env.backend(BackendName.LocalBaiduSim2, Sim2Argument.Dense_Matmul_Probability)
    elif backend_argument == 'Dense_Matmul_Accumulation':
        env.backend(BackendName.LocalBaiduSim2, Sim2Argument.Dense_Matmul_Accumulation)
    elif backend_argument == 'Dense_Matmul_Output_Probability':
        env.backend(BackendName.LocalBaiduSim2, Sim2Argument.Dense_Matmul_Output_Probability)  # Not supported in QCompute SDK
    elif backend_argument == 'Dense_Matmul_Output_State':
        env.backend(BackendName.LocalBaiduSim2, Sim2Argument.Dense_Matmul_Output_State)  # Not supported in QCompute SDK
    elif backend_argument == 'Sparse_Matmul_Output_State':
        env.backend(BackendName.LocalBaiduSim2, Sim2Argument.Sparse_Matmul_Output_State)  # Not supported in QCompute SDK

    # Initialize a 2-qubit circuit
    q = env.Q.createList(4)

    # A self-defined circuit
    H(q[0])
    CX(q[0], q[1])
    RX(0.5)(q[0])
    H(q[1])
    CX(q[1], q[2])
    H(q[2])
    CX(q[2], q[3])

    # Measure with the computational basis
    MeasureZ(*env.Q.toListPair())

    # Define 1-qubit noise objects
    bfobj = BitFlip(0.1)

    # Add noise
    env.noise(gateNameList=['H'], noiseList=[bfobj], qRegList=[0], positionList=[0])

    # Commit the task with 1024 shots
    shots = 1000
    res = env.commit(shots, fetchMeasure=True)
    return res


if __name__ == '__main__':
    # Not all supported in QCompute SDK
    # Supported_Backend_Arguements = ['Dense_Matmul_Probability', 'Dense_Matmul_Accumulation', 'Dense_Matmul_Output_Probability', 'Dense_Matmul_Output_State', 'Sparse_Matmul_Output_State']
    # All supported in QCompute SDK
    Supported_Backend_Arguements = ['Dense_Matmul_Probability', 'Dense_Matmul_Accumulation']
    
    for backend_argument in Supported_Backend_Arguements: 
        res = self_define_circuit(backend_argument)
        counts_str = ['0010', '1100', '1110', '1111', '0000', '1101', '0011', '0001'] 
        probability_each = 1 / len(counts_str)
        if backend_argument in ['Dense_Matmul_Probability', 'Dense_Matmul_Accumulation']:
            for str in counts_str:  
                assert abs(res['counts'][str] - 125) <= 20
        elif backend_argument == 'Dense_Matmul_Output_Probability':
            for str in counts_str:
                assert res['counts'][str] - probability_each <= 0.0025
        elif backend_argument in ['Dense_Matmul_Output_State', 'Sparse_Matmul_Output_State']:
            for str in counts_str:
                index = int(str, 2)  
                assert abs(res['state'][index, index]) - probability_each <= 0.0025