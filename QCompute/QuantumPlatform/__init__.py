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
Export the entire directory as a library
"""

from enum import Enum, unique


@unique
class BackendName(Enum):
    """
    Name of Backends

    Used in circuit computing tasks

    Example:

    env = QuantumEnvironment()

    env.backend(BackendName.LocalBaiduSim2)
    """

    LocalBaiduSim2 = 'local_baidu_sim2'
    """
    Local Baidu Sim2
    
    Param can be a Sim2Param enum or string.
    
    Default is Sim2Param.Dense_Matmul_Probability
    
    Example: 
    
    env = QuantumEnvironment()
    
    env.backend(BackendName.LocalBaiduSim2)
    
    or
    
    env.backend(BackendName.LocalBaiduSim2, Sim2Param.Dense_Matmul_Probability)
    
    or Added shots(must have space in string)
    
    env.backend(BackendName.LocalBaiduSim2, Sim2Param.Dense_Matmul_Probability.value + ' -s 1024')
    """

    

    CloudBaiduSim2 = 'cloud_baidu_sim2'
    """
    Cloud Baidu Sim2
    
    Param can be a Sim2Param enum or string.
    
    Example: 
    
    env = QuantumEnvironment()
    
    env.backend(BackendName.CloudBaiduSim2)
    
    or
    
    env.backend(BackendName.CloudBaiduSim2, Sim2Param.Dense_Matmul_Probability)
    
    or Added shots(must have space in string)
    
    env.backend(BackendName.CloudBaiduSim2, Sim2Param.Dense_Matmul_Probability.value + ' -s 1024')
    """

    CloudQpu = 'cloud_qpu'
    """
    Cloud QPU
    
    Param should be 0.
    
    Example: 
    
    env = QuantumEnvironment()
    
    env.backend(BackendName.CloudQpu, 0)
    """

    CloudAerAtBD = 'cloud_aer_at_bd'
    """
    Cloud Aer at Baidu
    
    Param can be one of '-q', '-s', '-u', but only '-q' is valid now.
    
    Example: 
    
    env = QuantumEnvironment()
    
    env.backend(BackendName.CloudAerAtBD)
    
    or
    
    env.backend(BackendName.CloudAerAtBD, '-q')
    """

    





@unique
class Sim2Param(Enum):
    """
    Params group of Baidu Sim2
    """

    Dense_Matmul_Probability = '-mt dense -a matmul -mm probability'
    Dense_Matmul_Accumulation = '-mt dense -a matmul -mm accumulation'
    Dense_Einsum_Probability = '-mt dense -a einsum -mm probability'
    Dense_Einsum_Accumulation = '-mt dense -a einsum -mm accumulation'
    
