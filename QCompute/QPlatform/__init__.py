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

    env = QEnv()

    env.backend(BackendName.LocalBaiduSim2)
    """

    LocalBaiduSim2 = 'local_baidu_sim2'
    """
    Local Baidu Sim2
    
    This backend name (LocalBaiduSim2) is only available >= v1.0.0
    
    Parameter can be a Sim2Argument enum or string.
    
    Default is Sim2Argument.Dense_Matmul_Probability
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.LocalBaiduSim2)
    
    or
    
    env.backend(BackendName.LocalBaiduSim2, Sim2Argument.Dense_Matmul_Probability)
    
    or Added shots(must have space in string)
    
    env.backend(BackendName.LocalBaiduSim2, Sim2Argument.Dense_Matmul_Probability.value + ' -s 20210210')
    """

    

    CloudBaiduSim2 = 'cloud_baidu_sim2'
    """
    Cloud Baidu Sim2 Deprecated
    
    This backend name (CloudBaiduSim2) is only available >= v1.0.0
    
    Alias of cloud_baidu_sim2_water
    
    Parameter can be a Sim2Argument enum or string.
    
    Default is Sim2Argument.Dense_Matmul_Probability
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudBaiduSim2)
    
    or
    
    env.backend(BackendName.CloudBaiduSim2, Sim2Argument.Dense_Matmul_Probability)
    
    or Added shots(must have space in string)
    
    env.backend(BackendName.CloudBaiduSim2, Sim2Argument.Dense_Matmul_Probability.value + ' -s 20210210')
    """

    CloudBaiduSim2Water = 'cloud_baidu_sim2_water'
    """
    Cloud Baidu Sim2 Water
    
    This backend name (CloudBaiduSim2Water) is only available >= v1.0.3
    
    Python simulator
    
    Parameter can be a Sim2Argument enum or string.
    
    Default is Sim2Argument.Dense_Matmul_Probability
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudBaiduSim2Water)
    
    or
    
    env.backend(BackendName.CloudBaiduSim2Water, Sim2Argument.Dense_Matmul_Probability)
    
    or Added shots(must have space in string)
    
    env.backend(BackendName.CloudBaiduSim2Water, Sim2Argument.Dense_Matmul_Probability.value + ' -s 20210210')
    """

    CloudBaiduSim2Earth = 'cloud_baidu_sim2_earth'
    """
    Cloud Baidu Sim2 Earth
    
    This backend name (CloudBaiduSim2Earth) is only available >= v1.0.3
    
    Python high performance simulator
    
    Parameter can be a Sim2Argument enum or string.
    
    Default is Sim2Argument.Dense_Matmul_Probability
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudBaiduSim2Earth)
    
    or
    
    env.backend(BackendName.CloudBaiduSim2Earth, Sim2Argument.Dense_Matmul_Probability)
    
    or Added shots(must have space in string)
    
    env.backend(BackendName.CloudBaiduSim2Earth, Sim2Argument.Dense_Matmul_Probability.value + ' -s 20210210')
    """

    CloudBaiduSim2Thunder = 'cloud_baidu_sim2_thunder'
    """
    Cloud Baidu Sim2 Thunder
    
    This backend name (CloudBaiduSim2Thunder) is only available >= v1.0.3
    
    Cpp dense simulator
    
    Dense_Matmul_Probability
    
    Parameter must be a string.
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudBaiduSim2Thunder)
    
    or
    
    env.backend(BackendName.CloudBaiduSim2Thunder, '-s 20210210')
    """

    CloudBaiduSim2Heaven = 'cloud_baidu_sim2_heaven'
    """
    Cloud Baidu Sim2 Heaven
    
    This backend name (CloudBaiduSim2Heaven) is only available >= v1.0.3
    
    Cpp cluster simulator
    
    Dense_Matmul_Probability
    
    Parameter must be a string.
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudBaiduSim2Heaven)
    
    or
    
    env.backend(BackendName.CloudBaiduSim2Heaven, '-s 20210210')
    """

    CloudBaiduSim2Wind = 'cloud_baidu_sim2_wind'
    """
    Cloud Baidu Sim2 Wind
    
    This backend name (CloudBaiduSim2Wind) is only available >= v1.1.0
    
    Cpp sparse simulator
    
    Sparse_Matmul_Probability
    
    Parameter must be a string.
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudBaiduSim2Wind)
    
    or
    
    env.backend(BackendName.CloudBaiduSim2Wind, '-s 20210210')
    """

    CloudQpu = 'cloud_qpu'
    """
    Cloud QPU
    
    This backend name (CloudQpu) is only available >= v1.0.0
    
    Parameter should be 0.
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudQpu, 0)
    """

    CloudQpu2 = 'cloud_qpu2'
    """
    Cloud QPU2
    
    This backend name (CloudQpu2) is only available >= v1.1.0
    
    Parameter should be 0.
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudQpu2, 0)
    """

    CloudAerAtBD = 'cloud_aer_at_bd'
    """
    Cloud Aer at Baidu
    
    This backend name (CloudAerAtBD) is only available >= v1.0.0
    
    Parameter can be one of '-q', '-s', '-u', but only '-q' is valid now.
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudAerAtBD)
    
    or
    
    env.backend(BackendName.CloudAerAtBD, '-q')
    """

    





@unique
class Sim2Argument(Enum):
    """
    Arguments group of Baidu Sim2
    """

    Dense_Matmul_Probability = '-mt dense -a matmul -mm probability'
    Dense_Matmul_Accumulation = '-mt dense -a matmul -mm accumulation'
    Dense_Einsum_Probability = '-mt dense -a einsum -mm probability'
    Dense_Einsum_Accumulation = '-mt dense -a einsum -mm accumulation'
    
