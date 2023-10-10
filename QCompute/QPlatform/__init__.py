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
Export the entire directory as a library
"""
from enum import Enum, unique

from QCompute.QPlatform import Error

ModuleErrorCode = 9
FileErrorCode = 14


def getBackendFromName(name: str):
    for value in BackendName.__members__.values():
        if value.name == name or value.value == name:
            return value
    raise Error.ArgumentError('Unknown backend name.', ModuleErrorCode, FileErrorCode, 1)

class BackendName(Enum):
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

    LocalCuQuantum = 'local_cuquantum'
    """
    Local CuQuantum
    
    This backend name (LocalCuQuantum) is only available >= v3.3.0
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.LocalCuQuantum)
    """

    LocalBaiduSimPhotonic = 'local_baidu_sim_photonic'
    """
    Local Baidu Sim Photonic Gaussian And Fock

    Example: 

    env = QEnv()

    env.backend(BackendName.LocalBaiduSimPhotonic)
    """

    LocalBaiduSim2WithNoise = 'local_baidu_sim2'
    """
    Local Baidu Sim2 With Noise (Deprecated, should use LocalBaiduSim2)

    Example: 

    env = QEnv()

    env.backend(BackendName.LocalBaiduSim2WithNoise)
    """

    

    CloudBaiduSim2Water = 'cloud_baidu_sim2_water'
    """
    Cloud Baidu Sim2 Water
    
    This backend name (CloudBaiduSim2Water) is only available >= v1.0.3
    
    Cpp dense simulator
    
    Dense_Matmul_Probability
    
    Parameter must be a string.
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudBaiduSim2Water)
    
    or Added shots(must have space in string)
    
    env.backend(BackendName.CloudBaiduSim2Water, '-s 20210210')
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

    CloudBaiduSim2Lake = 'cloud_baidu_sim2_lake'
    """
    Cloud Baidu Sim2 Lake
    
    This backend name (CloudBaiduSim2Lake) is only available >= v2.0.0
    
    Gpu dense simulator
    
    Dense_Matmul_Probability
    
    Parameter must be a string.
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudBaiduSim2Lake)
    
    or
    
    env.backend(BackendName.CloudBaiduSim2Lake, '-s 20210210')
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

    CloudBaiduQPUQian = 'cloud_baidu_qpu_qian'
    """
    Cloud Baidu QPU Qian

    This backend name (CloudBaiduQPUQian) is only available >= v3.0.0

    Example: 

    env = QEnv()

    env.backend(BackendName.CloudBaiduQPUQian)
    """

    CloudIoPCAS = 'cloud_iopcas'
    """
    Cloud IoPCAS
    
    This backend name (CloudIoPCAS) is only available >= v2.0.0
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudIoPCAS)
    """

    CloudIonAPM = 'cloud_ionapm'
    """
    Cloud IonAPM
    
    This backend name (IonAPM) is only available >= v3.0.0
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.CloudIonAPM)
    """

    ServiceUbqc = 'service_ubqc'
    """
    Service Ubqc
    
    This backend name (ServiceUbqc) is only available >= v2.0.4
    
    Example: 
    
    env = QEnv()
    
    env.backend(BackendName.ServiceUbqc)
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
    


@unique
class ServerModule(Enum):
    """
    Module at server
    """

    CompositeGate = 'CompositeGateModule'
    CompressGate = 'CompressGateModule'
    InverseCircuit = 'InverseCircuitModule'
    ReverseCircuit = 'ReverseCircuitModule'
    UnrollCircuit = 'UnrollCircuitModule'
    UnrollProcedure = 'UnrollProcedureModule'
    MappingToBaiduQPUQian = 'MappingToBaiduQPUQianModule'
    UnrollCircuitToBaiduQPUQian = 'UnrollCircuitToBaiduQPUQianModule'
    MappingToIoPCAS = 'MappingToIoPCASModule'
    UnrollCircuitToIoPCAS = 'UnrollCircuitToIoPCASModule'
    UnrollCircuitToIonAPM = 'UnrollCircuitToIonAPMModule'

    