# -*- coding: UTF-8 -*-
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

from .basic_circuit import BasicCircuit
from .encoding_circuit import IQPEncodingCircuit
from .encoding_circuit import BasisEncodingCircuit
from .kernel_estimation_circuit import KernelEstimationCircuit
from .parameterized_circuit import ParameterizedCircuit
from .pauli_measurement_circuit import PauliMeasurementCircuit
from .pauli_measurement_circuit import PauliMeasurementCircuitWithAncilla
from .pauli_measurement_circuit import SimultaneousPauliMeasurementCircuit
from .qaoa_ansatz import QAOAAnsatz
from .parameterized_circuit_template import UniversalCircuit, RealEntangledCircuit, ComplexEntangledCircuit
from .parameterized_circuit_template import RealAlternatingLayeredCircuit, ComplexAlternatingLayeredCircuit

__all__ = [
    'BasicCircuit',
    'IQPEncodingCircuit',
    'BasisEncodingCircuit',
    'KernelEstimationCircuit',
    'ParameterizedCircuit',
    'PauliMeasurementCircuit',
    'PauliMeasurementCircuitWithAncilla',
    'SimultaneousPauliMeasurementCircuit',
    'QAOAAnsatz',
    'UniversalCircuit',
    'RealEntangledCircuit',
    'ComplexEntangledCircuit',
    'RealAlternatingLayeredCircuit',
    'ComplexAlternatingLayeredCircuit'
]
