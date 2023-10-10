#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
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

"""init file for the tomography module."""
from Extensions.QuantumErrorProcessing.qcompute_qep.tomography.tomography import Tomography
from Extensions.QuantumErrorProcessing.qcompute_qep.tomography.basis import \
    PauliMeasBasis, MeasurementBasis, PreparationBasis, init_measurement_basis, init_preparation_basis
from Extensions.QuantumErrorProcessing.qcompute_qep.tomography.state_tomography import StateTomography
from Extensions.QuantumErrorProcessing.qcompute_qep.tomography.process_tomography import ProcessTomography
from Extensions.QuantumErrorProcessing.qcompute_qep.tomography.spectral_tomography import SpectralTomography
from Extensions.QuantumErrorProcessing.qcompute_qep.tomography.utils import plot_process_ptm, compare_process_ptm
from Extensions.QuantumErrorProcessing.qcompute_qep.tomography.gateset import \
    GateSet, STD1Q_GATESET_RXRY, STD1Q_GATESET_RXRYRX, STD1Q_GATESET_RXRYID, \
    STD2Q_GATESET_RXRYCX, STD2Q_GATESET_RXRYCZ, STD2Q_GATESET_RXRYSWAP
from Extensions.QuantumErrorProcessing.qcompute_qep.tomography.gateset_optimizer import GSTOptimizer, \
    LinearInversionOptimizer, MLEOptimizer, LinearApproximationOptimizer
from Extensions.QuantumErrorProcessing.qcompute_qep.tomography.gateset_tomography import GateSetTomography

__all__ = [
    'Tomography', 'StateTomography', 'ProcessTomography', 'GateSetTomography',
    'PauliMeasBasis', 'MeasurementBasis', 'PreparationBasis', 'GateSet', 'SpectralTomography',
    'init_measurement_basis', 'init_preparation_basis',
    'plot_process_ptm', 'compare_process_ptm',
    'STD1Q_GATESET_RXRY', 'STD1Q_GATESET_RXRYRX', 'STD1Q_GATESET_RXRYID',
    'STD2Q_GATESET_RXRYCX', 'STD2Q_GATESET_RXRYCZ', 'STD2Q_GATESET_RXRYSWAP',
    'GSTOptimizer',
    'LinearInversionOptimizer', 'MLEOptimizer', 'LinearApproximationOptimizer'
]
