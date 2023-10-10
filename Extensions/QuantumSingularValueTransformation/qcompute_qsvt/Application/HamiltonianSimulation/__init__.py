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
__init__ file of the `qcompute_qsvt.Application.HamiltonianSimulation` module.
"""

from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation. \
    HamiltonianGenerator import func_Hamiltonian_gen
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation. \
    HamiltonianSimulation import circ_HS_QSVT, func_HS_QSVT
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation. \
    SymmetricQSPHS import __HS_approx_data, func_LBFGS_QSP_HS

__all__ = [
    'func_Hamiltonian_gen', 'circ_HS_QSVT', 'func_HS_QSVT', '__HS_approx_data', 'func_LBFGS_QSP_HS'
]
