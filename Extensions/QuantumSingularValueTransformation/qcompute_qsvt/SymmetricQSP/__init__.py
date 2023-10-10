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
__init__ file of the `qcompute_qsvt.SymmetricQSP` module.
"""

from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.SymmetricQSP.SymmetricQSPExternal import \
    __func_Wx_map, __func_L, __func_gradL, __func_LBFGS_QSP, \
    __func_LBFGS_QSP_backtracking, __func_LBFGS_QSP_scipy, __func_LBFGS_QSP_interpolation
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.SymmetricQSP.SymmetricQSPInternalPy import \
    __func_expiZ_map, func_symQSP_A_map, func_symQSP_gradA_map

__all__ = [
    '__func_Wx_map', '__func_L', '__func_gradL', '__func_LBFGS_QSP',
    '__func_LBFGS_QSP_backtracking', '__func_LBFGS_QSP_scipy', '__func_LBFGS_QSP_interpolation',
    '__func_expiZ_map', 'func_symQSP_A_map', 'func_symQSP_gradA_map'
]
