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

"""
This is the init file for the "Quantum Error Correction" module.
Please import all classes and functions within this module here,
so that we can use `qcompute_qep.correction.xxx` to call the class or function `xxx` directly.
"""
from Extensions.QuantumErrorProcessing.qcompute_qep.correction.stabilizer import StabilizerCode
from Extensions.QuantumErrorProcessing.qcompute_qep.correction.basic import BasicCode, BitFlipCode, PhaseFlipCode, \
    FourOneTwoCode, FourTwoTwoCode, FiveQubitCode, SteaneCode, ShorCode
from Extensions.QuantumErrorProcessing.qcompute_qep.correction.utils import \
    pauli_list_to_check_matrix, check_matrix_to_standard_form, ColorTable

__all__ = [
    "StabilizerCode",
    "BasicCode",
    "BitFlipCode", "PhaseFlipCode", "FiveQubitCode", "SteaneCode", "ShorCode",
    "FourOneTwoCode", "FourTwoTwoCode",
    "pauli_list_to_check_matrix", "check_matrix_to_standard_form", "ColorTable"
]
