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
This is the init file for the "Quantum Estimation" module.
We import all classes and functions within this module here,
so that we can use `qcompute_qep.estimation.xxx` to call the class or function `xxx` directly.
"""
from qcompute_qep.estimation.estimation import Estimation
from qcompute_qep.estimation.dfe_state import DFEState
from qcompute_qep.estimation.dfe_process import DFEProcess
from qcompute_qep.estimation.cpe_state import CPEState

__all__ = [
    'Estimation',
    'DFEState', 'DFEProcess',
    'CPEState'
]

