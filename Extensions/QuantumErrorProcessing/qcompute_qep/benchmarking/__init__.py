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

"""__init__ file of the `qcompute_qep.benchmark` module."""

from qcompute_qep.benchmarking.benchmarking import RandomizedBenchmarking, default_prep_circuit, default_meas_circuit
from qcompute_qep.benchmarking.standardrb import StandardRB
from qcompute_qep.benchmarking.unitarityrb import UnitarityRB
from qcompute_qep.benchmarking.xeb import XEB
from qcompute_qep.benchmarking.interleavedrb import InterleavedRB

__all__ = [
    'RandomizedBenchmarking', 'StandardRB', 'UnitarityRB', 'XEB',
    'default_prep_circuit', 'default_meas_circuit',
    'InterleavedRB'
]
