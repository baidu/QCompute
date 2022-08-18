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

r"""
Module for different quantum circuit backends.
"""

from enum import Enum
import QCompute

__all__ = [
    "mbqc",
    "qcompute",
    "Backend"
]


class _QComputeBackend(Enum):
    r"""QCompute backends.
    """

    LocalBaiduSim2 = QCompute.BackendName.LocalBaiduSim2
    CloudBaiduSim2Water = QCompute.BackendName.CloudBaiduSim2Water
    CloudBaiduSim2Earth = QCompute.BackendName.CloudBaiduSim2Earth
    CloudBaiduSim2Thunder = QCompute.BackendName.CloudBaiduSim2Thunder
    CloudBaiduSim2Heaven = QCompute.BackendName.CloudBaiduSim2Heaven
    CloudBaiduSim2Wind = QCompute.BackendName.CloudBaiduSim2Wind
    CloudBaiduSim2Lake = QCompute.BackendName.CloudBaiduSim2Lake
    CloudAerAtBD = QCompute.BackendName.CloudAerAtBD
    CloudIoPCAS = QCompute.BackendName.CloudIoPCAS
    CloudIonAPM = QCompute.BackendName.CloudIonAPM
    CloudBaiduQPUQian = QCompute.BackendName.CloudBaiduQPUQian


class _MBQCBackend(Enum):
    r"""MBQC backends.
    """

    StateVector = "StateVector"


class Backend:
    r"""Backends for quantum circuit implementation.
    """

    QCompute = _QComputeBackend
    MBQC = _MBQCBackend
