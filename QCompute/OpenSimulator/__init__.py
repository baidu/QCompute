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
Quantum executor interface definition
"""
from QCompute.Define import sdkVersion


class QuantumResult:
    """
    The result of experiment
    """

    sdkVersion = sdkVersion
    """
    SDK Version from Define.sdkVersion
    """

    code = 0
    """
    error code
    """

    output = ''
    """
    output results
    """

    shots = 0
    """
    number of shots
    """

    counts = None
    """
    counts for results
    """

    seed = 0
    """
    random seed
    """

    startTimeUtc = ''
    """
    start utc time
    """

    endTimeUtc = ''
    """
    end utc time
    """


class QuantumImplement:
    """
    Implement params for quantum execution.

    Send to the simulator when submitting a task.
    """

    program = None
    """
    Protobuf format of the circuit
    """

    shots = 0
    """
    Number of shots
    """

    backendParam = None
    """
    The parameters of backend
    """

    result = QuantumResult()
    """
    The final result
    """
