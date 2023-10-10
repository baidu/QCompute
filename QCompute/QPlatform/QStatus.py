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
Quantum Status
"""
FileErrorCode = 11

import traceback
from typing import List, Dict, Any

import requests

from QCompute import Define
from QCompute.QPlatform import Error, ModuleErrorCode


def getDeviceStatus(backendList: List[str]) -> Dict[str, Any]:
    """
    Get Quantum Device Status

    Example:

    getDeviceStatus([
        BackendName.CloudBaiduSim2Water,  # Containerization
        BackendName.CloudBaiduSim2Earth,
        BackendName.CloudBaiduSim2Thunder,
        BackendName.CloudBaiduSim2Heaven,
        BackendName.CloudBaiduSim2Wind,
        BackendName.CloudBaiduSim2Lake,  # Containerization
        BackendName.CloudAerAtBD,  # Containerization
        BackendName.CloudBaiduQPUQian,
        BackendName.CloudIoPCAS,
        BackendName.CloudIonAPM,
    ])

    return value:

    {
        'error': 0, # Error code. 0 means ok, maybe has data value; others are failed, must have message value.
        'vendor': '', # Global Error Code.
        'message': '', # Error Message.

        'data': {
            'DeviceName0': {
                'State': 'Busy', # 'Busy', 'Idle', 'Maintaince', 'Up', 'Down'
                'Queue': 0, # Queue length.
                'Data': {
                    'Qubits': 8, # Maximum qubits count.
                    'Timelimit': 60, # Maximum task elapsed time, seconds.
                    'Status': {}, # Upstream device status.
                    'StaticsStatusTime': '2022-08-22T11:38:52.976389+00:00' # Upstream update time.
            },
            'DeviceName1': {}
            'DeviceName2': {}
        }
    }
    """

    req = {
        'backends': backendList
    }
    try:
        res = requests.post(f"{Define.quantumHubAddr}/backends/StatusAll", json=req)
        return res.json()
    except Exception:
        raise Error.NetworkError(traceback.format_exc(), ModuleErrorCode, FileErrorCode, 1)