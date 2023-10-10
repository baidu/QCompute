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
Backenc Filter
"""
FileErrorCode = 17

from QCompute.QPlatform import QEnv, BackendName


def filterCloudBackend(env: QEnv):
    if env.backendName is BackendName.CloudBaiduSim2Thunder:
        print('The quantum simulators CloudBaiduSim2Thunder (single instance C++ version) ' +
              'is deprecated from SDK this version [3.3.4]. ' +
              'Any task to the cloud simulator would be redirected to CloudBaiduSim2Water.')
        env.backendName = BackendName.CloudBaiduSim2Water
    elif env.backendName is BackendName.CloudBaiduSim2Lake:
        print('The quantum simulators CloudBaiduSim2Lake (single instance GPU version) ' +
              'is deprecated from SDK this version [3.3.4]. ' +
              'Any task to the cloud simulator would be redirected to CloudBaiduSim2Water.'
              )
        env.backendName = BackendName.CloudBaiduSim2Water