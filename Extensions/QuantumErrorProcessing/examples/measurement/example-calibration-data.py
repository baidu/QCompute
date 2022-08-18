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
This is a simple case of using calibration data
"""
import json

import sys
sys.path.append('../..')

from QCompute import Define
from QCompute.Calibration import CalibrationUpdate, CalibrationReadData

# Set the token. You must set your VIP token in order to access the hardware.
Define.hubToken = "Token"

# The quantum machine to be calibrated.
device = 'iopcas'

# Obtain the latest calibration data and store in the local directory.
CalibrationUpdate(device)

# Read all the calibration data in local directory.
ret = CalibrationReadData(device)

# The cloud server will store the latest 10 calibration data.
print(len(ret))

# Get the latest calibration data in dict form.
# print(json.dumps(ret[0].data, indent=2))
# OR
cal_data_latest = ret[0].readData()
print(json.dumps(cal_data_latest, indent=2))

# We can print the information about the calibration data, including the file path and the time of calibration.
for calibrationData in ret:
    print(calibrationData.file)
    print(calibrationData.time)
