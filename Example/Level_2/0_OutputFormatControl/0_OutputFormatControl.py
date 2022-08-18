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
This is a simple case of using formatReverseMeasure.
"""

from QCompute import *
from QCompute.Define import MeasureFormat
from QCompute.QPlatform.Processor.PostProcessor import formatReverseMeasure
from QCompute.Define import Settings
Settings.outputInfo = False

env = QEnv()
env.backend(BackendName.LocalBaiduSim2)

q = env.Q.createList(3)

X(q[0])
CX(q[0], q[1])
MeasureZ(*env.Q.toListPair())

# Get the calculation results and convert them
binResult = env.commit(1024, fetchMeasure=True)['counts']
decResult = formatReverseMeasure(binResult, 3, reverse=False, mFormat=MeasureFormat.Dec)
hexResult = formatReverseMeasure(binResult, 3, reverse=False, mFormat=MeasureFormat.Hex)
binResult_re = formatReverseMeasure(binResult, 3, reverse=True, mFormat=MeasureFormat.Bin)
decResult_re = formatReverseMeasure(binResult, 3, reverse=True, mFormat=MeasureFormat.Dec)
hexResult_re = formatReverseMeasure(binResult, 3, reverse=True, mFormat=MeasureFormat.Hex)

# Print the original calculation results
print(binResult)

# Print the result in decimal, big-endian
print(decResult)

# Print the result in hexadecimal, big-endian
print(hexResult)

# Print the result in binary, little-endian
print(binResult_re)

# Print the result in decimal, little-endian
print(decResult_re)

# Print the result in hexadecimal, little-endian
print(hexResult_re)
