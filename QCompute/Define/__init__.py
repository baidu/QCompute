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
Global Definitions
"""

import os
import sys
from enum import IntEnum, unique
from pathlib import Path

import QCompute
from QCompute.QPlatform import Error

env = 'prod'

"""
Environment

Do not modify by user.

Values: 'prod', 'test'

Used for prod or test environment.
"""
if env == "test":
    raise Error.RuntimeError('Not implemented')
else:
    # service address for production
    quantumHubAddr = 'https://quantum-hub.baidu.com/api'
    quantumBucket = 'quantum-task'

sdkVersion = 'Python 2.0.3'
"""
SDK Version

Do not modify by user.

Used for task submission.
"""

hubToken = os.environ.get('HUBTOKEN', '')
"""
Hub Token

Do not modify directly.

Used for Quantum hub task.

From http://quantum-hub.baidu.com

Token Management -> Creat/View Token

In circuit .py file, use: 

Define.hubToken = 'xxx'
"""

taskSource = os.environ.get('SOURCE', 'PySDK')
"""
Task Source

Do not modify by user.

Values: 'PySDK', 'PyOnline'

Used for distinguish PySDK or PyOnline.
"""

noLocalTask = os.environ.get('NOLOCALTASK', None)
"""
No Local Task

Do not modify by user.

Values: None or Other

Used for PyOnline.
"""

noWaitTask = os.environ.get('NOWAITTASK', None)
"""
No Wait Task

Do not modify by user.

Values: None or Other

Used for PyOnline.
"""

taskInside = os.environ.get('TASKINSIDE', False)
"""
Task Inside

Do not modify by user.

Values: False or True

Used for Inside.
"""

pollInterval = 5
"""
Poll Interval seconds

Do not modify by user.

Used for task check.
"""

waitTaskRetrys = 10
"""
Wait Task Retrys

Do not modify by user.

Retry count for waittask in case network failed.
"""

outputPath = Path('Output').absolute()

"""
Output Path

Do not modify by user.

Will be created, when not exist.
"""
if 'sphinx' in sys.modules:
    outputPath = Path()
else:
    os.makedirs(outputPath, mode=0o744, exist_ok=True)


@unique
class MeasureFormat(IntEnum):
    """
    Measure output format enum
    """
    Bin = 0
    Hex = Bin + 1
