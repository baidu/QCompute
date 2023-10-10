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
Global Definitions
"""
ModuleErrorCode = 3
FileErrorCode = 3


import os
import sys
from enum import IntEnum, unique
from pathlib import Path

import QCompute

env = 'prod'

"""
Environment

Do not modify by user.

Values: 'prod', 'test'

Used for prod or test environment.
"""
if env == "prod":
    # service address for production
    quantumHubAddr = 'https://quantum-hub.baidu.com/api'
    quantumBucket = 'quantum-task'
    blindCompAddr = 'wss://blindcomp.baidu.com'
else:
    from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented')

sdkVersion = 'Python 3.3.5'
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

Please provide the token before submitting the circuit task to quantum-hub, use:

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

waitTaskRetryTimes = 10
"""
Wait Task Retry Times

Do not modify by user.

Retry count for waittask in case network failed.
"""

waitTaskRetryDelaySeconds = 5
"""
Wait Task Retry Delay Seconds

Do not modify by user.

Retry delay for waittask in case network failed.
"""

outputDirPath = os.environ.get('OUTPUTPATH', None)
if outputDirPath is None:
    outputDirPath = Path('Output').absolute()
    
else:
    # it will caused a `str / str` error by lacking of this
    outputDirPath = Path(outputDirPath).absolute()
"""
Output Dir Path

Do not modify by user.

Will be created, when not exist.
"""

calibrationDirPath = Path('Calibration').absolute()

"""
Calibration Dir Path

Do not modify by user.

Will be created, when not exist.
"""



if 'sphinx' in sys.modules:
    outputDirPath = Path()
else:
    os.makedirs(outputDirPath, mode=0o744, exist_ok=True)

maxSeed = 2147483647
"""
Seed [0, 2147483647]

Do not modify by user.
"""

maxShots = 100000
"""
Shots [1, 100000]

Do not modify by user.
"""

maxNotesLen = 160
"""
NotesLen [1, 160]

Do not modify by user.
"""


@unique
class MeasureFormat(IntEnum):
    """
    Measure output format enum
    """
    Bin = 0
    Hex = Bin + 1
    Dec = Hex + 1  # formatReverseMeasure only