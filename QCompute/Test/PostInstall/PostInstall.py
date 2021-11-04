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
Post Install Test
"""
import sys
import traceback

from QCompute import Define
from QCompute.Define import Settings
from QCompute.QPlatform import Error
from QCompute.Test import ModuleErrorCode
from QCompute.Test.PostInstall.CloudFullTest import cloudFullTest
from QCompute.Test.PostInstall.LocalGateTest import localGateTest

FileErrorCode = 1


def testAll():
    """
    Post Install Test All
    """
    Settings.outputInfo = False
    localGateTest()
    print('Local test successed.')

    print('Please provide a token:')
    Define.hubToken = input()
    if Define.hubToken == '':
        print('Cloud test cancelled.')
        return

    try:
        cloudFullTest()
    except:
        [_, _, tb] = sys.exc_info()
        lastTb = traceback.extract_tb(tb)[-1]
        if lastTb.name == 'getSTSToken':
            raise Error.ArgumentError('Get STSToken from Quantum Cloud Error.', ModuleErrorCode, FileErrorCode, 1)
        elif lastTb.name == 'uploadCircuit':
            raise Error.NetworkError('Create Circuit on Quantum Cloud Error.', ModuleErrorCode, FileErrorCode, 2)
        elif lastTb.name == 'send_request':
            raise Error.NetworkError('Upload Circuit to Baidu BOS Error.', ModuleErrorCode, FileErrorCode, 3)
        else:
            raise
    print('Cloud test successed.')


if __name__ == '__main__':
    testAll()
