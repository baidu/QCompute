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
from QCompute.QuantumPlatform.Error import ParamError, NetworkError
from QCompute.Test.PostInstall.CloudFullTest import cloudFullTest
from QCompute.Test.PostInstall.LocalGateTest import localGateTest


def testAll():
    """
    Post Install Test All
    """
    Settings.outputInfo = False
    localGateTest()
    print('Local test successed.')

    print('Please provide a token:')
    Define.hubToken = input().replace('"', '').replace("'", '')
    if Define.hubToken == '':
        print('Cloud test cancelled.')
        return

    try:
        cloudFullTest()
    except:
        [_, _, tb] = sys.exc_info()
        lastTb = traceback.extract_tb(tb)[-1]
        if lastTb.name == '_getSTSToken':
            raise ParamError('Get STSToken from Quantum Cloud Error.')
        elif lastTb.name == '_uploadCircuit':
            raise NetworkError('Create Circuit on Quantum Cloud Error.')
        elif lastTb.name == 'send_request':
            raise NetworkError('Upload Circuit to Baidu BOS Error.')
        else:
            raise
    print('Cloud test successed.')


if __name__ == '__main__':
    testAll()
