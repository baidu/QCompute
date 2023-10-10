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
Post Install Test
"""
import sys
import traceback

from QCompute import Define
from QCompute.Define import Settings
from QCompute.QPlatform import Error
from Test.BaseSettings import inputHubToken
from Test.PostInstall.CloudFullTest import cloudFullTest
from Test.PostInstall.LocalGateTest import localGateTest
from Test.PostInstall.LocalNoiseTest import localNoiseTest
from Test.Photonic.GaussianFullTest import GaussianFullTest
from Test.Photonic.FockFullTest import FockFullTest
from QCompute import *

inputHubToken()
Settings.outputInfo = False


def test_local() -> None:
    """
    Local test
    """
    localGateTest()
    print('Local test successed.')
    
    localNoiseTest()
    print('Local noise test successed')

    GaussianFullTest(BackendName.LocalBaiduSimPhotonic)
    FockFullTest(BackendName.LocalBaiduSimPhotonic)
    print('Local photonic quantum simulator test successed')

def test_cloud() -> None:
    """
    Cloud test
    """
    if Define.hubToken == '':
        print('Cloud test closed because there is no token entered.')
        return

    try:
        cloudFullTest()
        # GaussianFullTest(BackendName.CloudBaiduSim2Fire)
        # FockFullTest(BackendName.CloudBaiduSim2Fire)
    except:
        [_, _, tb] = sys.exc_info()
        lastTb = traceback.extract_tb(tb)[-1]
        if lastTb.name == 'getSTSToken':
            raise Error.ArgumentError('Get STSToken from Quantum Cloud Error.')
        elif lastTb.name == 'uploadCircuit':
            raise Error.NetworkError('Create Circuit on Quantum Cloud Error.')
        elif lastTb.name == 'send_request':
            raise Error.NetworkError('Upload Circuit to Baidu BOS Error.')
        else:
            raise
    print('Cloud test successed.')


if __name__ == '__main__':
    test_local()
    test_cloud()
