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
Calibration service
"""
ModuleErrorCode = 1
FileErrorCode = 1

import json
import os
import re
from datetime import datetime
from os import path
from pathlib import Path
from typing import Dict, Set, List

import requests
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.bos.bos_client import BosClient

from QCompute import Define
from QCompute.QPlatform import Error


def CalibrationUpdate(device: str):
    """
    Calibration Data Update
    """

    # verify permissions, get key list and temporary signature needed for download
    re_sess = requests.session()
    res = re_sess.post(f'{Define.quantumHubAddr}/{device}/calibration/genDownloadSTS', json={'token': Define.hubToken})

    # exception handling
    content = res.json()
    if content['error'] != 0:
        raise Error.RuntimeError(f"Server error {content['error']}: {content['message']} {content['vendor']}", ModuleErrorCode, FileErrorCode, 1)

    # extract content
    config = content['data']
    keys = config['resource']['keys']
    bucket_name = config['resource']['bucket']

    # check the storage directory
    localCalibrationDirPath = Define.calibrationDirPath / device
    localKeysFilePath = localCalibrationDirPath / 'keys.json'
    localDataPath = localCalibrationDirPath / 'Data'
    if not localCalibrationDirPath.exists():
        os.makedirs(localCalibrationDirPath)
    if not localDataPath.exists():
        os.makedirs(localDataPath)

    # load key list from keys.json
    localKeys = None
    if localKeysFilePath.exists():
        localKeysStr = localKeysFilePath.read_text('utf-8')
        localKeys = json.loads(localKeysStr)

    # load online key list
    onlineKeysDict: Dict[str, str] = {}
    for file in keys:
        fileName = path.basename(file)
        onlineKeysDict[fileName] = file

    # load local storage file list
    localKeysSet: Set[str] = set()
    if localKeys:
        for file in localKeys:
            fileName = path.basename(file)
            localKeysSet.add(fileName)

    # delete expired files
    for localFile in localKeysSet:
        if localFile not in onlineKeysDict:
            filePath = localDataPath / localFile
            if (filePath.exists()):
                filePath.unlink()

    # create BosClient
    bosClient = BosClient(
        BceClientConfiguration(
            credentials=BceCredentials(
                str(
                    config['sts']['accessKeyId']),
                str(
                    config['sts']['secretAccessKey'])),
            endpoint='http://bd.bcebos.com',
            security_token=str(
                config['sts']['sessionToken'])))

    # download new files
    for fileName, file in onlineKeysDict.items():
        if fileName not in localKeysSet:
            bosClient.get_object_to_file(bucket_name, file, str(localDataPath / fileName))
    # update keys.json
    localKeysFilePath.write_text(json.dumps(keys), 'utf-8')


class CalibrationData:
    """
    Calibration Data
    """

    def __init__(self, file: Path, timeStr: str):
        self.file = file  # file path

        # convert time
        year = int(timeStr[:4])
        timeStrArray = re.findall(r'.{2}', timeStr[4:])
        timeIntArray = [int(Str) for Str in timeStrArray]
        self.time = datetime(year, *timeIntArray)

        # file data
        self.data: Dict = None

    def readData(self):
        """
        Read Data
        """

        if not self.data:  # if already read, do nothing
            text = self.file.read_text('utf-8')
            self.data = json.loads(text)
        return self.data


def CalibrationReadData(device: str) -> List[CalibrationData]:
    """
    Read Local Calibration Data
    """

    # get path
    localCalibrationDirPath = Define.calibrationDirPath / device
    localKeysFilePath = localCalibrationDirPath / 'keys.json'
    localDataPath = localCalibrationDirPath / 'Data'

    # check keys.json exists
    if not localKeysFilePath.exists():
        raise Error.RuntimeError(f'Local keys file {str(localKeysFilePath)} not exists!', ModuleErrorCode, FileErrorCode, 2)

    # load keys.json
    localKeysStr = localKeysFilePath.read_text('utf-8')
    localKeys = json.loads(localKeysStr)

    # build file information
    ret: List[CalibrationData] = []
    for file in localKeys:
        fileName = path.basename(file)
        ret.append(CalibrationData(localDataPath / fileName, fileName[:14]))
    # sort by data creation time
    ret.sort(key=lambda calibrationData: calibrationData.time, reverse=True)
    return ret