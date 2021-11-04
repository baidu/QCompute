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
Quantum Task
"""
import json
import time
import traceback
from enum import Enum, IntEnum, unique
from pathlib import Path
from typing import Callable, Tuple, Dict, Union, Any, List, Optional

import requests
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.bos.bos_client import BosClient

# Configuration of token
# This could be read from configure file or environment variable
from QCompute import Define
from QCompute.Define import quantumHubAddr, quantumBucket, pollInterval, sdkVersion, taskSource, outputPath
from QCompute.Define import waitTaskRetrys
# the url for cloud service
# SERVICE = "https://8yamgsew2cs2f.cfc-execute.gz.baidubce.com/"
# todo carefully demonstrate the upload logic
# Sign the files by cloud service. Then upload files to the cloud storage,
# submit the file id to cloud service, and finally get the results.
from QCompute.Define.Settings import outputInfo
from QCompute.QPlatform import Error, ModuleErrorCode

FileErrorCode = 7


def _invokeBackend(target: str, params: object) -> Dict:
    """
    Invoke the Backend Functions
    """

    try:
        ret = requests.post(
            f"{quantumHubAddr}/{target}", json=params).json()
    except Exception:
        raise Error.NetworkError(traceback.format_exc(), ModuleErrorCode, FileErrorCode, 1)

    if ret["error"] > 0:
        errCode = ret["error"]
        errMsg = ret["message"]
        if errCode == 401:
            raise Error.TokenError(errMsg, ModuleErrorCode, FileErrorCode, 2)
        else:
            raise Error.LogicError(errMsg, ModuleErrorCode, FileErrorCode, 3)

    return ret["data"]


def _retryWhileNetworkError(func: Callable) -> Callable:
    """
    The decorator for retrying function when network failed
    """

    def _func(*args, **kwargs):
        retryCount = 0
        while retryCount < waitTaskRetrys:
            try:
                return func(*args, **kwargs)
            except Error.NetworkError:
                # retry if that's a network related error
                # other errors will be raised
                print(f'Network error for {func.__name__}, retrying, {retryCount}')
                retryCount += 1
        else:
            return func(*args, **kwargs)

    return _func


@_retryWhileNetworkError
def _getSTSToken() -> Tuple[str, BosClient, str]:
    """
    Get the token to upload the file

    :return:
    """

    if not Define.hubToken:
        raise Error.ArgumentError('Please provide a valid token', ModuleErrorCode, FileErrorCode, 4)

    config = _invokeBackend("circuit/genSTS", {"token": Define.hubToken})

    bosClient = BosClient(
        BceClientConfiguration(
            credentials=BceCredentials(
                str(
                    config['accessKeyId']),
                str(
                    config['secretAccessKey'])),
            endpoint='http://bd.bcebos.com',
            security_token=str(
                config['sessionToken'])))

    return Define.hubToken, bosClient, config['dest']


def _downloadToFile(url: str, localFile: Path) -> Tuple[Path, int]:
    """
    Download from a url to a local file
    """

    total = 0
    with requests.get(url, stream=True) as req:
        req.raise_for_status()
        with open(localFile, 'wb') as fObj:
            for chunk in req.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    fObj.write(chunk)
                    total += len(chunk)
                    # f.flush()
    return localFile, total


class QTask:
    """
    Quantum Task
    """

    def __init__(self):
        self.token = ''
        self.circuitId = -1
        self.taskId = -1
        self.originFile = Path()
        self.measureFile = Path()

    @_retryWhileNetworkError
    def uploadCircuit(self, file: str) -> None:
        """
        Upload the file
        """

        self.token, client, dest = _getSTSToken()

        client.put_object_from_file(
            quantumBucket, f'tmp/{dest}', file)

        ret = _invokeBackend(
            "circuit/createCircuit",
            {
                "token": self.token,
                "dest": dest,
                "sdkVersion": sdkVersion,
                "source": taskSource
            }
        )

        self.circuitId = ret['circuitId']

    @_retryWhileNetworkError
    def createCircuitTask(self, shots: int, backend: str, backendParam: List[Union[str, Enum]] = None,
                          modules: List[Tuple[str, Any]] = None, debug: Optional[str] = None) -> None:
        """
        Create a task from the code

        debug: None/'shell'/'dump'
        """

        task = {
            "token": self.token,
            "circuitId": self.circuitId,
            "taskType": backend,
            "shots": shots,
            "sdkVersion": sdkVersion,
            "source": taskSource,
            "modules": modules,
        }

        if debug:
            task['debug'] = debug

        if backendParam:
            paramList = []
            for param in backendParam:
                if type(param) is str:
                    paramList.append(param)
                else:
                    paramList.append(param.value)
            task['backendParam'] = paramList

        ret = _invokeBackend(
            "task/createTask",
            task
        )

        self.taskId = ret['taskId']

    

    @_retryWhileNetworkError
    def _fetchResult(self) -> None:
        """
        Fetch the result files from the taskId
        """

        ret = _invokeBackend("task/getTaskInfo", {"token": self.token, "taskId": self.taskId})
        result = ret["result"]
        originUrl = result["originUrl"]
        # originSize = result["originSize"]
        try:
            self.originFile, downSize = _downloadToFile(originUrl, outputPath / f"remote.{self.taskId}.origin.json")
        except Exception:
            # TODO split the disk write error
            raise Error.NetworkError(traceback.format_exc(), ModuleErrorCode, FileErrorCode, 5)
        if outputInfo:
            print(f'Download origin success {self.originFile} size = {downSize}')

        measureUrl = result["measureUrl"]
        # measureSize = result["measureSize"]
        try:
            self.measureFile, downSize = _downloadToFile(measureUrl, outputPath / f"remote.{self.taskId}.measure.json")
        except Exception:
            # TODO split the disk write error
            raise Error.NetworkError(traceback.format_exc(), ModuleErrorCode, FileErrorCode, 6)
        if outputInfo:
            print(f'Download measure success {self.measureFile} size = {downSize}')

    def _fetchOriginResult(self) -> Optional[Dict]:
        """
        Dump the measurement content of the file from taskId
        """

        localFile = outputPath / f'remote.{self.taskId}.origin.json'
        if localFile.exists():
            with open(localFile, "rb") as fObj:
                data = json.loads(fObj.read())
                return data
        else:
            return None

    def _fetchMeasureResult(self) -> Optional[Dict]:
        """
        Dump the measurement content of the file from taskId
        """

        localFile = outputPath / f'remote.{self.taskId}.measure.json'
        if localFile.exists():
            with open(localFile, "rb") as fObj:
                data = json.loads(fObj.read())
                return data
        else:
            return None

    @_retryWhileNetworkError
    def wait(self, fetchMeasure: bool = False, downloadResult: bool = True) -> Dict:
        """
        Wait for a task from the taskId
        """
        if outputInfo:
            print(f'Task {self.taskId} is running, please wait...')

        task = {
            "token": self.token,
            "taskId": self.taskId
        }

        stepStatus = _Status.waiting
        while True:
            try:
                time.sleep(pollInterval)
                ret = _invokeBackend('task/checkTask', task)
                newStatus = _Status[ret["status"]]
                stepStatusName = _Status(stepStatus).name
                if newStatus > 0:
                    if outputInfo:
                        print(f'status changed {stepStatusName} => {ret["status"]}')
                    stepStatus = newStatus
                    result = {"taskId": self.taskId, "status": ret["status"]}

                    if newStatus == _Status.success and "originUrl" in ret.get("result", {}):
                        if downloadResult:
                            self._fetchResult()
                            result["origin"] = str(self.originFile)

                            originResult = self._fetchOriginResult()
                            result["moduleList"] = originResult["moduleList"]

                            if fetchMeasure:
                                result["counts"] = self._fetchMeasureResult()
                            else:
                                result["measure"] = str(self.measureFile)
                        break
                    elif newStatus == _Status.failed:
                        result = ret["reason"]
                        break
                    elif newStatus == _Status.manual_terminate:
                        break
                    else:
                        # go on loop
                        pass
                else:
                    if newStatus == stepStatus:
                        continue

                    if outputInfo:
                        print(f'status changed {stepStatusName} => {ret["status"]}')
                    stepStatus = newStatus

            except Error.Error as err:
                raise err

            except Exception:
                raise Error.RuntimeError(traceback.format_exc(), ModuleErrorCode, FileErrorCode, 7)

        return result


@unique
class _Status(IntEnum):
    waiting = 0
    executing = 1
    success = 2
    failed = 3
    manual_terminate = 4
