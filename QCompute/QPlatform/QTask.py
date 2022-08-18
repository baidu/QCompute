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
Quantum Task
"""
import base64
import hashlib
import json
import sys
import time
import traceback
from datetime import datetime
from enum import Enum, IntEnum, unique
from pathlib import Path
from typing import Callable, Tuple, Dict, Union, Any, List, Optional

import requests
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.http import bce_http_client
from baidubce.services.bos.bos_client import BosClient

# Configuration of token
# This could be read from configure file or environment variable
from QCompute import Define
# the url for cloud service
# SERVICE = "https://8yamgsew2cs2f.cfc-execute.gz.baidubce.com/"
# todo carefully demonstrate the upload logic
# Sign the files by cloud service. Then upload files to the cloud storage,
# submit the file id to cloud service, and finally get the results.
from QCompute.Define import Settings
from QCompute.QPlatform import Error, ModuleErrorCode

FileErrorCode = 7

_bosClient = None  # type: BceClientConfiguration


def _retryWhileNetworkError(func: Callable) -> Callable:
    """
    The decorator for retrying function when network failed
    """

    def _func(*args, **kwargs):
        retryCount = 0
        ret = None
        
        lastError = None  # type: Exception
        while retryCount < Define.waitTaskRetryTimes:
            try:
                ret = func(*args, **kwargs)
                lastError = None
                break

            except (Error.NetworkError, requests.RequestException) as e:
                # retry if that's a network related error
                # other errors will be raised
                retryCount += 1
                print(f'Network error for {func.__name__}, {retryCount} retrying to connect...')
                lastError = e
                time.sleep(Define.waitTaskRetryDelaySeconds)
            except bce_http_client.BceHttpClientError as e:
                retryCount += _bosClient.config.retry_policy.max_error_retry
                print(f'Network error for putObject, {retryCount} retrying to connect...')
                lastError = e
                time.sleep(Define.waitTaskRetryDelaySeconds)
        # else:
        #     ret = func(*args, **kwargs)

        if retryCount > 0:
            

            if lastError is None:
                print(f'Successfully reconnect to {func.__name__}')
            else:
                raise lastError
        return ret

    return _func


def _invokeBackend(target: str, params: object) -> Dict:
    """
    Invoke the Backend Functions
    """

    try:
        ret = requests.post(
            f"{Define.quantumHubAddr}/{target}", json=params).json()
    except Exception:
        raise Error.NetworkError(traceback.format_exc(), ModuleErrorCode, FileErrorCode, 1)

    if ret['error'] > 0:
        errCode = ret['error']
        errMsg = ret.get('message', '')
        vendor = ret.get('vendor', '')
        if errCode == 401:
            raise Error.TokenError(errMsg, ModuleErrorCode, FileErrorCode, 2)
        else:
            raise Error.LogicError(f'errCode: {errCode}; errMsg: {errMsg}; vendor: {vendor}', ModuleErrorCode,
                                   FileErrorCode, 3)

    return ret["data"]


@_retryWhileNetworkError
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


@_retryWhileNetworkError
def _downloadToJson(url: str) -> Tuple[Dict[str, any], int]:
    """
    Download from a url to a local file
    """
    with requests.get(url, stream=True) as req:
        req.raise_for_status()
        return req.json(), len(req.content)


class QTask:
    """
    Quantum Task
    """

    def __init__(self):
        if not Define.hubToken:
            raise Error.ArgumentError('Please provide a valid token', ModuleErrorCode, FileErrorCode, 4)

        self.circuitId = -1
        self.taskId = -1
        self.originFile = Path()
        self.measureFile = Path()

    def uploadCircuit(self, buf: bytes) -> None:
        """
        Upload the file
        """
        md5hash = hashlib.md5(buf)
        md5Buf = md5hash.digest()
        md5B64 = base64.b64encode(md5Buf)

        client, dest = self._getSTSToken()
        self._putObject(client, dest, buf, md5B64)
        ret = self._createCircuit(dest)

        self.circuitId = ret['circuitId']

    @_retryWhileNetworkError
    def _getSTSToken(self) -> Tuple[BosClient, str]:
        """
        Get the token to upload the file

        :return:
        """
        config = _invokeBackend("circuit/genSTS", {"token": Define.hubToken})

        global _bosClient
        _bosClient = BosClient(
            BceClientConfiguration(
                credentials=BceCredentials(
                    str(
                        config['accessKeyId']),
                    str(
                        config['secretAccessKey'])),
                endpoint='http://bd.bcebos.com',
                security_token=str(
                    config['sessionToken'])))

        return _bosClient, config['dest']

    @_retryWhileNetworkError
    def _putObject(self, client: BosClient, dest: str, buf: bytes, md5B64: str):
        client.put_object(
            Define.quantumBucket, f'tmp/{dest}', buf, len(buf), md5B64)

    @_retryWhileNetworkError
    def _createCircuit(self, dest: str):
        ret = _invokeBackend(
            "circuit/createCircuit",
            {
                "token": Define.hubToken,
                "dest": dest,
                "sdkVersion": Define.sdkVersion,
                "source": Define.taskSource
            }
        )
        return ret

    @_retryWhileNetworkError
    def createCircuitTask(self, shots: int, backend: str, qbits: int, backendParam: List[Union[str, Enum]] = None,
                          modules: List[Tuple[str, Any]] = None, notes: str = None,
                          debug: Optional[str] = None) -> None:
        """
        Create a task from the code

        debug: None/'shell'/'dump'
        """

        if notes is not None:
            if len(notes) <= 1:
                notes = None
            elif len(notes) > Define.maxNotesLen:
                print(f'Notes len warning, {len(notes)}/{Define.maxNotesLen}(current/max).')
                notes = notes[:Define.maxNotesLen]

        task = {
            "token": Define.hubToken,
            "circuitId": self.circuitId,
            "taskType": backend,
            "shots": shots,
            "sdkVersion": Define.sdkVersion,
            "source": Define.taskSource,
            "modules": modules,
            "qbits": qbits,
        }

        if notes is not None:
            task['notes'] = notes
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
    def createBlindTask(self, backend: str, params) -> None:
        """
        Create a blind task
        """

        task = {
            "token": Define.hubToken,
            "service": backend,
            "params": params,
            "sdkVersion": Define.sdkVersion,
            "source": Define.taskSource,
        }

        ret = _invokeBackend(
            "task/createBlind",
            task
        )

        self.taskId = ret['taskId']
        self.taskToken = ret['taskToken']

    

    def _fetchResult(self) -> None:
        """
        Fetch the result files from the taskId
        """

        ret = _invokeBackend("task/getTaskInfo", {"token": Define.hubToken, "taskId": self.taskId})
        result = ret["result"]
        originUrl = result["originUrl"]
        # originSize = result["originSize"]
        try:
            if not Settings.cloudTaskDoNotWriteFile:
                self.originFile, downSize = _downloadToFile(originUrl,
                                                            Define.outputDirPath / f"remote.{self.taskId}.origin.json")
            else:
                self.originJson, downSize = _downloadToJson(originUrl)
        except Exception:
            # TODO split the disk write error
            raise Error.NetworkError(traceback.format_exc(), ModuleErrorCode, FileErrorCode, 5)
        if Settings.outputInfo:
            print(f'Download origin success {self.originFile} size = {downSize}')

        measureUrl = result["measureUrl"]
        # measureSize = result["measureSize"]
        try:
            if not Settings.cloudTaskDoNotWriteFile:
                self.measureFile, downSize = _downloadToFile(measureUrl,
                                                             Define.outputDirPath / f"remote.{self.taskId}.measure.json")
            else:
                self.measureJson, downSize = _downloadToJson(measureUrl)
        except Exception:
            # TODO split the disk write error
            raise Error.NetworkError(traceback.format_exc(), ModuleErrorCode, FileErrorCode, 6)
        if Settings.outputInfo:
            print(f'Download measure success {self.measureFile} size = {downSize}')

    def _fetchOriginResult(self) -> Optional[Dict]:
        """
        Dump the measurement content of the file from taskId
        """

        localFile = Define.outputDirPath / f'remote.{self.taskId}.origin.json'
        if localFile.exists():
            text = localFile.read_text(encoding='utf-8')
            return json.loads(text)
        else:
            return None

    def _fetchMeasureResult(self) -> Optional[Dict]:
        """
        Dump the measurement content of the file from taskId
        """

        localFile = Define.outputDirPath / f'remote.{self.taskId}.measure.json'
        if localFile.exists():
            text = localFile.read_text(encoding='utf-8')
            return json.loads(text)
        else:
            return None

    @_retryWhileNetworkError
    def waitCircuitTask(self, fetchMeasure: bool = False, downloadResult: bool = True) -> Dict:
        """
        Wait for a task from the taskId
        """
        if Settings.outputInfo:
            print(f'Task {self.taskId} is running, please wait...')

        task = {
            "token": Define.hubToken,
            "taskId": self.taskId
        }

        stepStatus = _Status.waiting
        while True:
            try:
                time.sleep(Define.pollInterval)
                ret = _invokeBackend('task/checkTask', task)
                newStatus = _Status[ret["status"]]
                stepStatusName = _Status(stepStatus).name
                if newStatus > 0:
                    if Settings.outputInfo:
                        print(f'status changed {stepStatusName} => {ret["status"]}')
                    stepStatus = newStatus
                    result = {"taskId": self.taskId, "status": ret["status"]}

                    if newStatus == _Status.success and "originUrl" in ret.get("result", {}):
                        if downloadResult:
                            self._fetchResult()
                            if not Settings.cloudTaskDoNotWriteFile:
                                result["origin"] = str(self.originFile)
                                originResult = self._fetchOriginResult()
                            else:
                                originResult = self.originJson

                            result["moduleList"] = originResult["moduleList"]

                            if fetchMeasure:
                                if not Settings.cloudTaskDoNotWriteFile:
                                    result["counts"] = self._fetchMeasureResult()
                                else:
                                    result["counts"] = self.measureJson
                            elif not Settings.cloudTaskDoNotWriteFile:
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

                    if Settings.outputInfo:
                        print(f'status changed {stepStatusName} => {ret["status"]}')
                    stepStatus = newStatus

            except Error.Error as err:
                raise err

            except Exception:
                raise Error.RuntimeError(traceback.format_exc(), ModuleErrorCode, FileErrorCode, 7)

        return result

    @_retryWhileNetworkError
    def waitBlindTask(self, checkInterval) -> Union[Tuple[str, str], None]:
        """
        Wait for a task from the taskId
        """

        task = {
            "token": Define.hubToken,
            "taskId": self.taskId
        }

        stepStatus = _Status.waiting
        while True:
            try:
                time.sleep(checkInterval)
                ret = _invokeBackend('task/checkTask', task)
                newStatus = _Status[ret["status"]]
                stepStatusName = _Status(stepStatus).name

                if newStatus > 0:
                    if newStatus in (
                            _Status.success,
                            _Status.failed,
                            _Status.manual_terminate):
                        
                        print(f'resource unavailable\nDEBUG info: {self.taskId}, {ret}', file=sys.stderr)
                        return
                    else:
                        
                        return str(self.taskId), self.taskToken
                else:
                    continue

            except Error.Error as err:
                raise err

            except Exception:
                raise Error.RuntimeError(traceback.format_exc(), ModuleErrorCode, FileErrorCode, 7)


@unique
class _Status(IntEnum):
    waiting = 0
    executing = 1
    success = 2
    failed = 3
    manual_terminate = 4
