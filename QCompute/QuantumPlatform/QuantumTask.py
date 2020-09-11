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
import os
import time
import traceback
from enum import Enum
from pprint import pprint

import requests

# Configuration of token
# This could be read from configure file or environment variable
from QCompute import Define
from QCompute.Define import quantumHubAddr, quantumBucket, pollInterval, sdkVersion, taskSource
from QCompute.Define import waitTaskRetrys
# the url for cloud service
# SERVICE = "https://8yamgsew2cs2f.cfc-execute.gz.baidubce.com/"
# todo carefully demonstrate the upload logic
# Sign the files by cloud service. Then upload files to the cloud storage,
# submit the file id to cloud service, and finally get the results.
from QCompute.Define.Settings import outputInfo
from QCompute.QuantumPlatform import Error
from baidubce.auth.bce_credentials import BceCredentials
from baidubce.bce_client_configuration import BceClientConfiguration
from baidubce.services.bos.bos_client import BosClient


def invokeBackend(target, params):
    """
    Invoke the Backend Functions
    """

    try:
        ret = requests.post(
            f"{quantumHubAddr}/{target}", json=params).json()
    except Exception:
        raise Error.NetworkError(traceback.format_exc())

    if ret["error"] > 0:
        errCode = ret["error"]
        errMsg = ret["message"]
        if errCode == 401:
            raise Error.TokenError(errMsg)
        else:
            raise Error.LogicError(errMsg)

    return ret["data"]


def _retryWhileNetworkError(func):
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
                print(f"network error for {func.__name__}, retrying, {retryCount}")
                retryCount += 1
        else:
            return func(*args, **kwargs)

    return _func


@_retryWhileNetworkError
def _getSTSToken():
    """
    Get the token to upload the file

    :return:
    """

    if not Define.hubToken:
        raise Error.ParamError("please provide a valid token")

    config = invokeBackend("circuit/genSTS", {"token": Define.hubToken})

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

    return [Define.hubToken, bosClient, config['dest']]


@_retryWhileNetworkError
def _uploadCircuit(file):
    """
    Upload the file
    """

    token, client, dest = _getSTSToken()

    client.put_object_from_file(
        quantumBucket, f'tmp/{dest}', file)

    ret = invokeBackend(
        "circuit/createCircuit",
        {
            "token": token,
            "dest": dest,
            "sdkVersion": sdkVersion,
            "source": taskSource
        })

    if outputInfo:
        pprint(ret)
    return token, ret['circuitId']


@_retryWhileNetworkError
def _createTask(token, circuitId, shots, backend, backendParam=None, modules=None):
    """
    Create a task from the code
    """

    task = {
        "token": token,
        "circuitId": circuitId,
        "taskType": backend,
        "shots": shots,
        "sdkVersion": sdkVersion,
        "source": taskSource,
        "modules": modules,
    }

    if backendParam:
        paramList = []
        for param in backendParam:
            if isinstance(param, Enum):
                paramList.append(param.value)
            else:
                paramList.append(param)
        task['backendParam'] = paramList

    ret = invokeBackend(
        "task/createTask",
        task
    )

    if outputInfo:
        pprint(ret)
    return ret['taskId']


def _downloadToFile(url, localFile):
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
    return os.path.realpath(localFile), total


@_retryWhileNetworkError
def _fetchResult(token, taskId):
    """
    Fetch the result files from the taskId
    """

    params = {"token": token, "taskId": taskId}

    ret = invokeBackend("task/getTaskInfo", params)

    result = ret["result"]

    originUrl = result["originUrl"]
    # originSize = result["originSize"]
    try:
        originFile, downSize = _downloadToFile(originUrl, f"./Output/{taskId}.origin.json")
    except Exception:
        # TODO split the disk write error
        raise Error.NetworkError(traceback.format_exc())
    if outputInfo:
        print(f"Download result success {originFile} size = {downSize}")

    measureUrl = result["measureUrl"]
    # measureSize = result["measureSize"]
    try:
        measureFile, downSize = _downloadToFile(measureUrl, f"./Output/{taskId}.measure.json")
    except Exception:
        # TODO split the disk write error
        raise Error.NetworkError(traceback.format_exc())
    if outputInfo:
        print(f"Download result success {measureFile} size = {downSize}")
    # pprint(result)

    return originFile, measureFile


def _fetchMeasureResult(taskId):
    """
    Dump the measurement content of the file from taskId
    """

    localFile = f"./Output/{taskId}.measure.json"
    if os.path.exists(localFile):
        with open(localFile, "rb") as fObj:
            data = json.loads(fObj.read())
            return data
    else:
        return None


@_retryWhileNetworkError
def _waitTask(token, taskId, fetchMeasure=False, downloadResult=True):
    """
    Wait for a task from the taskId
    """

    task = {
        "token": token,
        "taskId": taskId
    }

    stepStatus = "waiting"
    while True:
        try:
            time.sleep(pollInterval)
            ret = invokeBackend("task/checkTask", task)
            if ret["status"] in ("success", "failed", "manual_term"):
                if outputInfo:
                    print(f"status changed {stepStatus} => {ret['status']}")
                stepStatus = ret["status"]
                result = {"status": ret["status"]}

                if ret["status"] == "success" and "originUrl" in ret.get("result", {}):
                    if downloadResult:
                        originFile, measureFile = _fetchResult(token, taskId)
                        result["origin"] = originFile
                        if fetchMeasure:
                            result["counts"] = _fetchMeasureResult(taskId)
                        else:
                            result["measure"] = measureFile
                    break
                elif ret["status"] == "failed":
                    result = ret["reason"]
                    break
                elif ret["status"] == "manual_term":
                    break
                else:
                    # go on loop
                    # pprint(ret)
                    pass
            else:
                if ret["status"] == stepStatus:
                    continue

                if outputInfo:
                    print(f"status changed {stepStatus} => {ret['status']}")
                stepStatus = ret["status"]

        except Error.Error as err:
            raise err

        except Exception:
            raise Error.RuntimeError(traceback.format_exc())

    return result
