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
Utils Functions
"""
import importlib
import re
from typing import List, Union

import QCompute

_reaesc = re.compile(r'\x1b[^m]*m')

_unierror = re.compile(r'((SV|SL)((\.\d+)+))')


def filterConsoleOutput(text: str):
    """
    Filter control characters in command output

    Example:

    text = '\t\u001b[0;35mbaidu.com\u001b[0m \u001b[0;36m127.0.0.1\u001b[0m'

    ret = FilterConsoleOutput(text)

    print(ret)

    '\tbaidu.com 127.0.0.1'

    :param text: To be filtered string
    :return: Clear string, which doesn't have control characters.
    """

    return _reaesc.sub('', text)


def findUniError(*texts: List[str]) -> Union[str, None]:
    """
    Find Any UniError Code from Inputs

    :param *texts: String array to be find
    :return: The founded code string or None
    """

    for text in texts:
        ret =  _unierror.findall(text)
        if ret:
            return ret[0][0]    # Only the first result would return
    else:
        return None


def loadPythonModule(moduleName: str):
    """
    Load module from file system.

    :param moduleName: Module name
    :return: Module object
    """

    moduleSpec = importlib.util.find_spec(moduleName)
    if moduleSpec is None:
        return None
    module = importlib.util.module_from_spec(moduleSpec)
    moduleSpec.loader.exec_module(module)
    return module


def matchSdkVersion(tagretVersion: str):
    """
    Match sdk version.
    """
    if QCompute.Define.sdkVersion != tagretVersion:
        import warnings

        warnings.warn(
            f'This example({tagretVersion}) '
            f'does not match the correct sdk version({QCompute.Define.sdkVersion}). '
            'Please update the sdk.',
            FutureWarning)
