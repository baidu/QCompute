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

import re

_reaesc = re.compile(r'\x1b[^m]*m')


def _filterConsoleOutput(text):
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
