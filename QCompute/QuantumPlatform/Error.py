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
Define Error
"""


class Error(Exception):
    """
    The basic class of the user-defined exceptions
    """

    code = 0

    def __init__(self, msg=None):
        self.message = msg


class ParamError(Error):
    """
    Parameters related error
    """

    code = 100


class NetworkError(Error):
    """
    Network related error
    """

    code = 200


class RuntimeError(Error):
    """
    Runtime related error
    """

    code = 300


class LogicError(Error):
    """
    Logical error
    """

    code = 400


class TokenError(LogicError):
    """
    Token related error
    """

    code = 401
