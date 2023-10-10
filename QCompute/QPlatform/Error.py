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
Define Error
"""
FileErrorCode = 3


class Error(Exception):
    """
    The basic class of the user-defined exceptions
    """

    type = 0x00

    def __init__(self, msg: str, module: int = 0, file: int = 0, error: int = 0, vendor: str = None):
        self.message = msg
        if vendor is None:
            self.code = 'QC.{}.{}.{}.{}'.format(self.type, module, file, error)
        else:
            self.code = vendor

    def __str__(self) -> str:
        return '###{}: {}###'.format(self.code, self.message)


class ArgumentError(Error):
    """
    Arguments related error
    """

    type = 0x01


class NetworkError(Error):
    """
    Network related error
    """

    type = 0x02


class RuntimeError(Error):
    """
    Runtime related error
    """

    type = 0x03


class LogicError(Error):
    """
    Logical error
    """

    type = 0x04


class TokenError(Error):
    """
    Token related error
    """

    type = 0x05