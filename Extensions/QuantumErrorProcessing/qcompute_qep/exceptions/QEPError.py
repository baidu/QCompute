#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
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
Supported Exceptions in the `qcompute_qep` module.
We classify the exceptions raised in `qcompute_qep` into different types for better recognition.
"""


class QEPError(Exception):
    r"""Basic exception class of the QEP project.

    Each error message is packed into a dictionary containing the following items:
    1. code number
    2. error message
    Example:
        error = {'code': 0,
                'message': 'Error occurred within the qcompute_qep project.'}
    """
    error = None

    def __init__(self, message: str = None):
        r"""Exceptions raised due to errors in result output.
        """
        self.error = dict()
        self.error['code'] = 0
        self.error['message'] = message
        super().__init__(message)

    def __str__(self):
        return 'Error Code {}: {}'.format(self.error['code'], self.error['message'])


class ArgumentError(QEPError):
    r"""Argument error exception class of the QEP project.

    Each error message is packed into a dictionary containing the following items:
    1. code number
    2. error message
    Example:
        error = {'code': 100,
                'message': 'Argument error description.'}
    """
    def __init__(self, message: str = None):
        """
        Exceptions raised due to errors in result output.
        """
        super().__init__(message)
        self.error['code'] = 100
        self.error['message'] = message

    def __str__(self):
        return 'Error Code {}: {}'.format(self.error['code'], self.error['message'])
