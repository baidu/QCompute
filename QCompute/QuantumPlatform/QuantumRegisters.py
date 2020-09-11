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
Quantum Register
"""


class QuantumRegisters:
    """
    The quantum register dict
    """

    def __init__(self, env):
        """
        The constructor of the QuantumRegister class

        :param env: the related quantum environment
        """

        self.env = env  # the quantum environment related with the quantum register dict
        self.registerDict = {}  # the inner data for quantum register dict

    def __getitem__(self, index):
        """
        Get the quantum register according to the index

        Create the register when it does not exists

        :param index: the quantum register index
        :return: QuantumRegisterStorage
        """

        value = self.registerDict.get(index)
        if value is not None:
            return value
        value = QuantumRegisterStorage(index, self.env)
        self.registerDict[index] = value
        return value


class QuantumRegisterStorage:
    """
    The storage for quantum register
    """

    def __init__(self, index, env):
        """
        The quantum register object needs to know its index and related quantum environment

        :param index: the quantum register index
        :param env: the related quantum environment
        """

        self.index = index
        self.env = env
