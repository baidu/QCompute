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
Procedure Parameter Pool
"""
from typing import Dict


class ProcedureParameterStorage:
    """
    The storage for procedure parameter
    """

    def __init__(self, index: int) -> None:
        """
        The quantum param object needs to know its index.
        :param index: the quantum register index
        """

        self.index = index


class ProcedureParameterPool:
    """
    The procedure parameter dict
    """

    def __init__(self) -> None:
        """
        The constructor of the ProcedureParameterPool class
        """

        # the inner data for procedure params dict
        self.parameterMap = {}  # type: Dict[int,ProcedureParameterStorage]

    def __getitem__(self, index: int) -> ProcedureParameterStorage:
        return self._get(index)

    def __call__(self, index: int) -> ProcedureParameterStorage:
        return self._get(index)

    def _get(self, index: int) -> ProcedureParameterStorage:
        """
        Get the procedure params according to the index.

        Create the register when it does not exist.

        :param index:
        :return: ProcedureParamStorage
        :return: ProcedureParamStorage
        """

        value = self.parameterMap.get(index)
        if value is None:
            value = ProcedureParameterStorage(index)
            self.parameterMap[index] = value
        return value
