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
Mapping To BaiduQPUQian
"""
FileErrorCode = 3

from typing import Dict, List, Optional

from QCompute.OpenModule import ModuleImplement
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBMeasure



class MappingToBaiduQPUQianModule(ModuleImplement):
    """
    Mapping Procedure

    Example:

    env.module(MappingToBaiduQPUQianModule())

    env.serverModule(ServerModule.MappingToBaiduQPUQian, {"disable": True})
    """

    

    def __init__(self, arguments: Optional[Dict[str, bool]] = None):
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.
        """
        super().__init__(arguments)

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program

        :return: mapped procedure
        """
        from QCompute.QPlatform import Error; raise Error.RuntimeError('Not implemented at local sdk')

    
