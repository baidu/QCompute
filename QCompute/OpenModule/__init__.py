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
Export the entire directory as a library
"""
from QCompute.Define.Utils import loadPythonModule
from QCompute.QPlatform import Error

ModuleErrorCode = 6
FileErrorCode = 1

from typing import Dict, Any


class ModuleImplement:
    """
    ConvertorImplement
    """
    arguments = None  # Any can serialize to json
    disable = False

    def __init__(self, arguments: Dict[str, Any]) -> None:
        self.arguments = arguments
        if arguments is not None and type(arguments) is dict:
            if 'disable' in arguments:
                self.disable = arguments['disable']

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        return program


def makeModuleObject(name: str, arguments: Dict[str, Any]) -> ModuleImplement:
    # import the module according to the module name
    module = loadPythonModule(f'QCompute.OpenModule.{name}')
    if module is None:
        module = loadPythonModule(f'QCompute.Module.{name}')
    if module is None:
        raise Error.ArgumentError(f'Invalid module => {name}!', ModuleErrorCode, FileErrorCode, 1)

    moduleClass = getattr(module, name)
    return moduleClass(arguments)