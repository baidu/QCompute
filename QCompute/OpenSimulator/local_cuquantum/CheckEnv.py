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
Check Env
"""
FileErrorCode = 27

import ctypes
import os


def __which(pgm):
    """
    locate the executable file associated with the given command by searching it in the path environment variable.

    returns the full path if the command exists.
    otherwise, returns None.
    """
    path = os.getenv('PATH')
    for p in path.split(os.path.pathsep):
        p = os.path.join(p, pgm)
        if os.path.exists(p) and os.access(p, os.X_OK):
            return p
    return None


def __so_exists(fn):
    """
    check whether a so function can be loaded.
    returns True or False.
    """
    try:
        ctypes.cdll.LoadLibrary(fn)
    except OSError:
        return False
    return True


def nvidia_gpu_driver_installed():
    """
    check whether nvidia gpu driver was installed.
    returns True or False.
    """
    return __which('nvidia-smi') is not None


def openmpi_installed():
    """
    check whether openmpi was installed.
    returns True or False.
    """
    return __which('mpirun') is not None


def cuda_11_installed():
    """
    check weather cuda 11 toolkit was installed.
    returns True or False.
    """
    return __so_exists('libcudart.so.11.0')