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
Statevector Initialization Process
"""

from enum import unique, IntEnum
from typing import Union

import numpy



from QCompute.QPlatform import Error


@unique
class MatrixType(IntEnum):
    """
    
    DEFINE the matrix type here.
    """

    Dense = 0
    


def initState_1_0(matrixType: MatrixType, n: int) -> Union[numpy.ndarray, 'COO']:
    """
    Generate an n-qubit state
    """

    if matrixType == MatrixType.Dense:
        return initStateDense_1_0(n)
    else:
        raise Error.RuntimeError('Not implemented')


def initStateDense_1_0(n: int) -> numpy.ndarray:
    """
    Generate an n-qubit state
    :param n: number of qubits
    :return: tensor of the state
    """

    state = numpy.zeros([2] * n, complex)
    state.reshape(-1)[0] = 1.0
    return state



