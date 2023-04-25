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
Gaussian state initialization
"""

from enum import unique, IntEnum
from typing import List
import numpy




@unique
class MatrixType(IntEnum):
    """
    Define matrix type
    """

    Dense = 0
    


def initState(matrixType: MatrixType, n: int):
    """
    Choose matrix type
    """

    if matrixType == MatrixType.Dense:
        return initStateDense(n)
    else:
        assert False


def initStateDense(n: int) -> List[numpy.ndarray]:
    r"""
    Generate the first and second moments of an :math:`N`-qumode gaussian state

    :param n: total number of qumodes
    :return state_list: the first and second moments of initial gaussian state.
            They are :math:`2N`-dimensional zero vector and :math:`2N \times 2N` identity matrix, respectively.
    """

    state_fir_mom = numpy.zeros((2 * n, 1), dtype=float)
    state_sec_mom = numpy.eye(2 * n, dtype=float)
    state_list = [state_fir_mom, state_sec_mom]
    return state_list
