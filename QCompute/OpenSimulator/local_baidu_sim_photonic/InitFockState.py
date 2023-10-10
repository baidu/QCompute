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
Fock state initialization
"""
FileErrorCode = 23

from enum import unique, IntEnum
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


def initStateDense(n: int) -> numpy.ndarray:
    r"""
    Generate an :math:`N`-qumode initial Fock state,

    :param n: total number of qumodes
    :return fock_state_vector: :math:`N`-dimensional zero vector
    """

    fock_state_vector = numpy.zeros((n, 1), dtype=int)
    return fock_state_vector