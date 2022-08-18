# -*- coding: UTF-8 -*-
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
Convert Pauli terms to a matrix
"""

from functools import reduce
from typing import List

import numpy as np


def pauli_terms_to_matrix(pauli_terms: List) -> np.ndarray:
    """Converts Pauli terms to a matrix

    :param pauli_terms: Pauli terms whose matrix is to be computed
    :return: Matrix form of the Pauli terms
    """
    pauli_dict = {'i': np.eye(2) + 0j, 'x': np.array([[0, 1], [1, 0]]) + 0j,
                  'y': np.array([[0, -1j], [1j, 0]]), 'z': np.array([[1, 0], [0, -1]]) + 0j}

    matrices = []
    for coeff, op_str in pauli_terms:
        sub_matrices = [pauli_dict[op] for op in op_str.lower()]
        if len(op_str) == 1:
            matrices.append(coeff * sub_matrices[0])
        else:
            matrices.append(coeff * reduce(np.kron, sub_matrices))

    return sum(matrices)
