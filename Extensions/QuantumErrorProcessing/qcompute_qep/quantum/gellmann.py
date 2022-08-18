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
This file aims to collect definitions and functions related to the Gell-Mann basis.
"""
from __future__ import annotations

import numpy as np
from math import log
from typing import List, Union
from scipy import sparse
import itertools
import random

from qcompute_qep.exceptions.QEPError import ArgumentError
from qcompute_qep.quantum.channel import QuantumChannel
from qcompute_qep.utils.linalg import tensor, dagger

# The normalized two-qubit Gell-Mann basis.
GELL_MANN_BASIS = {"S12": np.array([[0, 1, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0], ]).astype(complex),
                   "S13": np.array([[0, 0, 1, 0],
                                    [0, 0, 0, 0],
                                    [1, 0, 0, 0],
                                    [0, 0, 0, 0], ]).astype(complex),
                   "S14": np.array([[0, 0, 0, 1],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [1, 0, 0, 0], ]).astype(complex),
                   "S23": np.array([[0, 0, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 0, 0], ]).astype(complex),
                   "S24": np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 0, 0],
                                    [0, 1, 0, 0], ]).astype(complex),
                   "S34": np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 1],
                                    [0, 0, 1, 0], ]).astype(complex),
                   "A12": np.array([[0, -1j, 0, 0],
                                    [1j, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0], ]).astype(complex),
                   "A13": np.array([[0, 0, -1j, 0],
                                    [0, 0, 0, 0],
                                    [1j, 0, 0, 0],
                                    [0, 0, 0, 0], ]).astype(complex),
                   "A14": np.array([[0, 0, 0, -1j],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [1j, 0, 0, 0], ]).astype(complex),
                   "A23": np.array([[0, 0, 0, 0],
                                    [0, 0, -1j, 0],
                                    [0, 1j, 0, 0],
                                    [0, 0, 0, 0], ]).astype(complex),
                   "A24": np.array([[0, 0, 0, 0],
                                    [0, 0, 0, -1j],
                                    [0, 0, 0, 0],
                                    [0, 1j, 0, 0], ]).astype(complex),
                   "A34": np.array([[0, 0, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, -1j],
                                    [0, 0, 1j, 0], ]).astype(complex),
                   "D00": np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, 1], ]).astype(complex)/np.sqrt(2),
                   "D11": np.array([[1, 0, 0, 0],
                                    [0, -1, 0, 0],
                                    [0, 0, 0, 0],
                                    [0, 0, 0, 0], ]).astype(complex),
                   "D22": np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, -2, 0],
                                    [0, 0, 0, 0], ]).astype(complex)/np.sqrt(3),
                   "D33": np.array([[1, 0, 0, 0],
                                    [0, 1, 0, 0],
                                    [0, 0, 1, 0],
                                    [0, 0, 0, -3], ]).astype(complex)/np.sqrt(6)}
