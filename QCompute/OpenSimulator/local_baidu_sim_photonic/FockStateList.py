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
Auxiliary functions used for the quantum circuit based on fock states
"""
FileErrorCode = 17

import numpy
import sys
import math
from typing import Tuple
sys.setrecursionlimit(100000)


class ListOfInputFockState:
    """
    Class 'FockCombination' is used to generate all input and output states.
    """

    def __init__(self, initial_state: numpy.ndarray, num_cutoff: int, num_mode: int) -> None:
        """
        Define several necessary arguments
        """

        self.num_cutoff = num_cutoff
        self.num_mode = num_mode
        self.initial_state = initial_state

        self.mark_occupy_mode = numpy.nonzero(initial_state[:, 0])[0]
        self.state_occupy_mode = initial_state[self.mark_occupy_mode][:, 0]
        self.num_occupy_mode = len(self.state_occupy_mode)

        self.initial_state_list = []
        self.initial_state_num_list = []

    def backtrack_initial_state(self, path: numpy.ndarray) -> None:
        """
        To find all input states by backtrack algorithm.
        """
        if len(path) == self.num_occupy_mode and sum(path) == self.num_cutoff:
            state = numpy.zeros(self.num_mode, dtype=int)
            state[self.mark_occupy_mode] = path
            self.initial_state_list.append(state)

            diff = numpy.subtract(self.state_occupy_mode, path)
            num_repetition = 1
            for index in range(self.num_occupy_mode):
                num_repetition *= math.factorial(int(self.state_occupy_mode[index])) / \
                                  (math.factorial(path[index]) * math.factorial(int(diff[index])))
            self.initial_state_num_list = numpy.append(self.initial_state_num_list, num_repetition)
            return

        if len(path) >= self.num_occupy_mode or sum(path) > self.num_cutoff:
            return

        counts_next_mode = self.state_occupy_mode[len(path)]
        for counts in range(counts_next_mode + 1):
            path.append(counts)
            self.backtrack_initial_state(path)
            path.pop()

    def results_initial_state(self) -> Tuple[list, list]:
        """
        Return a list containing all input states
        """
        path = []
        self.backtrack_initial_state(path)

        return self.initial_state_list, self.initial_state_num_list