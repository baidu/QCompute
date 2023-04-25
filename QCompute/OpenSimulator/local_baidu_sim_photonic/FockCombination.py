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
import numpy
import sys
import math
from typing import List
sys.setrecursionlimit(100000)


class FockCombination:
    """
    Class 'FockCombination' is used to generate all input and output states.
    """

    def __init__(self, num_mode: int, initial_state: numpy.ndarray, num_coincidence: int) -> None:
        """
        Define several necessary arguments
        """

        self.num_coin = num_coincidence
        self.num_mode = num_mode
        self.initial_state = initial_state
        # self.mark_occupy_mode: obtain an array of the sequence of qumodes with photons.
        # self.state_occupy_mode: an array after deleting the qumodes without photons.
        # self.num_occupy_mode: calculate the number of qumodes with photons.
        self.mark_occupy_mode = numpy.nonzero(initial_state[:, 0])[0]
        self.state_occupy_mode = initial_state[self.mark_occupy_mode][:, 0]
        self.num_occupy_mode = len(self.state_occupy_mode)
        # store all the output states and input states, respectively.
        self.final_state_list = []
        self.initial_state_list = []

    def backtrack_final_state(self, current_num_photon: int, path: numpy.ndarray) -> None:
        """
        This function is used to generate all output states. The core of is backtrack algorithm.
        If the total number of photons of current state is equal to the number of coincident photons,
        and the length of path is equal to the number of qumodes, we store this state.

        :param current_num_photon: the number of photons in single shot
        :param path: a register for storing photons
        """

        if current_num_photon == self.num_coin and len(path) == self.num_mode:
            self.final_state_list.append(numpy.array(path))
            return

        if current_num_photon > self.num_coin or len(path) > self.num_mode:
            return
        # We throw different 'counts' into the temporary register 'path', and move to the next iteration.
        # If the conditions shown at the beginning of this function are meet, we store it.
        # Otherwise, we go back to previous iteration, and popup the 'counts' that was just filled.
        for counts in range(self.num_coin + 1):
            path.append(counts)
            current_num_photon += counts

            self.backtrack_final_state(current_num_photon, path)

            path.pop()
            current_num_photon -= counts

    def results_final_state(self) -> List[numpy.ndarray]:
        """
        Return a list containing all final states
        """

        self.backtrack_final_state(0, [])
        return self.final_state_list

    def backtrack_initial_state(self, current_num_photon: int, path: numpy.ndarray) -> None:
        r"""
        This function is used to find all input states. The core of is backtrack algorithm.
        Somewhat different from the function 'backtrack_final_state':
        (1) the length of temporary register maybe less than :math:`N` because we have deleted the qumode(s) without photons.
        (2) an additional condition must be considered: for each qumode the number of photons of input state
        must be less than or equal to the number of photons of original initial state.

        :param current_num_photon: the number of photons in single shot
        :param path: a register for storing photons
        """

        # The conditions of input state
        if (current_num_photon == self.num_coin) and (len(path) == self.num_occupy_mode):
            # Calculate the difference between temporary state and original input state.
            diff = self.state_occupy_mode - path
            # If ture, the additional condition (2) does not meet, and then we go back to previous iteration
            if numpy.any(diff < 0):
                return
            # Otherwise, the condition (2) is meet, and we store this state 'path'
            else:
                value = 1
                for index in range(len(diff)):
                    value *= math.factorial(int(self.state_occupy_mode[index])) / (math.factorial(path[index])
                                                                                   * math.factorial(int(diff[index])))

                state = numpy.zeros(self.num_mode, dtype=int)
                state[self.mark_occupy_mode] = path
                for _ in range(int(value)):
                    self.initial_state_list.append(state)
                return

        if current_num_photon > self.num_coin or len(path) >= self.num_occupy_mode:
            return

        # We throw different 'counts' into the temporary register 'path', and move to the next iteration.
        # If the conditions shown at the beginning of this function are meet, we store it.
        # Otherwise, we go back to previous iteration, and popup the 'counts' that was just filled.
        for counts in range(self.num_coin + 1):
            path.append(counts)
            current_num_photon += counts

            self.backtrack_initial_state(current_num_photon, path)

            path.pop()
            current_num_photon -= counts

    def results_initial_state(self) -> List[numpy.ndarray]:
        """
        Return a list containing all input states
        """

        self.backtrack_initial_state(0, [])

        return self.initial_state_list
