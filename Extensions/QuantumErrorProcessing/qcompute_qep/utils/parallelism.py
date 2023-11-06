#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
# Copyright (c) 2023 Baidu, Inc. All Rights Reserved.
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
Enable quantum program parallelization. More precisely, for a list of quantum programs that works on
disjoint subsets of qubits, we combine these quantum programs to run in parallel on the respective qubits.
This greatly saves the running time and the quantum computer resource.
"""

from typing import List, Optional

from Extensions.QuantumErrorProcessing.qcompute_qep.utils import execute
from Extensions.QuantumErrorProcessing.qcompute_qep.utils.types import QProgram, QComputer

from QCompute import *


# class ParallelProgram:
#     """Combine multiple quantum programs into a parallel quantum program.
#
#     Parallel experiments combine individual experiments on disjoint subsets
#     of qubits into a single composite experiment on the union of those qubits.
#     The component experiment circuits are combined to run in parallel on the
#     respective qubits.
#     """
#
#     def __init__(self, qp_list: List[QProgram], shots_list: List[int]):
#         """Initialize the analysis object.
#
#
#         :param qp_list: List[QProgram], a list of quantum programs.
#         :param qc: QComputer, the target quantum computer, on which the quantum program will execute
#         :return: shots_list, a list of running shots, each corresponds to a quantum program.
#         """
#         self.qp_list = qp_list
#         self.shots_list = shots_list
#         self._parallel_qp: QProgram = None
#
#     @property
#     def parallel_qp(self):
#         return self._parallel_qp
#
#     def split_counts(self):
#
#
# if __name__ == '__main__':
#
#     # Create Bell state in qubit [0, 1]
#     qp_1 = QEnv()
#     qp_1.Q.createList(2)
#     H(qp_1.Q[0])
#     CX(qp_1.Q[0], qp_1.Q[1])
#
#     # Create + state in qubit [2]
#     qp_2 = QEnv()
#     qp_2.Q.createList(3)
#     H(qp_2.Q[2])
#
#     # Create Bell state in qubit [3, 4]
#     qp_3 = QEnv()
#     qp_3.Q.createList(5)
#     H(qp_3.Q[3])
#     CX(qp_3.Q[3], qp_1.Q[4])
#
#     # Combine
#     pp_1 = ParallelProgram(qp_list=[qp_1, qp_2, qp_3])
#     pp_2 = ParallelProgram(qp_list=[qp_1, qp_2, qp_3])
#     pp.parallel_qp
#
#     counts_list = execute([pp_1.parallel_qp, pp_2.parallel_qp], shots=1024)
#
#     list[dict[str, int]] = pp_1.split_counts(counts)
