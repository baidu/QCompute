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
Quantum Operation
"""

import copy

from QCompute.QuantumPlatform.Error import Error


class QuantumOperation:
    """
    Basic classes for gates
    """

    def __call__(self, *qRegs):
        """
        Copy itself when it is called. Append to the circuit and the parameter list.

        :param qRegs: quantum register list
        """

        op = copy.copy(self)

        if len(qRegs) > 0:
            if hasattr(self, 'bits'):  # QuantumProcedure does not have bits configuration
                assert len(qRegs) == self.bits  # The number of quantum registers must match the setting

            qRegSet = set(qReg.index for qReg in qRegs)
            assert len(qRegs) == len(qRegSet)  # Quantum registers of operators in circuit are not repeatable

            targetEnv = qRegs[0].env  # Check if the registers belong to the same env
            if not all(r.env is targetEnv for r in qRegs):
                raise Error()

            op.qRegs = qRegs
            targetEnv.circuit.append(op)

        return op
