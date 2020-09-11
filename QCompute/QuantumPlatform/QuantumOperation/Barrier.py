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
Barrier Operation
"""

from QCompute.QuantumPlatform.QuantumOperation import QuantumOperation
from QCompute.QuantumPlatform.Utilities import _mergePBList
from QCompute.QuantumProtobuf.Library.PlatformStruct_pb2 import CircuitLine


class Barrier(QuantumOperation):
    """
    The barrier instruction

    Barrier does nothing for implementing circuits on simulator but does STOP optimization between two barriers
    """

    def __init__(self, *qRegs):
        """
        Hack initialize by calling parent classes
        :param qRegs: the quantum registers list
        """

        self.bits = len(qRegs)
        super().__call__(*qRegs)

    def _toPB(self, *qRegsIndex):
        """
        Convert to Protobuf object
        :param qRegsIndex: the quantum register list used in creating single circuit object
        """

        ret = CircuitLine()

        if len(qRegsIndex) == 0:  # fill in the register list
            # The circuit object is already in the Python env.
            # Directly generate the circuit in Protobuf format according to the member variables.
            for reg in self.qRegs:
                ret.qRegs.append(reg.index)
        else:
            # Insert the new circuit object in the module process.
            # Generate the Protobuf circuit according to the function parameters.
            _mergePBList(ret.qRegs, qRegsIndex)

        ret.barrier = True  # fill in the barrier field
        return ret
