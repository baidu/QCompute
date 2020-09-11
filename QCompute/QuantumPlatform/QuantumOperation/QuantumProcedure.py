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
Quantum Procedure
"""
import copy

from QCompute.QuantumPlatform.QuantumOperation import QuantumOperation
from QCompute.QuantumPlatform.Utilities import _mergePBList
from QCompute.QuantumProtobuf.Library.PlatformStruct_pb2 import CircuitLine


class QuantumProcedure(QuantumOperation):
    """
    Quantum procedure (sub-procedure)
    """

    def __init__(self, name, params, Q, circuit):
        """
        Initialize the sub-procedure of object
        """

        self.name = name

        # the procedure params
        self.params = params

        # the quantum registers
        self.Q = Q
        # change env to procedure
        Q.env = self
        for qRegisterStorage in Q.registerDict.values():
            qRegisterStorage.env = self

        # circuit
        self.circuit = circuit

    def __call__(self, *params):
        """
        Rotation params
        """

        op = copy.copy(self)
        op.procedureParams = params
        return op._callGate

    def _callGate(self, *qRegs):
        """
        Qreg params
        """

        super().__call__(*qRegs)

    def _toPB(self, *qRegsIndex):
        """
        Convert to Protobuf object
        :param qRegsIndex: the quantum register list used in creating single circuit object
        :return: Protobuf object
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

        paramValues = []
        paramIds = []
        for param in self.procedureParams:
            if isinstance(param, (int, float)):
                paramValues.append(param)
                paramIds.append(-1)
            else:
                paramValues.append(0)
                paramIds.append(param.index)  # ProcedureParamStorage
        if any(paramValues):
            _mergePBList(ret.paramValues, paramValues)
        if any([paramId != -1 for paramId in paramIds]):
            _mergePBList(ret.paramIds, paramIds)

        ret.procedureName = self.name
        return ret
