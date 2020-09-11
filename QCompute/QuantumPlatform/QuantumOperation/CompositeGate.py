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
Composite Gate Operation
"""

from QCompute.QuantumPlatform.QuantumOperation import QuantumOperation
from QCompute.QuantumPlatform.Utilities import _mergePBList
from QCompute.QuantumProtobuf.Library.PlatformStruct_pb2 import CircuitLine
from QCompute.QuantumProtobuf.Library.QuantumOperation_pb2 import CompositeGate as CompositeGateEnum


class CompositeGate(QuantumOperation):
    """
    The composite gate. Used for users' convenience.

    The programmer can define more composite gate in this file.

    The real implementation of Composite Gate can be given in the file OpenModule/CompositeGateModule/__init__.py

    An example "RZZ" has been given in this class.
    """

    def __init__(self):
        self.angles = []  # rotation parameter values
        self.procedureParams = []  # placeholder parameter No.

    def _setAngles(self, *angles):
        """
        Set angles and procedureParams
        """

        for angle in angles:
            if isinstance(angle, (int, float)):
                self.angles.append(angle)
                self.procedureParams.append(-1)
            else:
                self.angles.append(0)
                self.procedureParams.append(angle.index)  # ProcedureParamStorage
        if all([param != -1 for param in self.procedureParams]):
            self.angles = None
        if all([param == -1 for param in self.procedureParams]):
            self.procedureParams = None

    def _toPB(self, *qRegsIndex):
        """
        Convert to Protobuf object
        :param qRegsIndex: the quantum register list used in creating single circuit object
        :return: the circuit in Protobuf format.
                Filled with the name of composite gates and parameters of quantum registers.
        """

        ret = CircuitLine()

        if len(qRegsIndex) == 0:  # fill in the register list
            # The circuit object is already in the Python env.
            # Directly generate the circuit in Protobuf format according to the member variables.
            for reg in self.qRegs:
                ret.qRegs.append(reg.index)
        else:
            # Insert the new circuit object in the module process.
            # Generate the Protobuf circuit according to parameters.
            _mergePBList(ret.qRegs, qRegsIndex)  # fill in quantum registers

        assert len(ret.qRegs) == self.bits  # The number of quantum registers must conform to the setting.

        qRegSet = set(qReg for qReg in ret.qRegs)
        assert len(ret.qRegs) == len(
            qRegSet)  # The quantum registers of operators in circuit should not be duplicated.

        ret.compositeGate = CompositeGateEnum.Value(self.Gate)  # fill in name of the composite gate
        _mergePBList(ret.paramValues, self.angles)  # fill in rotation angles
        _mergePBList(ret.paramIds, self.procedureParams)  # fill in procedure parameters
        return ret


class RZZ(CompositeGate):
    """
    RZZ(xyz)(Q0, Q1)

    =

    CX(Q0, Q1)

    U(xyz)(Q1)

    CX(Q0, Q1)
    """

    Gate = 'RZZ'

    def __init__(self, *angles):
        """
        :param angles: parameters of the composite gate
        """
        super().__init__()

        self._setAngles(*angles)  # fill in rotation gate parameters
        self.bits = 2
