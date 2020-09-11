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
Measure Operation
"""

from QCompute.QuantumPlatform.QuantumOperation import QuantumOperation
from QCompute.QuantumPlatform.Utilities import _mergePBList
from QCompute.QuantumProtobuf.Library.PlatformStruct_pb2 import CircuitLine
from QCompute.QuantumProtobuf.Library.QuantumOperation_pb2 import Measure as PBMeasure


class Measure(QuantumOperation):
    """
    The measure instruction.

    This class is to be implemented on the next version.

    Currently, only computational basis measure, MeasureZ, can be used in the program.
    """

    def __call__(self, qRegs, cRegs):
        """
        Hack initialize by calling parent classes
        :param qRegs: the quantum register list
        :param cRegs: the classical register list
        """

        assert len(qRegs) == len(cRegs)  # The qreg count and creg count in Measure(qreg, creg) must be equal.
        self.bits = len(qRegs)
        gate = super().__call__(*qRegs)
        gate.cRegs = cRegs  # record the classical registers to the circuit
        targetEnv = qRegs[0].env
        targetEnv.ClassicRegister.update(cRegs)  # fill in the environment by classical registers in use

    def _toPB(self, qRegsIndex=None, cRegsIndex=None):
        """
        Convert to Protobuf object
        :param qRegsIndex: Quantum register list used when creating single circuit object
        :param cRegsIndex: Classical register list used when creating single circuit object
        :return: Protobuf object
        """

        ret = CircuitLine()
        if qRegsIndex is None:
            # The circuit object is already in the Python env.
            # Directly generate the circuit in Protobuf format according to the member variables.
            for reg in self.qRegs:  # fill in quantum registers
                ret.qRegs.append(reg.index)
            _mergePBList(ret.measure.cRegs, self.cRegs)  # fill in classical registers
        else:
            # Insert the new circuit object into the module process.
            # Generate the Protobuf circuit according to the function parameters.
            _mergePBList(ret.qRegs, qRegsIndex)  # fill in quantum registers
            _mergePBList(ret.measure.cRegs, cRegsIndex)  # fill in classical registers
        ret.measure.type = PBMeasure.Type.Value(self.Gate)  # fill in measure type
        return ret


MeasureZ = Measure()
"""
Z measure: measurement along computational basis
"""
MeasureZ.Gate = 'Z'
