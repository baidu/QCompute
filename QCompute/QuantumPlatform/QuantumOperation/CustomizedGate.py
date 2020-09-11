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
Customized Gate Operation
"""
from math import log2

import numpy

from QCompute.QuantumPlatform.QuantumOperation import QuantumOperation
from QCompute.QuantumPlatform.Utilities import _mergePBList, _numpyMatrixToProtobufMatrix, _protobufMatrixToNumpyMatrix
from QCompute.QuantumProtobuf.Library.PlatformStruct_pb2 import CircuitLine


class CustomizedGate(QuantumOperation):
    """
    Customized gate

    The current version does not support arbitrary unitary as the lack of decomposition process.

    The user should not use this feature.
    """

    def __init__(self, matrix):
        self.matrix = numpy.array(matrix, dtype=complex)
        bits = log2(len(matrix))
        self.bits = int(bits)
        assert bits == self.bits  # bits must be an integer

    def _toPB(self, *qRegsIndex):
        """
        Convert to Protobuf object
        :param qRegsIndex: the quantum register list used in creating single circuit object
        :return: the circuit in Protobuf format.
                Filled with the name of fixed gates and parameters of quantum registers.
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
            _mergePBList(ret.qRegs, qRegsIndex)  # fill in the quantum registers

        assert len(ret.qRegs) == self.bits  # The number of quantum registers must match to the setting.

        qRegSet = set(qReg for qReg in ret.qRegs)
        assert len(ret.qRegs) == len(qRegSet)  # Quantum registers of operators in circuit should not be duplicated

        _numpyMatrixToProtobufMatrix(self.matrix, ret.customizedGate.matrix)
        self.matrix = self.matrix
        mat = _protobufMatrixToNumpyMatrix(ret.customizedGate.matrix)
        return ret
