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
Composite Gate
"""

import copy

from QCompute.QuantumPlatform.QuantumOperation.FixedGate import CX
from QCompute.QuantumPlatform.Utilities import _mergePBList
from QCompute.QuantumProtobuf.Library.PlatformStruct_pb2 import CircuitLine
from QCompute.QuantumProtobuf.Library.QuantumOperation_pb2 import CompositeGate as CompositeGateEnum, U


class CompositeGate:
    """
    The decomposition of composite gate

    Example:

    env.module(CompositeGate())

    env.module(CompositeGate(['RZZ']))

    env.module(CompositeGate([PBCompositeGate.RZZ]))
    """

    def __init__(self, params=None):
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.

        :param params: The gate list to process. Let it be None to process all.
        """

        self.params = params

    def __call__(self, program):
        """
        Process the Module

        :param program: the program
        :return: decomposed circuit
        """

        ret = copy.deepcopy(program)
        ret.body.ClearField('circuit')
        for id, procedure in program.body.procedureMap.items():
            targetProcedure = ret.body.procedureMap[id]
            targetProcedure.ClearField('circuit')
            self._decompose(targetProcedure.circuit, procedure.circuit)
        self._decompose(ret.body.circuit, program.body.circuit)
        return ret

    def _decompose(self, circuitOut, circuitIn):
        """
        Decompose circuit

        :param circuitOut: output circuit
        :param circuitIn: input circuit
        """

        for circuitLine in circuitIn:
            if circuitLine.HasField('compositeGate') and (
                    self.params is None or circuitLine.compositeGate in self.params):
                # insert the decomposed circuit
                if circuitLine.compositeGate == CompositeGateEnum.RZZ:
                    """
                    
                    RZZ(xyz)(Q0, Q1)
                    
                    =
                    
                    CX(Q0, Q1)
                    
                    U(xyz)(Q1)
                    
                    CX(Q0, Q1)
                    """
                    circuitOut.append(CX._toPB(*circuitLine.qRegs))

                    newCircuitLine = CircuitLine()
                    newCircuitLine.rotationGate = U
                    newCircuitLine.qRegs.append(circuitLine.qRegs[1])
                    _mergePBList(newCircuitLine.paramValues, circuitLine.paramValues)
                    _mergePBList(newCircuitLine.paramIds, circuitLine.paramIds)
                    circuitOut.append(newCircuitLine)

                    circuitOut.append(CX._toPB(*circuitLine.qRegs))
                    continue

            # copy the original circuit
            circuitOut.append(circuitLine)
