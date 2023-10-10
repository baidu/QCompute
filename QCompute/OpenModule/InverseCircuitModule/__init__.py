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

FileErrorCode = 4

from copy import deepcopy
from typing import List, Dict, Optional

from QCompute.OpenModule import ModuleImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QPlatform.CircuitTools import gateToProtobuf
from QCompute.QPlatform.QOperation.CompositeGate import MS, createCompositeGateInstance
from QCompute.QPlatform.QOperation.CustomizedGate import CustomizedGateOP
from QCompute.QPlatform.QOperation.FixedGate import getFixedGateInstance
from QCompute.QPlatform.QOperation.RotationGate import createRotationGateInstance
from QCompute.QPlatform.Utilities import protobufMatrixToNumpyMatrix
from QCompute.QProtobuf import PBProgram, PBCircuitLine, PBFixedGate, PBRotationGate, PBCompositeGate


class InverseCircuitModule(ModuleImplement):
    """
    Inverse the circuit
    change the gate order, inverse the customized gate matrix, modify the angles of rotation gates

    Example:

    env.module(InverseCircuitModule())

    env.module(InverseCircuitModule({'disable': True}))  # Disable

    env.module(InverseCircuitModule({'errorOnUnsupported': True}))
    """
    errorOnUnsupported = True

    def __init__(self, arguments: Optional[Dict[str, bool]] = None):
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.
        """
        super().__init__(arguments)
        if arguments is not None and type(arguments) is dict:
            if 'errorOnUnsupported' in arguments:
                self.errorOnUnsupported = arguments['errorOnUnsupported']

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program
        :return: inversed circuit
        """
        if self.disable:
            return program

        ret = deepcopy(program)

        for name, procedure in program.body.procedureMap.items():
            targetProcedure = ret.body.procedureMap[name]
            del targetProcedure.circuit[:]
            self._inverse_circuit(targetProcedure.circuit, procedure.circuit)
        del ret.body.circuit[:]
        self._inverse_circuit(ret.body.circuit, program.body.circuit)
        return ret

    def _inverse_circuit(self, circuitOut: List['PBCircuitLine'], circuitIn: List['PBCircuitLine']):
        """
        Inverse a circuit

        :param circuitOut: input circuit
        :param circuitIn: output circuit
        """

        for circuitLine in reversed(circuitIn):
            circuitOut.append(self._inverse_gate(circuitLine))

    def _inverse_gate(self, circuitLine: 'PBCircuitLine'):
        """
        Inverse a single gate

        :param circuitLine: input circuitLine
        :return: output circuitLine
        """

        op = circuitLine.WhichOneof('op')
        if op == 'procedureName' or op == 'measure' or op == 'barrier':
            return deepcopy(circuitLine)
        if op == 'fixedGate':
            fixedGate: PBFixedGate = circuitLine.fixedGate
            gateName = PBFixedGate.Name(fixedGate)
            inversedGate = getFixedGateInstance(gateName).getInversed()
            return gateToProtobuf(inversedGate, circuitLine.qRegList)
        elif op == 'rotationGate':
            rotationGate: PBRotationGate = circuitLine.rotationGate
            gateName = PBRotationGate.Name(rotationGate)
            inversedGate = createRotationGateInstance(gateName, *circuitLine.argumentValueList).getInversed()
            return gateToProtobuf(inversedGate, circuitLine.qRegList)
        elif op == 'compositeGate':
            compositeGate: PBCompositeGate = circuitLine.compositeGate
            gateName = PBCompositeGate.Name(compositeGate)
            inversedGate = createCompositeGateInstance(gateName, *circuitLine.argumentValueList).getInversed()
            return gateToProtobuf(inversedGate, circuitLine.qRegList)
        elif op == 'customizedGate':
            mat = protobufMatrixToNumpyMatrix(circuitLine.customizedGate.matrix)
            return gateToProtobuf(CustomizedGateOP(mat).getInversed(), circuitLine.qRegList)

        else:
            # unsupported gate
            if self.errorOnUnsupported:
                # error
                raise Error.ArgumentError(
                    f'Unsupported operation {circuitLine}', ModuleErrorCode, FileErrorCode, 1)
            else:
                # ignore
                return deepcopy(circuitLine)