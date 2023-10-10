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
Convert the circuit to internal struct
"""
FileErrorCode = 3

from typing import List, TYPE_CHECKING

from QCompute import CustomizedGateOP, BarrierOP
from QCompute.OpenConvertor import ConvertorImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QPlatform.QOperation import CircuitLine
from QCompute.QPlatform.QOperation.FixedGate import getFixedGateInstance
from QCompute.QPlatform.QOperation.Measure import getMeasureInstance
from QCompute.QPlatform.QOperation.RotationGate import createRotationGateInstance
from QCompute.QPlatform.Utilities import protobufMatrixToNumpyMatrix
from QCompute.QProtobuf import PBFixedGate, PBRotationGate, PBMeasure

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBCircuitLine, PBCustomizedGate


class CircuitToInternalStruct(ConvertorImplement):
    """
    Circuit to internal struct
    """

    def convert(self, pbCircuitLineList: List['PBCircuitLine']) -> List[CircuitLine]:
        """
        Convert the circuit to internal struct.

        Example:

        env.publish()  # need to generate protobuf circuit data

        circuitLineList = CircuitToInternalStruct().convert(env.program)

        :param pbCircuitLineList: Protobuf format of the circuit
        :return: Internal circuit struct list
        """

        ret: List[CircuitLine] = []
        for pbCircuitLine in pbCircuitLineList:
            circuitLine = CircuitLine()
            op = pbCircuitLine.WhichOneof('op')
            if op == 'fixedGate':
                fixedGate: PBFixedGate = pbCircuitLine.fixedGate
                gateName = PBFixedGate.Name(fixedGate)
                circuitLine.data = getFixedGateInstance(gateName)
            elif op == 'rotationGate':
                rotationGate: PBRotationGate = pbCircuitLine.rotationGate
                gateName = PBRotationGate.Name(rotationGate)
                circuitLine.data = createRotationGateInstance(gateName, *pbCircuitLine.argumentValueList)
            elif op == 'customizedGate':
                customizedGate: PBCustomizedGate = pbCircuitLine.customizedGate
                circuitLine.data = CustomizedGateOP(protobufMatrixToNumpyMatrix(customizedGate.matrix))
            elif op == 'measure':
                measure: PBMeasure = pbCircuitLine.measure
                circuitLine.data = getMeasureInstance(PBMeasure.Type.Name(measure.type))
                circuitLine.cRegList = list(measure.cRegList)
            elif op == 'barrier':
                circuitLine.data = BarrierOP()
            else:
                raise Error.ArgumentError(f'InternalStruct Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 1)
            circuitLine.qRegList = list(pbCircuitLine.qRegList)
            ret.append(circuitLine)

        return ret