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
Convert the internal struct to circuit
"""
from typing import List

from QCompute.OpenConvertor import ConvertorImplement
from QCompute.QPlatform.QOperation import CircuitLine
from QCompute.QProtobuf import PBCircuitLine
from QCompute.QPlatform.QOperation.FixedGate import FixedGateOP
from QCompute.QPlatform.QOperation.RotationGate import RotationGateOP
from QCompute.QPlatform.QOperation.CustomizedGate import CustomizedGateOP
from QCompute.QPlatform.QOperation.Measure import MeasureOP
from QCompute.QPlatform.QOperation.Barrier import BarrierOP
from QCompute.QPlatform.CircuitTools import gateToProtobuf


class InternalStructToCircuit(ConvertorImplement):
    """
    Internal struct to circuit
    """

    def convert(self, circuitLineList: List[CircuitLine]) -> List[PBCircuitLine]:
        """
        Convert the internal struct to circuit.

        Example:

        circuit = CircuitToInternalStruct().convert(circuitLineList)

        :param circuitLineList: Internal circuit struct list
        :return: Protobuf format of the circuit
        """

        pbCircuit = []  # type: List[PBCircuitLine]
        for circuitLine in circuitLineList:
            pbCircuitLine = gateToProtobuf(circuitLine.data, circuitLine.qRegList,
                                           circuitLine.cRegList if hasattr(circuitLine, 'cRegList') else None)
            pbCircuit.append(pbCircuitLine)
        return pbCircuit
