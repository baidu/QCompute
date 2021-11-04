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
Compress Gate to Accelerate Simulator Process
"""
from copy import deepcopy
from typing import List, Dict, Optional

from QCompute.OpenModule import ModuleImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QPlatform.QOperation import CircuitLine
from QCompute.QPlatform.QOperation.Barrier import BarrierOP
from QCompute.QPlatform.QOperation.CustomizedGate import CustomizedGateOP
from QCompute.QPlatform.QOperation.FixedGate import FixedGateOP
from QCompute.QPlatform.QOperation.Measure import MeasureOP
from QCompute.QPlatform.QOperation.RotationGate import RotationGateOP
from QCompute.QPlatform.Utilities import contract1_1, contract1_2
from QCompute.QProtobuf import PBProgram

FileErrorCode = 1


class CompressGateModule(ModuleImplement):
    """
    Compress one-qubit gates into two qubit gates

    Example:

    env.module(CompressGateModule())

    env.module(CompressGateModule({'disable': True}))  # Disable
    """
    arguments = None  # type: Optional[Dict[str, bool]]

    def __init__(self, arguments: Optional[Dict[str, bool]] = None):
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.
        """

        self.arguments = arguments
        if arguments is not None and type(arguments) is dict:
            if 'disable' in arguments:
                self.disable = arguments['disable']

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program
        :return: compressed circuit
        """
        from QCompute.OpenConvertor.CircuitToInternalStruct import CircuitToInternalStruct
        from QCompute.OpenConvertor.InternalStructToCircuit import InternalStructToCircuit

        ret = deepcopy(program)

        del ret.body.circuit[:]
        circuitIn = CircuitToInternalStruct().convert(program.body.circuit)
        circuitOut = _compress(circuitIn, program.head.usingQRegList)
        pbCircuitOut = InternalStructToCircuit().convert(circuitOut)
        ret.body.circuit.extend(pbCircuitOut)

        return ret


def _compress(circuitIn: List[CircuitLine], qRegs: List[int]) -> List[CircuitLine]:
    """
    Compress the circuit.
    Contract all one-qubit gates into two-qubit gates, and output a circuit with only two-qubit gates.

    :param circuitIn: Input circuit
    :param qRegs: qReg list
    :return: Output circuit
    """

    n = len(qRegs)
    circuitMap = dict(zip(qRegs, n * [['Start']]))

    for num, circuitLine in enumerate(circuitIn):  # Separate operations in circuitIn and note them in circuitMap
        if isinstance(circuitLine.data, (FixedGateOP, RotationGateOP)):
            if len(circuitLine.qRegList) == 1:
                list_tmp = list(circuitMap[circuitLine.qRegList[0]])
                list_tmp.append(num)
                circuitMap[circuitLine.qRegList[0]] = list_tmp
            elif len(circuitLine.qRegList) == 2:
                for i in range(2):
                    twobit_gate = {'num': num, 'up_or_down': i}
                    list_tmp = list(circuitMap[circuitLine.qRegList[i]])
                    list_tmp.append(twobit_gate)
                    circuitMap[circuitLine.qRegList[i]] = list_tmp
        elif isinstance(circuitLine.data, MeasureOP):
            for qReg in circuitLine.qRegList:
                list_tmp = list(circuitMap[qReg])
                list_tmp.append([num])
                circuitMap[qReg] = list_tmp
        elif isinstance(circuitLine.data, BarrierOP):
            pass
        else:
            raise Error.ArgumentError('Unsupported operation at compress!', ModuleErrorCode, FileErrorCode, 1)

    for key, value in circuitMap.items():
        value.append("End")

    circuitIn_copy = list(circuitIn)  # Storage of circuitLines. Copy from circuitIn, revise and then output.

    for key, value in circuitMap.items():  # Separate operations in circuitMap and deal with them
        for tag_num, tag in enumerate(value):
            if isinstance(tag, int):  # This is a one-qubit gate. Check the next one. If str occurs,check backward.
                tag_next = value[tag_num + 1]
                if isinstance(tag_next, int):  # The next one is a one-qubit gate.
                    floating = circuitIn_copy[tag].data.getMatrix()
                    parking = circuitIn_copy[tag_next].data.getMatrix()
                    new_gate_matrix = contract1_1(matrixFloating=floating, matrixParking=parking)

                    new_circuitline = CircuitLine()
                    new_circuitline.data = CustomizedGateOP(new_gate_matrix)
                    new_circuitline.qRegList = circuitIn[tag_next].qRegList

                    circuitIn_copy[tag_next] = new_circuitline
                    circuitIn_copy[tag] = None
                    circuitMap[key][tag_num] = None
                    pass
                elif isinstance(tag_next, dict):  # The next one is a two-qubit gate.
                    floating = circuitIn_copy[tag].data.getMatrix()  # Floating describes a gate to be absorbed.
                    parking = circuitIn_copy[
                        tag_next['num']].data.getMatrix()  # Parking describes a gate that will stay there.
                    new_gate_matrix = contract1_2(matrixFloating=floating, matrixParking=parking,
                                                  leftOrRight=0, upOrDown=tag_next['up_or_down'])

                    new_circuitline = CircuitLine()
                    new_circuitline.data = CustomizedGateOP(new_gate_matrix)
                    new_circuitline.qRegList = circuitIn_copy[tag_next['num']].qRegList

                    circuitIn_copy[tag_next['num']] = new_circuitline
                    circuitIn_copy[tag] = None
                    circuitMap[key][tag_num] = None
                    pass
                elif isinstance(tag_next, str) or isinstance(tag_next, list):  # Blocked. Check backward.
                    tag_bef_num = tag_num - 1
                    while tag_bef_num >= 0:  # Check backward as checking forward, until the very beginning
                        # if not interrupted. ('Start', when tag_bef_num==0).
                        tag_bef = value[tag_bef_num]
                        if tag_bef is None:  # No gate here
                            pass
                        elif isinstance(tag_bef, dict):
                            floating = circuitIn_copy[tag].data.getMatrix()
                            parking = circuitIn_copy[tag_bef['num']].data.getMatrix()
                            new_gate_matrix = contract1_2(matrixFloating=floating, matrixParking=parking,
                                                          leftOrRight=1, upOrDown=tag_bef['up_or_down'])

                            new_circuitline = CircuitLine()
                            new_circuitline.data = CustomizedGateOP(new_gate_matrix)
                            new_circuitline.qRegList = circuitIn_copy[tag_bef['num']].qRegList

                            circuitIn_copy[tag_bef['num']] = new_circuitline
                            circuitIn_copy[tag] = None
                            circuitMap[key][tag_num] = None
                            break
                        elif isinstance(tag_bef, (str, list)):
                            break
                        else:
                            raise Error.ArgumentError('Wrong compression of gate!', ModuleErrorCode, FileErrorCode, 2)
                        tag_bef_num -= 1
                else:
                    raise Error.ArgumentError('Wrong construction of circuitMap!', ModuleErrorCode, FileErrorCode, 3)
            elif isinstance(tag, (dict, str, list)):
                pass
            else:
                raise Error.ArgumentError('Wrong construction of circuitMap!', ModuleErrorCode, FileErrorCode, 4)

    circuitOut = []  # type: List[CircuitLine]
    for circuitLine in circuitIn_copy:  # Get the compressed gates and other circuitLines
        if circuitLine is not None:
            circuitOut.append(circuitLine)
    return circuitOut
