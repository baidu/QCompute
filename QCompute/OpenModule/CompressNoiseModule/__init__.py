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
Compress Noisy Gate to Accelerate Simulator Process
"""
FileErrorCode = 3


from copy import deepcopy
from typing import List, Dict, Optional, Union

from QCompute.OpenModule import ModuleImplement, ModuleErrorCode
from QCompute.QPlatform import Error
from QCompute.QPlatform.QOperation import CircuitLine
from QCompute.QPlatform.Utilities import contract1_1, contract1_2
from QCompute.QPlatform.Utilities import numpyMatrixToProtobufMatrix, getProtobufCicuitLineMatrix
from QCompute.QProtobuf import PBProgram, PBCircuitLine

class CompressNoiseModule(ModuleImplement):
    """
    Compress one-qubit noiseless gates and reorder noisy circuit.

    Example:

    env.module(CompressNoiseModule())

    env.module(CompressNoiseModule({'disable': True}))  # Disable
    """

    def __init__(self, arguments: Optional[Dict[str, bool]] = None):
        """
        Initialize the Module.

        Json serialization is allowed by the requested parameter.
        """
        super().__init__(arguments)

    def __call__(self, program: 'PBProgram') -> 'PBProgram':
        """
        Process the Module

        :param program: the program
        :return: compressed circuit
        """
        if self.disable:
            return program

        ret = deepcopy(program)

        del ret.body.circuit[:]
        pbCircuitOut = _compressNoise(program.body.circuit, program.head.usingQRegList)
        ret.body.circuit.extend(pbCircuitOut)

        return ret


def _compressNoise(circuitIn: List['PBCircuitLine'], qRegs: List[int]) -> List['PBCircuitLine']:
    """
    Compress all one-qubit noiseless gates into two-qubit gates, and reorder circuit with more condense gates.

    :param circuitIn: Input circuit
    :param qRegs: qReg list
    :return: Output circuit
    """

    circuitIn_copy = list(circuitIn)  # Storage of circuitLines. Copy from circuitIn, revise and then output.

    circuitMap = _separateCircuit(circuitIn_copy, qRegs, True)

    circuitCompressed = _compressGate(circuitIn_copy, circuitMap)

    circuitOut: List[CircuitLine] = []
    for circuitLine in circuitCompressed:  # Get the compressed gates and other circuitLines
        if circuitLine is not None:
            circuitOut.append(circuitLine)

    circuitMap_compressed = _separateCircuit(circuitOut, qRegs, False)

    circuitReOrdered = _reOrderCircuit(circuitOut, circuitMap_compressed)
    return circuitReOrdered


def _separateCircuit(circuitIn: List['PBCircuitLine'],
                     qRegs: List[int],
                     withNoise: bool) -> Dict[int, Union[str, int, List]]:
    """
    Separate the noisy circuit by qRegs.

    :param circuitIn: Input circuit
    :param qRegs: qReg list
    :param withNoise: if True, separate circuit with consideration of noise; else, no.
    :return: a dict that contains abstract information of the input circuit
    """
    n = len(qRegs)

    circuitMap = dict(zip(qRegs, n * [['Start']]))

    for num, circuitLine in enumerate(circuitIn):
        op = circuitLine.WhichOneof('op')
        if op in ['fixedGate', 'rotationGate', 'customizedGate']:
            if len(circuitLine.qRegList) == 1:  # This is a 1-qubit gate.
                list_tmp = list(circuitMap[circuitLine.qRegList[0]])
                if withNoise:
                    if circuitLine.noiseList:  # Check if the gate is noisy.
                        list_tmp.append((num, True))  # Use tuple to represent a noisy gate
                    else:
                        list_tmp.append(num)  # Use int to represent a noiseless 1-qubit gate
                else:
                    list_tmp.append(num)
                circuitMap[circuitLine.qRegList[0]] = list_tmp
            elif len(circuitLine.qRegList) == 2:  # This is a 2-qubit gate.
                for i in range(2):
                    qRegs = circuitLine.qRegList
                    twobit_gate = {'num': num, 'up_or_down': i}  # Use dict to represent a noiseless 2-qubit gate
                    list_tmp = list(circuitMap[qRegs[i]])
                    if withNoise:
                        if circuitLine.noiseList:  # Check if the gate is noisy.
                            list_tmp.append((twobit_gate, True))  # Use tuple to represent a noisy gate
                        else:
                            list_tmp.append(twobit_gate)
                    else:
                        list_tmp.append(twobit_gate)
                    circuitMap[circuitLine.qRegList[i]] = list_tmp
            elif len(circuitLine.qRegList) == 3:  # This is a three-qubit gate.
                for qReg in circuitLine.qRegList:
                    list_tmp = list(circuitMap[qReg])
                    list_tmp.append({num})
                    circuitMap[qReg] = list_tmp  # Use set to represent a 3-qubit gate
        elif op == 'measure':
            for qReg in circuitLine.qRegList:
                list_tmp = list(circuitMap[qReg])
                list_tmp.append([num])
                circuitMap[qReg] = list_tmp
        elif op == 'barrier':
            pass
        else:
            raise Error.ArgumentError(
                f'Unsupported operation {circuitLine} at compress!', ModuleErrorCode, FileErrorCode, 1)

    for key, value in circuitMap.items():
        value.append("End")

    return circuitMap


def _reOrderCircuit(circuitIn: List['PBCircuitLine'],
                    circuitMap: Dict[int, Union[str, int, List]]) -> List['PBCircuitLine']:
    """
    Add barriers to noisy circuit.
    Collect all one-qubit noisy gates and two-qubit noisy gates

    :param circuitIn: Input circuit
    :param circuitMap: a dict of circuit
    :return: Output circuit
    """

    pointerMap = dict()  # Collect subcircuits
    markedGateList = []  # Collect gates that has already be collected
    for key, value in circuitMap.items():  # Separate operations in circuitMap and deal with them
        list_temp = []
        pointerKeyCurrent = None
        for tag_num, tag in enumerate(value):
            if isinstance(tag, int):  # This is a 1-qubit gate. Check next.
                list_temp.append(tag)  # Collect current 1-qubit gate.
                value[tag_num] = None  # Empty the value
                tag_next = value[tag_num + 1]
                if isinstance(tag_next, int):  # The next is a 1-qubit gate, collect it.
                    list_temp.append(tag_next)
                    value[tag_num + 1] = None
                elif isinstance(tag_next, (dict, set)):  # The next is a 2-qubit or 3-qubit gate, collect and stop.
                    if isinstance(tag_next, dict):  # This is a 2-qubit gate
                        tag_next_num = tag_next['num']
                    else:  # This is a 3-qubit gate
                        tag_next_num = list(tag_next)[0]
                    if tag_next_num in markedGateList:  # This gate is already collected. Collect current subcircuits.
                        list_temp += pointerMap[tag_next_num]  # Current subcircuit contains only 1-qubit gates.
                        pointerMap[
                            tag_next_num] = list_temp  # Collect subcircuit with key being the index of 2-qubit gate
                        if isinstance(tag, dict):
                            markedGateList.remove(tag_next_num)  # Remove it from markedGateList
                    else:  # This gate is not collected.
                        list_temp.append(
                            tag_next_num)  # Current subcircuit contains 1-qubit gates and end with a 2-qubit gate
                        pointerMap[tag_next_num] = list_temp
                        markedGateList.append(tag_next_num)  # Mark this gate as collected.

                    list_temp = []
                    pointerKeyCurrent = tag_next_num
                    value[tag_num + 1] = None
                elif isinstance(tag_next, (str, list)):  # The next is a measure or the end. Check stop constraint.
                    # Current subcircuit contains only 1-qubit state.
                    if pointerKeyCurrent:
                        pointerMap[
                            pointerKeyCurrent] += list_temp  # Append this subcircuit onto the last subcircuit on current qubit
                    else:
                        pointerMap[tag] = list_temp
                    list_temp = []
                else:
                    pass
            elif isinstance(tag,
                            (dict, set)):  # This is a 2-qubit or 3-qubit state. Collect current subcircuits and stop.
                if isinstance(tag, dict):
                    tag_circ_num = tag['num']
                else:
                    tag_circ_num = list(tag)[0]
                if tag_circ_num in markedGateList:
                    list_temp += pointerMap[tag_circ_num]
                    pointerMap[tag_circ_num] = list_temp
                    if isinstance(tag, dict):
                        markedGateList.remove(tag_circ_num)
                else:
                    list_temp.append(tag_circ_num)
                    pointerMap[tag_circ_num] = list_temp
                    markedGateList.append(tag_circ_num)

                list_temp = []
                pointerKeyCurrent = tag_circ_num
                value[tag_num] = None
            elif isinstance(tag, list) or tag == "End":
                if list_temp:
                    if pointerKeyCurrent:
                        pointerMap[pointerKeyCurrent] += list_temp
                    else:
                        pointerMap[list_temp[-1]] = list_temp
            else:
                pass

    # sort subcircuit by 2-qubit or 3-qubit gate
    sortedPointerMap = sorted(pointerMap.items(), key=lambda d: d[0])
    ret = []
    for num, tagList in sortedPointerMap:
        for tag in tagList:
            newCircuitLine = PBCircuitLine()
            newCircuitLine = circuitIn[tag]
            ret.append(newCircuitLine)

    # Add measure at the end
    if circuitIn[-1].WhichOneof('op') == 'measure':
        ret.append(circuitIn[-1])

    return ret


def _compressGate(circuitIn_copy: List['PBCircuitLine'],
                  circuitMap: Dict[int, Union[str, int, List]]) -> List['PBCircuitLine']:
    """
    Compress the noiseless gats in the input noisy circuit.
    Contract all one-qubit and two-qubit noiseless gates into two-qubit noiseless or noisy gates.

    :param circuitIn: Input circuit
    :param circuitMap: a circuit dict
    :return: Output circuit
    """

    for key, value in circuitMap.items():  # Separate operations in circuitMap and deal with them
        for tag_num, tag in enumerate(value):
            if isinstance(tag,
                          int):  # This is a one-qubit noiseless gate. Check the next one. If str occurs,check backward.
                tag_next = value[tag_num + 1]
                if isinstance(tag_next, int):  # The next one is a one-qubit noiseless gate.
                    # circuit[tag_next] * circuit[tag]
                    floating = getProtobufCicuitLineMatrix(circuitIn_copy[tag])
                    parking = getProtobufCicuitLineMatrix(circuitIn_copy[tag_next])
                    new_gate_matrix = contract1_1(matrixFloating=floating, matrixParking=parking)

                    # Create a customized gate with new_gate_matrix
                    new_pbcircuitline = circuitIn_copy[tag_next]
                    new_pbcircuitline.customizedGate.matrix.CopyFrom(numpyMatrixToProtobufMatrix(new_gate_matrix))

                    # Replace the circuit[tage] and circuit[tag_next] with the customized gate
                    circuitIn_copy[tag_next] = new_pbcircuitline
                    circuitIn_copy[tag] = None
                    circuitMap[key][tag_num] = None
                elif isinstance(tag_next, dict) or isinstance(tag_next, tuple):  # The next one is a two-qubit gate.
                    # circuit[tag_next] * circuit[tag]
                    floating = getProtobufCicuitLineMatrix(
                        circuitIn_copy[tag])  # Floating describes a gate to be absorbed.
                    if isinstance(tag_next, tuple):  # This is a noisy gate
                        tag_next = tag_next[0]
                    if isinstance(tag_next, int):  # The next is a noisy 1-qubit gate
                        tag_next_num = tag_next
                        parking = getProtobufCicuitLineMatrix(circuitIn_copy[tag_next_num])
                        new_gate_matrix = contract1_1(matrixFloating=floating, matrixParking=parking)
                    else:
                        tag_next_num = tag_next['num']
                        parking = getProtobufCicuitLineMatrix(circuitIn_copy[tag_next_num])
                        new_gate_matrix = contract1_2(matrixFloating=floating, matrixParking=parking,
                                                      leftOrRight=0, upOrDown=tag_next['up_or_down'])

                    # Create a customized gate with new_gate_matrix
                    new_pbcircuitline = PBCircuitLine()
                    new_pbcircuitline = circuitIn_copy[tag_next_num]
                    new_pbcircuitline.customizedGate.matrix.CopyFrom(numpyMatrixToProtobufMatrix(new_gate_matrix))

                    # Replace the circuit[tage] and circuit[tag_next] with the customized gate
                    circuitIn_copy[tag_next_num] = new_pbcircuitline
                    circuitIn_copy[tag] = None
                    circuitMap[key][tag_num] = None
                elif isinstance(tag_next, (str, list)):  # This is a measure or the end. Check backward.
                    tag_bef_num = tag_num - 1
                    while tag_bef_num >= 0:  # Check backward as checking forward, until the very beginning
                        # if not interrupted. ('Start', when tag_bef_num==0).
                        tag_bef = value[tag_bef_num]
                        if tag_bef is None:  # No gate here
                            pass
                        elif isinstance(tag_bef, dict):  # This is a 2-qubit gate.
                            # circuit[tag_nex] * circuit[tag]
                            floating = getProtobufCicuitLineMatrix(circuitIn_copy[tag])
                            tag_bef_num = tag_bef['num']
                            parking = getProtobufCicuitLineMatrix(circuitIn_copy[tag_bef_num])
                            new_gate_matrix = contract1_2(matrixFloating=floating, matrixParking=parking,
                                                          leftOrRight=1, upOrDown=tag_bef['up_or_down'])

                            # Create a customized gate with new_gate_matrix
                            new_pbcircuitline = PBCircuitLine()
                            new_pbcircuitline = circuitIn_copy[tag_bef_num]
                            new_pbcircuitline.customizedGate.matrix.CopyFrom(
                                numpyMatrixToProtobufMatrix(new_gate_matrix))

                            # Replace the circuit[tage] and circuit[tag_next] with the customized gate
                            circuitIn_copy[tag_bef_num] = new_pbcircuitline
                            circuitIn_copy[tag] = None
                            circuitMap[key][tag_num] = None
                            break
                        elif isinstance(tag_bef, (str, tuple, list)):
                            break
                        else:
                            raise Error.ArgumentError('Wrong compression of gate!', ModuleErrorCode, FileErrorCode, 2)
                        tag_bef_num -= 1
                else:
                    pass
            elif isinstance(tag, (dict, set, tuple, str, list)):
                pass
            else:
                raise Error.ArgumentError('Wrong construction of circuitMap!', ModuleErrorCode, FileErrorCode, 3)

    return circuitIn_copy