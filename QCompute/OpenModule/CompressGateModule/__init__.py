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

import copy

from QCompute import CustomizedGate
from QCompute.QuantumPlatform import Error
from QCompute.QuantumPlatform.QuantumOperation.FixedGate import *
from QCompute.QuantumPlatform.QuantumOperation.RotationGate import U
from QCompute.QuantumPlatform.Utilities import _contract1_1, _contract1_2, _protobufMatrixToNumpyMatrix
from QCompute.QuantumProtobuf.Library.QuantumOperation_pb2 import FixedGate as FixedGateEnum, \
    RotationGate as RotationGateEnum


def gate_from_circuitLine(circuitLine):  # do not support sparse
    if circuitLine.HasField('rotationGate'):
        if circuitLine.rotationGate != RotationGateEnum.U:
            raise Error.ParamError(
                f'unsupported operation {RotationGateEnum.Name(circuitLine.rotationGate)}')
        uGate = U(*circuitLine.paramValues)
        matrix = uGate.matrix
        return matrix
    if circuitLine.HasField('fixedGate'):
        operationDict = {}
        operationDict[FixedGateEnum.X] = X.matrix
        operationDict[FixedGateEnum.Y] = Y.matrix
        operationDict[FixedGateEnum.Z] = Z.matrix
        operationDict[FixedGateEnum.H] = H.matrix
        operationDict[FixedGateEnum.CX] = CX.matrix.reshape(2, 2, 2, 2)
        matrix = operationDict.get(circuitLine.fixedGate)
        if matrix is None:
            raise Error.ParamError(f'unsupported operation {FixedGateEnum.Name(circuitLine.fixedGate)}')
        return matrix
    if circuitLine.HasField('customizedGate'):
        return _protobufMatrixToNumpyMatrix(circuitLine.customizedGate.matrix)


class CompressGate:
    """
    Compress one-qubit gates into two qubit gates

    Example:

    env.module(CompressGate())
    """

    def __call__(self, program):
        """
        Process the Module

        :param program: the program
        :return: compressed circuit
        """

        ret = copy.deepcopy(program)
        ret.body.ClearField('circuit')
        for id, procedure in program.body.procedureMap.items():
            targetProcedure = ret.body.procedureMap[id]
            targetProcedure.ClearField('circuit')
            self._compress(targetProcedure.circuit, procedure.circuit, procedure.usingQRegs)
        self._compress(ret.body.circuit, program.body.circuit, program.head.usingQRegs)
        return ret

    def _compress(self, circuitOut, circuitIn, QRegs):
        """
        Compress the circuit.
        Contract all one-qubit gates into two-qubit gates, and output a circuit with only two-qubit gates.

        :param circuitOut: Output circuit.
        :param circuitIn: Input circuit.
        :param circuitMap: A 'note' of circuitlines, classify the circuitlines by qRegs. A dict with lists as values.
                            In the list, different circuitlines are stored according to the sequence,
                                    and they are marked with different types.
                            str: 'Start' or 'End' mark.
                            int: the index of a one-qubit gate
                            dict: marking a two-qubit gate
                            list: inrelavent operations, and will output directly into circuitOut.
        :param circuitIn_copy: A list constructed by copying circuitlines from circuitIn.
                When a one-qubit gate is contracted into another gate, its position in circuitIn_copy will be None.
                The other position will be a CustomizedGate.
        """
        QRegs = list(QRegs)
        n = len(QRegs)
        circuitMap = dict(zip(QRegs, n*[['Start']]))

        for num, circuitLine in enumerate(circuitIn):  # separate operations in circuitIn and note them in circuitMap
            if circuitLine.HasField('rotationGate') or circuitLine.HasField('fixedGate'):
                if len(circuitLine.qRegs) == 2:
                    qregs = circuitLine.qRegs
                    for i in range(2):
                        twobit_gate = {}
                        twobit_gate['num'] = num
                        twobit_gate['up_or_down'] = i
                        list_tmp = list(circuitMap[qregs[i]])
                        list_tmp.append(twobit_gate)
                        circuitMap[qregs[i]] = list_tmp

                elif len(circuitLine.qRegs) == 1:
                    qregs = circuitLine.qRegs[0]
                    list_tmp = list(circuitMap[qregs])
                    list_tmp.append(num)
                    circuitMap[qregs] = list_tmp

            elif circuitLine.HasField('measure'):
                qregs = circuitLine.qRegs
                for qreg in qregs:
                    list_tmp = list(circuitMap[qreg])
                    list_tmp.append([num])
                    circuitMap[qreg] = list_tmp

            elif circuitLine.HasField('barrier'):
                pass
            else:
                raise Error.ParamError('unsupported operation at compress')

        for key, value in circuitMap.items():
            value.append("End")

        circuitIn_copy = []  # Storage of circuitLines. Copy from circuitIn, revise and then output.
        for circuitLine in circuitIn:
            circuitIn_copy.append(circuitLine)

        for key, value in circuitMap.items():  # separate operations in circuitMap and deal with them
            for tag_num, tag in enumerate(value):
                if type(tag) is int:  # This is a one-qubit gate. Check the next one. If str occurs, check backward.
                    tag_next = value[tag_num+1]
                    if type(tag_next) is int:  # The next one is a one-qubit gate.
                        floating = gate_from_circuitLine(circuitIn_copy[tag])
                        parking = gate_from_circuitLine(circuitIn_copy[tag_next])
                        new_gate_matrix = _contract1_1(matrix_floating=floating, matrix_parking=parking)
                        new_circuitline = CustomizedGate(new_gate_matrix)._toPB(*circuitIn[tag_next].qRegs)
                        circuitIn_copy[tag_next] = new_circuitline
                        circuitIn_copy[tag] = None
                        circuitMap[key][tag_num] = None
                        pass
                    elif type(tag_next) is dict:  # The next one is a two-qubit gate.
                        floating = gate_from_circuitLine(
                            circuitIn_copy[tag])  # Floating describes a gate to be absorbed.
                        parking = gate_from_circuitLine(
                            circuitIn_copy[tag_next['num']])  # Parking describes a gate that will stay there.
                        new_gate_matrix = _contract1_2(matrix_floating=floating, matrix_parking=parking,
                                                       left_or_right=0, up_or_down=tag_next['up_or_down'])
                        new_circuitline = CustomizedGate(new_gate_matrix)._toPB(*circuitIn_copy[tag_next['num']].qRegs)
                        circuitIn_copy[tag_next['num']] = new_circuitline
                        circuitIn_copy[tag] = None
                        circuitMap[key][tag_num] = None
                        pass
                    elif type(tag_next) is str or type(tag_next) is list:  # Blocked. Check backward.
                        tag_bef_num = tag_num - 1
                        while tag_bef_num >= 0:  # Check backward as checking forward, until the very beginning
                                                 # if not interrupted. ('Start', when tag_bef_num==0).
                            tag_bef = value[tag_bef_num]
                            if tag_bef is None:  # No gate here
                                pass
                            elif type(tag_bef) is dict:
                                floating = gate_from_circuitLine(circuitIn_copy[tag])
                                parking = gate_from_circuitLine(circuitIn_copy[tag_bef['num']])
                                new_gate_matrix = _contract1_2(matrix_floating=floating, matrix_parking=parking,
                                                               left_or_right=1, up_or_down=tag_bef['up_or_down'])
                                new_circuitline = CustomizedGate(new_gate_matrix)._toPB(*circuitIn_copy[tag_bef['num']].qRegs)
                                circuitIn_copy[tag_bef['num']] = new_circuitline
                                circuitIn_copy[tag] = None
                                circuitMap[key][tag_num] = None
                                break
                            elif type(tag_bef) is str or list:
                                break
                            else:
                                raise Error.NetworkError('Wrong compression of gate')
                            tag_bef_num -= 1
                    else:
                        raise Error.NetworkError('Wrong construction of circuitMap')
                elif type(tag) is dict or str or list:
                    pass
                else:
                    raise Error.NetworkError('Wrong construction of circuitMap')

        for circuitLine in circuitIn_copy:  # get the compressed gates and other circuitLines
            if circuitLine is not None:
                circuitOut.append(circuitLine)
        return circuitOut