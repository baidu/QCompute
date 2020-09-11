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
Utilities
"""

import functools

import numpy

from QCompute.Define import MeasureFormat
from QCompute.Define.Settings import measureFormat
from QCompute.QuantumPlatform import Error


def nKron(AMatrix, BMatrix, *args):
    """
    Recursively execute kron n times. This function has at least two matrices.

    :param AMatrix: First matrix
    :param BMatrix: Second matrix
    :param args: If have more matrix, they are delivered by this matrix
    :return: The result of their tensor product
    """
    return functools.reduce(
        lambda result, index: numpy.kron(
            result, index), args, numpy.kron(
            AMatrix, BMatrix))


def _mergePBList(pbList, sourceList):
    """
    In python3.8: 'list' object has no attribute '_values'

    So MergeFrom is broken.

    :param pbList: Protobuf list
    :param sourceList: Source list
    """
    if sourceList is None:
        return
    for obj in sourceList:
        pbList.append(obj)


def _filterMeasure(counts, cRegList):
    cRegList.sort()
    cRegList.reverse()
    cRegCount = len(cRegList)

    sourceCRegCount = 0
    for key in counts.keys():
        sourceCRegCount = len(key)

    assert sourceCRegCount > 0

    zeroKey = '0' * cRegCount
    binRet = {}
    for k, v in counts.items():
        hit = False
        for cReg in cRegList:
            if k[sourceCRegCount - 1 - cReg] == '1':
                hit = True
                break
        if hit:
            keyList = ['0'] * cRegCount
            for index, qReg in enumerate(cRegList):
                keyList[index] = k[sourceCRegCount - 1 - qReg]
            key = ''.join(keyList)
        else:
            key = zeroKey
        if binRet.get(key) is None:
            binRet[key] = v
        else:
            binRet[key] += v
    return binRet


def _formatMeasure(counts, cRegCount, format=measureFormat):
    ret = {}
    for (k, v) in counts.items():
        if format == MeasureFormat.Bin and k.startswith('0x'):
            num = int(k, 16)
            ret[bin(num)[2:].zfill(cRegCount)] = v
        elif format == MeasureFormat.Hex and not k.startswith('0x'):
            num = int(k, 2)
            ret[hex(num)] = v
        else:
            ret[k] = v
    return ret


def _numpyMatrixToProtobufMatrix(numpyMatrix, protobufMatrix):
    _mergePBList(protobufMatrix.shape, numpyMatrix.shape)
    for value in numpy.nditer(numpyMatrix):
        val = value.reshape(1, 1)[0][0]
        complexVal = protobufMatrix.array.add()
        complexVal.real = val.real
        complexVal.imag = val.imag


def _protobufMatrixToNumpyMatrix(protobufMatrix):
    complexArray = [complex(complexVal.real, complexVal.imag) for complexVal in protobufMatrix.array]
    return numpy.array(complexArray).reshape(protobufMatrix.shape)


def _contract1_1(matrix_parking, matrix_floating):
    """
    The first indicator of the gate is the output,
    and the latter indicator is the input,
    just like writing matrix vector multiplication, U{ket0}
    """
    if matrix_floating.shape != (2, 2):
        raise Error.ParamError("Floating gate not a 1-qubit gate")
    if matrix_parking.shape != (2, 2):
        raise Error.ParamError("Parking gate not a 1-qubit gate")
    new_array = numpy.einsum('ab,bc->ac', matrix_parking,
                             matrix_floating)  # v is a vector, A B is a matrix, we must count ABv, here we count AB
    return new_array


def _contract1_2(matrix_parking, matrix_floating, up_or_down, left_or_right):
    """
    Matrix_parking is like 1010.
    The first half of the indicators are outputs,
    and the second half are inputs.
    """
    if matrix_floating.shape != (2, 2):
        raise Error.ParamError("Floating gate not a 1-qubit gate")
    if matrix_parking.shape == (4, 4):
        matrix_parking = numpy.reshape(matrix_parking, [2, 2, 2, 2])
    elif matrix_parking.shape != (2, 2, 2, 2):
        raise Error.ParamError("Parking gate not a 2-qubit gate")
    if left_or_right == 0:
        if up_or_down == 0:
            new_array = numpy.einsum("abcd,de->abce", matrix_parking, matrix_floating)
        elif up_or_down == 1:
            new_array = numpy.einsum("abcd,ce->abed", matrix_parking, matrix_floating)
    elif left_or_right == 1:
        if up_or_down == 0:
            new_array = numpy.einsum("eb,abcd->aecd", matrix_floating, matrix_parking)
        elif up_or_down == 1:
            new_array = numpy.einsum("ea,abcd->ebcd", matrix_floating, matrix_parking)
    return numpy.reshape(new_array, [4, 4])
