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
from typing import Dict, Optional, Union, Type

import numpy

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QProtobuf import PBMatrix

FileErrorCode = 6


def destoryObject(obj: object) -> None:
    """
    destoryObject
    """
    from itertools import chain
    for attrName in chain(dir(obj.__class__), dir(obj)):
        if attrName.startswith('__'):
            continue
        attr = getattr(obj, attrName)
        if callable(attr):
            setattr(obj, attrName, lambda *args, **kwargs: print('NoneType'))
        else:
            setattr(obj, attrName, None)


def numpyMatrixToProtobufMatrix(numpyMatrix: numpy.ndarray) -> PBMatrix:
    """
    Must be C-contiguous.
    """
    if not numpyMatrix.flags['C_CONTIGUOUS']:
        raise Error.ArgumentError('Matrix must be C-contiguous!', ModuleErrorCode, FileErrorCode, 1)

    protobufMatrix = PBMatrix()
    protobufMatrix.shape[:] = numpyMatrix.shape
    for value in numpy.nditer(numpyMatrix):
        val = value.reshape(1, 1)[0][0]
        complexValue = protobufMatrix.array.add()
        if isinstance(val, float):
            complexValue.real = val
        else:
            complexValue.real = val.real
            complexValue.imag = val.imag
    return protobufMatrix


def protobufMatrixToNumpyMatrix(protobufMatrix: PBMatrix) -> numpy.ndarray:
    """
    Must be C-contiguous.
    """
    complexArray = [
        complex(complexValue.real, complexValue.imag) if complexValue.HasField('imag') else complexValue.real for
        complexValue in protobufMatrix.array]
    return numpy.array(complexArray).reshape(protobufMatrix.shape)


def numpyMatrixToDictMatrix(numpyMatrix: numpy.ndarray) -> Dict:
    """
    Must be C-contiguous.
    """

    if not numpyMatrix.flags['C_CONTIGUOUS']:
        raise Error.ArgumentError('Matrix must be C-contiguous!', ModuleErrorCode, FileErrorCode, 2)

    if numpyMatrix.size == 0:
        return {}

    array = []
    dictMatrix = {
        'shape': list(numpyMatrix.shape),
        'array': array
    }

    for value in numpy.nditer(numpyMatrix):
        val = value.reshape(1, 1)[0][0]
        if isinstance(val, (int, float)):
            array.append({
                'real': val
            })
        else:
            array.append({
                'real': val.real,
                'imag': val.imag
            })
    return dictMatrix


def dictMatrixToNumpyMatrix(dictMatrix: Dict, valueType: Union[Type[complex], Type[float]]) -> numpy.ndarray:
    """
    Must be C-contiguous.
    """

    if len(dictMatrix) == 0:
        return numpy.empty(0, valueType)

    if valueType == complex:
        complexArray = [complex(complexValue['real'], complexValue['imag']) if not isinstance(complexValue, (
            int, float)) else complexValue for complexValue in dictMatrix['array']]
    else:
        complexArray = [complexValue['real'] if not isinstance(complexValue, (
            int, float)) else complexValue for complexValue in dictMatrix['array']]
    return numpy.array(complexArray).reshape(dictMatrix['shape'])


def normalizeNdarrayOrderForTranspose(matrix: numpy.ndarray) -> numpy.ndarray:
    old_shape = matrix.shape
    matrix = numpy.reshape(matrix, numpy.prod(matrix.shape))
    return numpy.reshape(matrix, old_shape)


def nKron(AMatrix: numpy.ndarray, BMatrix: numpy.ndarray, *args) -> numpy.ndarray:
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


def contract1_1(matrixParking: numpy.ndarray, matrixFloating: numpy.ndarray) -> numpy.ndarray:
    """
    The first indicator of the gate is the output,
    and the latter indicator is the input,
    just like writing matrix vector multiplication, U{ket0}
    """
    if matrixFloating.shape != (2, 2):
        raise Error.ArgumentError("Floating gate not a 1-qubit gate", ModuleErrorCode, FileErrorCode, 3)
    if matrixParking.shape != (2, 2):
        raise Error.ArgumentError("Parking gate not a 1-qubit gate", ModuleErrorCode, FileErrorCode, 4)
    newArray = numpy.einsum('ab,bc->ac', matrixParking,
                            matrixFloating)  # v is a vector, A B is a matrix, we must count ABv, here we count AB
    return newArray


def contract1_2(matrixParking: numpy.ndarray, matrixFloating: numpy.ndarray, upOrDown: int,
                leftOrRight: int) -> numpy.ndarray:
    """
    matrixParking is like 1010.
    The first half of the indicators are outputs,
    and the second half are inputs.
    """
    if matrixFloating.shape != (2, 2):
        raise Error.ArgumentError("Floating gate not a 1-qubit gate", ModuleErrorCode, FileErrorCode, 5)
    if matrixParking.shape == (4, 4):
        matrixParking = numpy.reshape(matrixParking, [2, 2, 2, 2])
    elif matrixParking.shape != (2, 2, 2, 2):
        raise Error.ArgumentError("Parking gate not a 2-qubit gate", ModuleErrorCode, FileErrorCode, 6)
    newArray = None  # type: Optional[numpy.ndarray]
    if leftOrRight == 0:
        if upOrDown == 0:
            newArray = numpy.einsum("abcd,de->abce", matrixParking, matrixFloating)
        elif upOrDown == 1:
            newArray = numpy.einsum("abcd,ce->abed", matrixParking, matrixFloating)
    elif leftOrRight == 1:
        if upOrDown == 0:
            newArray = numpy.einsum("eb,abcd->aecd", matrixFloating, matrixParking)
        elif upOrDown == 1:
            newArray = numpy.einsum("ea,abcd->ebcd", matrixFloating, matrixParking)
    return numpy.reshape(newArray, [4, 4])
