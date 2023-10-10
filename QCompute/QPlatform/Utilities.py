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
Utilities
"""
FileErrorCode = 13


import functools
from typing import Dict, TYPE_CHECKING, Optional, Tuple, Union, Type, List

import numpy

from QCompute.QPlatform import Error, ModuleErrorCode
from QCompute.QPlatform.QNoise import QNoise
from QCompute.QPlatform.QNoise.AmplitudeDamping import AmplitudeDamping
from QCompute.QPlatform.QNoise.BitFlip import BitFlip
from QCompute.QPlatform.QNoise.BitPhaseFlip import BitPhaseFlip
from QCompute.QPlatform.QNoise.CustomizedNoise import CustomizedNoise
from QCompute.QPlatform.QNoise.Depolarizing import Depolarizing
from QCompute.QPlatform.QNoise.PauliNoise import PauliNoise
from QCompute.QPlatform.QNoise.PhaseDamping import PhaseDamping
from QCompute.QPlatform.QNoise.PhaseFlip import PhaseFlip
from QCompute.QPlatform.QNoise.ResetNoise import ResetNoise
from QCompute.QPlatform.QOperation.FixedGate import getFixedGateInstance
from QCompute.QPlatform.QOperation.RotationGate import createRotationGateInstance
from QCompute.QProtobuf import PBMatrix, PBQNoise, PBFixedGate, PBRotationGate

if TYPE_CHECKING:
    from QCompute.QProtobuf import PBCircuitLine, PBCustomizedGate


def destoryObject(obj: object) -> None:
    """
    Destory Object
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
    Convert the matrix from numpy format to protobuf format. Must be C-contiguous.
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
    Convert the matrix from protobuf format to numpy format. Must be C-contiguous.
    """
    complexArray = [
        complex(complexValue.real, complexValue.imag) if complexValue.HasField('imag') else complexValue.real for
        complexValue in protobufMatrix.array]
    return numpy.array(complexArray).reshape(protobufMatrix.shape)


def numpyMatrixToDictMatrix(numpyMatrix: numpy.ndarray) -> Dict:
    """
    Convert the matrix from numpy format to dict format. Must be C-contiguous.
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
    Convert the matrix from dict format to numpy format. Must be C-contiguous.
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


def protobufQNoiseToQNoise(protobufQNoise: PBQNoise) -> QNoise:
    """
    Convert the noise from protobuf format to QNoise format.
    """
    noiseType = protobufQNoise.WhichOneof('noise')
    if noiseType == 'amplitudeDamping':
        quantumNoise = AmplitudeDamping(
            protobufQNoise.amplitudeDamping.probability)
    elif noiseType == 'bitFlip':
        quantumNoise = BitFlip(protobufQNoise.bitFlip.probability)
    elif noiseType == 'bitPhaseFlip':
        quantumNoise = BitPhaseFlip(
            protobufQNoise.bitPhaseFlip.probability)
    elif noiseType == 'customizedNoise':
        quantumNoise = CustomizedNoise(
            list(map(protobufMatrixToNumpyMatrix, protobufQNoise.customizedNoise.krauses)))
    elif noiseType == 'depolarizing':
        quantumNoise = Depolarizing(protobufQNoise.depolarizing.bits,
                                        protobufQNoise.depolarizing.probability)
    elif noiseType == 'pauliNoise':
        quantumNoise = PauliNoise(protobufQNoise.pauliNoise.probability1, protobufQNoise.pauliNoise.probability2,
                                    protobufQNoise.pauliNoise.probability3)
    elif noiseType == 'phaseDamping':
        quantumNoise = PhaseDamping(
            protobufQNoise.phaseDamping.probability)
    elif noiseType == 'phaseFlip':
        quantumNoise = PhaseFlip(protobufQNoise.phaseFlip.probability)
    elif noiseType == 'resetNoise':
        quantumNoise = ResetNoise(
            protobufQNoise.resetNoise.probability1, protobufQNoise.resetNoise.probability2)
    else:
        raise Error.ArgumentError(f'Unsupported protobufQNoise type {noiseType}!', ModuleErrorCode, FileErrorCode, 3)

    return quantumNoise

def contract1_1(matrixParking: numpy.ndarray, matrixFloating: numpy.ndarray) -> numpy.ndarray:
    r"""
    The first indicator of the gate is the output,
    and the latter indicator is the input,
    just like writing matrix vector multiplication, :math:`U |0\rangle`
    """
    if matrixFloating.shape != (2, 2):
        raise Error.ArgumentError('Floating gate not a 1-qubit gate', ModuleErrorCode, FileErrorCode, 4)
    if matrixParking.shape != (2, 2):
        raise Error.ArgumentError('Parking gate not a 1-qubit gate', ModuleErrorCode, FileErrorCode, 5)
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
        raise Error.ArgumentError('Floating gate not a 1-qubit gate', ModuleErrorCode, FileErrorCode, 6)
    if matrixParking.shape == (4, 4):
        matrixParking = numpy.reshape(matrixParking, [2, 2, 2, 2])
    elif matrixParking.shape != (2, 2, 2, 2):
        raise Error.ArgumentError('Parking gate not a 2-qubit gate', ModuleErrorCode, FileErrorCode, 7)
    newArray: Optional[numpy.ndarray] = None
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


def contract_in_3(matrixPre: numpy.ndarray, 
                qRegsPre: List[int],
                matrixPost: numpy.ndarray,
                qRegsPost: List[int]) -> Tuple[numpy.ndarray, List[int]]:
    """
    Contract two matrices on at most 3 qubits.

    :param matrixPre: a matrix

    :param qRegsPre: the qubits that matrixPre is applied on

    :param matrixPost: a matrix

    :param qRegsPost: the qubits that matrixPost is applied on

    :return: the multiplication of matrices and corresponding qRegs
    """   
    # Calculate maximum qRegs
    qRegsOut = list(set(qRegsPre).union(set(qRegsPost)))

    assert len(qRegsOut) <= 3, print('These two matrices work on more than 3 qubits')

    qRegsPreCopy = qRegsPre.copy()
    qRegsPostCopy = qRegsPost.copy()
    # Expand matrix to qubits qRegsOut
    if len(qRegsPre) != len(qRegsOut):
        matrixPre, qRegsPreCopy = expandMatrix(matrixPre, qRegsPre, qRegsOut)
    if len(qRegsPost) != len(qRegsOut):
        matrixPost, qRegsPostCopy = expandMatrix(matrixPost, qRegsPost, qRegsOut)

    qResPreShuffled = []
    qRegsPostShuffled = []
    # Shuffle matrix by ascent order of qubits
    matrixPreShuffled, qResPreShuffled = shuffleMatrix(matrixPre, qRegsPreCopy)
    matrixPostShuffled, qRegsPostShuffled = shuffleMatrix(matrixPost, qRegsPostCopy)

    assert qResPreShuffled == qRegsPostShuffled, print('Wrong in expandMatrix or shuffleMatrix')

    # Matrix multiplication
    matrixOut = numpy.matmul(matrixPostShuffled, matrixPreShuffled)
    return matrixOut, qRegsPostShuffled


def toTensorMatrix(matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Transform a matrix into a tensor matrix.
    If it already is, do nothing

    :param matrix: a matrix

    :return: a tensor matrix
    """
    bits = int(numpy.log2(numpy.sqrt(numpy.prod(matrix.shape))) + 0.5)

    if matrix.shape == (2 ** bits, 2 ** bits):
        matrix = matrix.reshape((2, 2) * bits)
    elif matrix.shape == (2, 2) * bits:
        pass
    else:
        print('Please input a n-qubit matrix')
    
    return matrix


def toSquareMatrix(matrix: numpy.ndarray) -> numpy.ndarray:
    """
    Transform a matrix into a square matrix.
    If it already is, do nothing

    :param matrix: a matrix

    :return: a square matrix
    """
    bits = int(numpy.log2(numpy.sqrt(numpy.prod(matrix.shape))) + 0.5)

    if matrix.shape == (2 ** bits, 2 ** bits):
        pass
    elif matrix.shape == (2, 2) * bits:
        matrix = matrix.reshape((2 ** bits, 2 ** bits))
    else:
        print('Please input a n-qubit matrix')
    
    return matrix


def shuffleMatrix(gateMatrix: numpy.ndarray,  qRegs: List[int]) -> numpy.ndarray:
    """
    shuffle a matrix to in ascent order of qubits

    :param gate: a matrix

    :param qReg: the bits that the gate is applied on

    :param target_qRegs: the bits that gate are supposed to reshuffle. If default,

    :return: reordered gate by ascent order of qRegs
    """
    bits = len(qRegs)
    # Transform to tensor matrix
    gateMatrix = toTensorMatrix(gateMatrix)

    qRegsOut = sorted(qRegs)

    # The first half indices are outputs, and the left are inputs.
    # The output indices change to target_qRegs, while the input indices change in the reversed way
    # Sort output index through first sorting index by values from big to small and then reverse it by values
    idxAscent = [x[0] for x in sorted(enumerate(qRegs), key=lambda x:x[1], reverse = True)]
    transpostOutputIndex = [bits - 1 - _ for _ in idxAscent]
    transpostInputIndex = [bits + _ for _ in transpostOutputIndex]
    transpostIndies = transpostOutputIndex + transpostInputIndex

    matrixOut = numpy.transpose(gateMatrix, transpostIndies).reshape((2 ** bits, 2 ** bits))
    return matrixOut, qRegsOut


def expandMatrix(matrix: numpy.ndarray, 
                qRegs: List[int], 
                qRegsTarget: List[int]) -> Tuple[numpy.ndarray, List[int]]:
    """
    Expand a matrix to a larger space.
    If a matrix 

    :param matrix: a matrix

    :param qRegs: the qubits that matrixPre is applied on

    :param qRegsTarget: the qubits that matrixPre is expanded to

    :return: a expanded matrix and corresponding qRegs
    """   

    # Transform to square matrix
    matrixOut = toSquareMatrix(matrix)
    qRegsOut = qRegs.copy()
    dim = 2 ** len(qRegs)

    for qRegsTemp in qRegsTarget:
        if qRegsTemp not in qRegs:
            # Expand n-qubit matrix to (n+1)-qubit matrix
            matrixTemp = numpy.zeros((2 * dim, 2 * dim), dtype=numpy.complex128)
            matrixTemp[0:dim, 0:dim] = matrixOut
            matrixTemp[dim:2*dim, dim:2*dim] = matrixOut 

            # Update matrixOut and qRegsOut
            matrixOut =  matrixTemp
            qRegsOut.append(qRegsTemp)
            dim = 2 * dim

    return matrixOut, qRegsOut


def getProtobufCicuitLineMatrix(pbCircuitLine: 'PBCircuitLine') -> numpy.ndarray:
    """
    get the pbCircuitLine matrix.

    :param pbCircuitLine: Protobuf format of the circuitLine

    :return: the matrix of the circuitLine
    """

    op = pbCircuitLine.WhichOneof('op')
    if op == 'fixedGate':
        fixedGate: PBFixedGate = pbCircuitLine.fixedGate
        gateName = PBFixedGate.Name(fixedGate)
        matrix = getFixedGateInstance(gateName).getMatrix()
    elif op == 'rotationGate':
        rotationGate: PBRotationGate = pbCircuitLine.rotationGate
        gateName = PBRotationGate.Name(rotationGate)
        matrix = createRotationGateInstance(gateName, *pbCircuitLine.argumentValueList).getMatrix()
    elif op == 'customizedGate':
        customizedGate: PBCustomizedGate = pbCircuitLine.customizedGate
        matrix = protobufMatrixToNumpyMatrix(customizedGate.matrix)
    else:
        raise Error.ArgumentError(f'InternalStruct Unsupported operation {op}!', ModuleErrorCode, FileErrorCode, 8)
    return matrix