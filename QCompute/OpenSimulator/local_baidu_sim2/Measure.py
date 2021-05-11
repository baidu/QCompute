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
Measurement
Two measurement methods (Meas_METHOD) are provided:
1) (SINGLE) Single shot accumulation
2) (PROB) Sampling based on probabilities
"""

from collections import Counter
from enum import IntEnum, unique
from typing import Union, Dict, Tuple, Optional

import numpy



from QCompute.OpenSimulator.local_baidu_sim2.InitState import MatrixType
from QCompute.OpenSimulator.local_baidu_sim2.Transfer import TransferProcessor, Algorithm
from QCompute.QPlatform import Error


@unique
class MeasureMethod(IntEnum):
    """
    Two measurement types: Meas_MED = MEAS_METHOD.PROB and Meas_MED = MEAS_METHOD.SINGLE.
    PROB is to sample many shots from the probability distribution while
    SINGLE is to do one shot to a circuit constructed and let the state collapse.
    """

    Probability = 0
    OutputProbability = Probability + 1
    OutputState = OutputProbability + 1
    Accumulation = OutputState + 1


def pow2(x: float) -> float:
    """
    Pow 2
    """

    return pow(x, 2)


def numToBinStr(num: int, n: int) -> str:
    """
    Oct to bin in the common order, not reversed

    :param num: original number
    :param n: total qubit number
    :return: bin_string
    """

    bin_string = bin(num)[2:].zfill(n)
    # return (''.join(reversed(bin_string)))
    return bin_string


class Measurer:
    """
    Quantum Measurer
    """

    def __init__(self, matrixType: MatrixType, algorithm: Algorithm, measureMethod: MeasureMethod) -> None:
        """
        To choose the algorithms by the parameters.
        """

        if measureMethod == MeasureMethod.Probability:
            if matrixType == MatrixType.Dense:
                self.proc = self._measureDenseByProbability
            else:
                raise Error.RuntimeError('Not implemented')
        elif measureMethod == MeasureMethod.OutputProbability:
            if matrixType == MatrixType.Dense:
                self.proc = self._measureDenseByOutputProbability
            else:
                raise Error.RuntimeError('Not implemented')
        elif measureMethod == MeasureMethod.OutputState:
            if matrixType == MatrixType.Dense:
                self.proc = self._measureDenseByOutputState
            else:
                raise Error.RuntimeError('Not implemented')
        elif measureMethod == MeasureMethod.Accumulation:
            self.Transfer = TransferProcessor(matrixType, algorithm)
            self.proc = self._measureBySingleAccumulation
        else:
            assert False

    def __call__(self, state: Union[numpy.ndarray, 'COO'], shots: int) -> \
            Optional[Union[Dict[str, int], Dict[str, float]]]:
        """
        To enable the object callable
        """

        return self.proc(state, shots)

    def _measureSingle(self, state: numpy.ndarray, bit: int) -> Tuple[int, Union[numpy.ndarray, 'COO']]:
        """
        One-qubit measurement
        """

        n = len(state.shape)
        axis = list(range(n))
        axis.remove(n - 1 - bit)
        probs = numpy.sum(numpy.abs(state) ** 2, axis=tuple(axis))
        rnd = numpy.random.rand()

        # measure single bit
        if rnd < probs[0]:
            out = 0
            prob = probs[0]
        else:
            out = 1
            prob = probs[1]

        # collapse single bit
        if out == 0:
            matrix = numpy.array([[1.0 / numpy.sqrt(prob), 0.0],
                                  [0.0, 0.0]], complex)
        else:
            matrix = numpy.array([[0.0, 0.0],
                                  [0.0, 1.0 / numpy.sqrt(prob)]], complex)
        state = self.Transfer(state, matrix, [bit])

        return out, state

    def _measureAll(self, state: numpy.ndarray) -> str:
        """
        Measure all by measuring qubit one by one
        """

        n = len(state.shape)
        outs = ''
        for i in range(n):
            # The collapse of bit0 after measuring bit0 affects the subsequent measurement of bit1 but does not affect the 1000 independent measurements of the previous layer
            out, state = self._measureSingle(state, i)
            outs = str(out) + outs  # Low to high
        return outs

    def _measureBySingleAccumulation(self, state: numpy.ndarray, shots: int) -> Dict[str, int]:
        """
        Measure by accumulation, one shot at a time
        """

        # print("Measure method Single Accu")
        result = {}  # type: Dict[str, int]
        for i in range(shots):
            outs = self._measureAll(state)
            if outs not in result:
                result[outs] = 0
            result[outs] += 1
        return result

    def _measureDenseByProbability(self, state: numpy.ndarray, shots: int) -> Dict[str, int]:
        """
        Measure by probability
        """

        # print("Measure method Probability")
        n = len(state.shape)

        prob_array = numpy.reshape(numpy.abs(state) ** 2, [2 ** n])

        """
        prob_key = []
        prob_values = []
        pos_list = list(numpy.nonzero(prob_array)[0])
        for index in pos_list:
            string = _numToBinStr(index, n)
            prob_key.append(string)
            prob_values.append(prob_array[index])

        # print("The sum prob is ", sum(prob_values))

        samples = numpy.random.choice(len(prob_key), shots, p=prob_values)
        """

        samples = numpy.random.choice(range(2 ** n), shots, p=prob_array)
        count_samples = Counter(samples)
        result = {}  # type: Dict[str, int]
        for idex in count_samples:
            """
            result[prob_key[idex]] = count_samples[idex]
            """
            result[numToBinStr(idex, n)] = count_samples[idex]
        return result

    

    def _measureDenseByOutputProbability(self, state: numpy.ndarray, shots: int) -> Dict[str, float]:
        """
        Output probability
        """

        # print("Measure method Probability")
        n = len(state.shape)

        prob_array = numpy.reshape(numpy.abs(state) ** 2, [2 ** n])

        """
        prob_key = []
        prob_values = []
        pos_list = list(numpy.nonzero(prob_array)[0])
        for index in pos_list:
            string = _numToBinStr(index, n)
            prob_key.append(string)
            prob_values.append(prob_array[index])

        # print("The sum prob is ", sum(prob_values))

        samples = numpy.random.choice(len(prob_key), shots, p=prob_values)
        """

        result = {}
        indices = numpy.nonzero(prob_array != 0)[0]
        for idex in indices:
            result[numToBinStr(idex, n)] = prob_array[idex]
        return result

    

    def _measureDenseByOutputState(self, state: numpy.ndarray, shots: int) -> None:
        """
        Output state
        """

        # return {'0' * len(state.shape): ascii(state.tolist())}

        return None

    
