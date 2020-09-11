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

import numpy



from QCompute.OpenSimulator.local_baidu_sim2.InitState import MatrixType
from QCompute.OpenSimulator.local_baidu_sim2.Transfer import TransferProcessor
from QCompute.QuantumPlatform import Error


@unique
class MeasureMethod(IntEnum):
    """
    Two measurement types: Meas_MED = MEAS_METHOD.PROB and Meas_MED = MEAS_METHOD.SINGLE.
    PROB is to sample many shots from the probability distribution while
    SINGLE is to do one shot to a circuit constructed and let the state collapse.
    """

    Probability = 0
    Accumulation = Probability + 1


def pow2(x):
    """
    Pow 2
    """

    return pow(x, 2)


def numToBinStr(num, n):
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

    def __init__(self, matrixType, algorithm, measureMethod):
        """
        To choose the algorithms by the parameters.
        """

        if measureMethod == MeasureMethod.Probability:
            if matrixType == MatrixType.Dense:
                self.proc = self.measureDenseByProbability
            else:
                raise Error.RuntimeError('Not implemented')
        else:
            self.Transfer = TransferProcessor(matrixType, algorithm)
            self.proc = self.measureBySingleAccumulation

    def __call__(self, state, shots):
        """
        To enable the object callable
        """

        return self.proc(state, shots)

    def measureSingle(self, state, bit):
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

    def measureAll(self, state):
        """
        Measure all by measuring qubit one by one
        """

        n = len(state.shape)
        outs = ''
        for i in range(n):
            # The collapse of bit0 after measuring bit0 affects the subsequent measurement of bit1 but does not affect the 1000 independent measurements of the previous layer
            out, state = self.measureSingle(state, i)
            outs = str(out) + outs  # Low to high
        return outs

    def measureBySingleAccumulation(self, state, shots):
        """
        Measure by accumulation, one shot at a time
        """

        # print("Measure method Single Accu")
        result = {}
        for i in range(shots):
            outs = self.measureAll(state)
            if outs not in result:
                result[outs] = 0
            result[outs] += 1
        return result

    def measureDenseByProbability(self, state, shots):
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
        result = {}
        for idex in count_samples:
            """
            result[prob_key[idex]] = count_samples[idex]
            """
            result[numToBinStr(idex, n)] = count_samples[idex]
        return result


