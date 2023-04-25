#!/usr/bin/python3
# -*- coding: utf-8 -*-
#
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
This is a simple example to demonstrate the five-qubit quantum error correction code, which is the smallest
quantum error correcting code that can protect a logical qubit from any arbitrary single qubit error.
It is originally invented by R Laflamme, C Miquel, JP Paz, and WH Zurek in the following paper:

.. [LMPZ96] Laflamme, Raymond, et al.
            "Perfect quantum error correcting code."
            Physical Review Letters 77.1 (1996): 198.

We will illustrate how to construct and simulate this code in QEP.
"""
import copy

import QCompute
from qcompute_qep.correction import FiveQubitCode, ColorTable
from qcompute_qep.utils import circuit


def print_counts(counts: dict, n: int, k: int):
    r"""Print measurement outcomes by splitting different roles.

    :param counts:
    :param n:
    :param k:
    :return:
    """
    for key, val in counts.items():
        # Print ancilla qubits
        print("'", end="")
        if n - k > 0:
            print("{}".format(ColorTable.ANCILLA + key[0:n - k] + ColorTable.END), end='')
            print("{}".format(ColorTable.PHYSICAL + key[n - k:2 * (n - k)] + ColorTable.END), end='')
        print("{}".format(ColorTable.ORIGINAL + key[2 * (n - k):2 * n - k] + ColorTable.END), end='')
        print("': {}".format(val))


def state_without_qec(qp: QCompute.QEnv, idx: int):
    r"""Test without correction.
    """
    qp = copy.deepcopy(qp)
    # Add Noise
    QCompute.ID(qp.Q[idx])
    qp.noise(gateNameList=['ID'], noiseList=[QCompute.Depolarizing(bits=1, probability=1)], qRegList=[idx])

    # Step 5. measure
    QCompute.MeasureZ(*qp.Q.toListPair())

    # Commit the computation task and fetch the results
    qp.backend(QCompute.BackendName.LocalBaiduSim2)
    result = qp.commit(shots=8192, fetchMeasure=True)
    # Obtain the 'counts' information for the computation result
    print_counts(result["counts"], n=1, k=1)


def state_with_qec(qp: QCompute.QEnv, idx: int):
    r"""Test five-qubit code error correction.
    """
    qec_code = FiveQubitCode()
    # Step 1. encode
    enc_qp = qec_code.encode(qp)
    # Step 2. Damaged by depolarizing error in the i-th qubit
    QCompute.ID(enc_qp.Q[idx])
    enc_qp.noise(gateNameList=['ID'],
                 noiseList=[QCompute.Depolarizing(bits=1, probability=1)],
                 qRegList=[idx])
    # Step 3. detect and correct
    cor_qp = qec_code.detect_and_correct(enc_qp)
    # Step 4. decode
    dec_qp = qec_code.decode(cor_qp)

    # Step 5. measure
    QCompute.MeasureZ(*dec_qp.Q.toListPair())
    counts = circuit.execute(qp=dec_qp, qc=QCompute.BackendName.LocalBaiduSim2, shots=8192)
    print_counts(counts, n=qec_code.n, k=qec_code.k)


if __name__ == '__main__':

    print("*******************************************************************************")
    print("We demonstrate a simple example that uses the five-qubit code to \n"
          "protect quantum state. Here is the five-qubit code information:")
    print(FiveQubitCode().info)

    # Assume we have the following single-qubit quantum state to be protected
    qp = QCompute.QEnv()
    qp.Q.createList(1)
    QCompute.X(qp.Q[0])
    print("*******************************************************************************")
    print("The original quantum circuit for preparing the quantum state is:")
    circuit.print_circuit(qp.circuit)
    print("*****************************************************************")
    print("Assume that Q[0] suffers from the completely depolarizing error.")
    print("Without QEC, the measurement results are (shots=8192):")
    state_without_qec(qp=qp, idx=0)

    indices = [0, 1, 2, 3, 4]
    for idx in indices:
        print("*******************************************************************************")
        print("Assume that Q[{}] suffers from the completely depolarizing error.".format(idx))
        print("With QEC, the measurement results are (shots=8192):")
        state_with_qec(qp, idx=idx)
        print("Qubits Order: [{}][{}][{}]".format(ColorTable.ANCILLA + "Ancilla" + ColorTable.END,
                                                  ColorTable.PHYSICAL + "Physical" + ColorTable.END,
                                                  ColorTable.ORIGINAL + "Original" + ColorTable.END))
        print("We see that the error is successfully detected and corrected.")
