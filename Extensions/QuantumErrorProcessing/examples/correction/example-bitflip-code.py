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
This is a simple example to demonstrate the three-qubit bit-flip quantum error correction code.
The three-qubit bit-flip code is a quantum error correction code that can protect a single qubit
against a single bit-flip error. To encode a single qubit,
we first prepare a state called the "code word" as follows:

.. math::

    \vert 0 \rangle \rightarrow \vert 000 \rangle

    \vert 1 \rangle \rightarrow \vert 111 \rangle

A valid logical state is a superposition of two possible states: :math:`\vert 000 \rangle`
and :math:`\vert 111 \rangle`, which are orthogonal to each other.
Any single bit-flip error that occurs on one of the three qubits can be detected and corrected by
measuring all three qubits at the end of a quantum computation.
The three-qubit bit-flip code is one of the simplest quantum error correction codes and
is often used as a building block for more complex codes.
"""

import QCompute

from Extensions.QuantumErrorProcessing.qcompute_qep.correction import BasicCode, ColorTable
import Extensions.QuantumErrorProcessing.qcompute_qep.utils.circuit as circuit


# To construct the three-qubit bit-flip code, we need to specify its stabilizers and detectable error types.
bitflip_code = BasicCode(stabilizers=["IZZ", "ZZI"], error_types=["X"], name="Bit-Flip Code")

# After we create the code, we can check its basic information
print(bitflip_code)

# You can check the encoding, detecting, correcting, and decoding quantum circuits individually
bitflip_code.print_encode_circuit()
bitflip_code.print_detect_circuit()
bitflip_code.print_correct_circuit()
bitflip_code.print_decode_circuit()

# You can also combine these methods together to see the composed quantum circuit
bitflip_code.print_encode_decode_circuit()
bitflip_code.print_detect_correct_circuit()
bitflip_code.print_encode_detect_correct_decode_circuit()

# Now we show to apply the bit-flip code to protect a single-qubit quantum state.
qp = QCompute.QEnv()
qp.Q.createList(1)
QCompute.X(qp.Q[0])

print("*******************************************************************************")
print("The raw quantum circuit is:")
circuit.print_circuit(qp.circuit, colors={"red": [0]})

# Step 1. Encode the quantum state
enc_qp = bitflip_code.encode(qp=qp)
print("After encoding, the quantum circuit is:")
circuit.print_circuit(enc_qp.circuit, colors={"red": [0], "blue": [1, 2]})

###########################################################
# Any single bit-flip error happens here ...
###########################################################

# Step 2. Detect single bit-flip error
det_qp = bitflip_code.detect(qp=enc_qp)
print("After detection, the quantum circuit is:")
circuit.print_circuit(det_qp.circuit, colors={"red": [0], "blue": [1, 2], "yellow": [3, 4]})

# Step 3. Correct the quantum error based on the detecting result
cor_qp = bitflip_code.correct(qp=det_qp)
print("After correction, the quantum circuit is:")
circuit.print_circuit(cor_qp.circuit, colors={"red": [0], "blue": [1, 2], "yellow": [3, 4]})

# Step 4. Correct the quantum error based on the detecting result
dec_qp = bitflip_code.decode(qp=cor_qp)
print("After decoding, the quantum circuit is:")
circuit.print_circuit(dec_qp.circuit, colors={"red": [0], "blue": [1, 2], "yellow": [3, 4]})

# Visualize the qubit order, as indicated by different colors
print(
    "Qubits Order: [{}][{}][{}]".format(
        ColorTable.ANCILLA + "Ancilla" + ColorTable.END,
        ColorTable.PHYSICAL + "Physical" + ColorTable.END,
        ColorTable.ORIGINAL + "Original" + ColorTable.END,
    )
)
