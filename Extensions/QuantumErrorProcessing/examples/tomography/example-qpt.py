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
In this script, we calibrate via process tomography a list of frequently used single- and two-qubit gates.
Current supported quantum gates are: {Z, H, CZ, CNOT, Toffoli, SWAP, CSWAP}.
For the gates that are native in PCAS hardware, {Z, H, CZ}, we calibrate them on qubits 0 and 1 directly.
For the gates that are not native in PCAS hardware, {CNOT, Toffoli, SWAP, CSWAP}, we offer two implementations:

1. We decompose them using the native gates and then perform process tomography for the large quantum circuit;
2. We use the corresponding gates supported in Quantum Leaf and rely on its mapping module.

These two different implementations enable us to compare the efficiency of different implementations.
We verify the decompositions for {CNOT, Toffoli, SWAP, CSWAP} in the auxiliary function `test_decomposition()`.
"""
import typing

from QCompute import *
import Extensions.QuantumErrorProcessing.qcompute_qep.tomography as tomography
import Extensions.QuantumErrorProcessing.qcompute_qep.utils.types as types
import Extensions.QuantumErrorProcessing.qcompute_qep.exceptions as exceptions


def process_tomography(qp: types.QProgram, qc: types.QComputer, gate_name: str):
    r"""Execute tomography for the quantum process specified by @qp on the quantum computer @qc.

    :param qp: QProgram, quantum program for creating the target quantum process
    :param qc: QComputer, the quantum computer instance
    :param gate_name: str, the quantum gate's name to do tomography
    """
    qc_name = types.get_qc_name(qc)
    print("*** Tomography the quantum gate {} in the quantum computer '{}' now ...".format(gate_name, qc_name))

    # Step 1. Perform quantum process tomography
    st = tomography.ProcessTomography()
    # Call the tomography procedure and obtain the noisy gate
    noisy_ptm = st.fit(qp, qc, prep_basis='Pauli', meas_basis='Pauli', method='inverse', shots=4096, ptm=True).data

    # Step 3. Analyze the data: compute the average gate fidelity of two quantum maps
    print("****** The average gate fidelity of quantum gate {} is: {:.5f}".format(gate_name, st.fidelity))

    # Visualize these PTMs
    tomography.compare_process_ptm(ptms=[st.ideal_ptm, noisy_ptm, st.ideal_ptm - noisy_ptm],
                                   titles=['Theoretical', qc_name, 'Difference'],
                                   show_labels=True,
                                   fig_name="QPT-{}-{}.png".format(qc_name, gate_name))

    print("*** Tomography the quantum gate {} in the quantum computer '{}' DONE!".format(gate_name, qc_name))


def cnot(qp: types.QProgram, indices: typing.List[int]):
    r"""Add a CNOT gate to the given quantum program.

    The qubit indices are assumed in the following order:

    ::

        indices[0]: ---@---
                       |
        indices[1]: ---X---

    We decompose the CNOT gate via Hadamard and CZ gates:

    ::

        indices[0]: -------@-------
                           |
        indices[1]: ---H---Z---H---

    :param qp: QProgram, quantum program for which a CNOT gate will be added to the end
    :param indices: List[int], the involved quantum qubits
    """
    # Extract the control and target qubit indices
    if len(indices) != 2:
        raise exceptions.ArgumentError("in cnot(): the number of qubits are not correct for the CNOT gate!")
    c = indices[0]
    t = indices[1]
    # Decompose CNOT via Hadamard and CZ
    H(qp.Q[t])
    CZ(qp.Q[c], qp.Q[t])
    H(qp.Q[t])


def toffoli(qp: types.QProgram, indices: typing.List[int]):
    r"""Add a Toffoli gate to the given quantum program.

    The qubit indices are assumed in the following order:

    ::

        indices[0]: ---@---
                       |
        indices[1]: ---@---
                       |
        indices[2]: ---X---

    We decompose the Toffoli gate using the Hadamard, phase, \pi/8 and CZ gates.
    Based on the quantum circuit decomposition in Figure 4.9 of Nielsen's book, illustrated as follows:

    ::

        indices[0]: -----------------@-----------------@---@---------@---T---
                                     |                 |   |         |
        indices[1]: -------@-----------------@---TDG-------X---TDG---X---S---
                           |         |       |         |
        indices[2]: ---H---X---TDG---X---T---X---TDG---X---T----H------------

    :param qp: QProgram, quantum program for which a Toffoli gate will be added to the end
    :param indices: List[int], the involved quantum qubits
    """
    # Extract the control and target qubit indices
    if len(indices) != 3:
        raise exceptions.ArgumentError("in toffoli(): the number of qubits are not correct for the Toffoli gate!")
    c_0 = indices[0]
    c_1 = indices[1]
    t = indices[2]

    H(qp.Q[t])
    # CNOT c_1 -> t
    cnot(qp, [c_1, t])

    TDG(qp.Q[t])
    # CNOT c_0 -> t
    cnot(qp, [c_0, t])

    T(qp.Q[t])
    # CNOT c_1 -> t
    cnot(qp, [c_1, t])

    TDG(qp.Q[t])
    # CNOT c_0 -> t
    cnot(qp, [c_0, t])

    TDG(qp.Q[c_1])
    # CNOT c_0 -> c_1
    cnot(qp, [c_0, c_1])

    TDG(qp.Q[c_1])
    # CNOT c_0 -> c_1
    cnot(qp, [c_0, c_1])

    T(qp.Q[c_0])
    S(qp.Q[c_1])
    T(qp.Q[t])
    H(qp.Q[t])


if __name__ == '__main__':

    ###################################################################################################################
    # Set the quantum hardware in process tomography
    ###################################################################################################################
    # For numeric test on the ideal simulator, change qc to BackendName.LocalBaiduSim2
    qc = BackendName.LocalBaiduSim2

    # For experiment on the real quantum device, change qc to BackendName.CloudBaiduQPUQian.
    # You must set your VIP token first in order to access the Baidu hardware.
    # Define.hubToken = "Token"
    # qc = BackendName.CloudBaiduQPUQian

    # For numeric test on the noisy simulator, change qc to Qiskit's FakeSantiago simulator
    # from qiskit.providers.fake_provider import FakeSantiago
    # qc = FakeSantiago()

    ###################################################################################################################
    # Example 1. Calibrating the Z gate
    ###################################################################################################################

    # Calibrating the native single-qubit Z gate
    qp = QEnv()
    qp.Q.createList(1)
    Z(qp.Q[0])

    # Quantum process tomography
    process_tomography(qp, qc, gate_name='Z')

    ###################################################################################################################
    # Example 2. Calibrating the H gate
    ###################################################################################################################

    # Calibrating the native single-qubit H gate
    qp = QEnv()
    qp.Q.createList(1)
    H(qp.Q[0])

    # Quantum process tomography
    process_tomography(qp, qc, gate_name='H')

    ###################################################################################################################
    # Example 3. Calibrating the CZ gate
    ###################################################################################################################

    # Calibrating the native CZ gate `q0 -> q1`.
    # CZ has no controlled direction, thus `CZ: q0 -> q1` and `CZ: q1 -> q2` have the same matrix representation.
    qp = QEnv()
    qp.Q.createList(2)
    CZ(qp.Q[0], qp.Q[1])

    # Quantum process tomography
    process_tomography(qp, qc, gate_name='CZ')

    ###################################################################################################################
    # Example 4. Calibrating the CNOT gate
    ###################################################################################################################

    # Calibrate the CNOT gate `q1 -> q0`. Notice that CNOT gate is not native in PCAS,
    # thus we have to manually decompose it using the native CZ and H gates.
    qp = QEnv()
    qp.Q.createList(2)
    cnot(qp, [1, 0])

    # Quantum process tomography
    process_tomography(qp, qc, gate_name='CNOT')

    ###################################################################################################################
    # Example 5. Calibrating the CNOT gate without manually decomposing it
    ###################################################################################################################

    # Calibrate the CNOT gate `q1 -> q0`. Notice that the CNOT gate originally supported in Quantum Leaf.
    # In this case, we can compare the efficiency of different implementations.
    qp = QEnv()
    qp.Q.createList(2)
    CX(qp.Q[0], qp.Q[1])

    # Quantum process tomography
    process_tomography(qp, qc, gate_name='CNOT-Mapping')

    ###################################################################################################################
    # Example 6. Calibrating the Toffoli gate
    ###################################################################################################################

    # Calibrate the Toffoli gate `(q2, q1) -> q0`. Notice that Toffoli gate is not native in PCAS,
    # thus we have to manually decompose it using Hadamard, phase, \pi/8 and CZ gates.
    # Based on the quantum circuit decomposition in Figure 4.9 of Nielsen's book.
    qp = QEnv()
    qp.Q.createList(3)
    toffoli(qp, [2, 1, 0])

    # Quantum process tomography
    process_tomography(qp, qc, gate_name='Toffoli')

    ###################################################################################################################
    # Example 7. Calibrating the Toffoli gate without manually decomposing it
    ###################################################################################################################

    # Calibrate the Toffoli gate `(q2, q1) -> q0`.
    # Notice that the Toffoli gate originally supported in Quantum Leaf, we do not decompose it manually as above.
    # In this case, we can compare the efficiency of different implementations.
    qp = QEnv()
    qp.Q.createList(3)
    CCX(qp.Q[2], qp.Q[1], qp.Q[0])

    # Quantum process tomography
    process_tomography(qp, qc, gate_name='Toffoli-Mapping')

    ###################################################################################################################
    # Example 8. Calibrating the SWAP gate
    ###################################################################################################################

    # Calibrate the SWAP gate `q0 <-> q1`. Notice that SWAP gate is not native in PCAS,
    # thus we have to manually decompose it using CZ and Hadamard gates and is illustrated as follows:
    #         q[1]: ---@---                   q[1]: ---@---X---@---
    #                  |          <===>                |   |   |
    #         q[0]: ---@---                   q[0]: ---X---@---X---

    qp = QEnv()
    qp.Q.createList(2)
    cnot(qp, [1, 0])
    cnot(qp, [0, 1])
    cnot(qp, [1, 0])

    # Quantum process tomography
    process_tomography(qp, qc, gate_name='SWAP')

    ###################################################################################################################
    # Example 9. Calibrating the SWAP gate without manually decomposing it
    ###################################################################################################################

    # Calibrate the SWAP gate `q0 <-> q1`.
    # Notice that the SWAP gate originally supported in Quantum Leaf, we do not decompose it manually as above.
    # In this case, we can compare the efficiency of different implementations.
    qp = QEnv()
    qp.Q.createList(2)
    SWAP(qp.Q[0], qp.Q[1])

    # Quantum process tomography
    process_tomography(qp, qc, gate_name='SWAP-Mapping')

    ###################################################################################################################
    # Example 10. Calibrating the CSWAP gate (controlled-SWAP gate)
    #               Notice that the CSWAP gate is also known as the Fredkin gate.
    ###################################################################################################################

    # Calibrate the CSWAP gate `q2 -> (q1, q0)`. Notice that CSWAP gate is not native in PCAS,
    # thus we have to manually decompose it using Hadamard, phase, \pi/8 and CZ gates.
    # Based on Figure 10 in `arXiv:quant-ph/0205095` and is illustrated as follows:
    #         q[2]: ---@---                   q[2]: -------@-------
    #                  |                                   |
    #         q[1]: ---@---       <===>       q[1]: ---X---@---X---
    #                  |                               |   |   |
    #         q[0]: ---X---                   q[0]: ---@---X---@---

    qp = QEnv()
    qp.Q.createList(3)
    # CNOT q0 -> q1
    cnot(qp, [0, 1])
    # Toffoli q2, q1 -> q0
    toffoli(qp, [2, 1, 0])
    # CNOT q0 -> q1
    cnot(qp, [0, 1])

    # Quantum process tomography
    process_tomography(qp, qc, gate_name='CSWAP')

    ###################################################################################################################
    # Example 11. Calibrating the CSWAP gate without manually decomposing it
    #               Notice that the CSWAP gate is also known as the Fredkin gate.
    ###################################################################################################################

    # Calibrate the CSWAP gate `q2 -> (q1, q0)`.
    # Notice that the CSWAP gate originally supported in Quantum Leaf, we do not decompose it manually as above.
    # In this case, we can compare the efficiency of different implementations.
    qp = QEnv()
    qp.Q.createList(3)
    CSWAP(qp.Q[2], qp.Q[1], qp.Q[0])

    # Quantum process tomography
    process_tomography(qp, qc, gate_name='CSWAP-Mapping')
