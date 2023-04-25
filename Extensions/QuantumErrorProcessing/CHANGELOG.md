# Changelog

## [1.1.0] - March-2023

``ADDED`` 

+ Quantum error correction code (QECC) simulator, enabling users to simulate stabilizer codes.

+ Quantum direct fidelity estimation method, enabling users to estimate the fidelities of 
  quantum states and quantum processes efficiently.

+ Quantum cross-platform fidelity estimation method, enabling users to estimate the fidelity of 
  two mixed quantum states prepared in separate experimental platforms. 

``IMPROVED`` 

+ Support callable quantum computer design. Any object that implements the `quantum_computer()` 
  interface serves as a valid quantum computer for all functions in QEP.

+ Support quantum circuits batch model, wherein the `execute()` function can accommodate a list 
  of quantum programs as input. This function runs the programs in a batch mode on the targeted 
  quantum computer.

``FIXED`` Quantum gateset tomography now can compute the fidelity of the computational basis 
measurement.

## [1.0.1] - August-2022

``FIXED`` Disable mapping and enable unrolling when calling quantum hardware.

## [1.0.0] - July-2022

``ADDED`` Interleaved randomized benchmarking, quantum detector tomography,
and quantum gateset tomography.

``FIXED`` Benchmarking and tomography methods now can specify target qubits. Adapt to QCompute 3.0.

``IMPROVED`` Improve user experience by adding progress bar to methods.

## [0.1.0] - May-2022

Release QEP, a Quantum Error Processing toolkit developed by the Institute for Quantum Computing at Baidu Research.

``ADDED``

+ Quantum randomized benchmarking (RB), including standard RB, cross-entropy RB, and unitarity RB.

+ Quantum error characterization, including state tomography, process tomography, and spectral tomography.

+ Quantum error mitigation (EM), including gate EM and measurement EM.
