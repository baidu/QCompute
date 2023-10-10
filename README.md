English | [简体中文](README_CN.md)

# Quantum Leaf (in Chinese: 量易伏) - QComputeSDK

- [Features](#features)
- [Install](#install)
   - [Environment Setup](#environment-setup)
   - [Install QComputeSDK](#install-qcomputeSDK)
   - [Run Example](#run-example)
   - [Breaking Change](#breaking-change)
- [Introduction and Developments](#introduction-and-developments)
   - [Tutorials](#tutorials)
   - [API Documentation](#api-documentation)
   - [Development](#development)
- [Discussion and Feedbacks](#discussion-and-feedbacks)
- [Develop with QComputeSDK](#develop-with-qcomputesdk)
- [Frequently Asked Questions](#faq)
- [Copyright and License](#copyright-and-license)

[Quantum Leaf](https://quantum-hub.baidu.com/) is the world's first cloud-native quantum computing platform developed by the [Institute for Quantum Computing at Baidu Research](https://quantum.baidu.com/). Users can use the Quantum Leaf for quantum programming, quantum simulation and running real quantum computers. Quantum Leaf aims to provide a quantum foundation development environment for QaaS (Quantum infrastructure as a Service).

![](https://release-data.cdn.bcebos.com/github-qleaf%2F%E9%87%8F%E6%98%93%E4%BC%8F%E5%9B%BE%E6%A0%87.png)

[![](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE) ![](https://img.shields.io/badge/build-passing-green) ![](https://img.shields.io/badge/Python-3.9--3.11-blue) ![](https://img.shields.io/badge/release-v3.3.5-blue) 

![](https://img.shields.io/badge/OS-MacOS%20|%20Windows%20|%20Linux-lightgrey.svg?style=flat-square)

The QComputeSDK installation package is a complete open-source quantum computing framework implemented in Python. It adopts the classic quantum hybrid programming method and presets various advanced modules. Users can use the quantum environment object (QEnv) to quickly build quantum circuits, and can also use it to develop various complex quantum algorithms. QComputeSDK has multiple interfaces for local simulators, cloud simulators and real machines, allowing users to quickly simulate and verify quantum algorithms in local, and submit circuit tasks to real quantum hardware (superconductors, ion traps) or high-performance simulators on cloud.

## Features

- Easy-to-use
   - Nearly 50 tutorial cases, and still increasing.
   - Quantum circuit local visualization.
   - Automatically call modules to complete quantum circuit compilation.
- Versatile
   - Support quantum circuit modularization.
   - The local high-performance simulator supports the simulation of 32 qubits.
   - The high-performance heterogeneous simulators on the cloud supports more qubit simulations.
   - Support the simulation of various quantum noise models.
   - Local GPU simulator based on NVIDIA cuQuantum.
   - Local photonic quantum simulator supports the Gaussian/Fork state.
- Real quantum computing power
   - Access to QPUQian, Baidu's superconducting quantum computer.
   - Access to IonAPM, the ion trap quantum computer of the Innovation Academy for Precision Measurement Science and Technology, CAS.
   - Access to IoPCAS, the superconducting quantum computer of the Institute of Physics, CAS.

## Install

### Environment Setup

We recommend using conda to manager virtual environments,
```bash
conda create -n qcompute_env python=3.10
conda activate qcompute_env
```
> Please refer to [Anaconda](https://www.anaconda.com/download#downloads)'s official installation.

> Note: Python version >= 3.9 

### Install QComputeSDK

Install QComputeSDK with `pip`,

```bash
pip install qcompute
```

or download all the files to install from sources. We recommend this installation. You can download from GitHub,
```bash
git clone https://github.com/baidu/QCompute.git
cd QCompute
pip install -e .
```
or download from Gitee,
```bash
git clone https://gitee.com/baidu/qcompute.git
cd qcompute
pip install -e .
```

### Run Example
If all the files have been downloaded, you can now try to run a program to verify whether the installation is successful. Here we run the test script provided by QComputeSDK,
```bash
python -m Test.PostInstall.PostInstall_test
```
User Token needs to be given on the command line before cloud testing, You can log in to [Quantum Leaf](https://quantum-hub.baidu.com/token)  to check your Token. If you don't need to do cloud testing, please type `Ctrl+c`.

> Note: Please skip this step if you installed with `pip`.

### Breaking Change

Starting with QComputeSDK 3.0.0, developers can run Baidu's superconducting quantum computer through QComputeSDK (device identifier: `CloudBaiduQPUQian`). The device provides services regularly, which can be viewed from the [Services Status](https://quantum-hub.baidu.com/services).

## Introduction and Developments

### Tutorials

QComputeSDK is a quantum computing development framework that implements backend access to real quantum hardware. It builds a bridge between quantum computing and quantum hardware, providing strong support for the research and development of quantum algorithms and applications, and also providing a wealth of cases for developers to learn from.

Here we provide primary, intermediate and advanced cases. With primary case, you can quickly get started with QComputeSDK, it includes quantum state preparation, classical quantum hybrid programming, circuit task submission to quantum computers, etc. The intermediate case is the use of QComputeSDK, including the calling of modules, the use of convertors, etc. The advanced case is the implementation of advanced quantum algorithms on QComputeSDK. We have provided detailed tutorial documents for these algorithms.

- [Primary Cases](./Example/Level_1)
  1. [GHZ state preparation (local)](./Example/Level_1/GHZ_Local.py)
  2. [GHZ state preparation (cloud)](./Example/Level_1/GHZ_Cloud.py)
  3. [Bell state preparation (local)](./Example/Level_1/HCNOT_Local.py)
  4. [Bell state preparation (cloud)](./Example/Level_1/HCNOT_Cloud.py)
  5. [Classical quantum hybrid language (local)](./Example/Level_1/Hybrid_Local.py)
  6. [Classical quantum hybrid language (cloud)](./Example/Level_1/Hybrid_Cloud.py)
  7. [Classical quantum information interaction (local)](./Example/Level_1/Interact_Local.py)
  8. [Classical quantum information interaction (cloud)](./Example/Level_1/Interact_Cloud.py)
  9. [QPU - BaiduQPUQian](./Example/Level_1/QPUCase_BaiduQPUQian.py)
  10. [QPU - IonAPM](./Example/Level_1/QPUCase_IonAPM.py)
  11. [QPU - IoPCAS](./Example/Level_1/QPUCase_IoPCAS.py)
  12. [Photonic quantum circuit simulation based on Fock state](./Example/Level_1/PhotonicFookCase_local.py)
  13. [Photonic quantum circuit simulation based on Gaussian state](./Example/Level_1/PhotonicGaussianCase_local.py)
  14. [GPU simulator based on cuQuantum](./Example/Level_1/SimulatorCase_CuQuantum_Local.py)
  15. [Universal blind quantum computation](./Example/Level_1/Ubqc.py)
  - [Quantum Noise Simulation](./Example/Level_1/Noise)
    1. [Adding noise to the circuit](./Example/Level_1/Noise/AddNoise.py)
    2. [Quantum noise compression module](./Example/Level_1/Noise/CompressNoiseTest.py)
    3. [One-qubit circuit with noise simulation](./Example/Level_1/Noise/OneQubitNoiseTest.py)
    4. [Noise simulation with multiple processes](./Example/Level_1/Noise/ParallelNoiseSimulationTest.py)
    5. [Two-qubit circuit with noise simulation](./Example/Level_1/Noise/TwoQubitNoiseTest.py)

- [Intermediate Cases](./Example/Level_2)
  - [Output Information Settings](./Example/Level_2/0_OutputFormatControl)
    1. [Tutorial](./Example/Level_2/0_OutputFormatControl/Tutorials/OutputFormatControl_EN.md)
    2. [Results printing information settings](./Example/Level_2/0_OutputFormatControl/0_OutputFormatControl.py)
    3. [Output file automatic cleaning](./Example/Level_2/0_OutputFormatControl/1_AutoClearOutputDir.py)
  - [General Modules](./Example/Level_2/1_OpenModules)
    1. [Tutorial](./Example/Level_2/1_OpenModules/Tutorials/OpenModules_EN.md)
    2. [Module usage examples](./Example/Level_2/1_OpenModules/0_OpenModules.py)
    3. [Quantum circuit inverse module](./Example/Level_2/1_OpenModules/1_InverseCircuitModule.py)
    4. [Quantum circuit reverse module](./Example/Level_2/1_OpenModules/2_ReverseCircuitModule.py)
    5. [Quantum procedure unroll module](./Example/Level_2/1_OpenModules/3_UnrollProcedureModule.py)
    6. [Quantum gate decomposition module](./Example/Level_2/1_OpenModules/4_UnrollCircuitModule.py)
    7. [Quantum gate compression module](./Example/Level_2/1_OpenModules/5_CompressGateModule.py)
  - [Convertors](./Example/Level_2/2_OpenConvertors)
    1. [Tutorial](./Example/Level_2/2_OpenConvertors/Tutorials/OpenConvertors_EN.md)
    2. [Circuit serialization](./Example/Level_2/2_OpenConvertors/0_Circuit.py)
    3. [Console drawing](./Example/Level_2/2_OpenConvertors/1_DrawConsole.py)
    4. [Convert circuit serialization and deserialization](./Example/Level_2/2_OpenConvertors/2_InternalStruct.py)
    5. [Convert circuit to JSON](./Example/Level_2/2_OpenConvertors/3_JSON.py)
    6. [Convert circuit to QASM](./Example/Level_2/2_OpenConvertors/4_QASM.py)
    7. [Convert circuit to QOBJ](./Example/Level_2/2_OpenConvertors/5_Qobj.py)
    8. [Convert circuit to IonQ](./Example/Level_2/2_OpenConvertors/6_IonQ.py)
    9. [Convert circuit to Xanadu](./Example/Level_2/2_OpenConvertors/7_XanaduSF.py)

- [Advanced Cases](./Example/Level_3)
  1. [Quantum Superdense Coding](./Example/Level_3/0_SuperdenseCoding/Tutorial-Superdense/Superdense_EN.md)
  2. [Deutsch-Jozsa Algorithm](./Example/Level_3/1_Deutsch-Jozsa/Tutorial-DJ/Deutsch-Jozsa_EN.md)
  3. [Quantum Phase Estimation (QPE)](./Example/Level_3/2_PhaseEstimation/Tutorial-phase/Phase_EN.md)
  4. [Grover's Search Algorithm](./Example/Level_3/3_Grover/Tutorial-Grover/Grover_EN.md)
  5. [Shor's Algorithm](./Example/Level_3/4_ShorAlgorithm/tutorial/Shor_EN.md)
  6. [Variational Quantum Eigensolver (VQE)](./Example/Level_3/5_VQE/Tutorial-VQE/VQE_EN.md)
  7. [Variational Quantum State Diagonalization (VQSD)](./Example/Level_3/6_VQSD/Tutorial-VQSD/VQSD_EN.md)

In recent updates, QComputeSDK has added a photonic quantum computing simulator (LocalBaiduSimPhotonic). Unlike traditional quantum circuit, photonic quantum computing has its own unique way of running. QComputeSDK supports the optical system on the architecture, and also becomes the first quantum development kit that integrates quantum computing and photonic quantum computing. Interested readers can refer to [Photonic Quantum Computing Simulator Tutorial](https://quantum-hub.baidu.com/pqs/tutorial-introduction).

### API Documentation

To learn more about how to use QComputeSDK, please refer to the [API documentation](https://quantum-hub.baidu.com/docs/qcompute/latest/), which contains detailed descriptions and usage of all functions and classes available to users.

### Development

QComputeSDK includes quantum computing architecture, quantum computing simulator, tutorials, and extensions. For developers who need to involve the code of the architecture or simulator, it is recommended to install from sources. For developers or researchers who use QComputeSDK to develop quantum algorithm applications, it is recommended to use [GHZ_Cloud.py](./Example/Level_1/GHZ_Cloud.py) as the code framework. Modifying and using this file can effectively help you learn the syntax of this quantum development kit. It is recommended that developers be familiar with the circuit construction of QComputeSDK, and pay attention to the qubit output order.

## Discussion and Feedbacks

- We welcome your questions, reports and suggestions. You can feedback through the following channels:
  - [GitHub Issues](https://github.com/baidu/QCompute/issues) / [Gitee Issues](https://gitee.com/baidu/qcompute/issues)
  - [Quantum Leaf Feedback](https://quantum-hub.baidu.com/feedback)
  -  Email: quantum@baidu.com
- You are welcomed to join our discussion QQ group (group number: 1147781135). You can scan QR code into the group.

![](https://release-data.cdn.bcebos.com/github-qleaf%2Fqrcode.png)

## Develop with QComputeSDK

We welcome developers to use QComputeSDK for quantum application development. If your work uses QComputeSDK, we also welcome you to contact us. The following are quantum applications developed based on QComputeSDK:
- [QEP (Quantum Error Processing)](https://quantum-hub.baidu.com/qep/tutorial-overview), a set of quantum noise processing tools developed by the Institute for Quantum Computing at Baidu Research. It offers four powerful functions: performance evaluation, quantum error characterization, quantum error mitigation, and quantum error correction.
- [UBQC (Universal Blind Quantum Computation)](https://quantum-hub.baidu.com/bqc/tutorial-bqc), a blind quantum computing proxy service based on the UBQC protocol developed by the Institute for Quantum Computing at Baidu Research.
- [QAPP](https://quantum-hub.baidu.com/qapp/tutorial-overview) is a set of quantum computing solution tools developed based on QComputeSDK, providing quantum computing solution services for a variety of field problems including quantum chemistry, combinatorial optimization, and machine learning.
- [QSVT (Quantum Singular Value Transformation)](https://quantum-hub.baidu.com/qsvt/tutorial-overview), a set of quantum singular value transformation tools developed by the Institute for Quantum Computing at Baidu Research, with main functions including quantum singular value transformation, symmetric quantum signal processing, and Hamiltonian quantum simulation.
- [QFinance](https://quantum-hub.baidu.com/qfinance/tutorial-option-pricing), a quantum finance library developed by the Institute for Quantum Computing at Baidu Research, providing a Quantum Monte Carlo method for price European options.
- [PQS (Photonic Quantum Simulator)](https://quantum-hub.baidu.com/pqs/tutorial-introduction), a photonic quantum computing simulator developed by the Institute for Quantum Computing at Baidu Research, supporting photonic quantum circuit simulation based on Gaussian state and Fock state.

## FAQ
1. Question: **What can be done with QComputeSDK? What are the applications?**

    Answer: QComputeSDK is a quantum computing development framework based on Python that can be used to build, run, and optimize quantum circuits. We have built a comprehensive and complete infrastructure in QComputeSDK to support the implementation of various quantum algorithms, it has a wide range of application scenarios in the development of quantum applications. Specific work can be referred to but not limited to the [Extensions](./Extensions) in QComputeSDK.

2. Question：**I want to use QComputeSDK for quantum programming, but I don't know much about quantum computing. How do I get started?**

    Answer: Quantum Computation and Quantum Information by Nielsen & Chuang is the classic introductory textbook to QC. We recommend readers to study Chapter 1, 2, and 4 of this book first. These chapters introduce the basic concepts, provide solid mathematical and physical foundations, and discuss the quantum circuit model widely used in QC. Readers can also learn on [QuLearn](https://qulearn.baidu.com/), which is an online quantum learning knowledge base that not only contains quantum computing tutorials, but also rich video sources. Readers can also download the Quantum Leaf APP (https://quantum-hub.baidu.com/qmobile), and the Playground on the APP contains a wealth of interesting quantum examples to help readers learn anytime, anywhere.

3. Question: **Is QComputeSDK free?**

    Answer: QComputeSDK is free. QComputeSDK is an open source SDK. It is free for users to execute local simulation tasks. When the user submits the task to the cloud simulator or the real machine through QComputeSDK, a certain number of points will be deducted. For detailed deduction rules, please refer to the [User Guide](https://quantum-hub.baidu.com/quickGuide/account). When the user creates an account, we will give away points. The point balance can be viewed in the [Personal Center](https://quantum-hub.baidu.com/profile).

4. Question: **How can I get more points?**

    Answer: Points are currently only used for resource control. If the points are insufficient, you can submit an application from the [Feedback](https://quantum-hub.baidu.com/feedback) on the Quantum Leaf website or the [User Feedback](https://quantum-hub.baidu.com/qmobile) of Quantum Leaf APP. We will deal with your request in three working days.


## Copyright and License

QComputeSDK uses [Apache-2.0 license](./LICENSE).
