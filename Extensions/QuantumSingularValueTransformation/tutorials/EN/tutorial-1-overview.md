# Overview

*Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

## About QSVT Toolkit

**QSVT** toolkit is a **Q**uantum **S**ingular **V**alue **T**ransformation toolkit based on [QCompute](https://quantum-hub.baidu.com/opensource) and developed by the [Institute for Quantum Computing](https://quantum.baidu.com) at [Baidu Research](http://research.baidu.com). It aims to implement quantum simulation and other algorithms on quantum devices or simulators more conveniently. Currently, it includes three main modules:

+ **Quantum Singular Value Transformation** (QSVT) is used for implementing singular value transformations of quantum operations, whose input and output are both block-encodings of quantum operations. 

+ **Symmetric Quantum Signal Processing** (SQSP) is used for encoding such transformation functions and completing such quantum circuits in QSVT. SQSP is introduced for implementing the encoding step more effectively.

+ **Hamiltonian Simulation** is one of the most significant applications for QSVT, and even quantum computing. It provides functions to generate quantum circuits for time evolution operators of Hamiltonians.

## Tutorials

QSVT toolkit provides [quick start](https://quantum-hub.baidu.com/qsvt/tutorial-quickstart) for using the Hamiltonian Simulation module, as well as a [brief introduction](https://quantum-hub.baidu.com/qsvt/tutorial-introduction) to the theories for users to learn and get started. The current content of the following tutorial is organized as follows, and it is recommended that beginners read and study in order:

- [Brief Introduction](https://quantum-hub.baidu.com/qsvt/tutorial-introduction)
- [Quantum Signal Processing](https://quantum-hub.baidu.com/qsvt/tutorial-qsp)
- [Block-Encoding and Linear Combination of Unitary Operations](https://quantum-hub.baidu.com/qsvt/tutorial-be)
- [Quantum Eigenvalue and Singular Value Transformation](https://quantum-hub.baidu.com/qsvt/tutorial-qet)
- [Hamiltonian Simulation](https://quantum-hub.baidu.com/qsvt/tutorial-hs)

We will supply more detailed and comprehensive tutorials in the future. 

## Frequently Asked Questions

**Q: What are the required packages to use QSVT toolkit?**

**A:** QSVT toolkit is based on [QCompute](https://quantum-hub.baidu.com/opensource), a Python-based open-source quantum computing platform SDK also developed by [Institute for Quantum Computing](https://quantum.baidu.com) at [Baidu Research](http://research.baidu.com). It provides a full-stack programming experience for senior users via hybrid quantum programming language features and high-performance simulators. You can install QCompute via [pypi](https://pypi.org/project/qcompute/). When you install QSVT toolkit, the dependency QCompute will be automatically installed. Please refer to QCompute's official [Open Source](https://quantum-hub.baidu.com/opensource) page for more details.

**Q: What should I do when running out of my credit points?**

**A:** Please contact us via [Quantum Hub](https://quantum-hub.baidu.com/). First, you should log into [Quantum Hub](https://quantum-hub.baidu.com/), then enter the "Feedback" page, choose "Get Credit Point", and record the necessary information. Submit your feedback and wait for our reply. We will reach you as soon as possible.

**Q: How should I cite QSVT toolkit in my research?**

**A:** We encourage researchers and developers to use QSVT toolkit to explore quantum algorithms. If your work uses QSVT toolkit, please feel free to send us a notice via [quantum@baidu.com](mailto:quantum@baidu.com) and cite us with the following BibTeX:

```BibTex
@misc{QSVT,
      title = {{Quantum Sigular Value Transformation toolkit in Baidu Quantum Platform}},
      year = {2022},
      url = {https://quantum-hub.baidu.com/qsvt/}
}
```

## Feedbacks
Users are encouraged to contact us via email [quantum@baidu.com](mailto:quantum@baidu.com) with general questions, unfixed bugs, and potential improvements. We hope to make QSVT toolkit better together with the community!

## Copyright and License

QSVT toolkit is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).