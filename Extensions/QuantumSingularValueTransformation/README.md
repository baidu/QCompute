*Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

[![](https://img.shields.io/badge/license-Apache%202.0-green)](./LICENSE) [![](https://img.shields.io/badge/build-passing-green)]() ![](https://img.shields.io/badge/Python-3.9-blue) ![](https://img.shields.io/badge/release-v0.1.1-red)

## About QSVT Toolkit

**QSVT** toolkit is a **Q**uantum **S**ingular **V**alue **T**ransformation toolkit based on [QCompute](https://quantum-hub.baidu.com/opensource) and developed by the [Institute for Quantum Computing](https://quantum.baidu.com) at [Baidu Research](http://research.baidu.com). It aims to implement quantum simulation and other algorithms on quantum devices or simulators more conveniently. Currently, it includes three main modules:

+ **Quantum Singular Value Transformation** (QSVT) is used for implementing singular value transformations of quantum operations, whose input and output are both block-encodings of quantum operations. 

+ **Symmetric Quantum Signal Processing** (SQSP) is used for encoding such transformation functions and completing such quantum circuits in QSVT. SQSP is introduced for implementing the encoding step more effectively.

+ **Hamiltonian Simulation** is one of the most significant applications for QSVT, and even quantum computing. It provides functions to generate quantum circuits for time evolution operators of Hamiltonians.

QSVT toolkit is based on [QCompute](https://quantum-hub.baidu.com/opensource), a Python-based open-source quantum computing platform SDK also developed by [Institute for Quantum Computing](https://quantum.baidu.com). It provides a full-stack programming experience for senior users via hybrid quantum programming language features and high-performance simulators. You can install QCompute via [pypi](https://pypi.org/project/qcompute/). When you install QSVT toolkit, the dependency QCompute will be automatically installed. Please refer to QCompute's official [Open Source](https://quantum-hub.baidu.com/opensource) page for more details.

## Installation

### Create Python Environment

We recommend using Anaconda as the development environment management tool for Python3. Anaconda supports multiple mainstream operating systems (Windows, macOS, and Linux). It provides Scipy, Numpy, Matplotlib, and many other scientific computing and drawing packages. Conda, a virtual environment manager, can be used to install and update the latest Python packages. Here we give simple instructions on how to use conda to create and manage virtual environments.

1. First, open the command line (Terminal) interface: Windows users can enter ``Anaconda Prompt``; macOS users can use the key combination ``command + space`` and then enter ``Terminal``.

2. After opening the Terminal window, enter

    ```bash
    conda create --name qsvt_env python=3.9
    ```

    to create a Python 3.9 environment named ``qsvt_env``. With the following command, we can activate the virtual environment created,

    ```bash
    conda activate qsvt_env
    ```

For more detailed instructions on conda, please refer to the [Official Tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

### Install QSVT Toolkit

QSVT toolkit is compatible with 64-bit Python 3.9, on Linux, macOS (10.14 or later) and Windows. We recommend installing it with ``pip``. After Activating the conda environment, then enter

```bash
pip install qcompute-qsvt
```

This will install the QSVT toolkit binaries as well as the QSVT toolkit package. 

For those using an older version of QSVT toolkit, update by installing with the `--upgrade` flag. The new version includes additional features and bug fixes.

### Run Examples

Now, you can try to write a simple program to check whether QSVT toolkit has been successfully installed. For example, run the following program,

```python
import numpy as np
from QCompute import *
from Extensions.QuantumSingularValueTransformation.qcompute_qsvt.Application.HamiltonianSimulation import func_HS_QSVT

func_HS_QSVT(list_str_Pauli_rep=[(1, 'X0X1'), (1, 'X0Z1'), (1, 'Z0X1'), (1, 'Z0Z1')],
             num_qubit_sys=2, float_tau=-np.pi / 8, float_epsilon=1e-6, circ_output=False)
```

we will operate a time evolution operator on initial state $|00\rangle$ for Hamiltonian $X\otimes X + X\otimes Z + Z\otimes X + Z\otimes Z$ and time $-\pi/8$ with precision `1e-6`, and then measure such final state.

Note that more examples are provided in the [source codes](https://github.com/baidu/QCompute/tree/master/Extensions/QuantumSingularValueTransformation), [tutorials](https://quantum-hub.baidu.com/docs/qsvt) and [API documentation](https://quantum-hub.baidu.com/docs/qsvt/). You can get started from there.

## Tutorials

QSVT toolkit provides [quick start](https://quantum-hub.baidu.com/qsvt/tutorial-quickstart) for using the Hamiltonian Simulation module, as well as a [brief introduction](https://quantum-hub.baidu.com/qsvt/tutorial-introduction) to the theories for users to learn and get started. The current content of the following tutorial is organized as follows, and it is recommended that beginners read and study in order:

- [Brief Introduction](https://quantum-hub.baidu.com/qsvt/tutorial-introduction)
- [Quantum Signal Processing](https://quantum-hub.baidu.com/qsvt/tutorial-qsp)
- [Block-Encoding and Linear Combination of Unitary Operations](https://quantum-hub.baidu.com/qsvt/tutorial-be)
- [Quantum Eigenvalue and Singular Value Transformation](https://quantum-hub.baidu.com/qsvt/tutorial-qet)
- [Hamiltonian Simulation](https://quantum-hub.baidu.com/qsvt/tutorial-hs)

We will supply more detailed and comprehensive tutorials in the future. 

## API Documentation

For those who are looking for explanation on the python classes and functions in QSVT toolkit, please refer to our [API documentation](https://quantum-hub.baidu.com/docs/qsvt/).

## Feedbacks

Users are encouraged to contact us via email quantum@baidu.com with general questions, unfixed bugs, and potential improvements. We hope to make QSVT toolkit better together with the community!

## Research based on QSVT Toolkit

We encourage researchers and developers to use QSVT toolkit to explore quantum algorithms. If your work uses QSVT toolkit, please feel free to send us a notice via quantum@baidu.com and cite us with the following BibTeX:

```BibTex
@misc{QSVT,
      title = {{Quantum Sigular Value Transformation toolkit in Baidu Quantum Platform}},
      year = {2022},
      url = {https://quantum-hub.baidu.com/qsvt/}
}
```

## Changelog

The changelog of this project can be found in [CHANGELOG.md](https://github.com/baidu/QCompute/blob/master/Extensions/QuantumSingularValueTransformation/CHANGELOG.md).

## Copyright and License

QSVT toolkit uses [Apache-2.0 license](https://github.com/baidu/QCompute/blob/master/Extensions/QuantumSingularValueTransformation/LICENSE).