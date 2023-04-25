English | [简体中文](README_CN.md)

# QCompute-QAPP User's Guide

*Copyright (c) 2021 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

<p align="center">
  <!-- docs -->
  <a href="https://quantum-hub.baidu.com/docs/qapp/">
    <img src="https://img.shields.io/badge/docs-link-green.svg?style=flat-square&logo=read-the-docs"/>
  </a>
  <!-- PyPI -->
  <a href="https://pypi.org/project/qcompute-qapp/">
    <img src="https://img.shields.io/badge/pypi-v0.0.1-orange.svg?style=flat-square&logo=pypi"/>
  </a>
  <!-- Python -->
  <a href="https://www.python.org/">
    <img src="https://img.shields.io/badge/Python-3.6+-blue.svg?style=flat-square&logo=python"/>
  </a>
  <!-- License -->
  <a href="./LICENSE">
    <img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg?style=flat-square&logo=apache"/>
  </a>
  <!-- Platform -->
  <a href="https://github.com/baidu/QCompute/tree/master/Extensions/QuantumApplication">
    <img src="https://img.shields.io/badge/OS-MacOS%20|%20Windows%20|%20Linux-lightgrey.svg?style=flat-square"/>
  </a>
</p>

## QAPP Introduction

QAPP is a quantum computing toolbox based on the [QCompute](https://quantum-hub.baidu.com/opensource) component of [Quantum Leaf](https://quantum-hub.baidu.com/), which provides quantum computing services for solving problems in many fields including quantum chemistry, combinatorial optimization, machine learning, etc. QAPP provides users with a one-stop quantum computing application development function, which directly connects to users' real requirements in artificial intelligence, financial technology, education and research.

## QAPP Architecture

QAPP architecture follows the complete development logic from application to real machine, including four modules: Application, Algorithm, Circuit, and Optimizer. The Application module converts the user requirements into the corresponding mathematical problem; the Algorithm module selects a suitable quantum algorithm to solve the mathematical problem; during the solution process, the user can specify the optimizer provided in the Optimizer module or design a custom optimizer; the quantum circuit required for the solution process is supported by the Circuit module. The Circuit module directly calls the [QCompute](https://quantum-hub.baidu.com/opensource) platform, and supports calls to the [Quantum Leaf](https://quantum-hub.baidu.com/services) simulators or QPUs.

![QAPP architecture](tutorials/figures/overview-fig-QAPPlandscape-EN.png "Figure 1: QAPP architecture")

## QAPP Use Cases

We provide QAPP practical cases such as solving [molecular ground state energy](tutorials/VQE_EN.md), solving [combinatorial optimization problem](tutorials/Max_Cut_EN.md), and solving [classification problem](tutorials/Kernel_Classifier_EN.md). These use cases are designed to help users quickly get started with calling QAPP modules and developing custom algorithms. Before we can run these use cases, there is some preparation work to do.

### Install Conda and Python environment

We use [Anaconda](https://www.anaconda.com/download) as the development environment management tool for Python. Anaconda supports multiple mainstream operating systems (Windows, macOS, and Linux). Here we provide a tutorial on how to use conda to create and manage virtual environments:

1. First, enter the command line (Terminal) interface: Windows users can enter `Anaconda Prompt`; Mac users can use the key combination `command⌘ + space` and then enter `Terminal`.

2. After opening the Terminal window, enter

   ```bash
   conda create --name qapp_env python=3.8
   ```
   
   to create a Python 3.8 environment named `qapp_env`. With the following command, we can enter the virtual environment created, 
   
   ```bash
   conda activate qapp_env
   ```

For more detailed instructions on conda, please refer to the [Official Tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).

### Credit Points

If you run QAPP with cloud servers, you will consume Quantum-hub credit points.  For more Quantum-hub credit points, please contact us via [Quantum Hub](https://quantum-hub.baidu.com). First, you should log into [Quantum Hub](https://quantum-hub.baidu.com), then enter the "Feedback" page, choose "Get Credit Point", and record the necessary information. Submit your feedback and wait for our reply. We will reach you as soon as possible.

### Install QCompute and packages required by QAPP

Install QAPP with `pip`:

```bash
pip install qcompute-qapp
```

> Some use cases may require additional packages, which are clarified in the corresponding tutorials.

### Run

Users can download the `tutorials` folder from GitHub, switch the path to the `tutorials` folder where the case is located in Terminal, and run it in Python. For example,

```bash
python vqe_example.py
```

## API Documentation

We provide QAPP's [API](API_Document.pdf) documentation for developers to look up. Users can also view the API documentation on the [Quantum Leaf website](https://quantum-hub.baidu.com/docs/qapp/).

## Citation

We encourage the researchers and developers to use QAPP for research & development on quantum computing applications. Please cite us by including the following BibTeX entry:

```bibtex
@misc{QAPP, 
  title = {{Quantum Application Python Package}}, 
  year = {2022}, 
  url = {https://quantum-hub-test.baidu.com/qapp/}, 
}
```

## Copyright and License

QAPP uses [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0).
