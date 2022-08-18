*Copyright (c) 2022 Institute for Quantum Computing, Baidu Inc. All Rights Reserved.*

![](https://img.shields.io/badge/release-v1.1.0-blue)
[![](https://img.shields.io/badge/docs-API-blue)](https://quantum-hub.baidu.com/docs/qnet/)
[![](https://img.shields.io/badge/Python-3.8+-green)](https://www.python.org/)
![](https://img.shields.io/badge/OS-MacOS%20|%20Windows%20|%20Linux-green)
[![](https://img.shields.io/badge/license-Apache%202.0-orange)](https://github.com/baidu/QCompute/blob/master/LICENSE)

## About QNET

QNET is a Quantum NETwork toolkit developed by the [Institute for Quantum Computing](https://quantum.baidu.com) at [Baidu Research](http://research.baidu.com/). It aims to accelerate the design of quantum network protocols, the testing of quantum network architectures and the deployment of quantum internet standards. QNET provides a fully-featured discrete-event simulation framework that allows for both accurate and efficient tracking of quantum network status. Its modular design provides a testbed for different quantum network architectures.


## Features

QNET is under active development and the latest version has the following key features:
* discrete-event simulation framework that allows for both accurate and efficient system tracking;
* quantum hardware interface that accelerates protocols testing and deployment;
* physical devices modeling that supports the simulation of realistic experiments;
* frequently-used templates that speed up the workflow of research and development.
* modular design that is compatible with different quantum network architectures;

## Installation

### Create Python environment

We recommend using Anaconda as the development environment management tool for Python3. Anaconda supports multiple mainstream operating systems (Windows, macOS, and Linux). It provides Scipy, Numpy, Matplotlib, and many other scientific computing and drawing packages. Conda, a virtual environment manager, can be used to install and update the latest Python packages. Here we give simple instructions on how to use conda to create and manage virtual environments.

1. First, open the command line (Terminal) interface: Windows users can enter ``Anaconda Prompt``; MacOS users can use the key combination ``command + space`` and then enter ``Terminal``.

2. After opening the Terminal window, enter
```bash
conda create --name qnet_env python=3.8
```
to create a Python 3.8 environment named qnet_env. With the following command, we can activate the virtual environment created,
```bash
conda activate qnet_env
```

For more detailed instructions on conda, please refer to the [Official Tutorial](https://docs.conda.io/projects/conda/en/latest/user-guide/getting-started.html).


### Install QNET

QNET is compatible with 64-bit Python 3.8+, on Linux, MacOS (10.14 or later) and Windows. We recommend installing it with ``pip``. Activate the conda environment and enter
```bash
pip install qcompute-qnet
```
This will install the QNET binaries as well as the QNET package. For those using an older version of QNET, keep up to date by installing with the ``--upgrade`` flag for additional features and bug fixes.

### Run examples

Now, you can try to write a simple program to check whether QNET has been successfully installed. For example,
```python
from qcompute_qnet.core.des import DESEnv
from qcompute_qnet.topology import Network, Node, Link

# Create a simulation environment
env = DESEnv("Simulation Environment", default=True)
# Create a network
network = Network("First Network")
# Create a node named Alice
alice = Node("Alice")
# Create another node named Bob
bob = Node("Bob")
# Create a link between Alice and Bob
link = Link("Alice_Bob", ends=(alice, bob))
# Build up the network from nodes and links
network.install([alice, bob, link])
# Initialize the simulation environment
env.init()
# Run the network simulation
env.run()
```

Note that more examples are provided in the [API documentation](https://quantum-hub.baidu.com/docs/qnet/). You can get started from there.

## Tutorials

We provide several [tutorials](https://quantum-hub.baidu.com/qnet/tutorial-introduction) to help users get started with QNET. These include:
* Introduction to discrete-event simulation
* Tour guide to quantum network simulation
* Micius quantum satellite experiment
* Quantum network architecture simulation
* Quantum network protocols on quantum hardware devices
* Quantum teleportation
* Quantum entanglement swapping
* CHSH game
* Magic square game

More tutorials and demonstrations will be included in the future release.

## API documentation

For those who are looking for explanation on the python classes and functions provided in QNET, please refer to our [API documentation](https://quantum-hub.baidu.com/docs/qnet/).


## Feedbacks

Users are encouraged to contact us via email quantum@baidu.com with general questions, unfixed bugs, and potential improvements. We hope to make QNET better together with the community!


## Research based on QNET

We also encourage researchers and developers to use QNET to explore quantum networks. If your work uses QNET, feel free to send us a notice via quantum@baidu.com and cite us with the following BibTeX:

```BibTex
@misc{QNET,
      title = {{Quantum NETwork in Baidu Quantum Platform}},
      year = {2022},
      url = {https://quantum-hub.baidu.com/qnet/}
}
```

## Copyright and License

QNET uses [Apache-2.0 license](https://github.com/baidu/QCompute/blob/master/LICENSE).
